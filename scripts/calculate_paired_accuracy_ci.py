#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


Z_95 = 1.959963984540054


def load_scores(path):
    """Load a JSON list or plain-text list containing only 0 and 1."""
    text = Path(path).read_text(encoding="utf-8").strip()

    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        values = [int(x) for x in re.findall(r"(?<!\d)[01](?!\d)", text)]

    if not isinstance(values, list):
        raise ValueError(f"Score file must contain a list: {path}")

    scores = np.asarray([int(x) for x in values], dtype=np.int8)
    if scores.size == 0:
        raise ValueError(f"No scores found in: {path}")
    if np.any((scores != 0) & (scores != 1)):
        raise ValueError(f"Scores must contain only 0 and 1: {path}")
    return scores


def load_types(path):
    """Load one question type per line or from a JSON list."""
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        values = json.loads(text)
        if isinstance(values, list):
            return [str(value).strip() for value in values]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_questions(path):
    """Load JSON, JSONL, or consecutive JSON objects."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        values = json.loads(text)
        if isinstance(values, list):
            return values
        if isinstance(values, dict):
            return [values]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    records = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        try:
            record, end_position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Could not parse {path} near character {position}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON objects in {path}")
        records.append(record)
        position = end_position
    return records


def get_dataset_name(question_id):
    question_id = str(question_id).strip()
    match = re.match(r"^(.*)_\d+$", question_id)
    return match.group(1) if match else question_id


def wilson_ci(correct, total, z=Z_95):
    """Wilson 95% confidence interval for one model's binomial accuracy."""
    if total == 0:
        return float("nan"), float("nan")
    proportion = correct / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - margin, center + margin


def is_lesion_grading(question_type):
    return str(question_type).strip().casefold() == "lesion grading"


def normalize_question_type(question_type):
    """Merge Lesion Grading into Disease Diagnosis."""
    question_type = str(question_type).strip()
    if is_lesion_grading(question_type):
        return "Disease Diagnosis"
    return question_type


def display_group_name(group):
    if group == "Disease Diagnosis":
        return "Disease Diagnosis + Lesion Grading"
    return group


def paired_bootstrap_difference(
    scores_a,
    scores_b,
    n_bootstrap=10000,
    confidence=0.95,
    seed=2026,
    chunk_size=500,
):
    """Paired percentile-bootstrap CI for accuracy difference A minus B."""
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    if scores_a.shape != scores_b.shape:
        raise ValueError("Paired score arrays must have identical shapes.")
    if scores_a.ndim != 1 or scores_a.size == 0:
        raise ValueError("Paired score arrays must be non-empty 1D arrays.")
    if n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be at least 1.")
    if not 0 < confidence < 1:
        raise ValueError("--confidence must be between 0 and 1.")

    paired_differences = scores_a - scores_b
    observed_difference = float(paired_differences.mean())

    rng = np.random.default_rng(seed)
    bootstrap_differences = np.empty(n_bootstrap, dtype=np.float64)
    n = paired_differences.size

    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n))
        bootstrap_differences[start:stop] = paired_differences[indices].mean(axis=1)

    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(
        bootstrap_differences,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return observed_difference, float(ci_low), float(ci_high), bootstrap_differences


def summarize_group(
    group,
    indices,
    scores_a,
    scores_b,
    label_a,
    label_b,
    n_bootstrap,
    confidence,
    seed,
):
    a = scores_a[indices]
    b = scores_b[indices]
    n = len(indices)

    a_correct = int(a.sum())
    b_correct = int(b.sum())
    a_accuracy = a_correct / n
    b_accuracy = b_correct / n
    a_ci_low, a_ci_high = wilson_ci(a_correct, n)
    b_ci_low, b_ci_high = wilson_ci(b_correct, n)

    difference, diff_ci_low, diff_ci_high, bootstrap = paired_bootstrap_difference(
        a,
        b,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )

    row = {
        "group": display_group_name(group),
        "n": n,
        f"{label_a}_correct": a_correct,
        f"{label_a}_accuracy": a_accuracy,
        f"{label_a}_wilson_ci_lower": a_ci_low,
        f"{label_a}_wilson_ci_upper": a_ci_high,
        f"{label_b}_correct": b_correct,
        f"{label_b}_accuracy": b_accuracy,
        f"{label_b}_wilson_ci_lower": b_ci_low,
        f"{label_b}_wilson_ci_upper": b_ci_high,
        "difference_a_minus_b": difference,
        "paired_bootstrap_ci_lower": diff_ci_low,
        "paired_bootstrap_ci_upper": diff_ci_high,
    }
    return row, bootstrap


def print_results(rows, label_a, label_b, confidence):
    group_width = max(38, max(len(row["group"]) for row in rows))
    ci_percent = confidence * 100

    print(
        f"\n{'Group':<{group_width}} "
        f"{'N':>7} "
        f"{label_a:>14} "
        f"{label_b:>14} "
        f"{'Difference':>12} "
        f"{f'{ci_percent:.1f}% paired CI':>24}"
    )
    print("-" * (group_width + 78))

    for row in rows:
        print(
            f"{row['group']:<{group_width}} "
            f"{row['n']:7d} "
            f"{row[f'{label_a}_accuracy'] * 100:13.2f}% "
            f"{row[f'{label_b}_accuracy'] * 100:13.2f}% "
            f"{row['difference_a_minus_b'] * 100:+11.2f} "
            f"[{row['paired_bootstrap_ci_lower'] * 100:+7.2f}, "
            f"{row['paired_bootstrap_ci_upper'] * 100:+7.2f}]"
        )


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written: {path}")


def write_npz(path, bootstrap_by_group):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    group_names = []
    for index, (group, values) in enumerate(bootstrap_by_group.items()):
        key = f"group_{index:02d}"
        arrays[key] = values
        group_names.append(group)
    arrays["group_names"] = np.asarray(group_names, dtype=str)
    np.savez_compressed(path, **arrays)
    print(f"Bootstrap arrays written: {path}")


def build_groups_generic(n):
    return {"Overall": np.arange(n, dtype=np.int64)}


def build_groups_typed(scores_length, questions, question_types, inspect_drimdb=False):
    if len(questions) != scores_length:
        raise ValueError(
            f"Length mismatch: {scores_length} scores versus "
            f"{len(questions)} questions."
        )
    if len(question_types) != scores_length:
        raise ValueError(
            f"Length mismatch: {scores_length} scores versus "
            f"{len(question_types)} question types."
        )

    kept_indices = []
    normalized_types = []
    drimdb_total = 0
    excluded_drimdb_grading = 0
    retained_drimdb_by_type = defaultdict(int)

    for index, question in enumerate(questions):
        question_id = question.get("question_id", "")
        dataset = get_dataset_name(question_id)
        original_type = str(question_types[index]).strip()

        if dataset == "DRIMDB":
            drimdb_total += 1
            if inspect_drimdb:
                question_text = question.get(
                    "text",
                    question.get("prompt", question.get("question", "")),
                )
                print(
                    f"\n[DRIMDB] {question_id}\n"
                    f"Type: {original_type}\n"
                    f"{question_text}"
                )
            if is_lesion_grading(original_type):
                excluded_drimdb_grading += 1
                continue
            retained_drimdb_by_type[normalize_question_type(original_type)] += 1

        kept_indices.append(index)
        normalized_types.append(normalize_question_type(original_type))

    if not kept_indices:
        raise ValueError(
            "No samples remain after excluding DRIMDB Lesion Grading items."
        )

    print("Mode: typed paired evaluation with OmniMedVQA filtering")
    print(f"Total DRIMDB samples: {drimdb_total}")
    print(f"Excluded DRIMDB Lesion Grading: {excluded_drimdb_grading} samples")
    print(f"Included samples: {len(kept_indices)}")

    if retained_drimdb_by_type:
        retained_text = ", ".join(
            f"{question_type}={count}"
            for question_type, count in sorted(retained_drimdb_by_type.items())
        )
        print(f"Retained DRIMDB by final type: {retained_text}")

    groups = {"Overall": np.asarray(kept_indices, dtype=np.int64)}
    indices_by_type = defaultdict(list)
    for index, question_type in zip(kept_indices, normalized_types):
        indices_by_type[question_type].append(index)

    for question_type, indices in sorted(
        indices_by_type.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        groups[question_type] = np.asarray(indices, dtype=np.int64)
    return groups


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two aligned 0/1 correctness-score files using paired "
            "bootstrap confidence intervals for accuracy differences. When "
            "--types is supplied, apply the OmniMedVQA rule: exclude only "
            "DRIMDB Lesion Grading samples, merge all remaining Lesion Grading "
            "samples into Disease Diagnosis, and report overall and category-level "
            "paired results. Without --types, report overall performance only."
        )
    )

    parser.add_argument("--scores-a", required=True, help="Scores for model A.")
    parser.add_argument("--scores-b", required=True, help="Scores for model B.")
    parser.add_argument("--label-a", default="MedPMC", help="Short model-A label.")
    parser.add_argument("--label-b", default="Baseline", help="Short model-B label.")
    parser.add_argument(
        "--types",
        help=(
            "Optional file containing one question type per sample. Supplying "
            "this enables OmniMedVQA filtering and category reporting."
        ),
    )
    parser.add_argument(
        "--questions",
        help=(
            "Original question JSON/JSONL. Required when --types is supplied; "
            "optional otherwise for length validation."
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of paired bootstrap replicates (default: 10000).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level as a fraction (default: 0.95).",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--output-csv", help="Optional summary CSV path.")
    parser.add_argument("--output-npz", help="Optional bootstrap-array NPZ path.")
    parser.add_argument(
        "--inspect-drimdb",
        action="store_true",
        help="Print DRIMDB samples before filtering in typed mode.",
    )

    args = parser.parse_args()
    scores_a = load_scores(args.scores_a)
    scores_b = load_scores(args.scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Paired files must have the same length: "
            f"{len(scores_a)} versus {len(scores_b)}."
        )

    questions = load_questions(args.questions) if args.questions else None

    if args.types:
        if questions is None:
            raise ValueError(
                "--questions is required when --types is supplied because "
                "DRIMDB samples are identified using question_id."
            )
        question_types = load_types(args.types)
        groups = build_groups_typed(
            scores_length=len(scores_a),
            questions=questions,
            question_types=question_types,
            inspect_drimdb=args.inspect_drimdb,
        )
    else:
        if args.inspect_drimdb:
            raise ValueError("--inspect-drimdb requires --types and --questions.")
        if questions is not None and len(questions) != len(scores_a):
            raise ValueError(
                f"Length mismatch: {len(scores_a)} scores versus "
                f"{len(questions)} questions."
            )
        print("Mode: generic paired evaluation")
        print(f"Included samples: {len(scores_a)}")
        groups = build_groups_generic(len(scores_a))

    rows = []
    bootstrap_by_group = {}
    for group_number, (group, indices) in enumerate(groups.items()):
        row, bootstrap = summarize_group(
            group=group,
            indices=indices,
            scores_a=scores_a,
            scores_b=scores_b,
            label_a=args.label_a,
            label_b=args.label_b,
            n_bootstrap=args.n_bootstrap,
            confidence=args.confidence,
            seed=args.seed + group_number,
        )
        rows.append(row)
        bootstrap_by_group[row["group"]] = bootstrap

    print_results(rows, args.label_a, args.label_b, args.confidence)

    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.output_npz:
        write_npz(args.output_npz, bootstrap_by_group)


if __name__ == "__main__":
    main()
