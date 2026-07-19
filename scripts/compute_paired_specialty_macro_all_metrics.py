#!/usr/bin/env python3
"""Paired specialty-macro CIs for accuracy, F1, and AUC.

This script compares MedPMC-CLIP with BMC-CLIP across the benchmarks selected
in benchmark_manifest.csv.

It does not rerun model inference.

- Accuracy and AUC bootstrap arrays are reused from the existing dataset-level
  CI outputs produced by compute_notebook_auc_ci.py and
  compute_microbench_legacy_ci.py.
- F1 is recalculated from the saved prediction files using the original
  notebook definitions and the same bootstrap design:
    * binary/multiclass: macro F1 across classes;
    * multilabel: positive-class F1 for each label, macro-averaged over labels;
    * MicroBench: macro F1 across answer-option indices, with resampling at the
      image level.
- The same resampled observations are applied to both models within each
  benchmark, yielding paired bootstrap differences.
- Benchmark metrics are averaged within specialty, followed by an unweighted
  macro-average across specialties.

Only trusted local PKL prediction files should be supplied because loading a
pickle can execute arbitrary code.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed


TRUTHY = {"1", "true", "yes", "y"}
METRICS = ("accuracy", "f1", "auc")


def _scalar(value: np.ndarray | Any) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.ndim == 0 or array.size == 1 else str(value)


def read_result_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    output: dict[str, dict[str, str]] = {}
    for row in rows:
        dataset = row.get("dataset", "").strip()
        if not dataset or dataset.upper().startswith("OVERALL"):
            continue
        if dataset in output:
            raise ValueError(f"Duplicate dataset in {path}: {dataset}")
        output[dataset] = row
    return output


def combine_results(
    nonmicro_path: Path,
    micro_path: Path,
) -> dict[str, tuple[dict[str, str], Path, str]]:
    combined: dict[str, tuple[dict[str, str], Path, str]] = {}

    for dataset, row in read_result_rows(nonmicro_path).items():
        combined[dataset] = (row, nonmicro_path, "nonmicrobench")

    if micro_path.exists():
        for dataset, row in read_result_rows(micro_path).items():
            if dataset in combined:
                raise ValueError(
                    "Duplicate dataset across non-MicroBench and MicroBench: "
                    f"{dataset}"
                )
            combined[dataset] = (row, micro_path, "microbench")

    return combined


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")

    required = {"include", "dataset", "specialty"}
    if not required.issubset(rows[0].keys()):
        raise ValueError(
            "Manifest must contain include,dataset,specialty columns"
        )

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row["include"].strip().lower() not in TRUTHY:
            continue

        dataset = row["dataset"].strip()
        specialty = row["specialty"].strip()
        if not dataset:
            raise ValueError("Included manifest row has an empty dataset")
        if not specialty:
            raise ValueError(
                f"Included dataset has no specialty assignment: {dataset}"
            )
        if dataset in seen:
            raise ValueError(f"Duplicate included dataset: {dataset}")

        seen.add(dataset)
        selected.append({"dataset": dataset, "specialty": specialty})

    if not selected:
        raise ValueError("No benchmarks are selected in the manifest")
    return selected


def resolve_local_path(raw: str, csv_path: Path, search_root: Path) -> Path:
    if not raw.strip():
        raise ValueError(f"Empty path in {csv_path}")

    candidate = Path(raw)
    attempts = [candidate, csv_path.parent / candidate, search_root / candidate]
    for attempt in attempts:
        if attempt.exists():
            return attempt.resolve()

    matches = list(search_root.rglob(candidate.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Ambiguous filename {candidate.name}; matches: {matches}"
        )
    raise FileNotFoundError(f"File not found: {raw}")


def load_bootstrap_metric(
    row: dict[str, str],
    csv_path: Path,
    search_root: Path,
    metric: str,
) -> np.ndarray:
    raw = row.get("bootstrap_path", "").strip()
    if not raw:
        raise ValueError(
            f"{row.get('dataset', '<unknown>')}: bootstrap_path is empty"
        )

    path = resolve_local_path(raw, csv_path, search_root)
    with np.load(path, allow_pickle=False) as payload:
        if metric not in payload.files:
            raise KeyError(f"{path}: missing bootstrap array '{metric}'")
        values = np.asarray(payload[metric], dtype=np.float64)

    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{path}: invalid '{metric}' bootstrap array")
    return values


def normalize_binary_scores(y_score: np.ndarray) -> np.ndarray:
    score = np.asarray(y_score, dtype=np.float64)
    if score.ndim == 1:
        return np.column_stack([1.0 - score, score])
    if score.ndim == 2 and score.shape[1] == 1:
        positive = score[:, 0]
        return np.column_stack([1.0 - positive, positive])
    if score.ndim == 2 and score.shape[1] == 2:
        return score
    raise ValueError(f"Invalid binary y_score shape: {score.shape}")


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray | None = None,
) -> float:
    """Equivalent to sklearn f1_score(..., average='macro')."""
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if len(true) != len(pred):
        raise ValueError("y_true/y_pred length mismatch")
    if len(true) == 0:
        raise ValueError("Cannot calculate F1 on an empty sample")

    if classes is None:
        classes = np.union1d(np.unique(true), np.unique(pred))
    else:
        classes = np.asarray(classes)

    values: list[float] = []
    for cls in classes:
        true_positive = np.sum((true == cls) & (pred == cls))
        false_positive = np.sum((true != cls) & (pred == cls))
        false_negative = np.sum((true == cls) & (pred != cls))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(
            0.0 if denominator == 0 else float(2 * true_positive / denominator)
        )
    return float(np.mean(values))


def multilabel_macro_positive_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean positive-class binary F1 across labels, matching the notebook."""
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.ndim != 2 or pred.shape != true.shape:
        raise ValueError(
            f"Expected matching multilabel matrices; got {true.shape}, {pred.shape}"
        )

    true_positive = np.sum((true == 1) & (pred == 1), axis=0)
    false_positive = np.sum((true == 0) & (pred == 1), axis=0)
    false_negative = np.sum((true == 1) & (pred == 0), axis=0)
    denominator = 2 * true_positive + false_positive + false_negative

    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )

    # The original multilabel notebook assigned F1=0 when a resampled label
    # had only one ground-truth class.
    has_positive = np.any(true == 1, axis=0)
    has_negative = np.any(true == 0, axis=0)
    f1[~(has_positive & has_negative)] = 0.0
    return float(np.mean(f1))


def stratified_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    labels = np.asarray(labels).reshape(-1)
    for value in np.unique(labels):
        class_indices = np.flatnonzero(labels == value)
        parts.append(
            rng.choice(class_indices, size=len(class_indices), replace=True)
        )
    sampled = np.concatenate(parts)
    rng.shuffle(sampled)
    return sampled


def grouped_indices(
    group_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = np.unique(group_ids)
    sampled_groups = rng.choice(groups, size=len(groups), replace=True)
    return np.concatenate(
        [np.flatnonzero(group_ids == group) for group in sampled_groups]
    )


def load_nonmicro_prediction(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "y_score", "task"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path}: missing keys {sorted(missing)}")

        task = _scalar(data["task"]).lower()
        y_true = np.asarray(data["y_true"])
        y_score = np.asarray(data["y_score"], dtype=np.float64)
        group_ids = (
            np.asarray(data["group_ids"])
            if "group_ids" in data.files
            else None
        )
        dataset = (
            _scalar(data["dataset"])
            if "dataset" in data.files
            else path.stem
        )

    if task == "binary":
        y_pred = normalize_binary_scores(y_score).argmax(axis=1)
    elif task == "multiclass":
        if y_score.ndim != 2:
            raise ValueError(f"{path}: multiclass y_score must be 2D")
        y_pred = y_score.argmax(axis=1)
    elif task == "multilabel":
        if y_true.ndim != 2 or y_score.shape != y_true.shape:
            raise ValueError(
                f"{path}: multilabel shapes differ: {y_true.shape}, {y_score.shape}"
            )
        y_pred = (y_score >= 0.5).astype(np.int64)
    else:
        raise ValueError(f"{path}: unsupported task '{task}'")

    if len(y_true) != len(y_pred):
        raise ValueError(f"{path}: prediction length mismatch")
    if group_ids is not None and len(group_ids) != len(y_true):
        raise ValueError(f"{path}: group_ids length mismatch")

    return {
        "dataset": dataset,
        "task": task,
        "y_true": y_true,
        "y_pred": y_pred,
        "group_ids": group_ids,
    }


def nonmicro_point_f1(payload: dict[str, Any]) -> float:
    if payload["task"] == "multilabel":
        return multilabel_macro_positive_f1(
            payload["y_true"], payload["y_pred"]
        )
    return macro_f1(payload["y_true"], payload["y_pred"])


def bootstrap_nonmicro_f1_pair(
    medpmc: dict[str, Any],
    bmc: dict[str, Any],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if medpmc["task"] != bmc["task"]:
        raise ValueError(
            f"Task mismatch: {medpmc['task']} vs {bmc['task']}"
        )
    if not np.array_equal(medpmc["y_true"], bmc["y_true"]):
        raise ValueError("Ground-truth arrays differ between models")

    medpmc_group = medpmc["group_ids"]
    bmc_group = bmc["group_ids"]
    if (medpmc_group is None) != (bmc_group is None):
        raise ValueError("Only one model contains group_ids")
    if medpmc_group is not None and not np.array_equal(medpmc_group, bmc_group):
        raise ValueError("group_ids differ between models")

    y_true = medpmc["y_true"]
    task = medpmc["task"]
    rng = np.random.default_rng(seed)
    medpmc_values = np.empty(n_bootstrap, dtype=np.float64)
    bmc_values = np.empty(n_bootstrap, dtype=np.float64)

    # Stratification preserves every ground-truth class for binary/multiclass.
    fixed_classes = None
    if task == "binary":
        fixed_classes = np.asarray([0, 1], dtype=np.int64)
    elif task == "multiclass":
        fixed_classes = np.unique(y_true.reshape(-1))

    for b in range(n_bootstrap):
        if medpmc_group is not None:
            indices = grouped_indices(medpmc_group, rng)
        elif task in {"binary", "multiclass"}:
            indices = stratified_indices(y_true.reshape(-1), rng)
        else:
            indices = rng.integers(0, len(y_true), size=len(y_true))

        if task == "multilabel":
            medpmc_values[b] = multilabel_macro_positive_f1(
                y_true[indices], medpmc["y_pred"][indices]
            )
            bmc_values[b] = multilabel_macro_positive_f1(
                y_true[indices], bmc["y_pred"][indices]
            )
        else:
            medpmc_values[b] = macro_f1(
                y_true[indices],
                medpmc["y_pred"][indices],
                classes=fixed_classes,
            )
            bmc_values[b] = macro_f1(
                y_true[indices],
                bmc["y_pred"][indices],
                classes=fixed_classes,
            )

    return medpmc_values, bmc_values


def load_micro_prediction(
    path: Path,
    *,
    n_caption_types: int,
) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected dictionary payload")
    if "answer_idxs" not in data or "similarity_scores" not in data:
        raise KeyError(
            f"{path}: answer_idxs and similarity_scores are required"
        )

    answer_idxs = np.asarray(data["answer_idxs"], dtype=np.int64).reshape(-1)
    scores = [
        np.asarray(score, dtype=np.float64).reshape(-1)
        for score in list(data["similarity_scores"])
    ]
    if len(answer_idxs) != len(scores) or not scores:
        raise ValueError(f"{path}: invalid answer/score lengths")

    for index, (answer, score) in enumerate(zip(answer_idxs, scores)):
        if score.size < 2 or answer < 0 or answer >= score.size:
            raise ValueError(f"{path}: invalid question {index}")

    predictions = np.asarray(
        data.get("preds", [int(score.argmax()) for score in scores]),
        dtype=np.int64,
    ).reshape(-1)
    if len(predictions) != len(scores):
        predictions = np.asarray(
            [int(score.argmax()) for score in scores], dtype=np.int64
        )

    if "question_sample_indices" in data:
        question_sample_indices = np.asarray(
            data["question_sample_indices"], dtype=np.int64
        ).reshape(-1)
        if len(question_sample_indices) != len(scores):
            raise ValueError(
                f"{path}: question_sample_indices length mismatch"
            )
        n_images = int(
            data.get("n_images", question_sample_indices.max() + 1)
        )
    else:
        if len(scores) % n_caption_types != 0:
            raise ValueError(
                f"{path}: question count is not divisible by "
                f"n_caption_types={n_caption_types}"
            )
        n_images = len(scores) // n_caption_types
        question_sample_indices = np.tile(
            np.arange(n_images, dtype=np.int64), n_caption_types
        )

    image_to_questions = [
        np.flatnonzero(question_sample_indices == image_index)
        for image_index in range(n_images)
    ]
    if any(len(indices) == 0 for indices in image_to_questions):
        raise ValueError(f"{path}: at least one image has no questions")

    return {
        "dataset": str(data.get("dataset", path.stem)),
        "answer_idxs": answer_idxs,
        "predictions": predictions,
        "question_sample_indices": question_sample_indices,
        "n_images": n_images,
        "image_to_questions": image_to_questions,
        "option_counts": np.asarray([score.size for score in scores]),
    }


def bootstrap_micro_f1_pair(
    medpmc: dict[str, Any],
    bmc: dict[str, Any],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.array_equal(medpmc["answer_idxs"], bmc["answer_idxs"]):
        raise ValueError("MicroBench ground-truth arrays differ between models")
    if medpmc["n_images"] != bmc["n_images"]:
        raise ValueError("MicroBench image counts differ between models")
    if not np.array_equal(
        medpmc["question_sample_indices"], bmc["question_sample_indices"]
    ):
        raise ValueError("MicroBench image-question mappings differ")
    if not np.array_equal(medpmc["option_counts"], bmc["option_counts"]):
        raise ValueError("MicroBench option counts differ between models")

    rng = np.random.default_rng(seed)
    n_images = medpmc["n_images"]
    image_to_questions = medpmc["image_to_questions"]
    labels = medpmc["answer_idxs"]

    medpmc_values = np.empty(n_bootstrap, dtype=np.float64)
    bmc_values = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        sampled_images = rng.integers(0, n_images, size=n_images)
        question_indices = np.concatenate(
            [image_to_questions[int(i)] for i in sampled_images]
        )
        selected_labels = labels[question_indices]

        # No explicit labels were passed in the original notebook, so use the
        # classes appearing in y_true or y_pred for each model.
        medpmc_values[b] = macro_f1(
            selected_labels,
            medpmc["predictions"][question_indices],
        )
        bmc_values[b] = macro_f1(
            selected_labels,
            bmc["predictions"][question_indices],
        )

    return medpmc_values, bmc_values


def process_dataset(
    item: dict[str, str],
    medpmc_entry: tuple[dict[str, str], Path, str],
    bmc_entry: tuple[dict[str, str], Path, str],
    *,
    search_root: Path,
    n_bootstrap: int,
    seed: int,
    n_caption_types: int,
) -> dict[str, Any]:
    dataset = item["dataset"]
    specialty = item["specialty"]
    medpmc_row, medpmc_csv, medpmc_source = medpmc_entry
    bmc_row, bmc_csv, bmc_source = bmc_entry

    if medpmc_source != bmc_source:
        raise ValueError(
            f"{dataset}: source mismatch ({medpmc_source} vs {bmc_source})"
        )

    medpmc_n = medpmc_row.get("n") or medpmc_row.get("n_images")
    bmc_n = bmc_row.get("n") or bmc_row.get("n_images")
    if medpmc_n and bmc_n and int(float(medpmc_n)) != int(float(bmc_n)):
        raise ValueError(
            f"{dataset}: sample counts differ ({medpmc_n} vs {bmc_n})"
        )

    accuracy_medpmc_boot = load_bootstrap_metric(
        medpmc_row, medpmc_csv, search_root, "accuracy"
    )
    accuracy_bmc_boot = load_bootstrap_metric(
        bmc_row, bmc_csv, search_root, "accuracy"
    )
    auc_medpmc_boot = load_bootstrap_metric(
        medpmc_row, medpmc_csv, search_root, "auc"
    )
    auc_bmc_boot = load_bootstrap_metric(
        bmc_row, bmc_csv, search_root, "auc"
    )

    lengths = {
        len(accuracy_medpmc_boot),
        len(accuracy_bmc_boot),
        len(auc_medpmc_boot),
        len(auc_bmc_boot),
    }
    if lengths != {n_bootstrap}:
        raise ValueError(
            f"{dataset}: expected bootstrap length {n_bootstrap}, got {lengths}"
        )

    medpmc_prediction_path = resolve_local_path(
        medpmc_row.get("path", ""), medpmc_csv, search_root
    )
    bmc_prediction_path = resolve_local_path(
        bmc_row.get("path", ""), bmc_csv, search_root
    )

    dataset_seed = (seed + zlib.crc32(dataset.encode("utf-8"))) % (2**32)

    if medpmc_source == "nonmicrobench":
        medpmc_prediction = load_nonmicro_prediction(medpmc_prediction_path)
        bmc_prediction = load_nonmicro_prediction(bmc_prediction_path)
        if medpmc_prediction["dataset"] != bmc_prediction["dataset"]:
            raise ValueError(
                f"{dataset}: prediction dataset names differ: "
                f"{medpmc_prediction['dataset']} vs {bmc_prediction['dataset']}"
            )
        medpmc_f1 = nonmicro_point_f1(medpmc_prediction)
        bmc_f1 = nonmicro_point_f1(bmc_prediction)
        medpmc_f1_boot, bmc_f1_boot = bootstrap_nonmicro_f1_pair(
            medpmc_prediction,
            bmc_prediction,
            n_bootstrap=n_bootstrap,
            seed=dataset_seed,
        )
    else:
        medpmc_prediction = load_micro_prediction(
            medpmc_prediction_path, n_caption_types=n_caption_types
        )
        bmc_prediction = load_micro_prediction(
            bmc_prediction_path, n_caption_types=n_caption_types
        )
        if medpmc_prediction["dataset"] != bmc_prediction["dataset"]:
            raise ValueError(
                f"{dataset}: prediction dataset names differ: "
                f"{medpmc_prediction['dataset']} vs {bmc_prediction['dataset']}"
            )
        medpmc_f1 = macro_f1(
            medpmc_prediction["answer_idxs"], medpmc_prediction["predictions"]
        )
        bmc_f1 = macro_f1(
            bmc_prediction["answer_idxs"], bmc_prediction["predictions"]
        )
        medpmc_f1_boot, bmc_f1_boot = bootstrap_micro_f1_pair(
            medpmc_prediction,
            bmc_prediction,
            n_bootstrap=n_bootstrap,
            seed=dataset_seed,
        )

    return {
        "dataset": dataset,
        "specialty": specialty,
        "source": medpmc_source,
        "points": {
            "accuracy": (
                float(medpmc_row["accuracy"]),
                float(bmc_row["accuracy"]),
            ),
            "f1": (medpmc_f1, bmc_f1),
            "auc": (float(medpmc_row["auc"]), float(bmc_row["auc"])),
        },
        "delta_boot": {
            "accuracy": accuracy_medpmc_boot - accuracy_bmc_boot,
            "f1": medpmc_f1_boot - bmc_f1_boot,
            "auc": auc_medpmc_boot - auc_bmc_boot,
        },
    }


def aggregate_results(
    processed: list[dict[str, Any]],
    *,
    confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    by_specialty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in processed:
        by_specialty[result["specialty"]].append(result)

    alpha = 1.0 - confidence
    quantiles = [alpha / 2.0, 1.0 - alpha / 2.0]
    rows: list[dict[str, Any]] = []
    overall_arrays: dict[str, np.ndarray] = {}

    for metric in METRICS:
        specialty_rows: list[dict[str, Any]] = []
        specialty_bootstraps: list[np.ndarray] = []

        for specialty, datasets in sorted(by_specialty.items()):
            medpmc_value = float(
                np.mean([result["points"][metric][0] for result in datasets])
            )
            bmc_value = float(
                np.mean([result["points"][metric][1] for result in datasets])
            )
            delta = medpmc_value - bmc_value
            delta_boot = np.mean(
                np.stack(
                    [result["delta_boot"][metric] for result in datasets],
                    axis=0,
                ),
                axis=0,
            )
            lower, upper = np.quantile(delta_boot, quantiles)

            row = {
                "metric": metric,
                "specialty": specialty,
                "n_benchmarks": len(datasets),
                "benchmarks": ";".join(
                    result["dataset"] for result in datasets
                ),
                "medpmc_value": medpmc_value,
                "bmc_value": bmc_value,
                "delta": delta,
                "delta_percentage_points": delta * 100,
                "delta_ci_lower": float(lower),
                "delta_ci_upper": float(upper),
                "delta_ci_lower_percentage_points": float(lower * 100),
                "delta_ci_upper_percentage_points": float(upper * 100),
                "confidence": confidence,
                "n_bootstrap": len(delta_boot),
            }
            specialty_rows.append(row)
            specialty_bootstraps.append(delta_boot)

        overall_medpmc = float(
            np.mean([row["medpmc_value"] for row in specialty_rows])
        )
        overall_bmc = float(
            np.mean([row["bmc_value"] for row in specialty_rows])
        )
        overall_delta = overall_medpmc - overall_bmc
        overall_delta_boot = np.mean(
            np.stack(specialty_bootstraps, axis=0), axis=0
        )
        overall_lower, overall_upper = np.quantile(
            overall_delta_boot, quantiles
        )

        overall_row = {
            "metric": metric,
            "specialty": "OVERALL_SPECIALTY_MACRO_AVERAGE",
            "n_benchmarks": len(processed),
            "benchmarks": "",
            "medpmc_value": overall_medpmc,
            "bmc_value": overall_bmc,
            "delta": overall_delta,
            "delta_percentage_points": overall_delta * 100,
            "delta_ci_lower": float(overall_lower),
            "delta_ci_upper": float(overall_upper),
            "delta_ci_lower_percentage_points": float(overall_lower * 100),
            "delta_ci_upper_percentage_points": float(overall_upper * 100),
            "confidence": confidence,
            "n_bootstrap": len(overall_delta_boot),
        }

        rows.extend(specialty_rows)
        rows.append(overall_row)
        overall_arrays[metric] = overall_delta_boot

    return rows, overall_arrays


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        type=Path,
        default=Path("results/MedPMC-CLIP/nonmicrobench_notebook_auc_ci.csv"),
    )
    parser.add_argument(
        "--bmc-nonmicro",
        type=Path,
        default=Path("results/BMC/nonmicrobench_notebook_auc_ci.csv"),
    )
    parser.add_argument(
        type=Path,
        default=Path("results/MedPMC-CLIP/microbench_legacy_auc_ci.csv"),
    )
    parser.add_argument(
        "--bmc-micro",
        type=Path,
        default=Path("results/BMC/microbench_legacy_auc_ci.csv"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmark_manifest.csv")
    )
    parser.add_argument("--search-root", type=Path, default=Path("."))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--n-caption-types",
        type=int,
        default=2,
        help="Used for legacy MicroBench PKLs without question_sample_indices.",
    )
    parser.add_argument("--expected-benchmarks", type=int, default=26)
    parser.add_argument("--expected-specialties", type=int, default=11)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/MedPMC_vs_BMC_specialty_macro_paired_all_metrics.csv"
        ),
    )
    parser.add_argument(
        "--save-bootstrap",
        type=Path,
        default=Path(
            "results/MedPMC_vs_BMC_specialty_macro_paired_all_metrics.npz"
        ),
    )
    args = parser.parse_args()

    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if args.n_bootstrap < 1:
        raise ValueError("n-bootstrap must be positive")

    medpmc = combine_results(args.medpmc_nonmicro, args.medpmc_micro)
    bmc = combine_results(args.bmc_nonmicro, args.bmc_micro)
    selected = read_manifest(args.manifest)

    if args.expected_benchmarks > 0 and len(selected) != args.expected_benchmarks:
        raise ValueError(
            f"Expected {args.expected_benchmarks} benchmarks, got {len(selected)}"
        )
    specialties = sorted({item["specialty"] for item in selected})
    if args.expected_specialties > 0 and len(specialties) != args.expected_specialties:
        raise ValueError(
            f"Expected {args.expected_specialties} specialties, "
            f"got {len(specialties)}: {specialties}"
        )

    work_items = []
    for item in selected:
        dataset = item["dataset"]
        if dataset not in medpmc:
            raise KeyError(f"Missing MedPMC-CLIP result: {dataset}")
        if dataset not in bmc:
            raise KeyError(f"Missing BMC-CLIP result: {dataset}")
        work_items.append((item, medpmc[dataset], bmc[dataset]))

    processed = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(process_dataset)(
            item,
            medpmc_entry,
            bmc_entry,
            search_root=args.search_root,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            n_caption_types=args.n_caption_types,
        )
        for item, medpmc_entry, bmc_entry in work_items
    )

    rows, overall_arrays = aggregate_results(
        processed, confidence=args.confidence
    )
    write_csv(args.output, rows)

    args.save_bootstrap.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.save_bootstrap,
        accuracy_difference=overall_arrays["accuracy"],
        f1_difference=overall_arrays["f1"],
        auc_difference=overall_arrays["auc"],
        confidence=np.asarray(args.confidence),
        benchmarks=np.asarray([item["dataset"] for item in selected]),
        specialties=np.asarray([item["specialty"] for item in selected]),
    )

    print("\nOverall specialty-macro paired differences:")
    for metric in METRICS:
        row = next(
            row
            for row in rows
            if row["metric"] == metric
            and row["specialty"] == "OVERALL_SPECIALTY_MACRO_AVERAGE"
        )
        wins = sum(
            1
            for candidate in rows
            if candidate["metric"] == metric
            and candidate["specialty"] != "OVERALL_SPECIALTY_MACRO_AVERAGE"
            and candidate["delta"] > 0
        )
        print(
            f"  {metric.upper():<8} "
            f"{row['delta_percentage_points']:+.2f} pp "
            f"({args.confidence * 100:.1f}% CI, "
            f"{row['delta_ci_lower_percentage_points']:+.2f} to "
            f"{row['delta_ci_upper_percentage_points']:+.2f} pp); "
            f"higher in {wins}/{len(specialties)} specialties"
        )

    print(f"\nSaved table: {args.output}")
    print(f"Saved bootstrap arrays: {args.save_bootstrap}")


if __name__ == "__main__":
    main()
