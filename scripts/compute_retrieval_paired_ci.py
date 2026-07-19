#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

Z_95 = 1.959963984540054


def load_dump(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON list.")
    for i, record in enumerate(data):
        if "query_index" not in record or "retrieved" not in record:
            raise ValueError(f"{path}: record {i} lacks query_index or retrieved.")
        if not isinstance(record["retrieved"], list):
            raise ValueError(f"{path}: record {i} has a non-list retrieved field.")
    return data


def index_by_query(records: List[dict], path: str) -> Dict[int, dict]:
    indexed = {}
    for record in records:
        query_index = int(record["query_index"])
        if query_index in indexed:
            raise ValueError(f"{path}: duplicate query_index={query_index}")
        indexed[query_index] = record
    return indexed


def normalized_labels(record: dict) -> Tuple[str, ...]:
    return tuple(sorted(str(x) for x in record.get("query_labels", [])))


def validate_alignment(indexed_a, indexed_b, path_a, path_b):
    keys_a = set(indexed_a)
    keys_b = set(indexed_b)
    if keys_a != keys_b:
        raise ValueError(
            "The two files do not contain the same query_index values.\n"
            f"Only in {path_a}: {sorted(keys_a - keys_b)[:10]}\n"
            f"Only in {path_b}: {sorted(keys_b - keys_a)[:10]}"
        )

    query_indices = sorted(keys_a)
    for query_index in query_indices:
        a = indexed_a[query_index]
        b = indexed_b[query_index]

        query_path_a = str(a.get("query_image_path", ""))
        query_path_b = str(b.get("query_image_path", ""))
        if query_path_a and query_path_b and query_path_a != query_path_b:
            raise ValueError(
                f"query_image_path mismatch at query_index={query_index}"
            )

        if normalized_labels(a) != normalized_labels(b):
            raise ValueError(f"query_labels mismatch at query_index={query_index}")

        if a.get("mode_for_relevance") != b.get("mode_for_relevance"):
            raise ValueError(
                f"mode_for_relevance mismatch at query_index={query_index}"
            )

        if a.get("jaccard_threshold") != b.get("jaccard_threshold"):
            raise ValueError(
                f"jaccard_threshold mismatch at query_index={query_index}"
            )

    return query_indices


def hit_at_k(record: dict, k: int) -> int:
    retrieved = sorted(record["retrieved"], key=lambda x: int(x["rank"]))
    top_k = [item for item in retrieved if int(item["rank"]) <= k]
    if len(top_k) < k:
        raise ValueError(
            f"query_index={record['query_index']} has fewer than {k} retrieved items."
        )
    return int(any(float(x.get("relevance_under_mode", 0.0)) > 0 for x in top_k))


def build_hits(indexed, query_indices, k):
    return np.asarray(
        [hit_at_k(indexed[q], k) for q in query_indices],
        dtype=np.int8,
    )


def wilson_interval(successes: int, total: int, z: float = Z_95):
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - margin, center + margin


def paired_bootstrap(hits_a, hits_b, n_bootstrap, confidence, seed, chunk_size=500):
    if hits_a.shape != hits_b.shape:
        raise ValueError("Paired hit arrays must have identical shapes.")
    differences = hits_a.astype(float) - hits_b.astype(float)
    observed = float(differences.mean())

    rng = np.random.default_rng(seed)
    n = len(differences)
    boot = np.empty(n_bootstrap, dtype=float)

    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = differences[indices].mean(axis=1)

    alpha = 1 - confidence
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return observed, float(low), float(high), boot


def write_csv(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {output}")


def write_npz(path, arrays, metadata):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata), dtype=str),
    )
    print(f"Bootstrap arrays written: {output}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Recall@K for two aligned retrieval dumps and paired-bootstrap "
            "confidence intervals for Model A minus Model B."
        )
    )
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--label-a", default="MedPMC-CLIP")
    parser.add_argument("--label-b", default="BMC-CLIP")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-csv",
        default="retrieval_paired_bootstrap_ci.csv",
    )
    parser.add_argument(
        "--output-npz",
        default="retrieval_paired_bootstrap_ci.npz",
    )
    args = parser.parse_args()

    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be at least 1.")
    if not 0 < args.confidence < 1:
        raise ValueError("--confidence must be between 0 and 1.")
    if any(k < 1 for k in args.ks) or len(set(args.ks)) != len(args.ks):
        raise ValueError("--ks must contain unique positive integers.")

    records_a = load_dump(args.model_a)
    records_b = load_dump(args.model_b)
    indexed_a = index_by_query(records_a, args.model_a)
    indexed_b = index_by_query(records_b, args.model_b)
    query_indices = validate_alignment(
        indexed_a, indexed_b, args.model_a, args.model_b
    )

    first = indexed_a[query_indices[0]]
    relevance_mode = first.get("mode_for_relevance", "unknown")
    threshold = first.get("jaccard_threshold", "unknown")

    print(f"Aligned queries: {len(query_indices)}")
    print(f"Relevance mode: {relevance_mode}")
    print(f"Jaccard threshold: {threshold}")
    print(f"Difference direction: {args.label_a} - {args.label_b}")

    rows = []
    arrays = {}

    for offset, k in enumerate(args.ks):
        hits_a = build_hits(indexed_a, query_indices, k)
        hits_b = build_hits(indexed_b, query_indices, k)
        n = len(query_indices)

        successes_a = int(hits_a.sum())
        successes_b = int(hits_b.sum())
        recall_a = successes_a / n
        recall_b = successes_b / n
        a_low, a_high = wilson_interval(successes_a, n)
        b_low, b_high = wilson_interval(successes_b, n)

        delta, ci_low, ci_high, boot = paired_bootstrap(
            hits_a,
            hits_b,
            args.n_bootstrap,
            args.confidence,
            args.seed + offset,
        )

        rows.append(
            {
                "metric": f"Recall@{k}",
                "n_queries": n,
                "relevance_mode": relevance_mode,
                f"{args.label_a}_hits": successes_a,
                f"{args.label_a}_recall": recall_a,
                f"{args.label_a}_wilson_ci_lower": a_low,
                f"{args.label_a}_wilson_ci_upper": a_high,
                f"{args.label_b}_hits": successes_b,
                f"{args.label_b}_recall": recall_b,
                f"{args.label_b}_wilson_ci_lower": b_low,
                f"{args.label_b}_wilson_ci_upper": b_high,
                "difference_a_minus_b": delta,
                "paired_bootstrap_ci_lower": ci_low,
                "paired_bootstrap_ci_upper": ci_high,
                "both_hit": int(np.sum((hits_a == 1) & (hits_b == 1))),
                "a_only_hit": int(np.sum((hits_a == 1) & (hits_b == 0))),
                "b_only_hit": int(np.sum((hits_a == 0) & (hits_b == 1))),
                "neither_hit": int(np.sum((hits_a == 0) & (hits_b == 0))),
            }
        )
        arrays[f"recall_at_{k}_difference"] = boot

        print(
            f"Recall@{k}: "
            f"{args.label_a}={recall_a * 100:.2f}%, "
            f"{args.label_b}={recall_b * 100:.2f}%, "
            f"delta={delta * 100:+.2f} pp "
            f"({args.confidence * 100:.1f}% paired CI, "
            f"{ci_low * 100:+.2f} to {ci_high * 100:+.2f} pp)"
        )

    write_csv(args.output_csv, rows)
    write_npz(
        args.output_npz,
        arrays,
        {
            "model_a": args.model_a,
            "model_b": args.model_b,
            "label_a": args.label_a,
            "label_b": args.label_b,
            "n_queries": len(query_indices),
            "ks": args.ks,
            "n_bootstrap": args.n_bootstrap,
            "confidence": args.confidence,
            "seed": args.seed,
            "relevance_mode": relevance_mode,
            "jaccard_threshold": threshold,
            "bootstrap_unit": "query",
        },
    )


if __name__ == "__main__":
    main()
