from __future__ import annotations

import csv
import json
import os
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm.auto import tqdm


METRIC_NAMES = ("accuracy", "f1", "auc")


def notebook_metrics(y_true: np.ndarray, y_score: np.ndarray, task: str) -> tuple[float, float, float]:
    """Return point metrics using the original notebook-compatible definitions."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if task in {"binary", "multiclass"}:
        labels = y_true.reshape(-1).astype(int)
        pred = np.argmax(y_score, axis=1)
        accuracy = float(accuracy_score(labels, pred))
        f1 = float(f1_score(labels, pred, average="macro", zero_division=0))
        one_hot = np.eye(y_score.shape[1], dtype=np.uint8)[labels]
        auc = float(roc_auc_score(one_hot.reshape(-1), y_score.reshape(-1)))
        return accuracy, f1, auc

    if task == "multilabel":
        truth = y_true.astype(np.uint8)
        pred = (y_score >= 0.5).astype(np.uint8)
        accuracy = float(np.mean(pred == truth))
        f1s = []
        aucs = []
        for column in range(truth.shape[1]):
            f1s.append(
                f1_score(
                    truth[:, column],
                    pred[:, column],
                    average="binary",
                    zero_division=0,
                )
            )
            if np.unique(truth[:, column]).size < 2:
                continue
            aucs.append(roc_auc_score(truth[:, column], y_score[:, column]))
        if not aucs:
            raise ValueError("No multilabel column contains both classes")
        return accuracy, float(np.mean(f1s)), float(np.mean(aucs))

    raise ValueError(f"Unsupported task: {task}")


def microbench_metrics(payload: dict[str, Any]) -> tuple[float, float, float]:
    """Return MicroBench metrics using sample-specific option predictions."""
    answers = np.asarray(payload["answer_idxs"], dtype=int)
    scores = payload["similarity_scores"]
    predictions = np.asarray([np.argmax(x) for x in scores], dtype=int)
    targets = []
    score_parts = []
    for answer, values in zip(answers, scores):
        target = np.zeros(len(values), dtype=np.uint8)
        target[answer] = 1
        targets.append(target)
        score_parts.append(np.asarray(values))
    return (
        float(np.mean(predictions == answers)),
        float(f1_score(answers, predictions, average="macro", zero_division=0)),
        float(roc_auc_score(np.concatenate(targets), np.concatenate(score_parts))),
    )


class WeightedAUC:
    """Fast weighted binary AUC for repeated bootstrap resamples.

    The score ordering is prepared once. Each bootstrap replicate passes only a
    nonnegative weight vector, avoiding repeated sklearn calls and array
    materialization with replacement.
    """

    def __init__(self, labels: np.ndarray, scores: np.ndarray):
        labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if labels.shape != scores.shape:
            raise ValueError("AUC labels/scores shape mismatch")
        if np.unique(labels).size != 2:
            raise ValueError("AUC requires both positive and negative labels")
        order = np.argsort(scores, kind="mergesort")
        self.order = order
        self.labels_sorted = labels[order].astype(bool)
        sorted_scores = scores[order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]

    def __call__(self, weights: np.ndarray) -> float:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        weights = weights[self.order]
        pos = np.where(self.labels_sorted, weights, 0.0)
        neg = np.where(self.labels_sorted, 0.0, weights)
        pos_total = float(pos.sum())
        neg_total = float(neg.sum())
        if pos_total <= 0 or neg_total <= 0:
            raise ValueError("AUC requires positive weight for both classes")

        pos_group = np.add.reduceat(pos, self.starts)
        neg_group = np.add.reduceat(neg, self.starts)
        cum_neg_before = np.cumsum(neg_group) - neg_group
        numerator = np.sum(pos_group * (cum_neg_before + 0.5 * neg_group))
        return float(numerator / (pos_total * neg_total))


def _weighted_macro_f1(
    true: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
    classes: np.ndarray | None = None,
) -> float:
    true = np.asarray(true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    active = weights > 0
    if classes is None:
        classes = np.union1d(np.unique(true[active]), np.unique(pred[active]))
    values = []
    for cls in classes:
        tp = np.sum(weights[(true == cls) & (pred == cls)])
        fp = np.sum(weights[(true != cls) & (pred == cls)])
        fn = np.sum(weights[(true == cls) & (pred != cls)])
        denom = 2 * tp + fp + fn
        values.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(np.mean(values)) if values else 0.0


def _stratified_bootstrap_weights(
    labels: np.ndarray,
    class_indices: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=np.float64)
    for idx in class_indices:
        counts = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)))
        weights[idx] = counts
    return weights


def _row_bootstrap_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)


def _group_bootstrap_weights(group_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups, inverse = np.unique(group_ids, return_inverse=True)
    group_counts = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)))
    return group_counts[inverse].astype(np.float64)


def bootstrap_npz(
    prediction: dict[str, Any],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast bootstrap from saved NPZ predictions using weighted resampling."""
    y_true = np.asarray(prediction["y_true"])
    y_score = np.asarray(prediction["y_score"], dtype=np.float64)
    task = str(prediction["task"])
    groups = prediction.get("group_ids")
    n = len(y_true)
    rng = np.random.default_rng(seed)
    acc = np.empty(n_bootstrap, dtype=np.float64)
    f1 = np.empty(n_bootstrap, dtype=np.float64)
    auc = np.empty(n_bootstrap, dtype=np.float64)

    if task in {"binary", "multiclass"}:
        labels = y_true.reshape(-1).astype(np.int64)
        pred = np.argmax(y_score, axis=1).astype(np.int64)
        classes = np.arange(y_score.shape[1])
        one_hot = np.eye(y_score.shape[1], dtype=np.uint8)[labels]
        auc_computer = WeightedAUC(one_hot.reshape(-1), y_score.reshape(-1))

        if groups is None:
            class_indices = [np.flatnonzero(labels == cls) for cls in np.unique(labels)]
        else:
            group_ids = np.asarray(groups)

        for b in range(n_bootstrap):
            if groups is None:
                weights = _stratified_bootstrap_weights(labels, class_indices, rng)
            else:
                weights = _group_bootstrap_weights(group_ids, rng)
            total = float(weights.sum())
            acc[b] = float(np.sum(weights * (pred == labels)) / total)
            f1[b] = _weighted_macro_f1(labels, pred, weights, classes=classes)
            auc[b] = auc_computer(np.repeat(weights, y_score.shape[1]))
        return acc, f1, auc

    if task == "multilabel":
        truth = y_true.astype(np.uint8)
        pred = (y_score >= 0.5).astype(np.uint8)
        n_labels = truth.shape[1]
        auc_computers = []
        valid_columns = []
        for column in range(n_labels):
            if np.unique(truth[:, column]).size == 2:
                valid_columns.append(column)
                auc_computers.append(WeightedAUC(truth[:, column], y_score[:, column]))
        if not auc_computers:
            raise ValueError("No multilabel column contains both classes")

        if groups is not None:
            group_ids = np.asarray(groups)

        for b in range(n_bootstrap):
            weights = (
                _group_bootstrap_weights(group_ids, rng)
                if groups is not None
                else _row_bootstrap_weights(n, rng)
            )
            total = float(weights.sum())
            acc[b] = float(np.sum(weights[:, None] * (pred == truth)) / (total * n_labels))

            tp = np.sum(weights[:, None] * ((truth == 1) & (pred == 1)), axis=0)
            fp = np.sum(weights[:, None] * ((truth == 0) & (pred == 1)), axis=0)
            fn = np.sum(weights[:, None] * ((truth == 1) & (pred == 0)), axis=0)
            denom = 2 * tp + fp + fn
            f1[b] = float(np.mean(np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)))

            auc_values = []
            for column, computer in zip(valid_columns, auc_computers):
                pos_w = float(np.sum(weights[truth[:, column] == 1]))
                neg_w = float(np.sum(weights[truth[:, column] == 0]))
                if pos_w > 0 and neg_w > 0:
                    auc_values.append(computer(weights))
            if not auc_values:
                # Extremely unlikely for these datasets; redraw this replicate.
                weights = np.ones(n, dtype=np.float64)
                auc_values = [computer(weights) for computer in auc_computers]
            auc[b] = float(np.mean(auc_values))
        return acc, f1, auc

    raise ValueError(f"Unsupported task: {task}")


def bootstrap_microbench(
    payload: dict[str, Any],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast image-level bootstrap for MicroBench prediction PKLs."""
    rng = np.random.default_rng(seed)
    n_images = int(payload["n_images"])
    q_image = np.asarray(payload["question_sample_indices"], dtype=np.int64)
    answers = np.asarray(payload["answer_idxs"], dtype=np.int64)
    scores = [np.asarray(x, dtype=np.float64) for x in payload["similarity_scores"]]
    preds = np.asarray([int(x.argmax()) for x in scores], dtype=np.int64)
    option_counts = np.asarray([len(x) for x in scores], dtype=np.int64)

    flat_labels = []
    flat_scores = []
    for answer, values in zip(answers, scores):
        target = np.zeros(len(values), dtype=np.uint8)
        target[int(answer)] = 1
        flat_labels.append(target)
        flat_scores.append(values)
    flat_labels = np.concatenate(flat_labels)
    flat_scores = np.concatenate(flat_scores)
    flat_question = np.repeat(np.arange(len(answers), dtype=np.int64), option_counts)
    auc_computer = WeightedAUC(flat_labels, flat_scores)

    acc = np.empty(n_bootstrap, dtype=np.float64)
    f1 = np.empty(n_bootstrap, dtype=np.float64)
    auc = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        image_weights = rng.multinomial(n_images, np.full(n_images, 1.0 / n_images))
        q_weights = image_weights[q_image].astype(np.float64)
        total_q = float(q_weights.sum())
        acc[b] = float(np.sum(q_weights * (preds == answers)) / total_q)
        f1[b] = _weighted_macro_f1(answers, preds, q_weights, classes=None)
        auc[b] = auc_computer(q_weights[flat_question])
    return acc, f1, auc


def _metric_definition(task: str, format_name: str) -> str:
    if format_name == "microbench":
        return "MicroBench pooled AUC over sample-specific answer options; image-level bootstrap"
    if task in {"binary", "multiclass"}:
        return "notebook pooled AUC: flattened one-hot labels and class scores"
    return "notebook multilabel AUC: mean of valid per-label positive-class AUCs"


def _bootstrap_unit(prediction: dict[str, Any]) -> str:
    if prediction["format"] == "microbench":
        return "image"
    if prediction.get("group_ids") is not None:
        return "group"
    if prediction["task"] in {"binary", "multiclass"}:
        return "stratified sample"
    return "sample row"


def _add_ci_columns(
    row: dict[str, Any],
    acc_b: np.ndarray,
    f1_b: np.ndarray,
    auc_b: np.ndarray,
    *,
    alpha: float,
    confidence: float,
    n_bootstrap: int,
) -> None:
    row.update({
        "accuracy_ci_lower": float(np.quantile(acc_b, alpha / 2)),
        "accuracy_ci_upper": float(np.quantile(acc_b, 1 - alpha / 2)),
        "f1_ci_lower": float(np.quantile(f1_b, alpha / 2)),
        "f1_ci_upper": float(np.quantile(f1_b, 1 - alpha / 2)),
        "auc_ci_lower": float(np.quantile(auc_b, alpha / 2)),
        "auc_ci_upper": float(np.quantile(auc_b, 1 - alpha / 2)),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    })


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _load_cached_bootstrap(path: Path, *, n_bootstrap: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["n_bootstrap"]) != int(n_bootstrap):
                return None
            if int(data["seed"]) != int(seed):
                return None
            return (
                np.asarray(data["accuracy"], dtype=np.float64),
                np.asarray(data["f1"], dtype=np.float64),
                np.asarray(data["auc"], dtype=np.float64),
            )
    except Exception:
        return None


def _save_cached_bootstrap(
    path: Path,
    *,
    n_bootstrap: int,
    seed: int,
    acc: np.ndarray,
    f1: np.ndarray,
    auc: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            n_bootstrap=np.asarray(n_bootstrap),
            seed=np.asarray(seed),
            accuracy=acc,
            f1=f1,
            auc=auc,
        )
    os.replace(tmp, path)


def _summarize_one_prediction(
    pred: dict[str, Any],
    *,
    ci: bool,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    specialty_map: dict[str, str],
    ci_cache_dir: Path | None,
    verbose: bool = False,
) -> dict[str, Any]:
    alpha = 1 - confidence
    dataset_name = str(pred["dataset"])
    dataset_seed = (seed + zlib.crc32(dataset_name.encode())) % (2**32)

    if pred["format"] == "microbench":
        accuracy, f1, auc = microbench_metrics(pred)
    else:
        accuracy, f1, auc = notebook_metrics(
            pred["y_true"], pred["y_score"], pred["task"]
        )

    row = {
        "dataset": dataset_name,
        "specialty": specialty_map.get(dataset_name, ""),
        "task": pred["task"],
        "n": int(pred.get("n_images", len(pred.get("y_true", [])))),
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
    }

    if ci:
        cache_path = None
        cached = None
        if ci_cache_dir is not None:
            cache_path = ci_cache_dir / f"{_safe_name(dataset_name)}__n{n_bootstrap}__s{dataset_seed}.npz"
            cached = _load_cached_bootstrap(
                cache_path, n_bootstrap=n_bootstrap, seed=dataset_seed
            )
        if cached is not None:
            acc_b, f1_b, auc_b = cached
            if verbose:
                print(f"[ci cache] {dataset_name}", flush=True)
        else:
            if verbose:
                print(f"[ci start] {dataset_name} n={row['n']} B={n_bootstrap}", flush=True)
            if pred["format"] == "microbench":
                acc_b, f1_b, auc_b = bootstrap_microbench(
                    pred, n_bootstrap=n_bootstrap, seed=dataset_seed
                )
            else:
                acc_b, f1_b, auc_b = bootstrap_npz(
                    pred, n_bootstrap=n_bootstrap, seed=dataset_seed
                )
            if cache_path is not None:
                _save_cached_bootstrap(
                    cache_path,
                    n_bootstrap=n_bootstrap,
                    seed=dataset_seed,
                    acc=acc_b,
                    f1=f1_b,
                    auc=auc_b,
                )
            if verbose:
                print(f"[ci done]  {dataset_name}", flush=True)

        _add_ci_columns(
            row, acc_b, f1_b, auc_b,
            alpha=alpha, confidence=confidence, n_bootstrap=n_bootstrap,
        )
        row.update({
            "bootstrap_unit": _bootstrap_unit(pred),
            "metric_definition": _metric_definition(pred["task"], pred["format"]),
            "_accuracy_bootstrap": acc_b,
            "_f1_bootstrap": f1_b,
            "_auc_bootstrap": auc_b,
        })
    return row


def summarize(
    predictions: list[dict[str, Any]],
    *,
    ci: bool,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    overall_only: bool,
    specialty_map: dict[str, str] | None = None,
    ci_jobs: int = 1,
    ci_cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Summarize benchmark metrics.

    CI uses cached weighted-bootstrap arrays under `ci_cache_dir` by default.
    This avoids rerunning completed benchmark CIs after interruption.
    """
    specialty_map = specialty_map or {}
    cache = Path(ci_cache_dir) if ci_cache_dir is not None else None

    if ci:
        print(
            f"[metrics] weighted bootstrap: {n_bootstrap} replicates "
            f"for {len(predictions)} benchmarks; jobs={ci_jobs}",
            flush=True,
        )
        if cache is not None:
            print(f"[metrics] bootstrap cache: {cache}", flush=True)

    if ci and ci_jobs and ci_jobs > 1:
        rows = Parallel(n_jobs=ci_jobs, prefer="processes")(
            delayed(_summarize_one_prediction)(
                pred,
                ci=ci,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=seed,
                specialty_map=specialty_map,
                ci_cache_dir=cache,
                verbose=False,
            )
            for pred in predictions
        )
    else:
        rows = []
        iterator = tqdm(predictions, desc="metrics/ci" if ci else "metrics")
        for pred in iterator:
            rows.append(
                _summarize_one_prediction(
                    pred,
                    ci=ci,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    seed=seed,
                    specialty_map=specialty_map,
                    ci_cache_dir=cache,
                    verbose=ci,
                )
            )

    overall = {
        "dataset": "OVERALL",
        "specialty": "",
        "task": "unweighted dataset mean",
        "n": int(sum(row["n"] for row in rows)),
        "accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "f1": float(np.mean([row["f1"] for row in rows])),
        "auc": float(np.mean([row["auc"] for row in rows])),
    }
    if ci:
        alpha = 1 - confidence
        acc_b = np.mean(np.stack([row["_accuracy_bootstrap"] for row in rows]), axis=0)
        f1_b = np.mean(np.stack([row["_f1_bootstrap"] for row in rows]), axis=0)
        auc_b = np.mean(np.stack([row["_auc_bootstrap"] for row in rows]), axis=0)
        _add_ci_columns(
            overall, acc_b, f1_b, auc_b,
            alpha=alpha, confidence=confidence, n_bootstrap=n_bootstrap,
        )
        overall.update({
            "bootstrap_unit": "within-dataset bootstrap; unweighted mean over benchmark rows",
            "metric_definition": "unweighted mean of benchmark-level metrics",
            "_accuracy_bootstrap": acc_b,
            "_f1_bootstrap": f1_b,
            "_auc_bootstrap": auc_b,
        })
    return [overall] if overall_only else rows + [overall]


def summarize_specialties(
    rows: list[dict[str, Any]],
    *,
    ci: bool,
    confidence: float,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    """Aggregate benchmark rows by specialty/domain using the manifest mapping."""
    dataset_rows = [
        row for row in rows
        if row.get("dataset") != "OVERALL" and row.get("specialty")
    ]
    if not dataset_rows:
        return []

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        groups[str(row["specialty"])].append(row)

    alpha = 1 - confidence
    output = []
    specialty_boots: dict[str, dict[str, np.ndarray]] = {}

    for specialty in sorted(groups):
        members = groups[specialty]
        out = {
            "specialty": specialty,
            "n_benchmarks": len(members),
            "n": int(sum(row["n"] for row in members)),
            "accuracy": float(np.mean([row["accuracy"] for row in members])),
            "f1": float(np.mean([row["f1"] for row in members])),
            "auc": float(np.mean([row["auc"] for row in members])),
            "aggregation": "unweighted mean over benchmarks in specialty",
        }

        if ci and all(f"_{metric}_bootstrap" in row for row in members for metric in METRIC_NAMES):
            boots = {}
            for metric in METRIC_NAMES:
                values = np.mean(
                    np.stack([row[f"_{metric}_bootstrap"] for row in members], axis=0),
                    axis=0,
                )
                boots[metric] = values
                out[f"{metric}_ci_lower"] = float(np.quantile(values, alpha / 2))
                out[f"{metric}_ci_upper"] = float(np.quantile(values, 1 - alpha / 2))
            out["n_bootstrap"] = n_bootstrap
            out["confidence"] = confidence
            out["bootstrap_unit"] = "within-dataset bootstrap; unweighted mean within specialty"
            specialty_boots[specialty] = boots

        output.append(out)

    overall = {
        "specialty": "OVERALL_SPECIALTY_MACRO",
        "n_benchmarks": int(sum(row["n_benchmarks"] for row in output)),
        "n": int(sum(row["n"] for row in output)),
        "accuracy": float(np.mean([row["accuracy"] for row in output])),
        "f1": float(np.mean([row["f1"] for row in output])),
        "auc": float(np.mean([row["auc"] for row in output])),
        "aggregation": "unweighted mean over specialties",
    }

    if ci and specialty_boots:
        for metric in METRIC_NAMES:
            values = np.mean(
                np.stack([specialty_boots[name][metric] for name in sorted(specialty_boots)], axis=0),
                axis=0,
            )
            overall[f"{metric}_ci_lower"] = float(np.quantile(values, alpha / 2))
            overall[f"{metric}_ci_upper"] = float(np.quantile(values, 1 - alpha / 2))
        overall["n_bootstrap"] = n_bootstrap
        overall["confidence"] = confidence
        overall["bootstrap_unit"] = "within-dataset bootstrap; mean within specialty; macro over specialties"

    output.append(overall)
    return output


def _public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]


def _ordered_keys(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "dataset", "specialty", "task", "n",
        "n_benchmarks",
        "accuracy", "accuracy_ci_lower", "accuracy_ci_upper",
        "f1", "f1_ci_lower", "f1_ci_upper",
        "auc", "auc_ci_lower", "auc_ci_upper",
        "confidence", "n_bootstrap", "bootstrap_unit", "metric_definition",
        "aggregation",
    ]
    seen = []
    available = {key for row in rows for key in row}
    for key in preferred:
        if key in available:
            seen.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def write_table(rows: list[dict[str, Any]], output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    public = _public_rows(rows)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    keys = _ordered_keys(public)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(public)
    json_path.write_text(json.dumps(public, indent=2), encoding="utf-8")
    return csv_path, json_path


def write_results(rows: list[dict[str, Any]], output_dir: Path, ci: bool) -> tuple[Path, Path]:
    stem = "classification_metrics_ci" if ci else "classification_metrics"
    return write_table(rows, output_dir, stem)


def write_specialty_results(rows: list[dict[str, Any]], output_dir: Path, ci: bool) -> tuple[Path, Path]:
    stem = "specialty_metrics_ci" if ci else "specialty_metrics"
    return write_table(rows, output_dir, stem)
