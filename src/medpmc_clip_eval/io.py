from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def save_npz(
    path: Path,
    *,
    dataset: str,
    task: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    sample_ids: list[str],
    group_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": np.asarray(dataset),
        "task": np.asarray(task),
        "y_true": np.asarray(y_true),
        "y_score": np.asarray(y_score, dtype=np.float32),
        "class_names": np.asarray(class_names),
        "sample_ids": np.asarray(sample_ids),
        "metadata_json": np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    }
    if group_ids is not None:
        payload["group_ids"] = np.asarray(group_ids)
    np.savez_compressed(path, **payload)


def load_prediction(path: Path) -> dict[str, Any]:
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            result = pickle.load(handle)
        result["format"] = "microbench"
        result["task"] = "microbench"
        return result

    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    for key in ("dataset", "task"):
        result[key] = str(result[key].item())
    if "group_ids" in result:
        result["group_ids"] = np.asarray(result["group_ids"], dtype=str)
    result["format"] = "npz"
    return result
