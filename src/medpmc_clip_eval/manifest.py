from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).with_name("benchmark_manifest.csv")


@dataclass(frozen=True)
class BenchmarkEntry:
    dataset: str
    specialty: str
    source: str
    include: bool


_STANDARD_DATASET_KEYS = {
    "breastmnist": "breastmnist",
    "chestmnist": "chestmnist",
    "dad": "dad",
    "deepdrid": "deepdrid",
    "ham10000": "ham10000",
    "lc25000colon": "lc25000_colon",
    "lc25000lung": "lc25000_lung",
    "pcam": "pcam",
    "rsnapneumonia": "rsna",
}


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _parse_include(value: str) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def load_manifest(path: Path | str | None = None) -> list[BenchmarkEntry]:
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"include", "dataset", "specialty", "source"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Benchmark manifest is missing columns: {sorted(missing)}"
            )

        entries = []
        for line_number, row in enumerate(reader, start=2):
            dataset = str(row.get("dataset", "")).strip()
            specialty = str(row.get("specialty", "")).strip()
            source = str(row.get("source", "")).strip().casefold()
            include = _parse_include(row.get("include", ""))

            if not dataset:
                raise ValueError(f"Blank dataset name at line {line_number}")
            if source not in {"nonmicrobench", "microbench"}:
                raise ValueError(
                    f"Unknown source {source!r} for {dataset} at line {line_number}"
                )
            if include and not specialty:
                raise ValueError(
                    f"Included dataset has no specialty assignment: {dataset}"
                )

            entries.append(
                BenchmarkEntry(
                    dataset=dataset,
                    specialty=specialty,
                    source=source,
                    include=include,
                )
            )

    if not entries:
        raise ValueError(f"Benchmark manifest is empty: {manifest_path}")
    return entries


def included_entries(path: Path | str | None = None) -> list[BenchmarkEntry]:
    return [entry for entry in load_manifest(path) if entry.include]


def selected_microbench_names(path: Path | str | None = None) -> list[str]:
    return [
        entry.dataset
        for entry in included_entries(path)
        if entry.source == "microbench"
    ]


def selected_standard_keys(path: Path | str | None = None) -> list[str]:
    keys = []
    for entry in included_entries(path):
        if entry.source != "nonmicrobench":
            continue
        normalized = _normalize_name(entry.dataset)
        try:
            key = _STANDARD_DATASET_KEYS[normalized]
        except KeyError as exc:
            raise ValueError(
                f"No evaluation registry key is defined for manifest dataset "
                f"{entry.dataset!r}"
            ) from exc
        if key not in keys:
            keys.append(key)
    return keys


def selected_specialties(path: Path | str | None = None) -> list[str]:
    return sorted({entry.specialty for entry in included_entries(path)})


def dataset_to_specialty(path: Path | str | None = None) -> dict[str, str]:
    mapping = {}
    for entry in included_entries(path):
        # Prediction files store standard datasets using display names
        # (for example, "HAM10000" and "LC25000_Colon") and MicroBench subsets
        # using raw subset names. Keep both the manifest/display name and the
        # internal registry key as aliases so every evaluated benchmark row is
        # included in domain-level summaries.
        mapping[entry.dataset] = entry.specialty

        if entry.source == "microbench":
            mapping[f"MicroBench/{entry.dataset}"] = entry.specialty
        else:
            normalized = _normalize_name(entry.dataset)
            try:
                key = _STANDARD_DATASET_KEYS[normalized]
            except KeyError as exc:
                raise ValueError(
                    f"No evaluation registry key is defined for manifest dataset "
                    f"{entry.dataset!r}"
                ) from exc
            mapping[key] = entry.specialty
    return mapping
