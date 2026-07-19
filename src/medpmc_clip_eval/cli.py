from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .datasets import ALIASES, REGISTRY
from .inference import run_microbench, run_standard
from .io import load_prediction
from .manifest import (
    DEFAULT_MANIFEST,
    dataset_to_specialty,
    included_entries,
    selected_microbench_names,
    selected_standard_keys,
)
from .metrics import (
    summarize,
    summarize_specialties,
    write_results,
    write_specialty_results,
)
from .model import MODEL_ALIASES, PUBLIC_BASELINE_MODELS, load_scorer, model_tag_for_key, normalize_model_key


def prepare_cache_dirs(cache_root: Path) -> dict[str, Path]:
    """Create and register all cache directories used by the public pipeline."""
    cache_root = cache_root.expanduser().resolve()
    paths = {
        "root": cache_root,
        "data": cache_root / "data",
        "huggingface": cache_root / "huggingface",
        "kagglehub": cache_root / "kagglehub",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(paths["huggingface"]))
    os.environ.setdefault("HF_HUB_CACHE", str(paths["huggingface"] / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(paths["huggingface"] / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(paths["huggingface"] / "transformers"))
    os.environ.setdefault("KAGGLEHUB_CACHE", str(paths["kagglehub"]))
    return paths


def load_full_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return config


def data_paths_from_config(config: dict[str, Any]) -> dict[str, Any]:
    data_paths = config.get("data_paths", {})
    if data_paths is None:
        data_paths = {}
    if not isinstance(data_paths, dict):
        raise ValueError("Config field `data_paths` must be a JSON object")
    return {str(k): v for k, v in data_paths.items()}


def parse_datasets(value: str, manifest_path: Path | None = None) -> list[str]:
    requested = [x.strip().lower() for x in value.split(",") if x.strip()]
    expanded = []
    for item in requested:
        if item == "medpmc-benchmarks":
            expanded.extend(selected_standard_keys(manifest_path))
            expanded.append("microbench")
        else:
            expanded.extend(ALIASES.get(item, [item]))

    result = []
    for item in expanded:
        if item not in REGISTRY and item != "microbench":
            raise ValueError(f"Unknown dataset: {item}")
        if item not in result:
            result.append(item)
    return result


def parse_models(args: argparse.Namespace) -> list[str]:
    raw: list[str]
    if args.models:
        raw = args.models
    else:
        raw = [args.model]

    expanded = []
    for item in raw:
        key = str(item).strip().lower()
        if key in {"all", "baselines"}:
            expanded.extend(PUBLIC_BASELINE_MODELS)
        else:
            expanded.append(normalize_model_key(key))

    output = []
    for key in expanded:
        key = normalize_model_key(key)
        if key not in output:
            output.append(key)
    return output


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="medpmc-benchmarks",
        help=(
            "Comma-separated dataset keys or aliases. The default "
            "`medpmc-benchmarks` alias uses the 26 benchmarks included "
            "in benchmark_manifest.csv. "
        ),
    )
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        help=(
            "Benchmark selection manifest. Defaults to the manifest bundled with "
            f"the package: {DEFAULT_MANIFEST}"
        ),
    )
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--model",
        default="medpmc",
        choices=sorted(MODEL_ALIASES),
        help="Model to evaluate. `medpmc` is the public Hugging Face MedPMC-CLIP checkpoint.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Evaluate multiple models sequentially. Use `all` for public models "
            "(medpmc, bmc, biomedclip, pmcclip, openclip, coca, medsiglip)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        help="Override local checkpoint for --model bmc.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Optional JSON file with data_paths and model settings. "
            "benchmark_manifest.csv controls benchmark selection; this config controls local file/model locations."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel benchmark-level CI jobs. Use 1 on memory-limited nodes.")
    parser.add_argument("--overall-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260704)
    return parser


def _prediction_and_result_dirs(
    *,
    output_dir: Path,
    model_key: str,
    n_models: int,
) -> tuple[Path, Path, bool]:
    """Return prediction/result dirs and whether model-specific subdirs are used.

    Backward compatibility: a single default `--model medpmc` run still uses
    `outputs/predictions` and `outputs/results`, so previously generated public
    MedPMC predictions can be reused. Any baseline or multi-model run uses
    `outputs/predictions/<ModelTag>` and `outputs/results/<ModelTag>`.
    """
    tag = model_tag_for_key(model_key)
    model_specific = n_models > 1 or model_key != "medpmc"
    if model_specific:
        return output_dir / "predictions" / tag, output_dir / "results" / tag, True
    return output_dir / "predictions", output_dir / "results", False


def _expected_path(prediction_dir: Path, key: str, microbench_name: str | None = None) -> Path:
    if key == "microbench":
        if microbench_name is None:
            raise ValueError("microbench_name is required")
        return prediction_dir / f"microbench__{microbench_name}.pkl"
    return prediction_dir / f"{key}.npz"


def _collect_prediction_paths(
    *,
    datasets: list[str],
    microbench_names: list[str],
    prediction_dir: Path,
    skip_failed: bool,
) -> list[Path]:
    paths = []
    for key in datasets:
        if key == "microbench":
            for dataset_name in microbench_names:
                path = _expected_path(prediction_dir, key, dataset_name)
                if path.exists():
                    paths.append(path)
                elif not skip_failed:
                    raise FileNotFoundError(path)
        else:
            path = _expected_path(prediction_dir, key)
            if path.exists():
                paths.append(path)
            elif not skip_failed:
                raise FileNotFoundError(path)
    return paths


def _run_one_model(
    *,
    model_key: str,
    n_models: int,
    args: argparse.Namespace,
    full_config: dict[str, Any],
    data_paths: dict[str, Any],
    datasets: list[str],
    microbench_names: list[str],
    cache_root: Path,
    cache_dirs: dict[str, Path],
    manifest_path: Path | None,
) -> None:
    model_tag = model_tag_for_key(model_key)
    output_dir = args.output_dir
    prediction_dir, result_dir, model_specific = _prediction_and_result_dirs(
        output_dir=output_dir,
        model_key=model_key,
        n_models=n_models,
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Model: {model_key} ({model_tag})")
    print(f"Prediction directory: {prediction_dir}")
    print(f"Result directory: {result_dir}")
    print("=" * 80)

    model = None
    prediction_paths: list[Path] = []

    for key in datasets:
        if args.metrics_only:
            continue

        try:
            if key == "microbench":
                if args.download_only:
                    from datasets import load_dataset

                    load_dataset(
                        "jnirschl/uBench",
                        split="test",
                        cache_dir=str(cache_dirs["huggingface"]),
                    )
                    continue

                selected_paths = [_expected_path(prediction_dir, key, name) for name in microbench_names]
                if not args.overwrite and all(path.exists() for path in selected_paths):
                    print(f"[skip] microbench: all {len(selected_paths)} selected subsets exist")
                    prediction_paths.extend(selected_paths)
                    continue

                if model is None:
                    _, model, model_source = load_scorer(
                        model_key,
                        full_config,
                        device=args.device,
                        cache_dir=cache_dirs["huggingface"],
                        checkpoint_override=args.checkpoint,
                        use_amp=not args.no_amp,
                    )
                    print(f"[model source] {model_source}")

                prediction_paths.extend(
                    run_microbench(
                        model=model,
                        data_dir=cache_root,
                        output_dir=prediction_dir,
                        batch_size=args.batch_size,
                        dataset_names=microbench_names,
                        overwrite=args.overwrite,
                    )
                )
                continue

            expected = _expected_path(prediction_dir, key)
            if expected.exists() and not args.overwrite:
                try:
                    load_prediction(expected)
                except Exception as exc:
                    print(f"[rerun] invalid file {expected}: {exc}")
                else:
                    print(f"[skip] {key}: {expected}")
                    prediction_paths.append(expected)
                    continue

            spec = REGISTRY[key]
            loader = None
            try:
                loader = spec.loader(cache_root / "data", args.batch_size, args.workers, data_paths)
                if args.download_only:
                    continue
                if model is None:
                    _, model, model_source = load_scorer(
                        model_key,
                        full_config,
                        device=args.device,
                        cache_dir=cache_dirs["huggingface"],
                        checkpoint_override=args.checkpoint,
                        use_amp=not args.no_amp,
                    )
                    print(f"[model source] {model_source}")
                run_standard(model=model, spec=spec, loader=loader, output_path=expected)
                prediction_paths.append(expected)
            finally:
                if loader is not None:
                    del loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as exc:
            if args.skip_failed:
                print(f"[skip] {model_key}/{key}: {exc}")
                continue
            raise

    if args.download_only:
        return

    # Metrics-only must collect from disk. After inference, collect again so
    # skipped and newly generated files are both represented exactly once.
    prediction_paths = _collect_prediction_paths(
        datasets=datasets,
        microbench_names=microbench_names,
        prediction_dir=prediction_dir,
        skip_failed=args.skip_failed,
    )
    if not prediction_paths:
        raise SystemExit(f"No prediction files are available under {prediction_dir}")

    predictions = [load_prediction(path) for path in prediction_paths]
    specialty_map = dataset_to_specialty(manifest_path)
    rows = summarize(
        predictions,
        ci=args.ci,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
        overall_only=args.overall_only,
        specialty_map=specialty_map,
        ci_jobs=args.jobs,
        ci_cache_dir=result_dir / "bootstrap_cache",
    )
    specialty_rows = summarize_specialties(
        rows,
        ci=args.ci,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
    )

    request_tokens = {token.strip().lower() for token in args.datasets.split(",") if token.strip()}
    if request_tokens & {"medpmc-benchmarks", "paper", "manuscript"}:
        expected_names = [entry.dataset for entry in included_entries(manifest_path)]
        present_names = {
            str(row["dataset"])
            for row in rows
            if row.get("dataset") != "OVERALL"
        }
        missing_names = [name for name in expected_names if name not in present_names]
        if missing_names:
            print(
                "\n[warning] Partial benchmark summary: "
                f"{len(expected_names) - len(missing_names)}/{len(expected_names)} "
                "manifest-selected benchmark rows are available.",
                flush=True,
            )
            print("[warning] Missing benchmark rows: " + ", ".join(missing_names), flush=True)

    csv_path, json_path = write_results(rows, result_dir, args.ci)
    specialty_csv_path = specialty_json_path = None
    if specialty_rows:
        specialty_csv_path, specialty_json_path = write_specialty_results(
            specialty_rows, result_dir, args.ci
        )

    for row in rows:
        if args.ci:
            print(
                f"{row['dataset']:<35} "
                f"Acc={row['accuracy']:.4f} "
                f"F1={row['f1']:.4f} "
                f"AUC={row['auc']:.4f} "
                f"[{row['auc_ci_lower']:.4f}, {row['auc_ci_upper']:.4f}]"
            )
        else:
            print(
                f"{row['dataset']:<35} "
                f"Acc={row['accuracy']:.4f} "
                f"F1={row['f1']:.4f} "
                f"AUC={row['auc']:.4f}"
            )

    if specialty_rows:
        print("\nBy specialty/domain:")
        for row in specialty_rows:
            if args.ci:
                print(
                    f"{row['specialty']:<35} "
                    f"n_benchmarks={row['n_benchmarks']:<2} "
                    f"Acc={row['accuracy']:.4f} "
                    f"F1={row['f1']:.4f} "
                    f"AUC={row['auc']:.4f} "
                    f"[{row['auc_ci_lower']:.4f}, {row['auc_ci_upper']:.4f}]"
                )
            else:
                print(
                    f"{row['specialty']:<35} "
                    f"n_benchmarks={row['n_benchmarks']:<2} "
                    f"Acc={row['accuracy']:.4f} "
                    f"F1={row['f1']:.4f} "
                    f"AUC={row['auc']:.4f}"
                )

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    if specialty_csv_path is not None:
        print(f"Saved: {specialty_csv_path}")
        print(f"Saved: {specialty_json_path}")


def main():
    args = build_parser().parse_args()
    manifest_path = args.benchmark_manifest
    full_config = load_full_config(args.config)
    data_paths = data_paths_from_config(full_config)

    if args.list_models:
        print("Available models:")
        for key in sorted(MODEL_ALIASES):
            print(f"  {key:<12} -> {MODEL_ALIASES[key]}")
        print("\n`--models all` runs: " + ", ".join(PUBLIC_BASELINE_MODELS))
        return

    if args.list_datasets:
        for key in list(REGISTRY) + ["microbench"]:
            print(key)
        print("\nAliases: medpmc-benchmarks, nonmicrobench, all")
        print(
            f"Selected MicroBench subsets in manifest: "
            f"{len(selected_microbench_names(manifest_path))}"
        )
        return

    model_keys = parse_models(args)
    datasets = parse_datasets(args.datasets, manifest_path)
    microbench_names = selected_microbench_names(manifest_path)

    cache_root = args.data_dir or Path(
        os.environ.get("MEDPMC_CLIP_EVAL_CACHE", "~/.cache/medpmc-clip-eval")
    )
    cache_dirs = prepare_cache_dirs(cache_root)
    cache_root = cache_dirs["root"]

    print(f"Cache root: {cache_root}")
    print(f"Hugging Face cache: {cache_dirs['huggingface']}")
    print(f"KaggleHub cache: {cache_dirs['kagglehub']}")
    print(f"Output directory: {args.output_dir}")
    if args.config:
        print(f"Path config: {args.config}")
        print(f"Configured dataset path keys: {sorted(data_paths)}")
        if full_config.get("models"):
            print(f"Configured model keys: {sorted(full_config['models'])}")
    print(f"Models: {', '.join(model_keys)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for model_key in model_keys:
        _run_one_model(
            model_key=model_key,
            n_models=len(model_keys),
            args=args,
            full_config=full_config,
            data_paths=data_paths,
            datasets=datasets,
            microbench_names=microbench_names,
            cache_root=cache_root,
            cache_dirs=cache_dirs,
            manifest_path=manifest_path,
        )


if __name__ == "__main__":
    main()
