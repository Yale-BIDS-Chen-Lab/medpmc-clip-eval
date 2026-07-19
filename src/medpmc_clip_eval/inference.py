from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .datasets import DatasetSpec
from .io import save_npz


DISPLAY_NAMES = {
    "breastmnist": "BreastMNIST",
    "chestmnist": "ChestMNIST",
    "pcam": "PCAM",
    "deepdrid": "DeepDRiD",
    "rsna": "RSNA_Pneumonia",
    "ham10000": "HAM10000",
    "lc25000_lung": "LC25000_Lung",
    "lc25000_colon": "LC25000_Colon",
    "dad": "DAD",
}


def run_standard(
    *,
    model,
    spec: DatasetSpec,
    loader,
    output_path: Path,
):
    truths = []
    scores = []
    sample_ids = []
    group_ids = []
    has_groups = False

    for batch in tqdm(loader, desc=spec.key):
        images = batch["images"]
        labels = batch["labels"]
        if spec.task == "multilabel":
            batch_scores = model.multilabel_scores(images, spec.prompts)
        else:
            batch_scores = model.multiclass_scores(images, spec.prompts)
        truths.append(np.asarray(labels))
        scores.append(np.asarray(batch_scores))
        sample_ids.extend(str(x) for x in batch["sample_ids"])
        groups = batch.get("group_ids")
        if groups is not None:
            has_groups = True
            group_ids.extend(str(x) for x in groups)

    save_npz(
        output_path,
        dataset=DISPLAY_NAMES.get(spec.key, spec.key),
        task=spec.task,
        y_true=np.concatenate(truths),
        y_score=np.concatenate(scores),
        class_names=spec.class_names,
        sample_ids=sample_ids,
        group_ids=group_ids if has_groups else None,
        metadata={"prompts": spec.prompts, "logit_scale": model.logit_scale},
    )



def _extract_microbench_caption(example, caption_type: int):
    """Return the MicroBench caption row for one sample and caption type.

    The original clip-ci legacy implementation reads the Hugging Face uBench
    schema as:

        example["captions"][f"classification_{caption_type}"]["options"]
        example["captions"][f"classification_{caption_type}"]["answer_idx"]

    Some locally flattened variants may expose classification_0_options and
    classification_0_answer directly; that fallback is accepted but is not the
    canonical uBench schema.
    """
    nested = example.get("captions")
    key = f"classification_{caption_type}"

    if isinstance(nested, dict) and key in nested:
        row = nested[key]
        if "options" not in row or "answer_idx" not in row:
            raise KeyError(f"MicroBench {key} must contain options and answer_idx")
        return [str(text) for text in row["options"]], int(row["answer_idx"])

    option_key = f"{key}_options"
    answer_key = f"{key}_answer"
    answer_idx_key = f"{key}_answer_idx"
    if option_key in example and (answer_key in example or answer_idx_key in example):
        options = [str(text) for text in example[option_key]]
        answer = example[answer_idx_key] if answer_idx_key in example else example[answer_key]
        if isinstance(answer, str):
            try:
                answer = options.index(answer)
            except ValueError as exc:
                raise ValueError(f"Answer string not found in {option_key}: {answer}") from exc
        return options, int(answer)

    available = sorted(str(k) for k in example.keys())
    raise KeyError(
        f"Missing MicroBench caption data for {key}. "
        f"Expected nested captions['{key}'] or flattened {option_key}. "
        f"Available top-level keys: {available}"
    )

def run_microbench(
    *,
    model,
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    dataset_names: list[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Run only the requested MicroBench subsets and return prediction paths."""
    from datasets import load_dataset

    hf_cache = data_dir / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "jnirschl/uBench",
        split="test",
        cache_dir=str(hf_cache),
    )
    available_names = sorted(set(str(x) for x in dataset["dataset"]))

    if dataset_names is None:
        selected_names = available_names
    else:
        selected_names = list(dict.fromkeys(str(x) for x in dataset_names))
        missing = [name for name in selected_names if name not in available_names]
        if missing:
            raise ValueError(
                "MicroBench subsets listed in the benchmark manifest were not "
                f"found in jnirschl/uBench: {missing}"
            )

    output_paths = []
    for dataset_name in selected_names:
        path = output_dir / f"microbench__{dataset_name}.pkl"
        output_paths.append(path)
        if path.exists() and not overwrite:
            continue

        subset = dataset.filter(lambda x: str(x["dataset"]) == dataset_name)
        records = list(subset)
        canonical_ids = [str(x.get("id", i)) for i, x in enumerate(records)]
        all_scores = []
        all_answers = []
        all_preds = []
        question_image = []
        question_types = []
        option_counts = []

        for caption_type in (0, 1):
            sample_ids_this_type = []
            for start in tqdm(
                range(0, len(records), batch_size),
                desc=f"microbench/{dataset_name}/ct{caption_type}",
            ):
                block = records[start : start + batch_size]
                images = [x["image"].convert("RGB") for x in block]
                options = []
                answers = []
                for example in block:
                    row_options, answer_idx = _extract_microbench_caption(example, caption_type)
                    if answer_idx < 0 or answer_idx >= len(row_options):
                        raise ValueError(
                            f"{dataset_name}/ct{caption_type}: "
                            f"answer_idx={answer_idx} for {len(row_options)} options"
                        )
                    options.append(row_options)
                    answers.append(answer_idx)

                scores = model.variable_option_scores(images, options)
                if len(scores) != len(answers):
                    raise RuntimeError("MicroBench prediction count mismatch")

                for local, (answer, values) in enumerate(zip(answers, scores)):
                    values = np.asarray(values, dtype=np.float32)
                    all_scores.append(values)
                    all_answers.append(int(answer))
                    all_preds.append(int(values.argmax()))
                    question_image.append(start + local)
                    question_types.append(caption_type)
                    option_counts.append(len(values))
                    sample_ids_this_type.append(canonical_ids[start + local])

            if sample_ids_this_type != canonical_ids:
                raise RuntimeError(
                    f"{dataset_name}: image order changed or sample count mismatch "
                    f"for caption type {caption_type}"
                )

        answer_array = np.asarray(all_answers, dtype=np.int16)
        pred_array = np.asarray(all_preds, dtype=np.int16)

        payload = {
            "format_version": 2,
            "format": "microbench",
            "protocol": "microbench_sample_specific_options_legacy",
            "dataset": str(dataset_name),
            "task": "microbench",
            "n_images": len(records),
            "caption_types": [0, 1],
            "question_order": "caption_type_major",
            "sample_ids": canonical_ids,
            "question_sample_indices": np.asarray(question_image, dtype=np.int32),
            "question_caption_types": np.asarray(question_types, dtype=np.int16),
            "question_option_counts": np.asarray(option_counts, dtype=np.int16),
            "answer_idxs": answer_array,
            "similarity_scores": all_scores,
            "preds": pred_array,
            "point_accuracy": float(np.mean(pred_array == answer_array)),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_paths
