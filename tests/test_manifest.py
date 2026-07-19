from medpmc_clip_eval.manifest import (
    included_entries,
    selected_microbench_names,
    selected_specialties,
    selected_standard_keys,
)


def test_paper_manifest_counts():
    assert len(included_entries()) == 26
    assert len(selected_standard_keys()) == 9
    assert len(selected_microbench_names()) == 17
    assert len(selected_specialties()) == 11


def test_excluded_microbench_subsets():
    selected = set(selected_microbench_names())
    assert "colocalization_benchmark" not in selected
    assert "kather_et_al_2018" not in selected
    assert "kather_et_al_2018_val7k" not in selected


def test_breastmnist_label_order():
    from medpmc_clip_eval.datasets import REGISTRY

    spec = REGISTRY["breastmnist"]
    assert spec.class_names == ["malignant", "benign_or_normal"]
    assert spec.prompts[0] == "breast ultrasound of a malignant tumor"
    assert spec.prompts[1] == "breast ultrasound of benign lesion or normal tissue"


def test_original_prompt_bank_alignment():
    from medpmc_clip_eval.datasets import REGISTRY

    assert REGISTRY["chestmnist"].class_names[12] == "pleural"
    assert REGISTRY["pcam"].class_names == ["negative", "positive"]
    assert REGISTRY["pcam"].prompts[1] == "histopathology image of metastatic tumor tissue"
    assert REGISTRY["deepdrid"].class_names == ["DR0", "DR1", "DR2", "DR3", "DR4"]
    assert REGISTRY["rsna"].prompts == ["chest x-ray of no pneumonia", "chest x-ray of pneumonia"]
    assert REGISTRY["dad"].class_names[1] == "inferior_mesenteric_artery"


def test_microbench_nested_caption_schema():
    from medpmc_clip_eval.inference import _extract_microbench_caption

    example = {
        "captions": {
            "classification_0": {
                "options": ["normal cell", "abnormal cell"],
                "answer_idx": 1,
            }
        }
    }
    options, answer = _extract_microbench_caption(example, 0)
    assert options == ["normal cell", "abnormal cell"]
    assert answer == 1


def test_microbench_flattened_caption_schema_fallback():
    from medpmc_clip_eval.inference import _extract_microbench_caption

    example = {
        "classification_1_options": ["a", "b", "c"],
        "classification_1_answer": "b",
    }
    options, answer = _extract_microbench_caption(example, 1)
    assert options == ["a", "b", "c"]
    assert answer == 1


def test_manifest_has_only_public_columns():
    import csv
    from medpmc_clip_eval.manifest import DEFAULT_MANIFEST
    with DEFAULT_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["include", "dataset", "specialty", "source"]


def test_display_names_map_to_specialties():
    from medpmc_clip_eval.manifest import dataset_to_specialty
    mapping = dataset_to_specialty()
    assert mapping["HAM10000"] == "Dermoscopy"
    assert mapping["BreastMNIST"] == "breast ultrasound"
    assert mapping["LC25000_Colon"] == "Neoplastic"
    assert mapping["PCAM"] == "Neoplastic"
