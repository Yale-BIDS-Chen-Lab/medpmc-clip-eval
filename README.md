# MedPMC-CLIP evaluation

Zero-shot evaluation code for the public MedPMC-CLIP release:

- Model: [`Yale-BIDS-Chen/medpmc-clip-l-14_jun24_v1`](https://huggingface.co/Yale-BIDS-Chen/medpmc-clip-l-14_jun24_v1)
- Default benchmark preset: `medpmc-benchmarks`
- Outputs: cached prediction scores plus accuracy, macro F1, and AUC summaries

The default benchmark preset is defined in `benchmark_manifest.csv` and contains the 26 benchmark rows used in the manuscript: 9 non-MicroBench datasets and 17 selected MicroBench subsets grouped into 11 domains.


## License and terms of use

The source code in this repository is released under the MIT License.

The MedPMC-CLIP model weights are released under the CC BY-NC-SA 4.0 license for non-commercial research use. The MIT License for this repository applies only to the code, not to the MedPMC-CLIP model weights.

This repository does not redistribute benchmark datasets or third-party baseline model weights. Each dataset and baseline model remains governed by its original license, access terms, and citation requirements. Users are responsible for reviewing and complying with the terms of the original sources before downloading or using them.

## Installation

We recommend a Python 3.10 environment. On GPU/HPC systems, install a PyTorch build compatible with your local CUDA driver.

```bash
conda create -n medpmc-clip-eval python=3.10 -y
conda activate medpmc-clip-eval

# Example CUDA build; adjust if needed for your system.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -e .
```

Optional test dependencies:

```bash
pip install -e '.[dev]'
pytest
```

## Dataset setup

This repository does not redistribute benchmark datasets. Prepare the datasets from their original sources and provide local paths in `config.json`. Each dataset is governed by its original license and access terms.

```bash
cp config.example.json config.json
# edit config.json
```

Only MedMNIST and MicroBench are downloaded programmatically by default. PCAM, DeepDRiD, RSNA Pneumonia, HAM10000/ISIC 2018 Task 3, LC25000, and DAD require local paths through `config.json`.

### Dataset sources

Please add or verify paper/source links before release.

| Benchmark row(s) | Paper/source link | Required local config keys | Notes |
|---|---|---|---|
| BreastMNIST, ChestMNIST | TODO: add MedMNIST paper/source link | `medmnist_root` | Uses the MedMNIST test split. BreastMNIST label order is `0=malignant`, `1=benign_or_normal`. |
| PCAM | TODO: add PatchCamelyon paper/source link | `pcam_x`, `pcam_y` | Uses official HDF5 test files: `camelyonpatch_level_2_split_test_x.h5` and `camelyonpatch_level_2_split_test_y.h5`. |
| DeepDRiD | TODO: add DeepDRiD / ISBI 2020 challenge link | `deepdrid_csv`, `deepdrid_images` | Labels may be `.csv` or `.xlsx`; images are loaded from the challenge image folder. |
| RSNA_Pneumonia | TODO: add RSNA Pneumonia Detection Challenge link | `rsna_csv`, `rsna_images` | Uses `stage_2_train_labels.csv` and a prepared image folder. |
| HAM10000 | TODO: add HAM10000 / ISIC 2018 Task 3 link | `ham10000_csv`, `ham10000_images` | Uses `ISIC2018_Task3_Test_GroundTruth.csv` and `ISIC2018_Task3_Test_Input/`. |
| LC25000_Colon, LC25000_Lung | TODO: add LC25000 paper/source link | `lc25000_root` | Expects a prepared `Test Set` folder with class subdirectories. |
| DAD | TODO: add Dresden Surgical Anatomy Dataset link | `dad_annotations`, `dad_images` | Uses COCO-style test annotations and test images. |
| MicroBench subsets | TODO: add MicroBench / uBench link | none | Loaded from `jnirschl/uBench`; only the subsets selected in `benchmark_manifest.csv` are evaluated. |

## Model selection

The default model is MedPMC-CLIP:

```bash
python -m medpmc_clip_eval.cli \
  --model medpmc \
  --datasets medpmc-benchmarks \
  --config config.json \
  --output-dir outputs \
  --device cuda
```

Supported model keys. Third-party baseline models are governed by their own licenses and terms of use:

| Key | Output tag | Default source |
|---|---|---|
| `medpmc` | `MedPMC-CLIP` | `Yale-BIDS-Chen/medpmc-clip-l-14_jun24_v1` |
| `bmc` | `BMC` | `BIOMEDICA/BMC_CLIP_CF` |
| `biomedclip` | `BioMedCLIP` | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| `pmcclip` | `PMC-CLIP` | `ryanyip7777/pmc_vit_l_14` |
| `openclip` | `OpenCLIP` | `ViT-L-14/commonpool_xl_clip_s13b_b90k` |
| `coca` | `CoCa` | `coca_ViT-L-14/mscoco_finetuned_laion2b_s13b_b90k` |
| `medsiglip` | `MedSigLIP` | `google/medsiglip-448` |

Run all supported public models sequentially:

```bash
python -u -m medpmc_clip_eval.cli \
  --models all \
  --datasets medpmc-benchmarks \
  --config config.json \
  --output-dir outputs \
  --device cuda \
  --workers 0 \
  --batch-size 32 \
  --skip-failed
```

## Running evaluation

Run the manuscript benchmark preset:

```bash
python -u -m medpmc_clip_eval.cli \
  --datasets medpmc-benchmarks \
  --config config.json \
  --output-dir outputs \
  --device cuda \
  --workers 0 \
  --batch-size 32
```

Recompute metrics from cached predictions without rerunning inference:

```bash
python -u -m medpmc_clip_eval.cli \
  --datasets medpmc-benchmarks \
  --config config.json \
  --output-dir outputs \
  --metrics-only
```

Compute bootstrap confidence intervals:

```bash
python -u -m medpmc_clip_eval.cli \
  --datasets medpmc-benchmarks \
  --config config.json \
  --output-dir outputs \
  --metrics-only \
  --ci \
  --n-bootstrap 10000 \
  --jobs 4
```

The CI path uses saved prediction scores only. It does not reload images or model weights. Dataset-level bootstrap arrays are cached under `results/bootstrap_cache`, so interrupted CI runs can resume without recomputing completed datasets.

## Outputs

Single-model MedPMC-CLIP runs store predictions and results as:

```text
<output-dir>/
├── predictions/
│   ├── <dataset>.npz
│   └── microbench__<subset>.pkl
└── results/
    ├── classification_metrics.csv
    ├── classification_metrics.json
    ├── specialty_metrics.csv
    └── specialty_metrics.json
```

With `--ci`, the result files are named:

```text
classification_metrics_ci.csv
classification_metrics_ci.json
specialty_metrics_ci.csv
specialty_metrics_ci.json
```

Multi-model runs store predictions and results under model-specific subdirectories:

```text
<output-dir>/
├── predictions/<ModelTag>/
└── results/<ModelTag>/
```

## Metrics

For binary and multiclass tasks, accuracy is computed from the top-scoring class, F1 is macro-averaged across classes, and AUC is computed by flattening one-hot labels and class scores across all sample-class pairs.

For multilabel tasks, predictions are thresholded at 0.5. Accuracy is computed across all sample-label pairs. Positive-class F1 and AUC are computed independently for each label and then macro-averaged across labels. Labels without both positive and negative samples are excluded from AUC calculation.

For MicroBench, each image-question pair is scored against its own shuffled answer options. Accuracy and macro F1 are computed over answer-option indices. Pooled AUC is computed after one-hot encoding and flattening all question-option scores.

The `OVERALL` row in `classification_metrics*.csv` is an unweighted mean over selected benchmark rows. Domain-level results are written to `specialty_metrics*.csv`; `OVERALL_SPECIALTY_MACRO` is the unweighted mean over domain rows.

## Dataset and model options

List available datasets:

```bash
python -m medpmc_clip_eval.cli --list-datasets
```

List available models:

```bash
python -m medpmc_clip_eval.cli --list-models
```

Available dataset aliases:

| Alias | Meaning |
|---|---|
| `medpmc-benchmarks` | Exact benchmark rows selected in `benchmark_manifest.csv` |
| `nonmicrobench` | The nine standard non-MicroBench datasets |
| `all` | All standard datasets plus selected MicroBench subsets |

Requesting `microbench` evaluates only the manifest-selected MicroBench subsets, not every subset in `jnirschl/uBench`.

## Optional comparison scripts

The `scripts/` directory contains utilities for paired bootstrap comparisons used in downstream manuscript analyses.

```bash
python scripts/compute_paired_specialty_macro_all_metrics.py --help
python scripts/calculate_paired_accuracy_ci.py --help
python scripts/compute_retrieval_paired_ci.py --help
```

These scripts operate on saved prediction/result files and may require path arguments depending on where outputs are stored.

## Citation

If you use MedPMC, MedPMC-CLIP, or this evaluation code, please cite:

```bibtex
@article{kim2026medpmc,
  title={MedPMC: A Systematic Framework for Scaling High-Fidelity Medical Multimodal Data for Foundation Models},
  author={Hyunjae Kim and Dain Kim and Pan Xiao and Serina S. Applebaum and Younjoon Chung and Xuguang Ai and Yu Yin and Roy Jiang and Yuexi Du and Yawen Wei and Yiming Kong and Tuo Guo and Zhiyuan Cao and Mengmeng Du and Yuelei Fu and Yan Hu and Rui Shi and Gui Yang and Kevin W. Jin and Yuntian Liu and Yuxuan Tian and Jonathan Marquez and Zhen Chen and Sheng Zhang and Hoifung Poon and Hua Xu and Jaewoo Kang and Qingyu Chen},
  journal={arXiv preprint arXiv:2607.07673},
  year={2026}
}
```

