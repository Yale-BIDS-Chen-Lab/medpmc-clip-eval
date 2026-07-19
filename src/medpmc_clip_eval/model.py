from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image


MEDPMC_MODEL_REPO = "Yale-BIDS-Chen/medpmc-clip-l-14_jun24_v1"
MEDPMC_MODEL_FILE = "open_clip_pytorch_model.safetensors"
DEFAULT_OPENCLIP_ARCH = "ViT-L-14"

MODEL_ALIASES = {
    "medpmc": "MedPMC-CLIP",
    "bmc": "BMC",
    "biomedclip": "BioMedCLIP",
    "pmcclip": "PMC-CLIP",
    "openclip": "OpenCLIP",
    "coca": "CoCa",
    "medsiglip": "MedSigLIP",
}

PUBLIC_BASELINE_MODELS = [
    "medpmc",
    "bmc",
    "biomedclip",
    "pmcclip",
    "openclip",
    "coca",
    "medsiglip",
]


def normalize_model_key(model_key: str) -> str:
    aliases = {
        "medpmc-clip": "medpmc",
        "medpmc_clip": "medpmc",
        "medpmcclip": "medpmc",
        "bmc-clip": "bmc",
        "bmc_clip": "bmc",
        "biomed": "biomedclip",
        "biomed-clip": "biomedclip",
        "pmc": "pmcclip",
        "pmc-clip": "pmcclip",
        "siglip": "medsiglip",
        "med-siglip": "medsiglip",
    }
    key = aliases.get(str(model_key).strip().lower(), str(model_key).strip().lower())
    if key not in MODEL_ALIASES:
        raise KeyError(f"Unknown model: {model_key}. Choices: {', '.join(MODEL_ALIASES)}")
    return key


def model_tag_for_key(model_key: str) -> str:
    return MODEL_ALIASES[normalize_model_key(model_key)]


def _resolve_device(device: str | torch.device) -> torch.device:
    device = str(device)
    return torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")


def _as_rgb_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    array = np.asarray(image)
    if (
        array.ndim == 3
        and array.shape[0] in {1, 3, 4}
        and array.shape[-1] not in {1, 3, 4}
    ):
        array = np.moveaxis(array, 0, -1)

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]

    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.dtype != np.uint8:
        array = array.astype(np.float32, copy=False)
        if array.size and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array).convert("RGB")


@dataclass
class OpenCLIPScorer:
    model: Any
    tokenizer: Any
    preprocess: Any
    device: torch.device
    use_amp: bool = True
    logit_scale: float = 100.0

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.model.eval()

    def _amp(self):
        if self.use_amp and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
        tensors = [self.preprocess(_as_rgb_pil(image)) for image in images]
        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        with torch.inference_mode(), self._amp():
            features = self.model.encode_image(batch)
        return F.normalize(features.float(), dim=-1)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.inference_mode(), self._amp():
            features = self.model.encode_text(tokens)
        return F.normalize(features.float(), dim=-1)

    def class_prototypes(self, prompts: Sequence[str | Sequence[str]]) -> torch.Tensor:
        rows = []
        for item in prompts:
            variants = [item] if isinstance(item, str) else list(item)
            embedded = self.encode_texts([str(x) for x in variants])
            rows.append(F.normalize(embedded.mean(0, keepdim=True), dim=-1)[0])
        return torch.stack(rows)

    def multiclass_scores(
        self,
        images: Sequence[Any],
        prompts: Sequence[str | Sequence[str]],
    ):
        image_features = self.encode_images(images)
        prototypes = self.class_prototypes(prompts)
        logits = self.logit_scale * image_features @ prototypes.T
        return logits.softmax(-1).cpu().numpy()

    def multilabel_scores(
        self,
        images: Sequence[Any],
        prompt_pairs: Sequence[Sequence[str]],
    ):
        image_features = self.encode_images(images)
        flat = [p for pair in prompt_pairs for p in pair]
        text_features = self.encode_texts(flat).reshape(len(prompt_pairs), 2, -1)
        logits = self.logit_scale * torch.einsum("bd,lpd->blp", image_features, text_features)
        return logits.softmax(-1)[..., 1].cpu().numpy()

    def variable_option_scores(
        self,
        images: Sequence[Any],
        options: Sequence[Sequence[str]],
    ):
        """Score each image against its own shuffled answer options."""
        if len(images) != len(options):
            raise ValueError("MicroBench image/option count mismatch")
        if len(images) == 0:
            return []

        option_counts = [len(x) for x in options]
        if min(option_counts) < 2:
            raise ValueError("Each MicroBench question must contain at least two options")

        image_features = self.encode_images(images)
        output = [None] * len(images)

        groups: dict[int, list[int]] = {}
        for index, count in enumerate(option_counts):
            groups.setdefault(int(count), []).append(index)

        for n_options, indices in groups.items():
            flat_texts = [
                str(text)
                for index in indices
                for text in options[index]
            ]
            text_features = self.encode_texts(flat_texts).reshape(
                len(indices), n_options, -1
            )
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=image_features.device)
            selected_images = image_features.index_select(0, index_tensor)
            logits = self.logit_scale * torch.bmm(
                selected_images.unsqueeze(1),
                text_features.transpose(1, 2),
            ).squeeze(1)
            probs = logits.softmax(dim=-1).detach().cpu().numpy()
            for local_index, original_index in enumerate(indices):
                output[original_index] = probs[local_index].astype("float32", copy=False)

        if any(value is None for value in output):
            raise RuntimeError("At least one MicroBench question was not scored")
        return [np.asarray(value, dtype=np.float32) for value in output]


@dataclass
class HFSigLIPScorer:
    model: Any
    processor: Any
    device: torch.device
    use_amp: bool = True
    logit_scale: float = 100.0

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.model.eval()

    def _amp(self):
        if self.use_amp and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
        prepared = [_as_rgb_pil(image) for image in images]
        inputs = self.processor(images=prepared, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode(), self._amp():
            features = self.model.get_image_features(**inputs)
        return F.normalize(features.float(), dim=-1)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=list(texts), padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode(), self._amp():
            features = self.model.get_text_features(**inputs)
        return F.normalize(features.float(), dim=-1)

    # Reuse the common scoring implementation.
    class_prototypes = OpenCLIPScorer.class_prototypes
    multiclass_scores = OpenCLIPScorer.multiclass_scores
    multilabel_scores = OpenCLIPScorer.multilabel_scores
    variable_option_scores = OpenCLIPScorer.variable_option_scores


def _create_openclip_with_checkpoint(model_name: str, checkpoint: str):
    import open_clip

    kwargs = {"model_name": model_name, "pretrained": checkpoint}
    try:
        return open_clip.create_model_and_transforms(**kwargs, weights_only=False)
    except TypeError:
        return open_clip.create_model_and_transforms(**kwargs)


def _load_medpmc_from_hf(
    *,
    device: torch.device,
    cache_dir: str | Path | None,
    use_amp: bool,
    config: Mapping[str, Any],
) -> tuple[str, OpenCLIPScorer, str]:
    import open_clip

    cfg = config.get("models", {}).get("medpmc", {})
    repo_id = str(cfg.get("repo_id", MEDPMC_MODEL_REPO))
    filename = str(cfg.get("filename", MEDPMC_MODEL_FILE))
    model_name = str(cfg.get("model_name", DEFAULT_OPENCLIP_ARCH))

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None)
    ckpt = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    state = load_file(ckpt, device="cpu")
    model.load_state_dict(state, strict=True)
    tokenizer = open_clip.get_tokenizer(model_name)
    scorer = OpenCLIPScorer(model, tokenizer, preprocess, device, use_amp=use_amp)
    print(f"[model] MedPMC-CLIP loaded from {ckpt}")
    return MODEL_ALIASES["medpmc"], scorer, ckpt


def load_scorer(
    model_key: str = "medpmc",
    config: Mapping[str, Any] | None = None,
    *,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
    checkpoint_override: str | None = None,
    use_amp: bool = True,
) -> tuple[str, Any, str]:
    """Return `(model_tag, scorer, source_description)` for all supported models.

    Supported public model keys:
      medpmc, bmc, biomedclip, pmcclip, openclip, coca, medsiglip

    Public MedPMC-CLIP is loaded with `--model medpmc` from Hugging Face.
    `--checkpoint` can be used to override the local BMC checkpoint only.
    """
    config = config or {}
    key = normalize_model_key(model_key)
    resolved = _resolve_device(device)

    if key == "medpmc":
        return _load_medpmc_from_hf(
            device=resolved,
            cache_dir=cache_dir,
            use_amp=use_amp,
            config=config,
        )

    model_config = config.get("models", {}).get(key, {})
    tag = MODEL_ALIASES[key]

    if key == "medsiglip":
        from transformers import AutoModel, AutoProcessor

        model_id = str(model_config.get("model_id", "google/medsiglip-448"))
        print(f"[model] loading {model_id}")
        model = AutoModel.from_pretrained(model_id, cache_dir=str(cache_dir) if cache_dir else None)
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        scorer = HFSigLIPScorer(model, processor, resolved, use_amp=use_amp)
        return tag, scorer, model_id

    if key == "bmc":
        model_name = str(model_config.get("model_name", DEFAULT_OPENCLIP_ARCH))
        checkpoint = checkpoint_override or model_config.get("checkpoint")
        if checkpoint:
            checkpoint = str(checkpoint)
            if not Path(checkpoint).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        else:
            checkpoint = hf_hub_download(
                repo_id=str(model_config.get("repo_id", "BIOMEDICA/BMC_CLIP_CF")),
                filename=str(model_config.get("filename", "BMC_CLIP_CF.pt")),
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        print(f"[model] loading {model_name} from {checkpoint}")
        import open_clip

        model, _, preprocess = _create_openclip_with_checkpoint(model_name, checkpoint)
        tokenizer = open_clip.get_tokenizer(model_name)
        scorer = OpenCLIPScorer(model, tokenizer, preprocess, resolved, use_amp=use_amp)
        return tag, scorer, checkpoint

    if key == "biomedclip":
        import open_clip

        model_id = str(
            model_config.get(
                "model_id",
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            )
        )
        print(f"[model] loading {model_id}")
        model, preprocess = open_clip.create_model_from_pretrained(model_id)
        tokenizer = open_clip.get_tokenizer(model_id)
        scorer = OpenCLIPScorer(model, tokenizer, preprocess, resolved, use_amp=use_amp)
        return tag, scorer, model_id

    if key == "pmcclip":
        import open_clip

        model_id = str(model_config.get("model_id", "hf-hub:ryanyip7777/pmc_vit_l_14"))
        print(f"[model] loading {model_id}")
        model, preprocess = open_clip.create_model_from_pretrained(model_id)
        tokenizer = open_clip.get_tokenizer(model_id)
        scorer = OpenCLIPScorer(model, tokenizer, preprocess, resolved, use_amp=use_amp)
        return tag, scorer, model_id

    if key == "openclip":
        import open_clip

        model_name = str(model_config.get("model_name", DEFAULT_OPENCLIP_ARCH))
        pretrained = str(model_config.get("pretrained", "commonpool_xl_clip_s13b_b90k"))
        print(f"[model] loading {model_name}/{pretrained}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        scorer = OpenCLIPScorer(model, tokenizer, preprocess, resolved, use_amp=use_amp)
        return tag, scorer, f"{model_name}/{pretrained}"

    if key == "coca":
        import open_clip

        model_name = str(model_config.get("model_name", "coca_ViT-L-14"))
        pretrained = str(model_config.get("pretrained", "mscoco_finetuned_laion2b_s13b_b90k"))
        print(f"[model] loading {model_name}/{pretrained}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )
        try:
            tokenizer = open_clip.get_tokenizer(model_name)
        except Exception:
            tokenizer = open_clip.get_tokenizer(DEFAULT_OPENCLIP_ARCH)
        scorer = OpenCLIPScorer(model, tokenizer, preprocess, resolved, use_amp=use_amp)
        return tag, scorer, f"{model_name}/{pretrained}"

    raise AssertionError(key)


class MedPMCClip(OpenCLIPScorer):
    """Backward-compatible alias for code importing MedPMCClip directly."""

    @classmethod
    def load(
        cls,
        *,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
        use_amp: bool = True,
    ) -> "MedPMCClip":
        _, scorer, _ = _load_medpmc_from_hf(
            device=_resolve_device(device),
            cache_dir=cache_dir,
            use_amp=use_amp,
            config={},
        )
        return cls(
            scorer.model,
            scorer.tokenizer,
            scorer.preprocess,
            scorer.device,
            use_amp=scorer.use_amp,
            logit_scale=scorer.logit_scale,
        )
