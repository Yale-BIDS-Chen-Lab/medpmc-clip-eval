from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    task: str
    class_names: list[str]
    prompts: list[Any]
    loader: Callable[[Path, int, int, Mapping[str, Any] | None], Iterable]
    restricted: bool = False


class ListDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class PathImageDataset(Dataset):
    """Lazy image-path dataset.

    Stores only paths and labels in memory. Images are opened in __getitem__.
    This avoids loading large image folders, especially RSNA, into CPU memory
    before inference starts.
    """

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label, sample_id, group_id = self.rows[index]
        path = Path(path)
        if path.suffix.lower() == ".dcm":
            image = _dicom_to_rgb(path)
        else:
            with Image.open(path) as img:
                image = img.convert("RGB")
        return image, label, sample_id, group_id


def _path_loader(rows, batch_size, workers):
    return DataLoader(
        PathImageDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=_collate,
    )


def _collate(batch):
    images, labels, ids, groups = zip(*batch)
    return {
        "images": list(images),
        "labels": np.asarray(labels),
        "sample_ids": list(ids),
        "group_ids": None if all(x is None for x in groups) else list(groups),
    }


def _loader(rows, batch_size, workers):
    return DataLoader(
        ListDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=_collate,
    )


def _path(data_paths: Mapping[str, Any] | None, key: str) -> Path | None:
    if not data_paths:
        return None
    value = data_paths.get(key)
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _require_existing(path: Path | None, *, key: str, kind: str = "path") -> Path:
    if path is None:
        raise FileNotFoundError(f"Missing required config data_paths[{key!r}]")
    if not path.exists():
        raise FileNotFoundError(f"Configured {kind} for {key!r} does not exist: {path}")
    return path


def _find_image(root: Path, stem: str, extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> Path | None:
    direct_candidates = [root / f"{stem}{ext}" for ext in extensions]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    for ext in extensions:
        matches = list(root.rglob(f"{stem}{ext}"))
        if matches:
            return sorted(matches)[0]
    return None


def _download(url: str, path: Path, *, dataset_name: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    try:
        urllib.request.urlretrieve(url, path)
    except urllib.error.HTTPError as exc:
        name = f" for {dataset_name}" if dataset_name else ""
        raise RuntimeError(
            f"Automatic download{name} failed with HTTP {exc.code}. "
            "Please download the original dataset files manually and provide "
            "their local paths through --config."
        ) from exc


def _unzip(path: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".complete"
    if marker.exists():
        return
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)
    marker.touch()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _dicom_to_rgb(path: Path) -> Image.Image:
    import pydicom

    arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
    low, high = np.percentile(arr, [0.5, 99.5])
    arr = np.clip((arr - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.repeat((arr * 255).astype(np.uint8)[..., None], 3, axis=-1)
    return Image.fromarray(rgb)


def medmnist_loader(
    flag: str,
    root: Path,
    batch_size: int,
    workers: int,
    data_paths: Mapping[str, Any] | None = None,
):
    import medmnist

    cls = getattr(medmnist, "BreastMNIST" if flag == "breastmnist" else "ChestMNIST")
    medmnist_root = _path(data_paths, "medmnist_root") or (root / "medmnist")
    medmnist_root.mkdir(parents=True, exist_ok=True)
    ds = cls(split="test", download=True, size=224, root=str(medmnist_root))
    rows = []
    for index in range(len(ds)):
        image, label = ds[index]
        value = np.asarray(label).reshape(-1)
        rows.append((
            image.convert("RGB"),
            value if len(value) > 1 else int(value[0]),
            f"{flag}:{index:08d}",
            None,
        ))
    return _loader(rows, batch_size, workers)



class PCAMH5Dataset(Dataset):
    """Lazy reader for PatchCamelyon HDF5 test files.

    The HDF5 test image file is close to 1 GB. Loading the whole array and
    materializing PIL images for every sample can exceed memory limits on
    shared HPC nodes, especially when DataLoader workers are used. This dataset
    opens the HDF5 files lazily inside each process and reads one sample at a
    time.
    """

    def __init__(self, x_path: Path, y_path: Path):
        import h5py

        self.x_path = str(x_path)
        self.y_path = str(y_path)
        with h5py.File(self.x_path, "r") as handle:
            self.x_key = "x" if "x" in handle else next(iter(handle.keys()))
            self.length = int(handle[self.x_key].shape[0])
        with h5py.File(self.y_path, "r") as handle:
            self.y_key = "y" if "y" in handle else next(iter(handle.keys()))
            y_length = int(handle[self.y_key].shape[0])
        if y_length != self.length:
            raise ValueError(f"PCAM x/y length mismatch: {self.length} vs {y_length}")
        self._x_file = None
        self._y_file = None
        self._x = None
        self._y = None

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_x_file"] = None
        state["_y_file"] = None
        state["_x"] = None
        state["_y"] = None
        return state

    def _ensure_open(self):
        if self._x_file is None or self._y_file is None:
            import h5py

            self._x_file = h5py.File(self.x_path, "r")
            self._y_file = h5py.File(self.y_path, "r")
            self._x = self._x_file[self.x_key]
            self._y = self._y_file[self.y_key]

    def __getitem__(self, index):
        self._ensure_open()
        image = Image.fromarray(np.asarray(self._x[index])).convert("RGB")
        label = int(np.asarray(self._y[index]).reshape(-1)[0])
        return image, label, f"pcam:{index:08d}", None


def pcam_loader(
    root: Path,
    batch_size: int,
    workers: int,
    data_paths: Mapping[str, Any] | None = None,
):
    x_path = _path(data_paths, "pcam_x")
    y_path = _path(data_paths, "pcam_y")
    if x_path is not None or y_path is not None:
        x_path = _require_existing(x_path, key="pcam_x", kind="file")
        y_path = _require_existing(y_path, key="pcam_y", kind="file")
        return DataLoader(
            PCAMH5Dataset(x_path, y_path),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=_collate,
        )

    raise RuntimeError(
        "PCAM requires prepared local HDF5 files for manuscript reproduction. "
        "Set data_paths.pcam_x and data_paths.pcam_y in --config."
    )


def deepdrid_loader(
    root: Path,
    batch_size: int,
    workers: int,
    data_paths: Mapping[str, Any] | None = None,
):
    csv_path = _path(data_paths, "deepdrid_csv")
    image_dir = _path(data_paths, "deepdrid_images")
    if csv_path is not None or image_dir is not None:
        csv_path = _require_existing(csv_path, key="deepdrid_csv", kind="file")
        image_dir = _require_existing(image_dir, key="deepdrid_images", kind="directory")
        df = _read_table(csv_path)
        rows = []
        skipped = 0
        for _, row in df.iterrows():
            image_id = str(
                row.get("image_id", row.get("image", row.get("ID", row.get("Image name", ""))))
            ).strip()
            if not image_id:
                skipped += 1
                continue

            label_value = row.get("DR_Levels", row.get("DR_Level", row.get("level", row.get("label"))))
            if pd.isna(label_value):
                skipped += 1
                continue
            label = int(float(label_value))

            stem = Path(image_id).stem
            patient_dir = stem.split("_")[0]
            original_path = image_dir / patient_dir / f"{stem}.jpg"
            image_path = original_path if original_path.exists() else _find_image(image_dir, stem)
            if image_path is None:
                skipped += 1
                continue
            rows.append((image_path, label, image_id, None))
        if not rows:
            raise RuntimeError(f"No labeled DeepDRiD images found using {csv_path} and {image_dir}")
        if skipped:
            print(f"DeepDRiD: skipped {skipped} rows with missing labels or images")
        return _path_loader(rows, batch_size, workers)

    raise RuntimeError(
        "DeepDRiD requires prepared local files for manuscript reproduction. "
        "Set data_paths.deepdrid_csv and data_paths.deepdrid_images in --config."
    )


def rsna_loader(
    root: Path,
    batch_size: int,
    workers: int,
    data_paths: Mapping[str, Any] | None = None,
):
    csv_path = _path(data_paths, "rsna_csv")
    image_dir = _path(data_paths, "rsna_images")
    if csv_path is not None or image_dir is not None:
        csv_path = _require_existing(csv_path, key="rsna_csv", kind="file")
        image_dir = _require_existing(image_dir, key="rsna_images", kind="directory")
        df = pd.read_csv(csv_path).drop_duplicates(subset=["patientId"])
        rows = []
        skipped = 0
        for _, row in df.iterrows():
            patient = str(row["patientId"])
            path = None
            for ext in (".dcm", ".png", ".jpg", ".jpeg"):
                candidate = image_dir / f"{patient}{ext}"
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                skipped += 1
                continue
            rows.append((path, int(row["Target"]), patient, patient))
        if not rows:
            raise RuntimeError(f"No RSNA images found using {csv_path} and {image_dir}")
        if skipped:
            print(f"RSNA: skipped {skipped} rows without matching images")
        return _path_loader(rows, batch_size, workers)

    raise RuntimeError(
        "RSNA requires prepared local files for manuscript reproduction. "
        "Set data_paths.rsna_csv and data_paths.rsna_images in --config. "
        "The paper used the Kaggle RSNA Pneumonia Detection Challenge files."
    )


def ham10000_loader(
    root: Path,
    batch_size: int,
    workers: int,
    data_paths: Mapping[str, Any] | None = None,
):
    csv_path = _path(data_paths, "ham10000_csv")
    image_dir = _path(data_paths, "ham10000_images")
    base = root / "ham10000"

    if csv_path is None and image_dir is None:
        raise RuntimeError(
            "HAM10000/ISIC 2018 Task 3 requires prepared local files for manuscript reproduction. "
            "Set data_paths.ham10000_csv and data_paths.ham10000_images in --config."
        )
    else:
        csv_path = _require_existing(csv_path, key="ham10000_csv", kind="file")
        image_dir = _require_existing(image_dir, key="ham10000_images", kind="directory")

    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    df = pd.read_csv(csv_path)
    rows = []
    skipped = 0
    for _, row in df.iterrows():
        image_id = str(row["image"])
        path = _find_image(image_dir, image_id)
        if path is None:
            skipped += 1
            continue
        label = int(np.argmax(row[classes].to_numpy(dtype=float)))
        rows.append((path, label, Path(image_id).stem, None))
    if not rows:
        raise RuntimeError(f"No HAM10000/ISIC images found using {csv_path} and {image_dir}")
    if skipped:
        print(f"HAM10000: skipped {skipped} rows without matching images")
    return _path_loader(rows, batch_size, workers)


def lc25000_loader(
    organ: str,
    root: Path,
    batch_size: int,
    workers: int,
    data_paths: Mapping[str, Any] | None = None,
):
    root_dir = _path(data_paths, "lc25000_root")
    if root_dir is not None:
        root_dir = _require_existing(root_dir, key="lc25000_root", kind="directory")
        if organ == "lung":
            class_map = {"lung_n": 0, "lung_aca": 1, "lung_scc": 2}
        else:
            class_map = {"colon_n": 0, "colon_aca": 1}
        rows = []
        for class_name, label in class_map.items():
            class_dir = root_dir / class_name
            files = []
            if class_dir.exists():
                files = list(class_dir.rglob("*.jpeg")) + list(class_dir.rglob("*.jpg")) + list(class_dir.rglob("*.png"))
            else:
                files = [p for p in root_dir.rglob("*") if p.is_file() and class_name.lower() in str(p.parent).lower()]
            for path in sorted(set(files)):
                rows.append((path, int(label), f"lc25000_{organ}:{class_name}:{path.stem}", None))
        if not rows:
            raise RuntimeError(f"No LC25000 {organ} images found under {root_dir}")
        return _path_loader(rows, batch_size, workers)

    raise RuntimeError(
        f"LC25000 {organ} requires a prepared local folder for manuscript reproduction. "
        "Set data_paths.lc25000_root in --config."
    )


def dad_loader(
    root: Path,
    batch_size: int,
    workers: int,
    data_paths: Mapping[str, Any] | None = None,
):
    annotations = _path(data_paths, "dad_annotations")
    image_root = _path(data_paths, "dad_images")
    if annotations is not None or image_root is not None:
        annotations = _require_existing(annotations, key="dad_annotations", kind="file")
        image_root = _require_existing(image_root, key="dad_images", kind="directory")
        coco = json.loads(annotations.read_text())
    else:
        raise RuntimeError(
            "DAD requires prepared local COCO-style test annotations/images for manuscript reproduction. "
            "Set data_paths.dad_annotations and data_paths.dad_images in --config."
        )

    image_to_categories = {}
    for ann in coco.get("annotations", []):
        image_to_categories.setdefault(int(ann["image_id"]), set()).add(int(ann["category_id"]))
    rows = []
    for info in coco["images"]:
        image_id = int(info["id"])
        target = np.zeros(10, dtype=np.uint8)
        for category_id in image_to_categories.get(image_id, set()):
            if 1 <= category_id <= 10:
                target[category_id - 1] = 1
        path = image_root / info["file_name"]
        rows.append((path, target, str(image_id), None))
    return _path_loader(rows, batch_size, workers)


# Prompt text and class order follow the original clip-ci evaluation code.
BREAST_CLASSES = ["malignant", "benign_or_normal"]
BREAST_PROMPTS = [
    "breast ultrasound of a malignant tumor",
    "breast ultrasound of benign lesion or normal tissue",
]

CHEST_CLASSES = [
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural",
    "hernia",
]
CHEST_PROMPTS = [[f"chest x-ray without {x}", f"chest x-ray of {x}"] for x in CHEST_CLASSES]

HAM_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
HAM_PROMPTS = [
    "dermoscopic image of melanoma",
    "dermoscopic image of a melanocytic nevi",
    "dermoscopic image of basal cell carcinoma",
    "dermoscopic image of Actinic Keratoses (Solar Keratoses) and Intraepithelial Carcinoma (Bowen's disease)",
    "dermoscopic image of a benign keratosis",
    "dermoscopic image of a dermatofibroma",
    "dermoscopic image of a vascular lesion",
]

DAD_CLASSES = [
    "colon",
    "inferior_mesenteric_artery",
    "intestinal_veins",
    "liver",
    "pancreas",
    "small_intestine",
    "spleen",
    "stomach",
    "ureter",
    "vesicular_glands",
]
DAD_PROMPTS = [
    [
        f"laparoscopic view without the {name.replace('_', ' ')}",
        f"laparoscopic view of the {name.replace('_', ' ')}",
    ]
    for name in DAD_CLASSES
]

REGISTRY = {
    "breastmnist": DatasetSpec(
        "breastmnist", "binary", BREAST_CLASSES, BREAST_PROMPTS,
        lambda root, b, w, p=None: medmnist_loader("breastmnist", root, b, w, p),
    ),
    "chestmnist": DatasetSpec(
        "chestmnist", "multilabel", CHEST_CLASSES, CHEST_PROMPTS,
        lambda root, b, w, p=None: medmnist_loader("chestmnist", root, b, w, p),
    ),
    "pcam": DatasetSpec(
        "pcam", "binary", ["negative", "positive"],
        ["histopathology image of normal lymph node tissue", "histopathology image of metastatic tumor tissue"],
        pcam_loader,
    ),
    "deepdrid": DatasetSpec(
        "deepdrid", "multiclass", ["DR0", "DR1", "DR2", "DR3", "DR4"],
        [
            "a colorful fundus photo of no diabetic retinopathy",
            "a colorful fundus photo of mild non-proliferative diabetic retinopathy",
            "a colorful fundus photo of moderate non-proliferative diabetic retinopathy",
            "a colorful fundus photo of severe non-proliferative diabetic retinopathy",
            "a colorful fundus photo of proliferative diabetic retinopathy",
        ],
        deepdrid_loader,
    ),
    "rsna": DatasetSpec(
        "rsna", "binary", ["negative", "positive"],
        ["chest x-ray of no pneumonia", "chest x-ray of pneumonia"],
        rsna_loader, restricted=True,
    ),
    "ham10000": DatasetSpec("ham10000", "multiclass", HAM_CLASSES, HAM_PROMPTS, ham10000_loader),
    "lc25000_lung": DatasetSpec(
        "lc25000_lung", "multiclass",
        ["lung_n", "lung_aca", "lung_scc"],
        ["histopathology image of benign lung tissue", "histopathology image of lung adenocarcinoma",
         "histopathology image of lung squamous cell carcinoma"],
        lambda root, b, w, p=None: lc25000_loader("lung", root, b, w, p),
    ),
    "lc25000_colon": DatasetSpec(
        "lc25000_colon", "binary", ["colon_n", "colon_aca"],
        ["histopathology image of benign colon tissue", "histopathology image of colon adenocarcinoma"],
        lambda root, b, w, p=None: lc25000_loader("colon", root, b, w, p),
    ),
    "dad": DatasetSpec("dad", "multilabel", DAD_CLASSES, DAD_PROMPTS, dad_loader, restricted=True),
}

ALIASES = {
    "all": list(REGISTRY) + ["microbench"],
    "nonmicrobench": list(REGISTRY),
}
