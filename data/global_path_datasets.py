# ============================================================
# GLOBAL FACE AGING DATASET + DATALOADER
# Final portable version
# ============================================================

import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(__file__).resolve().parent

DRIVE_ZIPS = [
    Path("/content/drive/MyDrive/ffhq_subset_6k_extremes-001.zip"),
    Path("/content/drive/MyDrive/ffhq_subset_6k_extremes-002.zip"),
    Path("/content/drive/MyDrive/ffhq_subset_6k_extremes-003.zip"),
    Path("/content/drive/MyDrive/ffhq_subset_6k_extremes-004.zip")]

GLOBAL_IMAGE_DIR = Path("/content/ffhq_subset_6k_extremes_local")
REPO_GLOBAL_CSV_PATH = DATA_DIR / "ffhq_predictions" / "ffhq_face_attribute_prompts.csv"
COLAB_GLOBAL_CSV_PATH = Path("/content/ffhq_face_attribute_prompts.csv")
GLOBAL_CSV_PATH = REPO_GLOBAL_CSV_PATH if REPO_GLOBAL_CSV_PATH.exists() else COLAB_GLOBAL_CSV_PATH

GLOBAL_RESOLUTION = 512
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)


# ============================================================
# Extract images from Drive ZIPs if needed
# ============================================================

def ensure_images_available(
    zip_paths: List[Path],
    output_dir: Path,
    force_reextract: bool = False) -> Path:

    """
    Ensures images are available locally.

    If output_dir already exists and contains images, it does nothing.
    Otherwise, it extracts all ZIP files into output_dir.
    """
    if output_dir.exists() and not force_reextract:
        existing_images = [
            p for p in output_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and not p.name.startswith("._")
            and "__MACOSX" not in str(p)
        ]

        if len(existing_images) > 0:
            print("[Images already available]")
            print("Local image dir:", output_dir)
            print("Local images:", len(existing_images))
            return output_dir

    missing_zips = [str(p) for p in zip_paths if not p.exists()]
    if len(missing_zips) > 0:
        raise FileNotFoundError(
            "Missing ZIP files. Make sure Drive is mounted and paths are correct:\n"
            + "\n".join(missing_zips)
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Extracting ZIP files]")
    for zip_path in zip_paths:
        print("Extracting:", zip_path.name)

        with zipfile.ZipFile(zip_path, "r") as z:
            bad_file = z.testzip()
            if bad_file is not None:
                raise RuntimeError(f"Corrupted file inside {zip_path}: {bad_file}")

            z.extractall(output_dir)

    all_files = [
        p for p in output_dir.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and "__MACOSX" not in str(p)
    ]

    image_files = [
        p for p in all_files
        if p.suffix.lower() in IMAGE_EXTS]

    print("\n[Extraction summary]")
    print("Extracted to:", output_dir)
    print("Total files:", len(all_files))
    print("Total images:", len(image_files))

    ext_counts = Counter(p.suffix.lower() for p in all_files)
    print("\n[Extension counts]")
    for ext, count in ext_counts.most_common(15):
        print(f"{ext if ext else '[no extension]'}: {count}")

    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found after extraction in: {output_dir}")

    return output_dir


def prepare_global_assets(
    zip_paths: List[Path] = DRIVE_ZIPS,
    output_dir: Path = GLOBAL_IMAGE_DIR,
    force_reextract: bool = False,
) -> Dict[str, Path]:
    """
    Extracts/indexes global images explicitly.

    The Drive paths are intentionally kept as Colab/Drive paths. Calling this
    on a local machine without mounted Drive should fail clearly, but importing
    this module should not.
    """
    image_dir = ensure_images_available(
        zip_paths=zip_paths,
        output_dir=output_dir,
        force_reextract=force_reextract,
    )
    return build_image_index(image_dir)


# ============================================================
# Basic helpers
# ============================================================

def valid_text(x: Any) -> bool:
    return (
        isinstance(x, str)
        and len(x.strip()) > 0
        and x.strip().lower() not in ["nan", "none", "null"]
    )


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def clean_gender_label(x: Any) -> Optional[str]:
    if x is None or pd.isna(x):
        return None

    label = str(x).strip().lower()

    if label in ["male", "man", "masculino", "m", "0", "label_0"]:
        return "male"

    if label in ["female", "woman", "femenino", "f", "1", "label_1"]:
        return "female"

    if "female" in label or "woman" in label:
        return "female"

    if "male" in label or "man" in label:
        return "male"

    return None


# ============================================================
# Image index
# ============================================================

def build_image_index(image_root: Path) -> Dict[str, Path]:
    """
    Builds an index:
        filename -> path
        stem     -> path
    """
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    image_paths = [
        p for p in image_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.name.startswith("._")
        and "__MACOSX" not in str(p)]

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in: {image_root}")

    index = {}
    duplicate_keys = []

    for path in image_paths:
        for key in [path.name, path.stem]:
            if key in index:
                duplicate_keys.append(key)
            index[key] = path

    print("\n[Image index]")
    print("Found image files:", len(image_paths))
    print("Index keys:", len(index))
    print("Duplicate keys overwritten:", len(duplicate_keys))

    if len(duplicate_keys) > 0:
        print("First duplicate keys:", duplicate_keys[:10])

    return index


global_image_index = None


# ============================================================
# Prompt builder
# ============================================================

def make_global_prompt_from_attributes(
    age: float,
    gender_label: Any = None,
    gender_confidence: Any = None,
    skin_tone_label: Any = None,
    skin_tone_confidence: Any = None,
    hair_label: Any = None,
    hair_confidence: Any = None,
    glasses_label: Any = None,
    glasses_confidence: Any = None,
    gender_threshold: float = 0.65,
    skin_threshold: float = 0.40,
    hair_threshold: float = 0.35,
    glasses_threshold: float = 0.55) -> str:

    """
    Builds final global prompt.

    Gender is included only if gender_confidence >= 0.65.
    Other visual attributes are included only if confidence is high enough.
    """
    age_int = int(round(float(age)))
    age_int = max(0, min(100, age_int))

    gender_confidence = safe_float(gender_confidence, default=None)
    gender_label = clean_gender_label(gender_label)

    if (
        gender_label == "male"
        and gender_confidence is not None
        and gender_confidence >= gender_threshold):

        person_phrase = f"a {age_int}-year-old man"

    elif (
        gender_label == "female"
        and gender_confidence is not None
        and gender_confidence >= gender_threshold):

        person_phrase = f"a {age_int}-year-old woman"

    else:
        person_phrase = f"a {age_int}-year-old person"

    attr_parts = []

    skin_confidence = safe_float(skin_tone_confidence, default=None)
    if (
        valid_text(skin_tone_label)
        and skin_confidence is not None
        and skin_confidence >= skin_threshold):

        attr_parts.append(str(skin_tone_label).strip())

    hair_confidence = safe_float(hair_confidence, default=None)
    if (
        valid_text(hair_label)
        and hair_confidence is not None
        and hair_confidence >= hair_threshold):

        attr_parts.append(str(hair_label).strip())

    glasses_confidence = safe_float(glasses_confidence, default=None)
    if (
        valid_text(glasses_label)
        and glasses_confidence is not None
        and glasses_confidence >= glasses_threshold
        and str(glasses_label).strip().lower() == "wearing glasses"):

        attr_parts.append("wearing glasses")

    if len(attr_parts) > 0:
        return "a portrait photo of " + person_phrase + ", " + ", ".join(attr_parts)

    return "a portrait photo of " + person_phrase


# ============================================================
# Build samples from final CSV
# ============================================================

def build_global_samples_from_attribute_csv(
    csv_path: Path,
    image_index: Dict[str, Path],
    filename_col: str = "filename",
    age_col: str = "age_pred",
    min_age: Optional[float] = None,
    max_age: Optional[float] = None,
    require_existing_image: bool = True) -> List[Dict[str, Any]]:

    if not csv_path.exists():
        fallback_csv_path = REPO_GLOBAL_CSV_PATH if REPO_GLOBAL_CSV_PATH.exists() else COLAB_GLOBAL_CSV_PATH
        if fallback_csv_path.exists():
            print(f"[WARN] CSV not found at {csv_path}. Falling back to: {fallback_csv_path}")
            csv_path = fallback_csv_path
        else:
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                f"Also checked repo CSV: {REPO_GLOBAL_CSV_PATH}\n"
                f"Also checked Colab CSV: {COLAB_GLOBAL_CSV_PATH}"
            )

    df = pd.read_csv(csv_path).copy()

    required_cols = [
        filename_col,
        age_col,
        "gender_label",
        "gender_confidence",
        "skin_tone_label",
        "skin_tone_confidence",
        "hair_label",
        "hair_confidence",
        "glasses_label",
        "glasses_confidence"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df = df.dropna(subset=[filename_col, age_col]).reset_index(drop=True)

    if min_age is not None:
        df = df[df[age_col] >= min_age]

    if max_age is not None:
        df = df[df[age_col] <= max_age]

    samples = []
    missing_images = []

    for _, row in df.iterrows():
        filename = str(row[filename_col])
        filename_name = Path(filename).name
        stem = Path(filename).stem
        age = float(row[age_col])

        image_path = image_index.get(filename, None)
        if image_path is None:
            image_path = image_index.get(filename_name, None)
        if image_path is None:
            image_path = image_index.get(stem, None)

        if image_path is None:
            missing_images.append(filename)

            if require_existing_image:
                continue
            else:
                raise FileNotFoundError(f"Image not found for filename={filename}")

        prompt = make_global_prompt_from_attributes(
            age=age,
            gender_label=row.get("gender_label", None),
            gender_confidence=row.get("gender_confidence", None),
            skin_tone_label=row.get("skin_tone_label", None),
            skin_tone_confidence=row.get("skin_tone_confidence", None),
            hair_label=row.get("hair_label", None),
            hair_confidence=row.get("hair_confidence", None),
            glasses_label=row.get("glasses_label", None),
            glasses_confidence=row.get("glasses_confidence", None),
            gender_threshold=0.65,
            skin_threshold=0.40,
            hair_threshold=0.35,
            glasses_threshold=0.55)

        sample = {
            "filename": filename,
            "stem": stem,
            "image_path": str(image_path),
            "age": age,
            "age_rounded": int(round(age)),
            "age_norm": age / 100.0,
            "prompt": prompt,

            # Metadata for debugging
            "gender_pred": row.get("gender_pred", None),
            "gender_label": row.get("gender_label", None),
            "gender_confidence": row.get("gender_confidence", None),
            "skin_tone_label": row.get("skin_tone_label", None),
            "skin_tone_confidence": row.get("skin_tone_confidence", None),
            "hair_label": row.get("hair_label", None),
            "hair_confidence": row.get("hair_confidence", None),
            "glasses_label": row.get("glasses_label", None),
            "glasses_confidence": row.get("glasses_confidence", None),
            "face_attribute_phrase": row.get("face_attribute_phrase", None),
            "enriched_prompt_original": row.get("enriched_prompt", None),
            "error": row.get("error", None)}

        samples.append(sample)

    print("\n[Dataset build summary]")
    print("CSV rows after filtering:", len(df))
    print("Matched global samples:", len(samples))
    print("Missing images:", len(missing_images))

    if len(missing_images) > 0:
        print("First missing images:", missing_images[:20])

    print("\n[Prompt check: first 10]")
    for s in samples[:10]:
        print(
            f"{s['filename']} | "
            f"gender={s['gender_label']} | "
            f"conf={s['gender_confidence']} | "
            f"{s['prompt']}"
        )

    if len(samples) == 0:
        raise ValueError("No valid global samples were created.")

    return samples


# ============================================================
# Dataset
# ============================================================

class GlobalAgingFaceDataset(Dataset):
    """
    Dataset for global face aging branch.

    One item = one full-face image.

    Returned:
        pixel_values: [3, resolution, resolution], normalized to [-1, 1]
        age: apparent age, float
        age_norm: apparent age / 100
        prompt: final controlled prompt
        filename: image filename
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        resolution: int = 512,
        train: bool = True,
        normalize_for_diffusion: bool = True,
    ):
        self.samples = samples
        self.resolution = int(resolution)
        self.train = bool(train)
        self.normalize_for_diffusion = bool(normalize_for_diffusion)

        if len(self.samples) == 0:
            raise ValueError("Global dataset has zero samples.")

        if train:
            transform_list = [
                transforms.Resize(
                    (self.resolution, self.resolution),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.035,
                            contrast=0.035,
                            saturation=0.025,
                            hue=0.004,
                        )
                    ],
                    p=0.25,
                ),
                transforms.ToTensor(),
            ]
        else:
            transform_list = [
                transforms.Resize(
                    (self.resolution, self.resolution),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]

        if self.normalize_for_diffusion:
            transform_list.append(
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            )

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.transform(image)

        age = float(sample["age"])
        prompt = sample["prompt"]

        return {
            "pixel_values": pixel_values,
            "age": torch.tensor(age, dtype=torch.float32),
            "age_norm": torch.tensor(age / 100.0, dtype=torch.float32),
            "prompt": prompt,
            "filename": sample["filename"],
            "stem": sample["stem"],
            "image_path": sample["image_path"],
            "gender_label": sample.get("gender_label", None),
            "gender_confidence": sample.get("gender_confidence", None),
        }


# ============================================================
# Collate functions
# ============================================================

def global_train_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "age": torch.stack([b["age"] for b in batch], dim=0),
        "age_norm": torch.stack([b["age_norm"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch]}


def global_debug_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "age": torch.stack([b["age"] for b in batch], dim=0),
        "age_norm": torch.stack([b["age_norm"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
        "filename": [b["filename"] for b in batch],
        "stem": [b["stem"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
        "gender_label": [b["gender_label"] for b in batch],
        "gender_confidence": [b["gender_confidence"] for b in batch]}
