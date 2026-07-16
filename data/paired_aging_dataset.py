"""Longitudinal face-aging pairs for optional supervised training.

The default path is intentionally conservative: pairs are split by identity,
only forward-aging pairs are produced, and no pixel-wise pair loss is assumed.
FG-NET and AgeDB contain the same person at different ages, but pose, crop,
lighting, background, and image quality are not aligned across time.
"""

from __future__ import annotations

import random
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
KAGGLE_REFS = {
    "fgnet": "aiolapo/fgnet-dataset",
    "agedb": "shukdevdatta/agedb-classwise-dataset",
}


@dataclass(frozen=True)
class AgingImageRecord:
    path: Path
    identity: str
    age: int
    gender: Optional[str] = None


@dataclass(frozen=True)
class AgingPairRecord:
    source: AgingImageRecord
    target: AgingImageRecord

    @property
    def age_gap(self) -> int:
        return int(self.target.age - self.source.age)


def download_kaggle_paired_dataset(
    dataset: str,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Download and extract one public Kaggle mirror without the Kaggle CLI."""
    dataset = str(dataset).lower().strip()
    if dataset not in KAGGLE_REFS:
        raise ValueError(f"dataset must be one of {sorted(KAGGLE_REFS)}, got {dataset!r}")

    output_dir = Path(output_dir)
    extracted_dir = output_dir / dataset
    if extracted_dir.exists() and any(extracted_dir.rglob("*")) and not force:
        return extracted_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{dataset}.zip"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_REFS[dataset]}"
    urllib.request.urlretrieve(url, archive_path)

    if force and extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Refuse zip-slip paths before extracting a third-party archive.
    root = extracted_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(root)

    return extracted_dir


def _find_image_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def discover_aging_records(
    root: str | Path,
    dataset: str = "auto",
    *,
    min_image_side: int = 128,
) -> List[AgingImageRecord]:
    """Parse exact identity/age labels from FG-NET or class-wise AgeDB names."""
    root = Path(root)
    files = _find_image_files(root)
    if not files:
        raise FileNotFoundError(f"No face images found under {root}")

    dataset = str(dataset).lower().strip()
    if dataset == "auto":
        dataset = "fgnet" if any(re.match(r"^\d{3}A\d{2}", p.stem, re.I) for p in files[:100]) else "agedb"
    if dataset not in KAGGLE_REFS:
        raise ValueError(f"dataset must be 'auto' or one of {sorted(KAGGLE_REFS)}")

    fgnet_pattern = re.compile(r"^(\d{3})A(\d{2})", re.I)
    agedb_pattern = re.compile(r"_(\d{1,3})_([mf])$", re.I)
    records: List[AgingImageRecord] = []

    for path in files:
        if min_image_side > 0:
            try:
                with Image.open(path) as image:
                    if min(image.size) < int(min_image_side):
                        continue
            except OSError:
                continue

        if dataset == "fgnet":
            match = fgnet_pattern.match(path.stem)
            if match is None:
                continue
            identity, age = match.group(1), int(match.group(2))
            gender = None
        else:
            match = agedb_pattern.search(path.stem)
            if match is None:
                continue
            identity, age = path.parent.name, int(match.group(1))
            gender = "male" if match.group(2).lower() == "m" else "female"

        records.append(AgingImageRecord(path=path, identity=identity, age=age, gender=gender))

    if not records:
        raise ValueError(f"No valid {dataset} identity/age filenames found under {root}")
    return records


def split_records_by_identity(
    records: Sequence[AgingImageRecord],
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[AgingImageRecord], List[AgingImageRecord]]:
    """Identity-disjoint split; prevents the same person leaking into validation."""
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    identities = sorted({record.identity for record in records})
    random.Random(seed).shuffle(identities)
    n_val = 0 if val_fraction == 0 else max(1, round(len(identities) * val_fraction))
    val_ids = set(identities[:n_val])
    train = [record for record in records if record.identity not in val_ids]
    val = [record for record in records if record.identity in val_ids]
    return train, val


def make_aging_pairs(
    records: Sequence[AgingImageRecord],
    *,
    min_age_gap: int = 5,
    max_age_gap: int = 40,
    max_pairs_per_identity: Optional[int] = 8,
    seed: int = 42,
) -> List[AgingPairRecord]:
    """Create forward-aging pairs and cap combinatorial growth per identity."""
    if min_age_gap <= 0 or max_age_gap < min_age_gap:
        raise ValueError("Require 0 < min_age_gap <= max_age_gap")

    grouped: Dict[str, List[AgingImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.identity, []).append(record)

    pairs: List[AgingPairRecord] = []
    for identity in sorted(grouped):
        images = sorted(grouped[identity], key=lambda item: (item.age, str(item.path)))
        candidates = [
            AgingPairRecord(source, target)
            for source in images
            for target in images
            if min_age_gap <= target.age - source.age <= max_age_gap
        ]
        if max_pairs_per_identity is not None and len(candidates) > max_pairs_per_identity:
            identity_seed = seed + sum(ord(char) for char in identity)
            candidates = random.Random(identity_seed).sample(candidates, int(max_pairs_per_identity))
            candidates.sort(key=lambda pair: (pair.source.age, pair.target.age, str(pair.source.path)))
        pairs.extend(candidates)
    if not pairs:
        raise ValueError("No longitudinal pairs satisfy the requested age-gap limits")
    return pairs


def make_age_prompt(age: int, gender: Optional[str] = None) -> str:
    person = "person"
    if gender == "male":
        person = "man"
    elif gender == "female":
        person = "woman"
    return f"a portrait photo of a {int(age)}-year-old {person}"


def make_default_image_transform(resolution: int) -> Callable[[Image.Image], torch.Tensor]:
    """Deterministic resize/center-crop without adding a torchvision dependency."""
    resolution = int(resolution)

    def transform(image: Image.Image) -> torch.Tensor:
        width, height = image.size
        scale = resolution / min(width, height)
        resized = image.resize(
            (max(resolution, round(width * scale)), max(resolution, round(height * scale))),
            resample=Image.Resampling.BICUBIC,
        )
        left = (resized.width - resolution) // 2
        top = (resized.height - resolution) // 2
        cropped = resized.crop((left, top, left + resolution, top + resolution))
        array = np.asarray(cropped, dtype=np.float32).copy() / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    return transform


class PairedAgingDataset(Dataset):
    """A small, branch-agnostic paired bundle for supervised diffusion loss."""

    def __init__(
        self,
        pairs: Sequence[AgingPairRecord],
        *,
        resolution: int = 512,
        transform: Optional[Callable] = None,
        prompt_builder: Callable[[int, Optional[str]], str] = make_age_prompt,
    ):
        if not pairs:
            raise ValueError("pairs cannot be empty")
        self.pairs = list(pairs)
        self.prompt_builder = prompt_builder
        self.transform = transform or make_default_image_transform(resolution)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index: int) -> Dict[str, object]:
        pair = self.pairs[index]
        gender = pair.target.gender or pair.source.gender
        return {
            "source_pixel_values": self._load(pair.source.path),
            "target_pixel_values": self._load(pair.target.path),
            "source_prompt": self.prompt_builder(pair.source.age, gender),
            "target_prompt": self.prompt_builder(pair.target.age, gender),
            "source_age": torch.tensor(float(pair.source.age), dtype=torch.float32),
            "target_age": torch.tensor(float(pair.target.age), dtype=torch.float32),
            "age_gap": torch.tensor(float(pair.age_gap), dtype=torch.float32),
            "identity": pair.source.identity,
            "source_path": str(pair.source.path),
            "target_path": str(pair.target.path),
        }


def build_paired_aging_dataloaders(
    root: str | Path,
    dataset: str = "auto",
    *,
    resolution: int = 512,
    batch_size: int = 2,
    val_fraction: float = 0.15,
    min_age_gap: int = 5,
    max_age_gap: int = 40,
    max_pairs_per_identity: Optional[int] = 8,
    min_image_side: int = 128,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, object]:
    records = discover_aging_records(root, dataset=dataset, min_image_side=min_image_side)
    train_records, val_records = split_records_by_identity(
        records, val_fraction=val_fraction, seed=seed
    )
    train_pairs = make_aging_pairs(
        train_records,
        min_age_gap=min_age_gap,
        max_age_gap=max_age_gap,
        max_pairs_per_identity=max_pairs_per_identity,
        seed=seed,
    )
    val_pairs = [] if not val_records else make_aging_pairs(
        val_records,
        min_age_gap=min_age_gap,
        max_age_gap=max_age_gap,
        max_pairs_per_identity=max_pairs_per_identity,
        seed=seed + 1,
    )
    train_dataset = PairedAgingDataset(train_pairs, resolution=resolution)
    val_dataset = PairedAgingDataset(val_pairs, resolution=resolution) if val_pairs else None
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None if val_dataset is None else DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
    }
