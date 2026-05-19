from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.local_path_dataset import (
    LOCAL_RESOLUTION,
    REGION_ORDER,
    make_local_prompt,
    make_zone_prompt,
    normalize_region_skip_list,
    region_aware_bbox_to_square,
)


GLOBAL_CSV_PATH = Path(__file__).resolve().parent / "ffhq_predictions" / "ffhq_face_attribute_prompts.csv"
FUSED_FULL_RESOLUTION = 512


def make_soft_rect_mask(size: int = 256, feather: int = 24) -> torch.Tensor:
    mask = torch.ones(1, size, size, dtype=torch.float32)

    feather = int(max(1, min(feather, size // 2)))
    ramp = torch.linspace(0.0, 1.0, feather, dtype=torch.float32)

    mask[:, :feather, :] *= ramp.view(1, feather, 1)
    mask[:, -feather:, :] *= ramp.flip(0).view(1, feather, 1)
    mask[:, :, :feather] *= ramp.view(1, 1, feather)
    mask[:, :, -feather:] *= ramp.flip(0).view(1, 1, feather)

    return mask.clamp(0.0, 1.0)


def make_bbox_soft_mask(
    *,
    size: int,
    bbox: Dict[str, Any],
    crop_box: tuple[int, int, int, int],
    feather: int = 24,
    expansion: float = 1.12,
) -> torch.Tensor:
    """
    Soft mask for the annotated local region inside a square context crop.

    The crop image contains context around the annotation, but insertion should
    not paste that whole square back into the face. Otherwise later zones can
    overwrite earlier zones and make it look like a region was not generated.
    """
    left, top, right, bottom = [float(v) for v in crop_box]
    crop_w = max(1.0, right - left)
    crop_h = max(1.0, bottom - top)

    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])

    expansion = max(1.0, float(expansion))
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    w = w * expansion
    h = h * expansion
    x1_f = cx - 0.5 * w
    x2_f = cx + 0.5 * w
    y1_f = cy - 0.5 * h
    y2_f = cy + 0.5 * h

    x1 = int(round((x1_f - left) / crop_w * size))
    x2 = int(round((x2_f - left) / crop_w * size))
    y1 = int(round((y1_f - top) / crop_h * size))
    y2 = int(round((y2_f - top) / crop_h * size))

    x1 = max(0, min(size - 1, x1))
    x2 = max(x1 + 1, min(size, x2))
    y1 = max(0, min(size - 1, y1))
    y2 = max(y1 + 1, min(size, y2))

    mask = torch.zeros(1, size, size, dtype=torch.float32)
    patch_h = y2 - y1
    patch_w = x2 - x1
    patch = torch.ones(1, patch_h, patch_w, dtype=torch.float32)

    feather = int(max(0, min(feather, patch_h // 2, patch_w // 2)))
    if feather > 0:
        ramp = torch.linspace(0.0, 1.0, feather, dtype=torch.float32)
        patch[:, :feather, :] *= ramp.view(1, feather, 1)
        patch[:, -feather:, :] *= ramp.flip(0).view(1, feather, 1)
        patch[:, :, :feather] *= ramp.view(1, 1, feather)
        patch[:, :, -feather:] *= ramp.flip(0).view(1, 1, feather)

    mask[:, y1:y2, x1:x2] = patch
    return mask.clamp(0.0, 1.0)


def load_global_prompt_lookup(csv_path: Path = GLOBAL_CSV_PATH) -> Dict[str, str]:
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        return {}

    prompt_col = "enriched_prompt" if "enriched_prompt" in df.columns else None
    if prompt_col is None:
        return {}

    lookup: Dict[str, str] = {}
    for row in df.to_dict("records"):
        filename = str(row.get("filename", "")).strip()
        prompt = row.get(prompt_col)
        if not filename or not isinstance(prompt, str) or not prompt.strip():
            continue
        lookup[filename] = prompt
        lookup[Path(filename).stem] = prompt
    return lookup


def load_global_attribute_phrase_lookup(csv_path: Path = GLOBAL_CSV_PATH) -> Dict[str, str]:
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "face_attribute_phrase" not in df.columns:
        return {}

    lookup: Dict[str, str] = {}
    for row in df.to_dict("records"):
        filename = str(row.get("filename", "")).strip()
        phrase = row.get("face_attribute_phrase")
        if not filename or not isinstance(phrase, str) or not phrase.strip():
            continue
        lookup[filename] = phrase.strip()
        lookup[Path(filename).stem] = phrase.strip()
    return lookup


def group_samples_by_image(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    image_paths: Dict[str, str] = {}
    image_ids: Dict[str, str] = {}

    for sample in samples:
        stem = str(sample["image_stem"])
        grouped[stem].append(sample)
        image_paths[stem] = str(sample["image_path"])
        image_ids[stem] = str(sample["image_id"])

    out = []
    for stem in sorted(grouped):
        ordered = sorted(
            grouped[stem],
            key=lambda s: (
                REGION_ORDER.index(s["region_key"]) if s["region_key"] in REGION_ORDER else 999,
                str(s.get("box_index", "")),
            ),
        )
        out.append({
            "image_stem": stem,
            "image_id": image_ids[stem],
            "image_path": image_paths[stem],
            "samples": ordered,
        })
    return out


def filter_samples_by_skip_regions(
    samples: List[Dict[str, Any]],
    skip_regions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    skipped = set(normalize_region_skip_list(skip_regions))
    if not skipped:
        return samples
    return [s for s in samples if str(s["region_key"]) not in skipped]


def select_one_sample_per_region(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_region: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        region = str(sample["region_key"])
        if region not in by_region:
            by_region[region] = sample

    return [
        by_region[region]
        for region in REGION_ORDER
        if region in by_region
    ]


def sort_samples_by_region_and_box(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        samples,
        key=lambda s: (
            REGION_ORDER.index(s["region_key"]) if s["region_key"] in REGION_ORDER else 999,
            int(s.get("box_index", 0) or 0),
        ),
    )


class LocalAlignedFusedDataset(Dataset):
    """
    Person/image-aligned local dataset for the optional fused local loss.

    This dataset is separate from the random crop-level local dataset. Each
    item returns one full image plus all local crops/boxes/scores for that
    image, so deterministic fusion can reinsert generated local crops.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        full_resolution: int = FUSED_FULL_RESOLUTION,
        local_resolution: int = LOCAL_RESOLUTION,
        context_scale: float = 1.20,
        global_csv_path: Path = GLOBAL_CSV_PATH,
        max_crops_per_image: Optional[int] = None,
        mask_feather: int = 24,
        skip: Optional[List[str]] = None,
        skip_regions: Optional[List[str]] = None,
    ):
        self.skip_regions = normalize_region_skip_list(
            skip_regions if skip_regions is not None else skip
        )
        self.groups = group_samples_by_image(
            filter_samples_by_skip_regions(samples, self.skip_regions)
        )
        if len(self.groups) == 0:
            raise ValueError("LocalAlignedFusedDataset received no grouped samples.")

        self.full_resolution = int(full_resolution)
        self.local_resolution = int(local_resolution)
        self.context_scale = float(context_scale)
        self.max_crops_per_image = max_crops_per_image
        self.mask_feather = int(mask_feather)
        self.global_prompt_lookup = load_global_prompt_lookup(Path(global_csv_path))

        self.full_transform = transforms.Compose([
            transforms.Resize((self.full_resolution, self.full_resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.crop_transform = transforms.Compose([
            transforms.Resize((self.local_resolution, self.local_resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        group = self.groups[idx]
        image = Image.open(group["image_path"]).convert("RGB")
        orig_w, orig_h = image.size

        full_pixel_values = self.full_transform(image)
        scale_x = self.full_resolution / float(orig_w)
        scale_y = self.full_resolution / float(orig_h)

        crop_tensors = []
        masks = []
        boxes = []
        target_scores = []
        prompts = []
        zone_prompts = []
        region_keys = []

        samples = group["samples"]
        if self.max_crops_per_image is not None:
            samples = samples[: int(self.max_crops_per_image)]

        for sample in samples:
            left, top, right, bottom = region_aware_bbox_to_square(
                bbox=sample["bbox"],
                region_key=sample["region_key"],
                image_width=orig_w,
                image_height=orig_h,
                context_scale=self.context_scale,
            )

            crop = image.crop((left, top, right, bottom))
            crop_tensors.append(self.crop_transform(crop))
            masks.append(make_bbox_soft_mask(
                size=self.local_resolution,
                bbox=sample["bbox"],
                crop_box=(left, top, right, bottom),
                feather=self.mask_feather,
            ))
            boxes.append(torch.tensor([
                round(left * scale_x),
                round(top * scale_y),
                round(right * scale_x),
                round(bottom * scale_y),
            ], dtype=torch.long))

            score = float(sample["score_raw"])
            target_scores.append(torch.tensor(score / 100.0, dtype=torch.float32))
            prompts.append(make_local_prompt(sample["region_key"], score, sample.get("ethnicity_raw", None)))
            zone_prompts.append(make_zone_prompt(sample["region_key"]))
            region_keys.append(sample["region_key"])

        global_prompt = self.global_prompt_lookup.get(
            group["image_stem"],
            self.global_prompt_lookup.get(group["image_id"], "a portrait photo of a person"),
        )

        return {
            "full_pixel_values": full_pixel_values,
            "pixel_values": torch.stack(crop_tensors, dim=0),
            "masks": torch.stack(masks, dim=0),
            "boxes": torch.stack(boxes, dim=0),
            "target_scores": torch.stack(target_scores, dim=0),
            "prompt": prompts,
            "zone_prompt": zone_prompts,
            "region_key": region_keys,
            "image_id": group["image_id"],
            "image_stem": group["image_stem"],
            "image_path": group["image_path"],
            "global_prompt": global_prompt,
        }


def local_fused_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    max_crops = max(item["pixel_values"].shape[0] for item in batch)
    crop_shape = batch[0]["pixel_values"].shape[1:]
    mask_shape = batch[0]["masks"].shape[1:]

    pixel_values = torch.zeros(batch_size, max_crops, *crop_shape, dtype=torch.float32)
    masks = torch.zeros(batch_size, max_crops, *mask_shape, dtype=torch.float32)
    boxes = torch.zeros(batch_size, max_crops, 4, dtype=torch.long)
    target_scores = torch.zeros(batch_size, max_crops, dtype=torch.float32)
    valid_mask = torch.zeros(batch_size, max_crops, dtype=torch.bool)

    prompts: List[List[str]] = []
    zone_prompts: List[List[str]] = []
    region_keys: List[List[str]] = []

    for bidx, item in enumerate(batch):
        n = item["pixel_values"].shape[0]
        pixel_values[bidx, :n] = item["pixel_values"]
        masks[bidx, :n] = item["masks"]
        boxes[bidx, :n] = item["boxes"]
        target_scores[bidx, :n] = item["target_scores"]
        valid_mask[bidx, :n] = True
        prompts.append(item["prompt"])
        zone_prompts.append(item["zone_prompt"])
        region_keys.append(item["region_key"])

    return {
        "full_pixel_values": torch.stack([item["full_pixel_values"] for item in batch], dim=0),
        "pixel_values": pixel_values,
        "masks": masks,
        "boxes": boxes,
        "target_scores": target_scores,
        "valid_mask": valid_mask,
        "prompt": prompts,
        "zone_prompt": zone_prompts,
        "region_key": region_keys,
        "image_id": [item["image_id"] for item in batch],
        "image_stem": [item["image_stem"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "global_prompt": [item["global_prompt"] for item in batch],
    }


class SinglePersonSamplingDataset(Dataset):
    """
    One-person monitoring dataset for deterministic reconstruction sampling.

    The returned item is compatible with both parse_sampling_global_batch and
    parse_sampling_local_batch, so the same loader can be passed as both
    sampling_loader_global and sampling_loader_local.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_stem: str = "09501",
        target_age: int = 75,
        local_target_score: float | Dict[str, float] = 85.0,
        full_resolution: int = FUSED_FULL_RESOLUTION,
        local_resolution: int = LOCAL_RESOLUTION,
        context_scale: float = 1.20,
        global_csv_path: Path = GLOBAL_CSV_PATH,
        mask_feather: int = 24,
        skip: Optional[List[str]] = None,
        skip_regions: Optional[List[str]] = None,
    ):
        self.image_stem = str(image_stem)
        self.target_age = int(target_age)
        self.local_target_score = local_target_score
        self.full_resolution = int(full_resolution)
        self.local_resolution = int(local_resolution)
        self.context_scale = float(context_scale)
        self.mask_feather = int(mask_feather)
        self.skip_regions = normalize_region_skip_list(
            skip_regions if skip_regions is not None else skip
        )
        self.attribute_lookup = load_global_attribute_phrase_lookup(Path(global_csv_path))

        matching = [
            s
            for s in filter_samples_by_skip_regions(samples, self.skip_regions)
            if str(s["image_stem"]) == self.image_stem
        ]
        if not matching:
            available = sorted({str(s["image_stem"]) for s in samples})[:10]
            raise ValueError(
                f"No local samples found for image_stem={self.image_stem}. "
                f"First available stems: {available}"
            )

        selected = sort_samples_by_region_and_box(matching)
        if len(selected) == 0:
            raise ValueError(f"No valid region samples found for image_stem={self.image_stem}.")

        self.samples = selected
        self.image_path = Path(selected[0]["image_path"])
        self.image_id = str(selected[0]["image_id"])

        self.full_transform = transforms.Compose([
            transforms.Resize((self.full_resolution, self.full_resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self.crop_transform = transforms.Compose([
            transforms.Resize((self.local_resolution, self.local_resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return 1

    def _global_prompt(self) -> str:
        phrase = self.attribute_lookup.get(self.image_stem, "")
        if phrase:
            return f"a portrait photo of a {self.target_age}-year-old person, {phrase}"
        return f"a portrait photo of a {self.target_age}-year-old person"

    def _score_for_region(self, region_key: str) -> float:
        if isinstance(self.local_target_score, dict):
            return float(self.local_target_score.get(region_key, self.local_target_score.get("default", 85.0)))
        return float(self.local_target_score)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = Image.open(self.image_path).convert("RGB")
        orig_w, orig_h = image.size
        x_orig = self.full_transform(image)
        scale_x = self.full_resolution / float(orig_w)
        scale_y = self.full_resolution / float(orig_h)

        zones = []
        for sample in self.samples:
            target_score = self._score_for_region(str(sample["region_key"]))
            left, top, right, bottom = region_aware_bbox_to_square(
                bbox=sample["bbox"],
                region_key=sample["region_key"],
                image_width=orig_w,
                image_height=orig_h,
                context_scale=self.context_scale,
            )
            crop = image.crop((left, top, right, bottom))
            bbox = (
                int(round(left * scale_x)),
                int(round(top * scale_y)),
                int(round(right * scale_x)),
                int(round(bottom * scale_y)),
            )
            zones.append({
                "zone_name": str(sample["region_key"]),
                "box_index": sample.get("box_index", None),
                "crop": self.crop_transform(crop),
                "prompt": make_local_prompt(
                    region_key=sample["region_key"],
                    score=target_score,
                    ethnicity_text=sample.get("ethnicity_raw", None),
                ),
                "bbox": bbox,
                "mask": make_bbox_soft_mask(
                    size=self.local_resolution,
                    bbox=sample["bbox"],
                    crop_box=(left, top, right, bottom),
                    feather=self.mask_feather,
                ),
                "target_score": target_score / 100.0,
            })

        return {
            "x_orig": x_orig,
            "image": x_orig,
            "global_prompt": self._global_prompt(),
            "image_id": self.image_id,
            "id": self.image_stem,
            "zones": zones,
        }


def single_person_sampling_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("Single-person sampling loader must use batch_size=1.")
    return batch[0]
