from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from src.inference.image_tensor_utils import (
    gaussian_blur_tensor,
    image_to_tensor01,
    normalize_mask01,
    resize_tensor_image,
)

# ============================================================
# Residual global fusion
# ============================================================

def compute_low_frequency_global_residual_fusion(
    x_orig: torch.Tensor,
    x_global: torch.Tensor,
    face_mask: Optional[torch.Tensor] = None,
    residual_alpha: float = 0.35,
    residual_sigma: float = 9.0,
    face_mask_blur_sigma: float = 3.0,
    use_face_mask: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Our contribution:

        residual_raw = x_global - x_orig
        residual_low = GaussianBlur(residual_raw)
        x_coarse = x_orig + alpha * M_face * residual_low

    This is deterministic and has 0 trainable parameters.
    """
    x_orig = x_orig.float().clamp(0, 1)
    x_global = x_global.float().clamp(0, 1)

    if x_global.shape[-2:] != x_orig.shape[-2:]:
        x_global = resize_tensor_image(x_global, size_hw=x_orig.shape[-2:])

    residual_raw = x_global - x_orig
    residual_low = gaussian_blur_tensor(residual_raw, sigma=residual_sigma)

    if face_mask is None or not use_face_mask:
        m = torch.ones_like(x_orig[:, :1])
    else:
        m = face_mask.float().clamp(0, 1)

        if m.shape[-2:] != x_orig.shape[-2:]:
            m = resize_tensor_image(m, size_hw=x_orig.shape[-2:])

        if face_mask_blur_sigma is not None and face_mask_blur_sigma > 0:
            m = gaussian_blur_tensor(m, sigma=face_mask_blur_sigma)

        m = m.clamp(0, 1)

    x_coarse = x_orig + float(residual_alpha) * m * residual_low
    x_coarse = x_coarse.clamp(0, 1)

    return {
        "x_coarse": x_coarse,
        "residual_raw": residual_raw,
        "residual_low": residual_low,
        "face_mask": m,
    }


# ============================================================
# Local crop insertion
# ============================================================

def masked_mean_std(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes per-channel mean/std for image tensor [1,3,H,W].

    Args:
        x:
            [1,3,H,W]

        mask:
            [1,1,H,W] in [0,1], optional.

    Returns:
        mean: [1,3,1,1]
        std:  [1,3,1,1]
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"x must be [1,3,H,W], got {tuple(x.shape)}")

    if mask is None:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(var + eps)
        return mean, std

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be [1,1,H,W], got {tuple(mask.shape)}")

    mask = mask.to(device=x.device, dtype=x.dtype).clamp(0, 1)

    if mask.shape[-2:] != x.shape[-2:]:
        mask = resize_tensor_image(mask, size_hw=x.shape[-2:])

    denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(eps)

    mean = (x * mask).sum(dim=(2, 3), keepdim=True) / denom
    var = (((x - mean) ** 2) * mask).sum(dim=(2, 3), keepdim=True) / denom
    std = torch.sqrt(var + eps)

    return mean, std


def masked_mean_std_color_match(
    source_crop: torch.Tensor,
    target_region: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    strength: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Matches source crop color statistics to target region.

    Formula:
        normalized = (source - mean_source) / std_source
        matched = normalized * std_target + mean_target
        output = (1-strength) * source + strength * matched

    Args:
        source_crop:
            Generated local crop [1,3,h,w] in [0,1].

        target_region:
            Region from current canvas [1,3,h,w] in [0,1].

        mask:
            Local soft mask [1,1,h,w]. If provided, statistics are computed
            mostly inside the pasted region.

        strength:
            0.0 = no color matching.
            1.0 = full mean/std matching.
            Recommended: 0.5–0.85.
    """
    strength = float(strength)

    if strength <= 0:
        return source_crop.clamp(0, 1)

    source_crop = source_crop.float().clamp(0, 1)
    target_region = target_region.float().clamp(0, 1)

    if source_crop.shape != target_region.shape:
        raise ValueError(
            f"source_crop and target_region must have same shape. "
            f"Got {tuple(source_crop.shape)} vs {tuple(target_region.shape)}"
        )

    if mask is not None:
        mask = mask.to(device=source_crop.device, dtype=source_crop.dtype).clamp(0, 1)
        if mask.shape[-2:] != source_crop.shape[-2:]:
            mask = resize_tensor_image(mask, size_hw=source_crop.shape[-2:])

    src_mean, src_std = masked_mean_std(source_crop, mask=mask, eps=eps)
    tgt_mean, tgt_std = masked_mean_std(target_region, mask=mask, eps=eps)

    matched = (source_crop - src_mean) / src_std.clamp_min(eps)
    matched = matched * tgt_std + tgt_mean
    matched = matched.clamp(0, 1)

    out = (1.0 - strength) * source_crop + strength * matched
    return out.clamp(0, 1)


def paste_local_crop_feathered(
    base: torch.Tensor,
    aged_crop,
    bbox_xyxy: Tuple[int, int, int, int],
    mask=None,
    insert_alpha: float = 1.0,
    mask_blur_sigma: float = 5.0,

    # New color matching controls.
    color_match: bool = True,
    color_match_strength: float = 0.75,
) -> torch.Tensor:
    """
    Inserts aged crop into base with:
        1. optional masked color matching
        2. soft mask feathering

    base:
        [1,3,H,W] in [0,1]

    aged_crop:
        PIL or tensor.

    bbox_xyxy:
        (x1, y1, x2, y2)

    mask:
        Optional soft local mask.
    """
    if base.ndim != 4 or base.shape[0] != 1:
        raise ValueError(f"base must be [1,3,H,W], got {tuple(base.shape)}")

    _, _, H, W = base.shape

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

    x1 = max(0, min(W - 1, x1))
    x2 = max(x1 + 1, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(y1 + 1, min(H, y2))

    crop_h = y2 - y1
    crop_w = x2 - x1

    crop_t = image_to_tensor01(
        aged_crop,
        device=base.device,
        dtype=base.dtype,
    )
    crop_t = resize_tensor_image(crop_t, size_hw=(crop_h, crop_w))

    if mask is None:
        mask_t = torch.ones(
            (1, 1, crop_h, crop_w),
            device=base.device,
            dtype=base.dtype,
        )
    else:
        mask_t = normalize_mask01(
            mask,
            device=base.device,
            dtype=base.dtype,
        )

        if mask_t.shape[-2:] != (crop_h, crop_w):
            mask_t = resize_tensor_image(mask_t, size_hw=(crop_h, crop_w))

    if mask_blur_sigma is not None and mask_blur_sigma > 0:
        mask_t = gaussian_blur_tensor(mask_t, sigma=mask_blur_sigma)

    mask_t = mask_t.clamp(0, 1)

    out = base.clone()
    target_region = out[:, :, y1:y2, x1:x2]

    # ------------------------------------------------------------
    # New: color match generated crop to target region.
    # ------------------------------------------------------------
    if color_match:
        crop_t = masked_mean_std_color_match(
            source_crop=crop_t,
            target_region=target_region,
            mask=mask_t,
            strength=color_match_strength,
        )

    # ------------------------------------------------------------
    # Feathered insertion.
    # ------------------------------------------------------------
    mask_eff = (mask_t * float(insert_alpha)).clamp(0, 1)
    blended = mask_eff * crop_t + (1.0 - mask_eff) * target_region

    out[:, :, y1:y2, x1:x2] = blended
    return out.clamp(0, 1)


def insert_local_outputs_feathered(
    x_coarse: torch.Tensor,
    local_outputs: Sequence[Dict[str, Any]],
    local_insert_alpha: float = 1.0,
    local_mask_blur_sigma: float = 5.0,

    # New color matching controls.
    color_match: bool = True,
    color_match_strength: float = 0.75,
) -> torch.Tensor:
    """
    Sequential local insertion.

    This is Dos-Santos-style feathering, but inserted over our residual base:

        x^(0) = x_coarse

        x^(z) = M_z * color_match(crop_z, region_z)
                 + (1-M_z) * x^(z-1)

    The only difference between deterministic and refiner mode is whether
    x_blend later goes through a prompt-conditioned refiner.
    """
    x = x_coarse

    if local_outputs is None:
        return x

    for item in local_outputs:
        if "aged_crop" not in item:
            raise KeyError("Each local output must contain key 'aged_crop'.")

        if "bbox" not in item:
            raise KeyError("Each local output must contain key 'bbox'.")

        x = paste_local_crop_feathered(
            base=x,
            aged_crop=item["aged_crop"],
            bbox_xyxy=item["bbox"],
            mask=item.get("mask", None),
            insert_alpha=float(item.get("insert_alpha", local_insert_alpha)),
            mask_blur_sigma=float(item.get("mask_blur_sigma", local_mask_blur_sigma)),
            color_match=bool(item.get("color_match", color_match)),
            color_match_strength=float(item.get("color_match_strength", color_match_strength)),
        )

    return x.clamp(0, 1)
