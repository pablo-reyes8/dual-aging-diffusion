from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ============================================================
# Image utilities
# ============================================================

def _is_pil_image(x: Any) -> bool:
    return PIL_AVAILABLE and isinstance(x, Image.Image)


def image_to_tensor01(
    image,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts PIL/tensor to [1,3,H,W] in [0,1].
    """
    if torch.is_tensor(image):
        x = image.detach()

        if x.ndim == 3:
            if x.shape[-1] == 3 and x.shape[0] != 3:
                x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)

        elif x.ndim == 4:
            pass

        else:
            raise ValueError(f"Unsupported image tensor shape: {tuple(x.shape)}")

        x = x.float()

        if x.max().item() > 2.0:
            x = x / 255.0

        if x.min().item() < -0.05:
            x = (x + 1.0) / 2.0

        x = x.clamp(0, 1)

        if device is not None:
            x = x.to(device=device, dtype=dtype)

        return x

    if _is_pil_image(image):
        import numpy as np

        arr = np.array(image.convert("RGB")).astype("float32") / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        if device is not None:
            x = x.to(device=device, dtype=dtype)

        return x

    raise TypeError(f"Unsupported image type: {type(image)}")


def tensor01_to_pil(x: torch.Tensor):
    """
    Converts [1,3,H,W] or [3,H,W] tensor in [0,1] to PIL.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL is not available.")

    import numpy as np

    x = x.detach().float().cpu()

    if x.ndim == 4:
        x = x[0]

    x = x.clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")

    return Image.fromarray(arr)


def resize_tensor_image(
    x: torch.Tensor,
    size_hw: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(x, size=size_hw, mode=mode, align_corners=False)
    return F.interpolate(x, size=size_hw, mode=mode)


def normalize_mask01(
    mask,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts mask to [1,1,H,W] in [0,1].
    """
    if mask is None:
        return None

    if torch.is_tensor(mask):
        m = mask.detach().float()

    elif _is_pil_image(mask):
        import numpy as np
        m = torch.from_numpy(np.array(mask.convert("L")).astype("float32") / 255.0)

    else:
        m = torch.tensor(mask, dtype=torch.float32)

    if m.ndim == 2:
        m = m.unsqueeze(0).unsqueeze(0)

    elif m.ndim == 3:
        if m.shape[-1] == 1:
            m = m.permute(2, 0, 1).unsqueeze(0)
        elif m.shape[0] == 1:
            m = m.unsqueeze(0)
        else:
            m = m[:1].unsqueeze(0)

    elif m.ndim == 4:
        if m.shape[1] != 1:
            m = m[:, :1]

    else:
        raise ValueError(f"Unsupported mask shape: {tuple(m.shape)}")

    if m.max().item() > 2.0:
        m = m / 255.0

    m = m.clamp(0, 1)

    if device is not None:
        m = m.to(device=device, dtype=dtype)

    return m


# ============================================================
# Gaussian blur
# ============================================================

def gaussian_kernel1d(
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    truncate: float = 3.0,
) -> torch.Tensor:
    sigma = float(sigma)

    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)

    radius = max(1, int(truncate * sigma + 0.5))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)

    kernel = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum().clamp_min(1e-8)

    return kernel


def gaussian_blur_tensor(
    x: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Separable Gaussian blur for [B,C,H,W].
    """
    if sigma is None or sigma <= 0:
        return x

    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")

    _, c, _, _ = x.shape

    k1 = gaussian_kernel1d(
        sigma=sigma,
        device=x.device,
        dtype=x.dtype,
    )

    k = k1.numel()
    pad = k // 2

    weight_x = k1.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    weight_y = k1.view(1, 1, k, 1).repeat(c, 1, 1, 1)

    _, _, h, w = x.shape
    pad_x_mode = "reflect" if pad < w else "replicate"
    pad_y_mode = "reflect" if pad < h else "replicate"

    x = F.pad(x, (pad, pad, 0, 0), mode=pad_x_mode)
    x = F.conv2d(x, weight_x, groups=c)

    x = F.pad(x, (0, 0, pad, pad), mode=pad_y_mode)
    x = F.conv2d(x, weight_y, groups=c)

    return x
