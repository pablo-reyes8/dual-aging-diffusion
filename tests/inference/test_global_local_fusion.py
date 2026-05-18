import torch
from PIL import Image

from src.inference.global_local_fusion import fuse_global_local_outputs
from src.inference.image_tensor_utils import (
    gaussian_blur_tensor,
    image_to_tensor01,
    normalize_mask01,
    tensor01_to_pil,
)


def test_image_tensor_roundtrip_and_mask_normalization():
    pil = Image.new("RGB", (16, 12), color=(64, 128, 192))
    x = image_to_tensor01(pil)
    assert x.shape == (1, 3, 12, 16)
    assert x.min() >= 0
    assert x.max() <= 1

    pil_back = tensor01_to_pil(x)
    assert pil_back.size == (16, 12)

    mask = normalize_mask01(torch.ones(12, 16) * 255)
    assert mask.shape == (1, 1, 12, 16)
    assert torch.allclose(mask.max(), torch.tensor(1.0))


def test_gaussian_blur_handles_small_inputs_with_large_sigma():
    x = torch.rand(1, 1, 8, 8)
    y = gaussian_blur_tensor(x, sigma=5.0)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_deterministic_global_local_fusion_shapes_and_ranges():
    x_orig = torch.zeros(1, 3, 32, 32)
    x_global = torch.ones(1, 3, 32, 32) * 0.25
    local_outputs = [
        {
            "aged_crop": torch.ones(1, 3, 8, 8) * 0.75,
            "bbox": (8, 8, 16, 16),
            "mask": torch.ones(1, 1, 8, 8),
        }
    ]

    out = fuse_global_local_outputs(
        x_orig=x_orig,
        x_global=x_global,
        local_outputs=local_outputs,
        fusion_bundle=None,
        device="cpu",
        return_pil=True,
        verbose=False,
    )

    assert out["mode"] == "deterministic"
    for key in ["x_orig", "x_global", "x_coarse", "x_blend", "x_final"]:
        assert out[key].shape == (1, 3, 32, 32)
        assert out[key].min() >= 0
        assert out[key].max() <= 1

    assert out["pil"]["x_final"].size == (32, 32)
    assert out["fusion_model_config"] is None
