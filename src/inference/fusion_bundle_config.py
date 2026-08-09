from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Union

import torch

from src.inference.diffusion_inversion import InversionConfig

@dataclass
class FusionModelConfig:
    model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    device: str = "cuda"
    torch_dtype: str = "auto"

    # Low-strength img2img refinement.
    strength: float = 0.055
    guidance_scale: float = 1.5
    num_inference_steps: int = 12

    # Prompt is intentionally about harmonization, not changing identity.
    prompt: str = (
        "ultra-realistic portrait photo of the same person, natural facial aging, "
        "consistent skin texture, seamless blending, realistic wrinkles, "
        "identity-preserving face, natural lighting"
    )

    negative_prompt: str = (
        "changed identity, different person, deformed face, distorted eyes, "
        "distorted mouth, plastic skin, waxy skin, blurry, artifacts, "
        "erased wrinkles, over-smoothed skin, unrealistic texture"
    )

    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_model_cpu_offload: bool = False

    # Independent, experimental anchor for the already-fused image. OFF by default.
    inversion: InversionConfig = field(
        default_factory=lambda: InversionConfig(
            enabled=False,
            num_steps=20,
            strength=0.15,
            inversion_guidance_scale=1.0,
            fallback_to_img2img=True,
        )
    )
    inversion_source_prompt: str = "a realistic portrait photo of the same person"


def _resolve_fusion_device(device: Union[str, torch.device] = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device

    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if str(device) == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(device)


def _resolve_fusion_dtype(
    torch_dtype: Union[str, torch.dtype] = "auto",
    device: Union[str, torch.device] = "auto",
) -> torch.dtype:
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    device = _resolve_fusion_device(device)
    torch_dtype = str(torch_dtype).lower().strip()

    if torch_dtype == "auto":
        if device.type == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    if torch_dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16

    if torch_dtype in {"fp16", "float16"}:
        return torch.float16

    if torch_dtype in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported torch_dtype={torch_dtype}")


def _count_params(module) -> Dict[str, int]:
    if module is None or not hasattr(module, "parameters"):
        return {"total": 0, "trainable": 0}

    total = 0
    trainable = 0

    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    return {"total": int(total), "trainable": int(trainable)}


def _count_fusion_pipe_params(pipe) -> Dict[str, int]:
    total = 0
    trainable = 0

    for name in ["unet", "vae", "text_encoder", "text_encoder_2"]:
        module = getattr(pipe, name, None)
        pc = _count_params(module)
        total += pc["total"]
        trainable += pc["trainable"]

    return {"total": int(total), "trainable": int(trainable)}
