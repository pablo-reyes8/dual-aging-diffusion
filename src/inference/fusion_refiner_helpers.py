from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

from src.inference.image_tensor_utils import image_to_tensor01, resize_tensor_image, tensor01_to_pil

def apply_fusion_refiner_if_available(
    x_blend: torch.Tensor,
    fusion_bundle: Optional[Dict[str, Any]],
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    strength: Optional[float] = None,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    If fusion_bundle is None:
        returns x_blend.

    If fusion_bundle exists:
        applies low-strength img2img refiner to x_blend.
    """
    if fusion_bundle is None:
        return x_blend

    pipe = fusion_bundle.get("pipe", None)
    cfg = fusion_bundle.get("config", None)

    if pipe is None or cfg is None:
        raise ValueError("fusion_bundle must contain keys 'pipe' and 'config'.")

    if device is None:
        device = x_blend.device
    else:
        device = torch.device(device)

    if hasattr(pipe, "to"):
        pipe.to(device)

    for name in ["unet", "vae", "text_encoder", "text_encoder_2"]:
        module = getattr(pipe, name, None)
        if module is not None:
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    prompt = prompt if prompt is not None else cfg.prompt
    negative_prompt = negative_prompt if negative_prompt is not None else cfg.negative_prompt
    strength = float(strength if strength is not None else cfg.strength)
    guidance_scale = float(guidance_scale if guidance_scale is not None else cfg.guidance_scale)
    num_inference_steps = int(num_inference_steps if num_inference_steps is not None else cfg.num_inference_steps)

    min_effective_strength = (1.0 / max(1, num_inference_steps)) + 1e-3
    if 0.0 < strength < min_effective_strength:
        print(
            "[WARN] Fusion refiner strength is too low for the requested step count. "
            f"Clamping strength from {strength:.6f} to {min_effective_strength:.6f} "
            "to avoid an empty SDXL img2img timestep schedule."
        )
        strength = min_effective_strength

    image_pil = tensor01_to_pil(x_blend)

    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

    if hasattr(out, "images"):
        out_img = out.images[0]
    elif isinstance(out, (tuple, list)):
        out_img = out[0]
    else:
        out_img = out

    x_final = image_to_tensor01(
        out_img,
        device=x_blend.device,
        dtype=x_blend.dtype,
    )

    if x_final.shape[-2:] != x_blend.shape[-2:]:
        x_final = resize_tensor_image(x_final, size_hw=x_blend.shape[-2:])

    return x_final.clamp(0, 1)
