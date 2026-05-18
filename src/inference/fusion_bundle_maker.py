# ============================================================
# FUSION BUNDLE BUILDER
#
# Builds ONLY the optional prompt-conditioned fusion/refiner model.
# Deterministic global-local fusion does not need this bundle.
# ============================================================

from __future__ import annotations

import gc
from typing import Any, Dict, Optional, Union

import torch

from src.inference.fusion_bundle_config import (
    FusionModelConfig,
    _count_fusion_pipe_params,
    _resolve_fusion_device,
    _resolve_fusion_dtype,
)

def build_fusion_bundle(
    model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
    device: Union[str, torch.device] = "auto",
    torch_dtype: Union[str, torch.dtype] = "auto",

    strength: float = 0.055,
    guidance_scale: float = 1.5,
    num_inference_steps: int = 12,

    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,

    enable_attention_slicing: bool = True,
    enable_vae_slicing: bool = True,
    enable_model_cpu_offload: bool = False,

    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Builds the optional fusion/refiner bundle.

    This function downloads/loads ONLY the fusion model.

    Use:
        fusion_bundle = build_fusion_bundle(...)

    Then:
        out = fuse_global_local_outputs(..., fusion_bundle=fusion_bundle)

    If you call fuse_global_local_outputs(..., fusion_bundle=None),
    the exact same residual + local feathering pipeline is used, but without
    the final model refinement step.
    """
    device = _resolve_fusion_device(device)
    dtype = _resolve_fusion_dtype(torch_dtype, device=device)

    if prompt is None:
        prompt = (
            "ultra-realistic portrait photo of the same person, natural facial aging, "
            "consistent skin texture, seamless blending, realistic wrinkles, "
            "identity-preserving face, natural lighting"
        )

    if negative_prompt is None:
        negative_prompt = (
            "changed identity, different person, deformed face, distorted eyes, "
            "distorted mouth, plastic skin, waxy skin, blurry, artifacts, "
            "erased wrinkles, over-smoothed skin, unrealistic texture"
        )

    cfg = FusionModelConfig(
        model_id=model_id,
        device=str(device),
        torch_dtype=str(dtype).replace("torch.", ""),
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        prompt=str(prompt),
        negative_prompt=str(negative_prompt),
        enable_attention_slicing=bool(enable_attention_slicing),
        enable_vae_slicing=bool(enable_vae_slicing),
        enable_model_cpu_offload=bool(enable_model_cpu_offload),
    )

    try:
        from diffusers import AutoPipelineForImage2Image
    except ImportError as e:
        raise ImportError(
            "diffusers is required to build the fusion bundle. "
            "Install with: pip install diffusers transformers accelerate safetensors"
        ) from e

    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    pipe.set_progress_bar_config(disable=True)

    if enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    if enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if enable_model_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    for name in ["unet", "vae", "text_encoder", "text_encoder_2"]:
        module = getattr(pipe, name, None)
        if module is not None:
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    param_count = _count_fusion_pipe_params(pipe)

    fusion_bundle = {
        "type": "prompt_conditioned_fusion_refiner",
        "pipe": pipe,
        "config": cfg,
        "param_count": param_count,
    }

    if verbose:
        print("\n" + "=" * 100)
        print("Fusion bundle")
        print("=" * 100)
        print("Model id              :", cfg.model_id)
        print("Device                :", cfg.device)
        print("Dtype                 :", cfg.torch_dtype)
        print("Strength              :", cfg.strength)
        print("Guidance scale        :", cfg.guidance_scale)
        print("Inference steps       :", cfg.num_inference_steps)
        print("Trainable params      :", param_count["trainable"])
        print("Total loaded params   :", param_count["total"])
        print("-" * 100)
        print("Prompt                :", cfg.prompt)
        print("Negative prompt       :", cfg.negative_prompt)
        print("=" * 100)

    return fusion_bundle


def offload_fusion_bundle(fusion_bundle: Optional[Dict[str, Any]]) -> None:
    """
    Moves fusion model to CPU and clears CUDA cache.
    """
    if fusion_bundle is None:
        return

    pipe = fusion_bundle.get("pipe", None)

    if pipe is not None and hasattr(pipe, "to"):
        pipe.to("cpu")

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
