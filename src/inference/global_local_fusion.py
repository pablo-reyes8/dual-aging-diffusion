# ============================================================
# GLOBAL-LOCAL OUTPUT FUSION
#
# Public entrypoint for deterministic global-local output fusion.
# Deterministic mode is used when fusion_bundle=None.
# Passing a fusion_bundle adds the optional low-strength refiner after
# the deterministic residual + local crop blending stage.
# ============================================================

from __future__ import annotations

import gc
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence, Union

import torch

from src.inference.fusion_bundle_maker import offload_fusion_bundle
from src.inference.deterministic_fusion_ops import *
from src.inference.fusion_refiner_helpers import apply_fusion_refiner_if_available
from src.inference.image_tensor_utils import image_to_tensor01, normalize_mask01, tensor01_to_pil

# ============================================================
# Main fusion function
# ============================================================

def fuse_global_local_outputs(
    *,
    x_orig,
    x_global,
    local_outputs: Sequence[Dict[str, Any]],

    face_mask=None,

    # Optional model bundle. If None => deterministic mode.
    fusion_bundle: Optional[Dict[str, Any]] = None,

    # Residual fusion parameters.
    residual_alpha: float = 0.35,
    residual_alpha_inside_local: Optional[float] = None,
    residual_alpha_outside_local: Optional[float] = None,
    residual_sigma: float = 9.0,
    local_union_blur_sigma: float = 7.0,
    use_face_mask: bool = True,
    face_mask_blur_sigma: float = 3.0,

    # Local feathering parameters.
    local_insert_alpha: float = 1.0,
    local_mask_blur_sigma: float = 5.0,

    # New color matching.
    color_match: bool = True,
    color_match_strength: float = 0.75,

    # Optional fusion model overrides.
    fusion_prompt: Optional[str] = None,
    fusion_negative_prompt: Optional[str] = None,
    fusion_strength: Optional[float] = None,
    fusion_guidance_scale: Optional[float] = None,
    fusion_num_inference_steps: Optional[int] = None,

    # Runtime.
    device: Union[str, torch.device] = "auto",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
    return_pil: bool = True,
    offload_fusion_after: bool = False,
    verbose: bool = True) -> Dict[str, Any]:

    """
    Builds final aged image from already-generated global/local outputs.

    Modes:
        fusion_bundle=None:
            deterministic residual + color-matched local feathering.

        fusion_bundle!=None:
            deterministic residual + color-matched local feathering
            + low-strength prompt-conditioned refiner.

    Important:
        The refiner is never responsible for creating aging from scratch.
        It only harmonizes x_blend.
    """
    if str(device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    if verbose:
        print("\n" + "═" * 110)
        print("Global-local fusion")
        print("═" * 110)
        print("Mode                 :", "model" if fusion_bundle is not None else "deterministic")
        print("Device               :", device)
        print("Residual alpha       :", residual_alpha)
        print("Residual alpha local :", residual_alpha_inside_local)
        print("Residual alpha outer :", residual_alpha_outside_local)
        print("Residual sigma       :", residual_sigma)
        print("Local union sigma    :", local_union_blur_sigma)
        print("Use face mask        :", use_face_mask)
        print("Local crops          :", len(local_outputs) if local_outputs is not None else 0)
        print("Local insert α       :", local_insert_alpha)
        print("Local mask sigma     :", local_mask_blur_sigma)
        print("Color match          :", color_match)
        print("Color match strength :", color_match_strength)

        if fusion_bundle is not None:
            cfg = fusion_bundle["config"]
            print("Fusion model         :", cfg.model_id)
            print("Fusion strength      :", cfg.strength if fusion_strength is None else fusion_strength)
            print("Fusion guidance      :", cfg.guidance_scale if fusion_guidance_scale is None else fusion_guidance_scale)
            print("Fusion steps         :", cfg.num_inference_steps if fusion_num_inference_steps is None else fusion_num_inference_steps)

        print("─" * 110)


    with torch.inference_mode():
        x_orig_t = image_to_tensor01(
            x_orig,
            device=device,
            dtype=dtype,
        )

        x_global_t = image_to_tensor01(
            x_global,
            device=device,
            dtype=dtype,
        )

        face_mask_t = None
        if face_mask is not None:
            face_mask_t = normalize_mask01(
                face_mask,
                device=device,
                dtype=dtype,
            )

        # --------------------------------------------------------
        # Global low-frequency residual.
        # --------------------------------------------------------
        residual_pack = compute_low_frequency_global_residual_fusion(
            x_orig=x_orig_t,
            x_global=x_global_t,
            local_outputs=local_outputs,
            face_mask=face_mask_t,
            residual_alpha=residual_alpha,
            residual_alpha_inside_local=residual_alpha_inside_local,
            residual_alpha_outside_local=residual_alpha_outside_local,
            residual_sigma=residual_sigma,
            local_union_blur_sigma=local_union_blur_sigma,
            face_mask_blur_sigma=face_mask_blur_sigma,
            use_face_mask=use_face_mask)

        x_coarse = residual_pack["x_coarse"]

        # --------------------------------------------------------
        # Local crop insertion with color matching.
        # --------------------------------------------------------
        x_blend = insert_local_outputs_feathered(
            x_coarse=x_coarse,
            local_outputs=local_outputs,
            local_insert_alpha=local_insert_alpha,
            local_mask_blur_sigma=local_mask_blur_sigma,
            color_match=color_match,
            color_match_strength=color_match_strength)

        # --------------------------------------------------------
        # Deterministic final, before optional low-strength refiner.
        # --------------------------------------------------------
        x_final = x_blend.clamp(0, 1)

        # --------------------------------------------------------
        # Optional low-strength refiner. Kept separate from x_final so
        # grids can show what deterministic fusion did before refinement.
        # --------------------------------------------------------
        x_refined = apply_fusion_refiner_if_available(
            x_blend=x_final,
            fusion_bundle=fusion_bundle,
            prompt=fusion_prompt,
            negative_prompt=fusion_negative_prompt,
            strength=fusion_strength,
            guidance_scale=fusion_guidance_scale,
            num_inference_steps=fusion_num_inference_steps,
            generator=generator,
            device=device,
        )

        x_refined = x_refined.clamp(0, 1)

    out = {
        "mode": "model" if fusion_bundle is not None else "deterministic",

        "x_orig": x_orig_t.detach().cpu(),
        "x_global": x_global_t.detach().cpu(),
        "x_coarse": x_coarse.detach().cpu(),
        "x_blend": x_blend.detach().cpu(),
        "x_final": x_final.detach().cpu(),
        "x_refined": x_refined.detach().cpu(),

        "residual_raw": residual_pack["residual_raw"].detach().cpu(),
        "residual_low": residual_pack["residual_low"].detach().cpu(),
        "face_mask": residual_pack["face_mask"].detach().cpu(),
        "local_union_mask": residual_pack["local_union_mask"].detach().cpu(),
        "alpha_map": residual_pack["alpha_map"].detach().cpu(),

        "local_outputs": local_outputs,
        "fusion_model_config": None
            if fusion_bundle is None
            else asdict(fusion_bundle["config"]),
    }

    if return_pil:
        out["pil"] = {
            "x_orig": tensor01_to_pil(out["x_orig"]),
            "x_global": tensor01_to_pil(out["x_global"]),
            "x_coarse": tensor01_to_pil(out["x_coarse"]),
            "x_blend": tensor01_to_pil(out["x_blend"]),
            "x_final": tensor01_to_pil(out["x_final"]),
            "residual_raw_vis": tensor01_to_pil((out["residual_raw"] * 0.5 + 0.5).clamp(0, 1)),
            "residual_low_vis": tensor01_to_pil((out["residual_low"] * 0.5 + 0.5).clamp(0, 1)),
            "local_union_mask_vis": tensor01_to_pil(out["local_union_mask"].repeat(1, 3, 1, 1).clamp(0, 1)),
            "alpha_map_vis": tensor01_to_pil((out["alpha_map"] / out["alpha_map"].max().clamp_min(1e-6)).repeat(1, 3, 1, 1).clamp(0, 1)),
        }
        if fusion_bundle is not None:
            out["pil"]["x_refined"] = tensor01_to_pil(out["x_refined"])

    if offload_fusion_after:
        offload_fusion_bundle(fusion_bundle)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print("Fusion finished")
        print("Returned: x_orig | x_global | x_coarse | x_blend | x_final | x_refined | residuals")
        print("═" * 110)

    return out
