from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch

from src.inference.fusion_bundle_maker import build_fusion_bundle, offload_fusion_bundle
from src.inference.global_local_fusion import fuse_global_local_outputs
from src.training.chekpoints import (
    load_inference_checkpoint,
    restore_inference_checkpoint_into_bundle,
)
from src.training.training_memory_helpers import (
    move_bundle_modules_only_to_device,
    offload_bundle_modules_only,
)
from src.training.training_sampling_helpers import (
    _call_sample_forward_with_supported_kwargs,
    default_sample_global_forward,
    default_sample_local_forward,
    parse_sampling_global_batch,
    parse_sampling_local_batch,
)


DEFAULT_INFERENCE_WRAPPER_CONFIG: Dict[str, Any] = {
    "checkpoints": {
        "strict_adapter": True,
    },
    "generation": {
        "global_strength": 0.30,
        "global_guidance_scale": 5.0,
        "global_num_inference_steps": 35,
        "global_negative_prompt": None,
        "local_strength": 0.20,
        "local_guidance_scale": 0.8,
        "local_num_inference_steps": 40,
        "local_negative_prompt": None,
        "local_recycle_passes": 1,
        "local_recycle_strength": None,
        "local_recycle_guidance_scale": None,
        "local_recycle_num_inference_steps": None,
        "seed": 123,
    },
    "fusion": {
        "residual_alpha": 0.35,
        "residual_sigma": 9.0,
        "use_face_mask": True,
        "face_mask_blur_sigma": 3.0,
        "local_insert_alpha": 1.0,
        "local_mask_blur_sigma": 5.0,
        "color_match": True,
        "color_match_strength": 0.75,
        "fusion_prompt": None,
        "fusion_negative_prompt": None,
        "fusion_strength": None,
        "fusion_guidance_scale": None,
        "fusion_num_inference_steps": None,
    },
    "refiner": {
        "enabled": False,
        "model_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "torch_dtype": "auto",
        "strength": 0.055,
        "guidance_scale": 1.5,
        "num_inference_steps": 12,
        "prompt": None,
        "negative_prompt": None,
        "enable_attention_slicing": True,
        "enable_vae_slicing": True,
        "enable_model_cpu_offload": False,
        "offload_after": True,
    },
    "runtime": {
        "offload_after_each_stage": True,
        "return_pil": True,
        "verbose": True,
    },
}


def _deep_update(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    if override is None:
        return out

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_sampling_loader(sampling_objects):
    if isinstance(sampling_objects, dict) and "loader" in sampling_objects:
        return sampling_objects["loader"]
    return sampling_objects


def _first_batch(loader):
    if isinstance(loader, dict):
        return loader
    return next(iter(loader))


def _normalize_checkpoint_paths(checkpoint_paths) -> Dict[str, Optional[Path]]:
    if checkpoint_paths is None:
        return {"global": None, "local": None}

    if isinstance(checkpoint_paths, (list, tuple)):
        if len(checkpoint_paths) != 2:
            raise ValueError("checkpoint_paths tuple/list must be: (global_path, local_path).")
        return {
            "global": Path(checkpoint_paths[0]),
            "local": Path(checkpoint_paths[1]),
        }

    if isinstance(checkpoint_paths, dict):
        return {
            "global": None if checkpoint_paths.get("global") is None else Path(checkpoint_paths["global"]),
            "local": None if checkpoint_paths.get("local") is None else Path(checkpoint_paths["local"]),
        }

    raise TypeError("checkpoint_paths must be None, dict, or (global_path, local_path).")


def _adapter_config_from_checkpoint(path: Path) -> Dict[str, Any]:
    ckpt = load_inference_checkpoint(path)
    metadata = ckpt.get("metadata", {})
    adapter_config = metadata.get("adapter_config")
    adapter_type = metadata.get("adapter_type")

    if adapter_config is None or adapter_type is None:
        raise ValueError(
            f"Checkpoint does not contain adapter metadata needed for injection: {path}"
        )

    out = dict(adapter_config)
    out["adapter_type"] = adapter_type
    return out


def _ensure_adapter_injected_from_checkpoint(
    *,
    bundle: Dict[str, Any],
    checkpoint_path: Optional[Path],
    strict_adapter: bool,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    if checkpoint_path is None:
        return None

    if bundle.get("adapter_type", None) is None:
        from src.diffusion_pipeline.load_diffusion_models import apply_adapter_to_existing_bundle

        adapter_cfg = _adapter_config_from_checkpoint(checkpoint_path)
        apply_adapter_to_existing_bundle(
            bundle=bundle,
            adapter_type=adapter_cfg["adapter_type"],
            rank=adapter_cfg["rank"],
            alpha=adapter_cfg["alpha"],
            dropout=adapter_cfg.get("dropout", 0.0),
            target_suffixes=adapter_cfg.get("target_suffixes", ["to_q", "to_k", "to_v", "to_out.0"]),
            train_mode=False,
            verbose=verbose,
        )

    return restore_inference_checkpoint_into_bundle(
        bundle=bundle,
        checkpoint_path=checkpoint_path,
        strict_adapter=strict_adapter,
    )


def _build_refiner_from_config(
    *,
    config: Dict[str, Any],
    device: torch.device,
    verbose: bool,
):
    refiner_cfg = config["refiner"]
    if not bool(refiner_cfg.get("enabled", False)):
        return None

    return build_fusion_bundle(
        model_id=refiner_cfg.get("model_id", "stabilityai/stable-diffusion-xl-refiner-1.0"),
        device=device,
        torch_dtype=refiner_cfg.get("torch_dtype", "auto"),
        strength=float(refiner_cfg.get("strength", 0.055)),
        guidance_scale=float(refiner_cfg.get("guidance_scale", 1.5)),
        num_inference_steps=int(refiner_cfg.get("num_inference_steps", 12)),
        prompt=refiner_cfg.get("prompt"),
        negative_prompt=refiner_cfg.get("negative_prompt"),
        enable_attention_slicing=bool(refiner_cfg.get("enable_attention_slicing", True)),
        enable_vae_slicing=bool(refiner_cfg.get("enable_vae_slicing", True)),
        enable_model_cpu_offload=bool(refiner_cfg.get("enable_model_cpu_offload", False)),
        verbose=verbose,
    )


def save_inference_wrapper_outputs(
    fusion_out: Dict[str, Any],
    output_dir: Union[str, Path],
    prefix: str = "inference",
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "pil" not in fusion_out:
        raise ValueError("fusion_out must contain PIL outputs. Use runtime.return_pil=True.")

    paths: Dict[str, Path] = {}
    for key, image in fusion_out["pil"].items():
        path = output_dir / f"{prefix}_{key}.png"
        image.save(path)
        paths[key] = path

    if "x_final" in fusion_out["pil"]:
        final_path = output_dir / "final.png"
        fusion_out["pil"]["x_final"].save(final_path)
        paths["final"] = final_path

    return paths


def run_sampling_objects_inference(
    *,
    sampling_objects,
    mixed_global_bundle: Dict[str, Any],
    mixed_local_bundle: Dict[str, Any],
    checkpoint_paths=None,
    config: Optional[Dict[str, Any]] = None,
    fusion_bundle: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    output_prefix: str = "inference",
    device: Optional[Union[str, torch.device]] = None,
    sample_global_forward_fn: Optional[Callable] = None,
    sample_local_forward_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    High-level inference from a one-person sampling loader.

    Args:
        sampling_objects:
            Usually the dict returned by build_single_person_sampling_loader(...).
            Passing sampling_objects["loader"] directly also works.

        mixed_global_bundle / mixed_local_bundle:
            Already constructed global/local diffusion bundles. Adapters may be
            injected already, or they can be injected from checkpoint metadata.

        checkpoint_paths:
            None, {"global": path, "local": path}, or (global_path, local_path).
            When provided, the wrapper restores inference adapter weights.

        config:
            Generation/fusion/refiner config. Missing keys inherit
            DEFAULT_INFERENCE_WRAPPER_CONFIG.

        fusion_bundle:
            Optional prebuilt refiner bundle. If None and config["refiner"]["enabled"]
            is true, this wrapper builds one.
    """
    config = _deep_update(DEFAULT_INFERENCE_WRAPPER_CONFIG, config)
    runtime_cfg = config["runtime"]
    gen_cfg = config["generation"]
    fusion_cfg = config["fusion"]

    verbose = bool(runtime_cfg.get("verbose", True))
    device = _resolve_device(device)
    checkpoint_paths = _normalize_checkpoint_paths(checkpoint_paths)

    restore_reports = {
        "global": _ensure_adapter_injected_from_checkpoint(
            bundle=mixed_global_bundle,
            checkpoint_path=checkpoint_paths["global"],
            strict_adapter=bool(config["checkpoints"].get("strict_adapter", True)),
            verbose=verbose,
        ),
        "local": _ensure_adapter_injected_from_checkpoint(
            bundle=mixed_local_bundle,
            checkpoint_path=checkpoint_paths["local"],
            strict_adapter=bool(config["checkpoints"].get("strict_adapter", True)),
            verbose=verbose,
        ),
    }

    loader = _resolve_sampling_loader(sampling_objects)
    batch = _first_batch(loader)
    global_batch = parse_sampling_global_batch(batch)
    local_zones = parse_sampling_local_batch(batch)

    sample_global_forward_fn = sample_global_forward_fn or default_sample_global_forward
    sample_local_forward_fn = sample_local_forward_fn or default_sample_local_forward

    generator = None
    if gen_cfg.get("seed", None) is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(gen_cfg["seed"]))

    if verbose:
        print("\n" + "=" * 100)
        print("High-level sampling inference")
        print("=" * 100)
        print("Device        :", device)
        print("Sample id     :", global_batch["sample_id"])
        print("Local zones   :", len(local_zones))
        print("Refiner       :", bool(config["refiner"].get("enabled", False) or fusion_bundle is not None))
        print("=" * 100)

    with torch.inference_mode():
        move_bundle_modules_only_to_device(mixed_global_bundle, device, eval_mode=True)
        x_global = sample_global_forward_fn(
            mixed_global_bundle=mixed_global_bundle,
            x_orig=global_batch["x_orig"],
            global_prompt=global_batch["global_prompt"],
            device=device,
            strength=float(gen_cfg["global_strength"]),
            guidance_scale=float(gen_cfg["global_guidance_scale"]),
            num_inference_steps=int(gen_cfg["global_num_inference_steps"]),
            negative_prompt=gen_cfg.get("global_negative_prompt"),
            generator=generator,
        )

        if bool(runtime_cfg.get("offload_after_each_stage", True)):
            offload_bundle_modules_only(mixed_global_bundle, label="global inference")

        move_bundle_modules_only_to_device(mixed_local_bundle, device, eval_mode=True)
        local_outputs = _call_sample_forward_with_supported_kwargs(
            sample_local_forward_fn,
            {
                "mixed_local_bundle": mixed_local_bundle,
                "zones": local_zones,
                "device": device,
                "strength": float(gen_cfg["local_strength"]),
                "guidance_scale": float(gen_cfg["local_guidance_scale"]),
                "num_inference_steps": int(gen_cfg["local_num_inference_steps"]),
                "negative_prompt": gen_cfg.get("local_negative_prompt"),
                "recycle_passes": int(gen_cfg.get("local_recycle_passes", 1)),
                "recycle_strength": gen_cfg.get("local_recycle_strength"),
                "recycle_guidance_scale": gen_cfg.get("local_recycle_guidance_scale"),
                "recycle_num_inference_steps": gen_cfg.get("local_recycle_num_inference_steps"),
                "generator": generator,
            },
        )

        if bool(runtime_cfg.get("offload_after_each_stage", True)):
            offload_bundle_modules_only(mixed_local_bundle, label="local inference")

        owns_fusion_bundle = fusion_bundle is None
        fusion_bundle = fusion_bundle or _build_refiner_from_config(
            config=config,
            device=device,
            verbose=verbose,
        )

        fusion_out = fuse_global_local_outputs(
            x_orig=global_batch["x_orig"],
            x_global=x_global,
            local_outputs=local_outputs,
            face_mask=global_batch.get("face_mask", None),
            fusion_bundle=fusion_bundle,
            residual_alpha=float(fusion_cfg["residual_alpha"]),
            residual_sigma=float(fusion_cfg["residual_sigma"]),
            use_face_mask=bool(fusion_cfg["use_face_mask"]),
            face_mask_blur_sigma=float(fusion_cfg["face_mask_blur_sigma"]),
            local_insert_alpha=float(fusion_cfg["local_insert_alpha"]),
            local_mask_blur_sigma=float(fusion_cfg["local_mask_blur_sigma"]),
            color_match=bool(fusion_cfg["color_match"]),
            color_match_strength=float(fusion_cfg["color_match_strength"]),
            fusion_prompt=fusion_cfg.get("fusion_prompt"),
            fusion_negative_prompt=fusion_cfg.get("fusion_negative_prompt"),
            fusion_strength=fusion_cfg.get("fusion_strength"),
            fusion_guidance_scale=fusion_cfg.get("fusion_guidance_scale"),
            fusion_num_inference_steps=fusion_cfg.get("fusion_num_inference_steps"),
            device=device,
            seed=gen_cfg.get("seed"),
            return_pil=bool(runtime_cfg.get("return_pil", True)),
            offload_fusion_after=False,
            verbose=verbose,
        )

        if owns_fusion_bundle and bool(config["refiner"].get("offload_after", True)):
            offload_fusion_bundle(fusion_bundle)

    paths = {}
    if output_dir is not None:
        paths = save_inference_wrapper_outputs(
            fusion_out=fusion_out,
            output_dir=output_dir,
            prefix=output_prefix,
        )

    return {
        "sample_id": global_batch["sample_id"],
        "global_batch": global_batch,
        "local_zones": local_zones,
        "local_outputs": local_outputs,
        "fusion_out": fusion_out,
        "paths": paths,
        "restore_reports": restore_reports,
        "config": config,
    }

