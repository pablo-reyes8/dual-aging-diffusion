from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from scripts.common import deep_update, ensure_dir, load_config, print_config_summary


DEFAULT_CONFIG: Dict[str, Any] = {
    "device": {"device": "auto", "torch_dtype": "auto"},
    "models": {
        "global_model_id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        "global_vae_id": "stabilityai/sd-vae-ft-mse",
        "local_model_id": "runwayml/stable-diffusion-v1-5",
        "local_vae_id": None,
    },
    "checkpoints": {
        "global": None,
        "local": None,
        "strict_adapter": True,
    },
    "adapters": {
        "global": None,
        "local": None,
    },
    "generation": {
        "global_strength": 0.30,
        "global_guidance_scale": 5.0,
        "global_num_inference_steps": 35,
        "local_strength": 0.20,
        "local_guidance_scale": 0.8,
        "local_num_inference_steps": 40,
        "local_generation_method": "img2img",
        "local_inversion": {
            "enabled": False,
            "method": "ddim",
            "num_steps": 40,
            "strength": 0.45,
            "inversion_guidance_scale": 1.0,
            "edit_guidance_scale": None,
            "source_score_mode": "auto",
            "source_prompt_fallback": "zone",
            "negative_prompt_during_inversion": False,
            "return_source_reconstruction": False,
            "cache_enabled": True,
            "post_edit_img2img_passes": 0,
            "fallback_to_img2img": True,
        },
        "negative_prompt": "",
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
        "inversion_source_prompt": "a realistic portrait photo of the same person",
        "inversion": {
            "enabled": False,
            "method": "ddim",
            "num_steps": 20,
            "strength": 0.15,
            "inversion_guidance_scale": 1.0,
            "negative_prompt_during_inversion": False,
            "return_source_reconstruction": False,
            "fallback_to_img2img": True,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run high-level global-local aging inference.")
    parser.add_argument("--config", type=str, default="configs/inference/default_inference.yaml")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--global-prompt", type=str, required=True)
    parser.add_argument("--local-spec", type=str, required=True, help="JSON file with crop/bbox/prompt specs.")
    parser.add_argument("--output-dir", type=str, default="outputs/inference")
    parser.add_argument("--global-checkpoint", type=str, default=None)
    parser.add_argument("--local-checkpoint", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def resolve_device_dtype(config: Dict[str, Any]):
    from src.training.mixed_precision import resolve_device

    device = resolve_device(config["device"]["device"])
    dtype_name = str(config["device"].get("torch_dtype", "auto")).lower()
    if dtype_name == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif dtype_name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif dtype_name in {"fp16", "float16"}:
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def pil_to_minus1_1(image: Image.Image, size: int | None = None) -> torch.Tensor:
    if size is not None:
        image = image.resize((size, size), Image.BICUBIC)
    from src.inference.image_tensor_utils import image_to_tensor01

    return image_to_tensor01(image).squeeze(0) * 2.0 - 1.0


def load_local_specs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("crops", payload.get("zones", []))
    if not isinstance(payload, list):
        raise ValueError("local spec must be a list or a dict with key 'crops'/'zones'.")
    return payload


def load_and_prepare_bundles(config: Dict[str, Any], device: torch.device, dtype: torch.dtype):
    from src.diffusion_pipeline.load_diffusion_models import (
        apply_adapter_to_existing_bundle,
        build_global_local_bundles,
    )
    from src.training.chekpoints import load_inference_checkpoint, restore_inference_checkpoint_into_bundle

    model_cfg = config["models"]
    global_bundle, local_bundle = build_global_local_bundles(
        global_model_id=model_cfg["global_model_id"],
        global_vae_id=model_cfg["global_vae_id"],
        local_model_id=model_cfg["local_model_id"],
        local_vae_id=model_cfg.get("local_vae_id"),
        device=device,
        dtype=dtype,
        print_memory=True,
    )

    ckpt_cfg = config["checkpoints"]
    for branch, bundle in [("global", global_bundle), ("local", local_bundle)]:
        ckpt_path = ckpt_cfg.get(branch)
        adapter_cfg = config.get("adapters", {}).get(branch)

        if ckpt_path is None:
            raise ValueError(f"Missing checkpoints.{branch}; inference needs adapter .pt paths.")

        if adapter_cfg is None:
            ckpt = load_inference_checkpoint(Path(ckpt_path))
            adapter_cfg = ckpt.get("metadata", {}).get("adapter_config")
            if adapter_cfg is None:
                raise ValueError(
                    f"No adapter config found for {branch}. Provide adapters.{branch} in config."
                )

        apply_adapter_to_existing_bundle(
            bundle=bundle,
            adapter_type=adapter_cfg["adapter_type"],
            rank=adapter_cfg["rank"],
            alpha=adapter_cfg["alpha"],
            dropout=adapter_cfg.get("dropout", 0.0),
            target_suffixes=adapter_cfg.get("target_suffixes", ["to_q", "to_k", "to_v", "to_out.0"]),
            train_mode=False,
            verbose=True,
        )
        restore_inference_checkpoint_into_bundle(
            bundle=bundle,
            checkpoint_path=Path(ckpt_path),
            strict_adapter=bool(ckpt_cfg.get("strict_adapter", True)),
        )
        bundle["inference_checkpoint_id"] = str(Path(ckpt_path).resolve())

    return global_bundle, local_bundle


def main() -> None:
    args = parse_args()
    config = deep_update(DEFAULT_CONFIG, load_config(args.config))
    if args.global_checkpoint is not None:
        config["checkpoints"]["global"] = args.global_checkpoint
    if args.local_checkpoint is not None:
        config["checkpoints"]["local"] = args.local_checkpoint

    if args.print_config or args.dry_run:
        print_config_summary(config)
        print("image:", args.image)
        print("local_spec:", args.local_spec)
        print("output_dir:", args.output_dir)

    if args.dry_run:
        print("[DRY RUN] Config and required CLI arguments validated. Models were not loaded.")
        return

    from src.diffusion_pipeline import smoke_forward_models
    from src.inference.global_local_fusion import fuse_global_local_outputs
    from src.inference.image_tensor_utils import tensor01_to_pil

    device, dtype = resolve_device_dtype(config)
    smoke_forward_models.device = device
    smoke_forward_models.dtype = dtype
    output_dir = ensure_dir(args.output_dir)
    image = Image.open(args.image).convert("RGB")
    local_specs = load_local_specs(args.local_spec)
    global_bundle, local_bundle = load_and_prepare_bundles(config, device, dtype)

    gen = config["generation"]
    x_orig = pil_to_minus1_1(image)
    x_global = smoke_forward_models.img2img_single_bundle(
        bundle=global_bundle,
        image_tensor=x_orig,
        prompt=args.global_prompt,
        negative_prompt=gen.get("negative_prompt", ""),
        strength=float(gen["global_strength"]),
        guidance_scale=float(gen["global_guidance_scale"]),
        num_inference_steps=int(gen["global_num_inference_steps"]),
        seed=int(gen["seed"]),
    )

    zones = []
    for idx, spec in enumerate(local_specs):
        bbox = tuple(spec["bbox"])
        crop_path = spec.get("crop_path")
        if crop_path:
            crop_img = Image.open(crop_path).convert("RGB")
        else:
            crop_img = image.crop(bbox)
        crop_tensor = pil_to_minus1_1(crop_img)
        mask = None
        if spec.get("mask_path"):
            mask = Image.open(spec["mask_path"]).convert("L")

        zone = {
            "zone_name": spec.get("zone_name", f"zone_{idx}"),
            "crop": crop_tensor,
            "bbox": bbox,
            "mask": mask,
            "prompt": spec["prompt"],
            "negative_prompt": spec.get("negative_prompt", gen.get("negative_prompt", "")),
            "strength": float(spec.get("strength", gen["local_strength"])),
            "guidance_scale": float(spec.get("guidance_scale", gen["local_guidance_scale"])),
            "num_inference_steps": int(
                spec.get("num_inference_steps", gen["local_num_inference_steps"])
            ),
            "seed": int(spec.get("seed", gen["seed"] + idx + 1)),
        }
        for key in (
            "source_prompt",
            "zone_prompt",
            "source_score",
            "target_score",
            "inversion_cache_key",
        ):
            if key in spec:
                zone[key] = spec[key]
        zones.append(zone)

    from src.training.training_sampling_helpers import default_sample_local_forward

    local_generator = torch.Generator(device=device)
    local_generator.manual_seed(int(gen["seed"]))
    local_outputs = default_sample_local_forward(
        mixed_local_bundle=local_bundle,
        zones=zones,
        device=device,
        strength=float(gen["local_strength"]),
        guidance_scale=float(gen["local_guidance_scale"]),
        num_inference_steps=int(gen["local_num_inference_steps"]),
        negative_prompt=gen.get("negative_prompt", ""),
        generation_method=gen.get("local_generation_method", "img2img"),
        inversion_config=gen.get("local_inversion"),
        generator=local_generator,
    )

    fusion_bundle = None
    if bool(config.get("refiner", {}).get("enabled", False)):
        from src.inference.inference_wrapper import _build_refiner_from_config

        fusion_bundle = _build_refiner_from_config(config=config, device=device, verbose=True)

    fusion_out = fuse_global_local_outputs(
        x_orig=image,
        x_global=((x_global.detach().cpu() + 1.0) / 2.0).clamp(0, 1),
        local_outputs=local_outputs,
        fusion_bundle=fusion_bundle,
        device=device,
        seed=int(gen["seed"]),
        return_pil=True,
        verbose=True,
        **config["fusion"],
    )

    for key, pil_img in fusion_out["pil"].items():
        pil_img.save(output_dir / f"{key}.png")
    tensor01_to_pil(fusion_out["x_final"]).save(output_dir / "final.png")
    print(f"[OK] Saved inference outputs to: {output_dir}")


if __name__ == "__main__":
    main()
