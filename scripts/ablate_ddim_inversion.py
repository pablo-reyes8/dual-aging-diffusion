from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image, ImageDraw

from scripts.common import deep_update, ensure_dir, load_config, print_config_summary
from scripts.inference_cli import (
    DEFAULT_CONFIG,
    load_local_specs,
    pil_to_minus1_1,
    resolve_device_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare historical local img2img against partial DDIM inversion on a GPU host."
    )
    parser.add_argument("--config", default="configs/inference/ddim_inversion.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--local-spec", required=True)
    parser.add_argument("--local-checkpoint", required=True)
    parser.add_argument("--score-net-checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/ddim_inversion_ablation")
    parser.add_argument("--strengths", default="0.25,0.35,0.45,0.55")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--compute-lpips",
        action="store_true",
        help="Compute LPIPS for source reconstruction (requires the optional lpips package).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def _parse_strengths(raw: str) -> List[float]:
    values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("--strengths must contain comma-separated values in [0, 1].")
    return values


def _load_local_bundle(config: Dict[str, Any], checkpoint_path: Path, device, dtype):
    from src.diffusion_pipeline.load_diffusion_models import (
        apply_adapter_to_existing_bundle,
        build_diffusion_bundle,
        is_no_vae_checkpoint,
    )
    from src.training.chekpoints import load_inference_checkpoint, restore_inference_checkpoint_into_bundle

    model_cfg = config["models"]
    local_model_id = model_cfg["local_model_id"]
    local_vae_id = model_cfg.get("local_vae_id")
    if local_vae_id is None and is_no_vae_checkpoint(local_model_id):
        local_vae_id = model_cfg.get("global_vae_id")

    bundle = build_diffusion_bundle(
        name=f"Local_ablation_{Path(local_model_id).name}",
        model_id=local_model_id,
        vae_id=local_vae_id,
        device=device,
        dtype=dtype,
    )
    checkpoint = load_inference_checkpoint(checkpoint_path)
    adapter_cfg = config.get("adapters", {}).get("local")
    if adapter_cfg is None:
        adapter_cfg = checkpoint.get("metadata", {}).get("adapter_config")
    if adapter_cfg is None:
        raise ValueError(
            "The checkpoint has no adapter_config metadata. Set adapters.local in the config."
        )
    adapter_type = adapter_cfg.get(
        "adapter_type", checkpoint.get("metadata", {}).get("adapter_type", "dora")
    )
    apply_adapter_to_existing_bundle(
        bundle=bundle,
        adapter_type=adapter_type,
        rank=adapter_cfg["rank"],
        alpha=adapter_cfg["alpha"],
        dropout=adapter_cfg.get("dropout", 0.0),
        target_suffixes=adapter_cfg.get("target_suffixes", ["to_q", "to_k", "to_v", "to_out.0"]),
        train_mode=False,
        verbose=True,
    )
    restore_inference_checkpoint_into_bundle(
        bundle=bundle,
        checkpoint_path=checkpoint_path,
        strict_adapter=bool(config["checkpoints"].get("strict_adapter", True)),
    )
    bundle["inference_checkpoint_id"] = str(checkpoint_path.resolve())
    return bundle


def _timed_call(fn, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    value = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_mb = None
    return value, time.perf_counter() - start, peak_mb


def _score(score_net, image) -> Optional[float]:
    if score_net is None:
        return None
    from src.inference.diffusion_inversion import evaluate_score_net
    from src.inference.image_tensor_utils import image_to_tensor01

    x = image_to_tensor01(image, device=next(score_net.parameters()).device) * 2.0 - 1.0
    result = evaluate_score_net(score_net, x)
    return None if result is None else 100.0 * result


def _lpips_value(lpips_model, source, reconstruction, device) -> Optional[float]:
    if lpips_model is None or reconstruction is None:
        return None
    from src.inference.image_tensor_utils import image_to_tensor01

    source_m11 = image_to_tensor01(source, device=device) * 2.0 - 1.0
    rec_m11 = image_to_tensor01(reconstruction, device=device) * 2.0 - 1.0
    with torch.inference_mode():
        return float(lpips_model(source_m11, rec_m11).view(-1)[0].detach().cpu().item())


def _save_grid(images: List[tuple[str, Image.Image]], path: Path) -> None:
    if not images:
        return
    width, height = images[0][1].size
    label_h = 30
    canvas = Image.new("RGB", (width * len(images), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        image = image.convert("RGB").resize((width, height), Image.BICUBIC)
        canvas.paste(image, (index * width, label_h))
        draw.text((index * width + 6, 8), label, fill="black")
    canvas.save(path)


def run_ablation(args: argparse.Namespace) -> Dict[str, Any]:
    strengths = _parse_strengths(args.strengths)
    config = deep_update(DEFAULT_CONFIG, load_config(args.config))
    config["checkpoints"]["local"] = args.local_checkpoint
    if args.seed is not None:
        config["generation"]["seed"] = int(args.seed)

    if args.print_config or args.dry_run:
        print_config_summary(config)
        print("image:", args.image)
        print("local_spec:", args.local_spec)
        print("local_checkpoint:", args.local_checkpoint)
        print("strengths:", strengths)
    if args.dry_run:
        print("[DRY RUN] Ablation inputs validated. Models were not loaded.")
        return {"dry_run": True}

    device, dtype = resolve_device_dtype(config)
    if device.type != "cuda":
        print("[WARN] CUDA is not active; this ablation is intended for Vast.ai/GPU execution.")
    output_dir = ensure_dir(args.output_dir)
    image = Image.open(args.image).convert("RGB")
    specs = load_local_specs(args.local_spec)
    bundle = _load_local_bundle(config, Path(args.local_checkpoint), device, dtype)

    score_net = None
    if args.score_net_checkpoint:
        from src.score_net.load_scorenet import load_score_net_safely

        score_net = load_score_net_safely(
            checkpoint_path=args.score_net_checkpoint,
            device=str(device),
            dtype=torch.float32,
            freeze=True,
            eval_mode=True,
        )
        bundle["score_net"] = score_net

    lpips_model = None
    if args.compute_lpips:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("--compute-lpips requires: pip install lpips") from exc
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    from src.inference.image_tensor_utils import tensor01_to_pil
    from src.training.training_sampling_helpers import default_sample_local_forward
    from src.training.target_prompt_building import extract_score_from_local_prompt

    gen = config["generation"]
    seed = int(gen["seed"])
    report: Dict[str, Any] = {
        "checkpoint": str(Path(args.local_checkpoint).resolve()),
        "model_id": bundle.get("model_id"),
        "adapter_type": bundle.get("adapter_type"),
        "strengths": strengths,
        "zones": [],
    }

    for index, spec in enumerate(specs):
        bbox = tuple(spec["bbox"])
        crop_image = (
            Image.open(spec["crop_path"]).convert("RGB")
            if spec.get("crop_path")
            else image.crop(bbox)
        )
        zone_dir = ensure_dir(output_dir / f"{index:02d}_{spec.get('zone_name', 'zone')}")
        crop_image.save(zone_dir / "source.png")
        crop_tensor = pil_to_minus1_1(crop_image)
        zone = dict(spec)
        zone.update({"crop": crop_tensor, "bbox": bbox, "mask": None})
        zone_seed = int(spec.get("seed", seed + index + 1))

        def sample(method: str, inversion: Optional[Dict[str, Any]] = None):
            generator = torch.Generator(device=device).manual_seed(zone_seed)
            return default_sample_local_forward(
                mixed_local_bundle=bundle,
                zones=[zone],
                device=device,
                strength=float(spec.get("strength", gen["local_strength"])),
                guidance_scale=float(spec.get("guidance_scale", gen["local_guidance_scale"])),
                num_inference_steps=int(
                    spec.get("num_inference_steps", gen["local_num_inference_steps"])
                ),
                negative_prompt=spec.get("negative_prompt", gen.get("negative_prompt", "")),
                generation_method=method,
                inversion_config=inversion,
                generator=generator,
            )[0]

        historical, elapsed, peak_mb = _timed_call(lambda: sample("img2img"), device)
        historical_pil = tensor01_to_pil(
            ((historical["aged_crop"].detach().cpu() + 1.0) / 2.0).clamp(0, 1)
        )
        historical_pil.save(zone_dir / "img2img.png")
        target_score = spec.get("target_score")
        if target_score is None:
            target_score = extract_score_from_local_prompt(spec["prompt"])
        zone_report: Dict[str, Any] = {
            "zone_name": spec.get("zone_name", f"zone_{index}"),
            "target_prompt": spec["prompt"],
            "source_score": _score(score_net, crop_tensor),
            "target_score": target_score,
            "img2img": {
                "seconds": elapsed,
                "peak_cuda_memory_mb": peak_mb,
                "score": _score(score_net, historical["aged_crop"]),
            },
            "ddim_inversion": [],
        }
        grid_items = [("source", crop_image), ("img2img", historical_pil)]

        for strength in strengths:
            inversion_cfg = dict(gen.get("local_inversion", {}))
            inversion_cfg.update(
                {
                    "enabled": True,
                    "method": "ddim",
                    "strength": strength,
                    "return_source_reconstruction": True,
                    "fallback_to_img2img": False,
                }
            )
            result, inv_elapsed, inv_peak_mb = _timed_call(
                lambda cfg=inversion_cfg: sample("ddim_inversion", cfg), device
            )
            edited_pil = tensor01_to_pil(
                ((result["aged_crop"].detach().cpu() + 1.0) / 2.0).clamp(0, 1)
            )
            label = f"ddim_{strength:.2f}"
            edited_pil.save(zone_dir / f"{label}.png")
            reconstruction = result.get("source_reconstruction")
            reconstruction_score = None
            reconstruction_lpips = None
            if reconstruction is not None:
                rec_pil = tensor01_to_pil(((reconstruction.cpu() + 1.0) / 2.0).clamp(0, 1))
                rec_pil.save(zone_dir / f"reconstruction_{strength:.2f}.png")
                reconstruction_score = _score(score_net, reconstruction)
                reconstruction_lpips = _lpips_value(
                    lpips_model, crop_tensor, reconstruction, device
                )
                grid_items.append((f"rec {strength:.2f}", rec_pil))
            grid_items.append((label, edited_pil))
            diagnostics = result.get("inversion_diagnostics", {})
            edited_score = _score(score_net, result["aged_crop"])
            target_score = zone_report["target_score"]
            zone_report["ddim_inversion"].append(
                {
                    "strength": strength,
                    "seconds": inv_elapsed,
                    "peak_cuda_memory_mb": inv_peak_mb,
                    "score": edited_score,
                    "target_score_error": None
                    if edited_score is None or target_score is None
                    else abs(float(edited_score) - float(target_score)),
                    "reconstruction_score": reconstruction_score,
                    "reconstruction_lpips": reconstruction_lpips,
                    "diagnostics": diagnostics,
                }
            )

        _save_grid(grid_items, zone_dir / "comparison.png")
        report["zones"].append(zone_report)

    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] DDIM inversion ablation saved to: {output_dir}")
    return report


def main() -> None:
    run_ablation(parse_args())


if __name__ == "__main__":
    main()
