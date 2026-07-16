from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from scripts.common import REPO_ROOT, deep_update, load_config, print_config_summary


DEFAULT_CONFIG: Dict[str, Any] = {
    "run": {"name": "face_aging_global_local", "checkpoint_root": "training_checkpoints"},
    "device": {"device": "auto", "amp_enabled": True, "amp_dtype": "bf16"},
    "data": {
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
        "fused_batch_size": 1,
        "fused_max_crops_per_image": None,
    },
    "models": {
        "global_model_id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        "global_vae_id": "stabilityai/sd-vae-ft-mse",
        "local_model_id": "runwayml/stable-diffusion-v1-5",
        "local_vae_id": None,
        "torch_dtype": "auto",
    },
    "adapters": {
        "global": {
            "adapter_type": "lora",
            "rank": 8,
            "alpha": 8,
            "dropout": 0.05,
            "target_suffixes": ["to_q", "to_k", "to_v", "to_out.0"],
        },
        "local": {
            "adapter_type": "dora",
            "rank": 16,
            "alpha": 16,
            "dropout": 0.05,
            "target_suffixes": ["to_q", "to_k", "to_v", "to_out.0"],
        },
        "optimizer": {"lr": 7.0e-5, "betas": [0.9, 0.999], "weight_decay": 1.0e-2},
        "global_optimizer": {"lr": 5.0e-5},
        "local_optimizer": {"lr": 7.0e-5},
    },
    "score_net": {
        "checkpoint_path": "models/score net/score_net_best_overall.pt",
        "base_channels": 32,
        "dropout": 0.15,
        "strict": True,
        "freeze": True,
    },
    "losses": {
        "local": {
            "lambda_full": 1.0,
            "lambda_zone": 0.25,
            "lambda_score": 0.05,
            "lambda_cycle": 0.0,
            "score_timestep_min": 5,
            "score_timestep_max": 150,
            "score_loss_mode": "1_step_per_loss",
            "use_min_snr": True,
            "min_snr_gamma": 5.0,
        },
        "global": {
            "use_aux_bundle": True,
            "aux": {"use_age": True, "use_identity": True, "use_lpips": False},
            "lambda_diff": 1.0,
            "lambda_id": 0.35,
            "lambda_age": 0.10,
            "lambda_delta_age": 0.15,
            "lambda_perc": 0.0,
            "semantic_timestep_min": 5,
            "semantic_timestep_max": 120,
            "semantic_anchor_to_source": True,
            "delta_age_target_mode": "chronological_gap",
            "use_min_snr": True,
        },
    },
    "training": {
        "num_epochs": 5,
        "local_num_epochs": 5,
        "global_num_epochs": 2,
        "train_order": ["local", "global"],
        "train_local": True,
        "train_global": True,
        "local_grad_accum_steps": 4,
        "global_grad_accum_steps": 4,
        "local_max_batches": None,
        "global_max_batches": None,
        "inner_print_every": 10,
        "inner_verbose": False,
        "print_first_batch": False,
        "use_fused_loss": False,
        "fused_loss_epoch": 15,
        "fused_loss_every_n_steps": 1,
        "lambda_fuse_score": 0.03,
        "lambda_fuse_seam": 0.01,
        "global_p_diff": 0.70,
        "global_p_semantic": 0.30,
        "global_p_neutral": 0.05,
        "global_p_double_diff": 0.05,
        "min_target_age": 18,
        "max_target_age": 85,
    },
    "paired_supervision": {
        "enabled": False,
        "config_path": "configs/data/paired_fgnet.yaml",
    },
    "sampling": {
        "enabled": False,
        "sample_every_epochs": 0,
        "sampling_output_dir": None,
        "sample_global_strength": 0.25,
        "sample_global_guidance_scale": 3.5,
        "sample_global_num_inference_steps": 40,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-level training CLI for global/local aging.")
    parser.add_argument("--config", type=str, default="configs/training/default_train.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate config without loading models.")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def resolve_device_and_dtype(config: Dict[str, Any]):
    from src.training.mixed_precision import get_effective_amp_dtype, resolve_device

    device_name = config["device"]["device"]
    device = resolve_device(device_name)

    dtype_name = str(config["models"].get("torch_dtype", "auto")).lower()
    if dtype_name == "auto":
        dtype = get_effective_amp_dtype(
            amp_dtype=config["device"]["amp_dtype"],
            device=device,
        ) or torch.float32
    elif dtype_name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif dtype_name in {"fp16", "float16"}:
        dtype = torch.float16
    elif dtype_name in {"fp32", "float32"}:
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported models.torch_dtype={dtype_name}")

    return device, dtype


def build_data(config: Dict[str, Any]):
    from data.create_data import (
        build_global_dataloaders,
        build_local_dataloaders,
        build_local_fused_dataloaders,
    )

    data_cfg = config["data"]
    train_cfg = config["training"]
    skip_regions = data_cfg.get("skip_regions")
    local_objects = build_local_dataloaders(
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=bool(data_cfg["pin_memory"]),
        skip_regions=skip_regions,
    )
    local_fused_objects = None
    if bool(train_cfg.get("use_fused_loss", False)):
        local_fused_objects = build_local_fused_dataloaders(
            batch_size=int(data_cfg.get("fused_batch_size", 1)),
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
            max_crops_per_image=data_cfg.get("fused_max_crops_per_image"),
            skip_regions=skip_regions,
        )
    global_objects = build_global_dataloaders(
        batch_size=min(int(data_cfg["batch_size"]), 4),
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=bool(data_cfg["pin_memory"]),
        skip_regions=skip_regions,
    )
    return local_objects, global_objects, local_fused_objects


def build_model_bundles(config: Dict[str, Any], device: torch.device, dtype: torch.dtype):
    from src.diffusion_pipeline.load_diffusion_models import (
        build_global_local_bundles,
        build_mixed_lora_dora_training_setup,
    )

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

    adapter_cfg = config["adapters"]
    optimizer_config = dict(adapter_cfg["optimizer"])
    optimizer_config["betas"] = tuple(optimizer_config.get("betas", (0.9, 0.999)))

    mixed_global_bundle, mixed_local_bundle = build_mixed_lora_dora_training_setup(
        global_bundle=global_bundle,
        local_bundle=local_bundle,
        global_adapter_config=adapter_cfg["global"],
        local_adapter_config=adapter_cfg["local"],
        optimizer_config=optimizer_config,
        global_optimizer_config=adapter_cfg.get("global_optimizer"),
        local_optimizer_config=adapter_cfg.get("local_optimizer"),
        freeze_before_injection=True,
        print_memory=True,
        print_reports=True,
        verbose=True,
    )
    return mixed_global_bundle, mixed_local_bundle


def build_losses(
    config: Dict[str, Any],
    mixed_global_bundle: Dict[str, Any],
    mixed_local_bundle: Dict[str, Any],
    device: torch.device,
):
    from src.loss.global_loss import GlobalAgingLoss
    from src.loss.local_loss import LDLALocalAgingLoss
    from src.score_net.load_scorenet import load_score_net_safely

    score_cfg = config["score_net"]
    score_net = None
    if score_cfg.get("checkpoint_path"):
        score_net = load_score_net_safely(
            checkpoint_path=score_cfg["checkpoint_path"],
            device=str(device),
            dtype=torch.float32,
            base_channels=int(score_cfg.get("base_channels", 32)),
            dropout=float(score_cfg.get("dropout", 0.15)),
            strict=bool(score_cfg.get("strict", True)),
            freeze=bool(score_cfg.get("freeze", True)),
        )

    local_loss = LDLALocalAgingLoss(
        local_bundle=mixed_local_bundle,
        score_net=score_net,
        device=str(device),
        **config["losses"]["local"],
    )

    global_loss = None
    global_cfg = config["losses"]["global"]
    if global_cfg.get("use_aux_bundle", True):
        from src.loss.global_aux_bundle import GlobalLossAuxBundle

        aux_cfg = dict(global_cfg.get("aux", {}))
        aux_bundle = GlobalLossAuxBundle(device=str(device), dtype=torch.float32, **aux_cfg)
        loss_kwargs = {
            key: value
            for key, value in global_cfg.items()
            if key not in {"use_aux_bundle", "aux"}
        }
        global_loss = GlobalAgingLoss(
            global_bundle=mixed_global_bundle,
            global_loss_bundle=aux_bundle,
            device=str(device),
            **loss_kwargs,
        )

    return local_loss, global_loss


def resolve_paired_supervision_config(config):
    inline = dict(config.get("paired_supervision", {}))
    config_path = inline.pop("config_path", None)
    file_config = load_config(config_path) if config_path else {}
    # The dedicated YAML supplies reusable defaults; explicit high-level values
    # remain valid overrides (useful for short notebook experiments).
    paired_cfg = deep_update(file_config, inline) if config_path else inline
    paired_cfg["enabled"] = bool(config.get("paired_supervision", {}).get("enabled", False))
    for key in ("root", "cache_dir"):
        value = paired_cfg.get(key)
        if value:
            path = Path(value)
            paired_cfg[key] = str(path if path.is_absolute() else (REPO_ROOT / path))
    if config_path:
        paired_cfg["config_path"] = config_path
    return paired_cfg


def load_training_config(path: str | Path = "configs/training/default_train.yaml"):
    """Load one complete training config, including optional data sub-configs."""
    config = deep_update(DEFAULT_CONFIG, load_config(str(path)))
    config["paired_supervision"] = resolve_paired_supervision_config(config)
    return config


def prepare_paired_dataset(config):
    """Resolve/download paired data before expensive model allocation."""
    paired_cfg = resolve_paired_supervision_config(config)
    if not bool(paired_cfg.get("enabled", False)):
        return paired_cfg

    from data.paired_aging_dataset import ensure_paired_aging_dataset

    root = ensure_paired_aging_dataset(
        dataset=paired_cfg.get("dataset", "fgnet"),
        root=paired_cfg.get("root"),
        cache_dir=paired_cfg.get("cache_dir", REPO_ROOT / "data/external/paired_aging"),
        download_if_missing=bool(paired_cfg.get("download_if_missing", True)),
    )
    paired_cfg["root"] = str(root)
    return paired_cfg


def build_global_paired_supervision(
    config,
    mixed_global_bundle,
    device,
    paired_cfg=None,
):
    paired_cfg = prepare_paired_dataset(config) if paired_cfg is None else paired_cfg
    if not bool(paired_cfg.get("enabled", False)):
        return None, None, paired_cfg

    from data.paired_aging_dataset import build_paired_aging_dataloaders
    from src.loss.paired_supervision_loss import PairedDiffusionSupervisionLoss

    objects = build_paired_aging_dataloaders(
        root=paired_cfg["root"],
        dataset=paired_cfg.get("dataset", "auto"),
        resolution=int(paired_cfg.get("resolution", 512)),
        batch_size=int(paired_cfg.get("batch_size", 2)),
        val_fraction=float(paired_cfg.get("val_fraction", 0.15)),
        min_age_gap=int(paired_cfg.get("min_age_gap", 5)),
        max_age_gap=int(paired_cfg.get("max_age_gap", 40)),
        max_pairs_per_identity=paired_cfg.get("max_pairs_per_identity", 8),
        min_image_side=int(paired_cfg.get("min_image_side", 128)),
        seed=int(paired_cfg.get("seed", 42)),
        num_workers=int(paired_cfg.get("num_workers", config["data"].get("num_workers", 0))),
        pin_memory=bool(paired_cfg.get("pin_memory", config["data"].get("pin_memory", False))),
    )
    loss_cfg = paired_cfg.get("loss", {})
    loss_fn = PairedDiffusionSupervisionLoss(
        mixed_global_bundle,
        lambda_target_diff=float(loss_cfg.get("lambda_target_diff", 1.0)),
        lambda_source_diff=float(loss_cfg.get("lambda_source_diff", 0.25)),
        lambda_latent_delta=float(loss_cfg.get("lambda_latent_delta", 0.0)),
        use_min_snr=bool(loss_cfg.get("use_min_snr", True)),
        min_snr_gamma=float(loss_cfg.get("min_snr_gamma", 5.0)),
        device=str(device),
    )
    return objects["train_loader"], loss_fn, paired_cfg


def run_training(config: str | Path | Dict[str, Any]):
    """Run the complete global/local pipeline from a YAML path or config dict."""
    if isinstance(config, (str, Path)):
        config = load_training_config(config)
    else:
        config = deep_update(DEFAULT_CONFIG, config)
    config["paired_supervision"] = resolve_paired_supervision_config(config)
    # Network/cache work happens before allocating diffusion models on GPU.
    config["paired_supervision"] = prepare_paired_dataset(config)

    device, dtype = resolve_device_and_dtype(config)
    local_objects, global_objects, local_fused_objects = build_data(config)
    mixed_global_bundle, mixed_local_bundle = build_model_bundles(config, device, dtype)
    local_loss, global_loss = build_losses(config, mixed_global_bundle, mixed_local_bundle, device)
    global_paired_loader, global_paired_loss, paired_cfg = build_global_paired_supervision(
        config,
        mixed_global_bundle,
        device,
        paired_cfg=config["paired_supervision"],
    )

    from src.training.train_aging_model import train_global_local_face_aging

    train_cfg = config["training"]
    sampling_cfg = config.get("sampling", {})
    sampling_kwargs = {
        "sample_every_epochs": int(sampling_cfg.get("sample_every_epochs", 0)),
        "sampling_output_dir": sampling_cfg.get("sampling_output_dir"),
        "sample_global_strength": float(sampling_cfg.get("sample_global_strength", 0.25)),
        "sample_global_guidance_scale": float(sampling_cfg.get("sample_global_guidance_scale", 3.5)),
        "sample_global_num_inference_steps": int(sampling_cfg.get("sample_global_num_inference_steps", 40)),
        "sample_global_negative_prompt": sampling_cfg.get("sample_global_negative_prompt"),
        "sample_local_strength": float(sampling_cfg.get("sample_local_strength", 0.20)),
        "sample_local_guidance_scale": float(sampling_cfg.get("sample_local_guidance_scale", 0.8)),
        "sample_local_num_inference_steps": int(sampling_cfg.get("sample_local_num_inference_steps", 40)),
        "sample_local_negative_prompt": sampling_cfg.get("sample_local_negative_prompt"),
        "sample_local_recycle_passes": int(sampling_cfg.get("sample_local_recycle_passes", 1)),
        "sample_local_recycle_strength": sampling_cfg.get("sample_local_recycle_strength"),
        "sample_local_recycle_guidance_scale": sampling_cfg.get("sample_local_recycle_guidance_scale"),
        "sample_local_recycle_num_inference_steps": sampling_cfg.get("sample_local_recycle_num_inference_steps"),
        "sample_residual_alpha": float(sampling_cfg.get("sample_residual_alpha", 0.35)),
        "sample_residual_sigma": float(sampling_cfg.get("sample_residual_sigma", 9.0)),
        "sample_use_face_mask": bool(sampling_cfg.get("sample_use_face_mask", True)),
        "sample_face_mask_blur_sigma": float(sampling_cfg.get("sample_face_mask_blur_sigma", 3.0)),
        "sample_local_insert_alpha": float(sampling_cfg.get("sample_local_insert_alpha", 1.0)),
        "sample_local_mask_blur_sigma": float(sampling_cfg.get("sample_local_mask_blur_sigma", 5.0)),
        "sample_color_match": bool(sampling_cfg.get("sample_color_match", True)),
        "sample_color_match_strength": float(sampling_cfg.get("sample_color_match_strength", 0.75)),
        "sample_seed": sampling_cfg.get("sample_seed", 123),
        "sample_save_grid": bool(sampling_cfg.get("sample_save_grid", True)),
    }

    result = train_global_local_face_aging(
        mixed_local_bundle=mixed_local_bundle,
        mixed_global_bundle=mixed_global_bundle,
        local_train_loader=local_objects["train_loader"],
        global_train_loader=global_objects["train_loader"],
        local_fused_train_loader=(
            None if local_fused_objects is None else local_fused_objects["train_loader"]
        ),
        local_loss_fn=local_loss,
        global_loss_fn=global_loss,
        global_paired_train_loader=global_paired_loader,
        global_paired_loss_fn=global_paired_loss,
        global_paired_every_n_steps=int(
            paired_cfg.get("every_n_steps", 0)
        ) if global_paired_loader is not None else 0,
        global_paired_weight=float(
            paired_cfg.get("weight", 0.0)
        ) if global_paired_loader is not None else 0.0,
        device=device,
        amp_enabled=bool(config["device"]["amp_enabled"]),
        amp_dtype=str(config["device"]["amp_dtype"]),
        run_name=config["run"]["name"],
        checkpoint_root=config["run"]["checkpoint_root"],
        sampling_loader_global=None,
        sampling_loader_local=None,
        **sampling_kwargs,
        **train_cfg,
    )

    return result


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)

    if args.run_name is not None:
        config["run"]["name"] = args.run_name
    if args.checkpoint_root is not None:
        config["run"]["checkpoint_root"] = args.checkpoint_root
    if args.device is not None:
        config["device"]["device"] = args.device

    if args.print_config or args.dry_run:
        print_config_summary(config)

    if args.dry_run:
        print("[DRY RUN] Config validated. Models/data were not loaded or downloaded.")
        return

    result = run_training(config)
    print("\n[TRAINING RESULT]")
    print(result)


if __name__ == "__main__":
    main()
