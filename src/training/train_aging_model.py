# ============================================================
# FINAL TRAINING WRAPPER - GLOBAL + LOCAL FACE AGING
#
# Responsibilities:
#   - Train local/global branches stage-wise.
#   - Move only active branch to GPU.
#   - Move inactive branch + optimizer states to CPU.
#   - Enable gradient checkpointing.
#   - Build schedulers if needed.
#   - Instantiate losses if factories are provided.
#   - Save LAST and BEST checkpoints only.
#   - Clean CUDA memory aggressively between branches.
# ============================================================

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from src.training.train_one_epoch_global import *
from src.training.scheduler_warmup import *
from src.training.chekpoints import *
from src.training.training_display_helpers import *
from src.training.training_memory_helpers import *
from src.training.training_loss_helpers import *
from src.training.training_sampling_helpers import *

# ============================================================
# Final wrapper
# ============================================================

def train_global_local_face_aging(
    *,
    # Core objects.
    mixed_local_bundle: Dict[str, Any],
    mixed_global_bundle: Dict[str, Any],
    local_train_loader,
    global_train_loader,

    # Losses: either pass existing instances or factories.
    local_loss_fn=None,
    global_loss_fn=None,
    local_loss_factory: Optional[Callable] = None,
    global_loss_factory: Optional[Callable] = None,
    local_loss_kwargs: Optional[Dict[str, Any]] = None,
    global_loss_kwargs: Optional[Dict[str, Any]] = None,

    # Auxiliary objects/bundles that must move CPU/GPU with each branch.
    local_aux_objects: Optional[Sequence[Any]] = None,
    global_aux_objects: Optional[Sequence[Any]] = None,

    # Device / precision.
    device: Optional[torch.device] = None,
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    scaler=None,

    # Schedule.
    run_name: str = "face_aging_global_local",
    num_epochs: int = 5,
    local_num_epochs: Optional[int] = None,
    global_num_epochs: Optional[int] = None,
    start_epoch: int = 0,
    train_order: Tuple[str, ...] = ("local", "global"),
    train_local: bool = True,
    train_global: bool = True,

    # Gradient accumulation.
    local_grad_accum_steps: int = 4,
    global_grad_accum_steps: int = 4,
    local_grad_clip: Optional[float] = 1.0,
    global_grad_clip: Optional[float] = 1.0,

    # Local loss mode config.
    local_p_full: float = 0.50,
    local_p_score: float = 0.35,
    local_p_zone: float = 0.15,
    local_enable_full: bool = True,
    local_enable_score: bool = True,
    local_enable_zone: bool = True,
    local_p_neutral: float = 0.10,
    local_p_double_full: float = 0.15,

    # Directional/contrastive score loss (Error 3). The numeric hyperparameters
    # (lambda_direction, margin, high/low score, timesteps) are configured on the
    # local loss via local_loss_kwargs; these control how often it runs.
    local_p_direction: float = 0.0,
    local_enable_direction: bool = False,

    # True classifier-free-guidance dropout for the local branch (empty-prompt).
    local_p_uncond: float = 0.0,

    # Optional supervised longitudinal crops. Disabled unless all four values
    # are configured; old notebook calls therefore keep identical behavior.
    local_paired_train_loader=None,
    local_paired_loss_fn=None,
    local_paired_every_n_steps: int = 0,
    local_paired_weight: float = 0.0,

    # Optional local deterministic fused loss.
    local_fused_train_loader=None,
    local_fused_loss_fn=None,
    use_fused_loss: bool = False,
    fused_loss_epoch: int = 15,
    fused_loss_every_n_steps: int = 1,
    lambda_fuse_score: float = 0.03,
    lambda_fuse_seam: float = 0.01,
    fused_global_forward_fn: Optional[Callable] = None,

    # Global loss mode config.
    global_p_diff: float = 0.70,
    global_p_semantic: float = 0.30,
    global_enable_diff: bool = True,
    global_enable_semantic: bool = True,
    global_semantic_components: Tuple[str, ...] = ("age", "delta_age", "id"),
    global_p_neutral: float = 0.05,
    global_p_double_diff: float = 0.05,
    min_target_age: int = 18,
    max_target_age: int = 85,

    # Optional supervised longitudinal full faces (FG-NET/AgeDB).
    global_paired_train_loader=None,
    global_paired_loss_fn=None,
    global_paired_every_n_steps: int = 0,
    global_paired_weight: float = 0.0,

    # Schedulers.
    build_schedulers_if_missing: bool = True,
    local_warmup_ratio: float = 0.05,
    global_warmup_ratio: float = 0.05,
    local_min_lr: float = 1e-6,
    global_min_lr: float = 1e-6,
    min_warmup_steps: int = 10,
    max_warmup_steps: Optional[int] = None,

    # Checkpoints.
    checkpoint_root: str = "/content/drive/MyDrive/face_aging_checkpoints",
    checkpoint_managers: Optional[Dict[str, Any]] = None,
    save_latest: bool = True,
    save_best: bool = True,
    save_epoch_checkpoints: bool = True,
    save_inference_copy: bool = True,
    local_monitor_key: str = "loss/total",
    global_monitor_key: str = "loss/total",

    # Memory.
    enable_gradient_checkpointing_flag: bool = True,
    offload_after_each_branch: bool = True,
    print_memory: bool = True,

    # Loop control.
    local_max_batches: Optional[int] = None,
    global_max_batches: Optional[int] = None,

    # Reconstruction sampling.
    sampling_loader_global=None,
    sampling_loader_local=None,
    sample_every_epochs: int = 0,
    sample_after_epoch_zero: bool = False,
    sampling_output_dir: Optional[str] = None,
    sample_global_forward_fn: Optional[Callable] = None,
    sample_local_forward_fn: Optional[Callable] = None,

    # Sampling generation params.
    sample_global_strength: float = 0.30,
    sample_global_guidance_scale: float = 5.0,
    sample_global_num_inference_steps: int = 35,
    sample_global_negative_prompt: Optional[str] = None,

    sample_local_strength: float = 0.20,
    sample_local_guidance_scale: float = 0.8,
    sample_local_num_inference_steps: int = 40,
    sample_local_negative_prompt: Optional[str] = None,
    sample_local_recycle_passes: int = 1,
    sample_local_recycle_strength: Optional[float] = None,
    sample_local_recycle_guidance_scale: Optional[float] = None,
    sample_local_recycle_num_inference_steps: Optional[int] = None,

    # Sampling deterministic fusion params.
    sample_residual_alpha: float = 0.35,
    sample_residual_sigma: float = 9.0,
    sample_use_face_mask: bool = True,
    sample_face_mask_blur_sigma: float = 3.0,
    sample_local_insert_alpha: float = 1.0,
    sample_local_mask_blur_sigma: float = 5.0,
    sample_color_match: bool = True,
    sample_color_match_strength: float = 0.75,
    sample_seed: Optional[int] = 123,
    sample_save_grid: bool = True,

    # Printing.
    inner_print_every: int = 0,
    inner_verbose: bool = False,
    print_first_batch: bool = False,
    verbose: bool = True,

    # Counters.
    local_start_global_step: int = 0,
    local_start_optimizer_step: int = 0,
    global_start_global_step: int = 0,
    global_start_optimizer_step: int = 0,
) -> Dict[str, Any]:
    """
    Final training wrapper for both branches.

    New behavior:
        - local_num_epochs and global_num_epochs can differ.
        - If one branch finishes earlier, it stops training and remains available
          for checkpoints/sampling.
        - Optional deterministic reconstruction sampling every N epochs.

    Sampling assumptions:
        sampling_loader_global:
            one full-face person:
                image/x_orig/pixel_values
                global_prompt/prompt
                optional face_mask

        sampling_loader_local:
            same person's local crops:
                either batch["zones"] list
                or crop/prompt/bbox/mask arrays/lists.

    Sampling does:
        global eval forward -> x_global
        local eval forward  -> local_outputs
        deterministic fusion -> saved images

    It does NOT:
        use refiner
        use ScoreNet
        track gradients
        move optimizer states to GPU
    """
    # ------------------------------------------------------------
    # Epoch configuration.
    # ------------------------------------------------------------
    if local_num_epochs is None:
        local_num_epochs = int(num_epochs)

    if global_num_epochs is None:
        global_num_epochs = int(num_epochs)

    local_num_epochs = int(local_num_epochs)
    global_num_epochs = int(global_num_epochs)

    if local_num_epochs < 0 or global_num_epochs < 0:
        raise ValueError("local_num_epochs and global_num_epochs must be >= 0.")

    total_loop_epochs = max(local_num_epochs, global_num_epochs)

    if total_loop_epochs <= 0:
        raise ValueError("At least one of local_num_epochs/global_num_epochs must be > 0.")

    # ------------------------------------------------------------
    # Device.
    # ------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint_root = Path(checkpoint_root)

    if sampling_output_dir is None:
        sampling_output_dir = checkpoint_root / "samples"
    else:
        sampling_output_dir = Path(sampling_output_dir)

    sampling_enabled = (
        sample_every_epochs is not None
        and int(sample_every_epochs) > 0
        and sampling_loader_global is not None
        and sampling_loader_local is not None
    )

    # ------------------------------------------------------------
    # Checkpoint managers.
    # ------------------------------------------------------------
    if checkpoint_managers is None:
        checkpoint_managers = build_global_local_checkpoint_managers(
            root_dir=checkpoint_root,
            global_metric_mode="min",
            local_metric_mode="min",
        )

    local_ckpt_manager = checkpoint_managers["local"]
    global_ckpt_manager = checkpoint_managers["global"]

    local_config = {
        "p_full": local_p_full,
        "p_score": local_p_score,
        "p_zone": local_p_zone,
        "p_double_full": local_p_double_full,
        "num_epochs": local_num_epochs,
    }

    global_config = {
        "p_diff": global_p_diff,
        "p_semantic": global_p_semantic,
        "p_double_diff": global_p_double_diff,
        "num_epochs": global_num_epochs,
    }

    if verbose:
        print_run_header_v2(
            run_name=run_name,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            total_epochs=total_loop_epochs,
            local_num_epochs=local_num_epochs,
            global_num_epochs=global_num_epochs,
            start_epoch=start_epoch,
            local_grad_accum_steps=local_grad_accum_steps,
            global_grad_accum_steps=global_grad_accum_steps,
            local_config=local_config,
            global_config=global_config,
            checkpoint_root=checkpoint_root,
            sampling_enabled=sampling_enabled,
            sample_every_epochs=int(sample_every_epochs),
        )

    # ------------------------------------------------------------
    # Enable gradient checkpointing once.
    # ------------------------------------------------------------
    if enable_gradient_checkpointing_flag:
        enable_unet_gradient_checkpointing(mixed_local_bundle, branch_name="local")
        enable_unet_gradient_checkpointing(mixed_global_bundle, branch_name="global")

    # ------------------------------------------------------------
    # Build schedulers if missing.
    # Important:
    #   Each branch scheduler is built with its own num_epochs.
    # ------------------------------------------------------------
    if build_schedulers_if_missing:
        if train_local and local_num_epochs > 0 and "scheduler" not in mixed_local_bundle:
            build_bundle_scheduler_from_loader(
                bundle=mixed_local_bundle,
                train_loader=local_train_loader,
                num_epochs=local_num_epochs,
                grad_accum_steps=local_grad_accum_steps,
                warmup_ratio=local_warmup_ratio,
                min_lr=local_min_lr,
                min_warmup_steps=min_warmup_steps,
                max_warmup_steps=max_warmup_steps,
            )

        if train_global and global_num_epochs > 0 and "scheduler" not in mixed_global_bundle:
            build_bundle_scheduler_from_loader(
                bundle=mixed_global_bundle,
                train_loader=global_train_loader,
                num_epochs=global_num_epochs,
                grad_accum_steps=global_grad_accum_steps,
                warmup_ratio=global_warmup_ratio,
                min_lr=global_min_lr,
                min_warmup_steps=min_warmup_steps,
                max_warmup_steps=max_warmup_steps,
            )

    # ------------------------------------------------------------
    # Instantiate losses if needed.
    # ------------------------------------------------------------
    if train_local and local_num_epochs > 0:
        local_loss_fn = maybe_build_loss(
            existing_loss=local_loss_fn,
            loss_factory=local_loss_factory,
            loss_kwargs=local_loss_kwargs,
            name="local",
        )

    if train_global and global_num_epochs > 0:
        global_loss_fn = maybe_build_loss(
            existing_loss=global_loss_fn,
            loss_factory=global_loss_factory,
            loss_kwargs=global_loss_kwargs,
            name="global",
        )

    if use_fused_loss:
        if local_fused_train_loader is None:
            raise ValueError("use_fused_loss=True requires local_fused_train_loader.")
        if local_fused_loss_fn is None:
            from src.loss.local_fused_loss import LocalFusedFusionLoss

            score_net = getattr(local_loss_fn, "score_net", None)
            local_fused_loss_fn = LocalFusedFusionLoss(
                score_net=score_net,
                lambda_fuse_score=lambda_fuse_score,
                lambda_fuse_seam=lambda_fuse_seam,
            )

    # ------------------------------------------------------------
    # Initial CPU offload.
    # ------------------------------------------------------------
    print_section("Initial CPU offload")
    move_bundle_modules_to_device(mixed_local_bundle, torch.device("cpu"))
    move_bundle_modules_to_device(mixed_global_bundle, torch.device("cpu"))

    if local_aux_objects is not None:
        for obj in local_aux_objects:
            move_any_to_device(obj, torch.device("cpu"))

    if global_aux_objects is not None:
        for obj in global_aux_objects:
            move_any_to_device(obj, torch.device("cpu"))

    hard_cuda_cleanup(label="after initial CPU offload", reset_peak=True)

    # ------------------------------------------------------------
    # Counters.
    # ------------------------------------------------------------
    local_global_step = int(local_start_global_step)
    local_optimizer_step = int(local_start_optimizer_step)

    global_global_step = int(global_start_global_step)
    global_optimizer_step = int(global_start_optimizer_step)

    history = {
        "local": [],
        "global": [],
        "samples": [],
    }

    total_start_time = time.time()

    # ============================================================
    # Epoch loop.
    # ============================================================

    for epoch in range(start_epoch, total_loop_epochs):
        if verbose:
            print_epoch_header(epoch, total_loop_epochs)

        local_active_this_epoch = train_local and epoch < local_num_epochs
        global_active_this_epoch = train_global and epoch < global_num_epochs

        if verbose:
            print(
                f"Branch activity -> "
                f"local={local_active_this_epoch} ({min(epoch + 1, local_num_epochs)}/{local_num_epochs}) | "
                f"global={global_active_this_epoch} ({min(epoch + 1, global_num_epochs)}/{global_num_epochs})"
            )

        for branch in train_order:
            branch = str(branch).lower().strip()

            # ====================================================
            # LOCAL BRANCH
            # ====================================================
            if branch == "local":
                if not local_active_this_epoch:
                    if verbose and train_local and epoch >= local_num_epochs:
                        print("└─ [LOCAL] training budget finished; skipping branch.")
                    continue

                prepare_branch_for_training(
                    active_bundle=mixed_local_bundle,
                    active_aux_objects=local_aux_objects,
                    inactive_bundle=mixed_global_bundle,
                    inactive_aux_objects=global_aux_objects,
                    device=device,
                    branch_name="local",
                    enable_gradient_checkpointing_flag=enable_gradient_checkpointing_flag,
                    print_memory=print_memory,
                )

                start = time.time()

                local_result = train_one_epoch_local(
                    local_bundle=mixed_local_bundle,
                    local_loss_fn=local_loss_fn,
                    train_loader=local_train_loader,
                    device=device,
                    epoch=epoch,

                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    scaler=scaler,

                    p_full=local_p_full,
                    p_score=local_p_score,
                    p_zone=local_p_zone,
                    p_direction=local_p_direction,
                    enable_full=local_enable_full,
                    enable_score=local_enable_score,
                    enable_zone=local_enable_zone,
                    enable_direction=local_enable_direction,

                    p_neutral=local_p_neutral,
                    p_uncond=local_p_uncond,
                    p_double_full=local_p_double_full,

                    grad_accum_steps=local_grad_accum_steps,
                    grad_clip=local_grad_clip,

                    step_scheduler=True,

                    max_batches=local_max_batches,
                    start_global_step=local_global_step,
                    start_optimizer_step=local_optimizer_step,

                    fused_train_loader=local_fused_train_loader,
                    local_fused_loss_fn=local_fused_loss_fn,
                    use_fused_loss=use_fused_loss,
                    fused_loss_epoch=fused_loss_epoch,
                    fused_loss_every_n_steps=fused_loss_every_n_steps,
                    fused_global_forward_fn=fused_global_forward_fn,

                    paired_train_loader=local_paired_train_loader,
                    paired_loss_fn=local_paired_loss_fn,
                    paired_every_n_steps=local_paired_every_n_steps,
                    paired_weight=local_paired_weight,

                    print_every=inner_print_every,
                    print_first_batch=print_first_batch,
                    verbose=inner_verbose,
                )

                elapsed = time.time() - start

                local_global_step = local_result["global_step"]
                local_optimizer_step = local_result["optimizer_step"]

                local_metrics = local_result["epoch_metrics"]
                local_metric_value = get_monitor_metric(
                    metrics=local_metrics,
                    monitor_key=local_monitor_key,
                )

                improved = False
                best_paths = {}

                if save_best and local_metric_value is not None:
                    improved, best_paths = local_ckpt_manager.save_best_if_improved(
                        bundle=mixed_local_bundle,
                        metric_value=local_metric_value,
                        epoch=epoch,
                        global_step=local_global_step,
                        optimizer_step=local_optimizer_step,
                        loss_value=local_metric_value,
                        scaler=scaler,
                        extra_metadata={
                            "run_name": run_name,
                            "branch": "local",
                            "epoch": epoch,
                            "monitor_key": local_monitor_key,
                            "mode_config": local_config,
                            "grad_accum_steps": local_grad_accum_steps,
                            "branch_num_epochs": local_num_epochs,
                        },
                        save_inference_copy=save_inference_copy,
                    )

                latest_paths = {}

                if save_latest:
                    latest_paths = local_ckpt_manager.save_latest(
                        bundle=mixed_local_bundle,
                        epoch=epoch,
                        global_step=local_global_step,
                        optimizer_step=local_optimizer_step,
                        loss_value=local_metric_value,
                        metric_value=local_metric_value,
                        scaler=scaler,
                        extra_metadata={
                            "run_name": run_name,
                            "branch": "local",
                            "epoch": epoch,
                            "monitor_key": local_monitor_key,
                            "mode_config": local_config,
                            "grad_accum_steps": local_grad_accum_steps,
                            "branch_num_epochs": local_num_epochs,
                        },
                        save_inference_copy=save_inference_copy,
                    )

                epoch_paths = {}

                if save_epoch_checkpoints:
                    epoch_paths = local_ckpt_manager.save_epoch(
                        bundle=mixed_local_bundle,
                        epoch=epoch,
                        global_step=local_global_step,
                        optimizer_step=local_optimizer_step,
                        loss_value=local_metric_value,
                        metric_value=local_metric_value,
                        scaler=scaler,
                        extra_metadata={
                            "run_name": run_name,
                            "branch": "local",
                            "epoch": epoch,
                            "monitor_key": local_monitor_key,
                            "mode_config": local_config,
                            "grad_accum_steps": local_grad_accum_steps,
                            "branch_num_epochs": local_num_epochs,
                        },
                        save_inference_copy=save_inference_copy,
                    )

                if verbose:
                    print_branch_summary(
                        branch_name="local",
                        result=local_result,
                        elapsed=elapsed,
                        bundle=mixed_local_bundle,
                        monitor_key=local_monitor_key,
                        metric_value=local_metric_value,
                        best_metric=local_ckpt_manager.best_metric,
                        improved=improved,
                    )

                    print_checkpoint_report(
                        branch_name="local",
                        latest_paths={**latest_paths, **epoch_paths},
                        best_paths=best_paths,
                        improved=improved,
                    )

                history["local"].append({
                    "epoch": epoch,
                    "result": local_result,
                    "monitor_value": local_metric_value,
                    "improved": improved,
                    "elapsed": elapsed,
                })

                if offload_after_each_branch:
                    offload_branch_after_training(
                        bundle=mixed_local_bundle,
                        aux_objects=local_aux_objects,
                        branch_name="local",
                        print_memory=print_memory,
                    )

            # ====================================================
            # GLOBAL BRANCH
            # ====================================================
            elif branch == "global":
                if not global_active_this_epoch:
                    if verbose and train_global and epoch >= global_num_epochs:
                        print("└─ [GLOBAL] training budget finished; skipping branch.")
                    continue

                prepare_branch_for_training(
                    active_bundle=mixed_global_bundle,
                    active_aux_objects=global_aux_objects,
                    inactive_bundle=mixed_local_bundle,
                    inactive_aux_objects=local_aux_objects,
                    device=device,
                    branch_name="global",
                    enable_gradient_checkpointing_flag=enable_gradient_checkpointing_flag,
                    print_memory=print_memory,
                )

                start = time.time()

                global_result = train_one_epoch_global(
                    global_bundle=mixed_global_bundle,
                    global_loss_fn=global_loss_fn,
                    train_loader=global_train_loader,
                    device=device,
                    epoch=epoch,

                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    scaler=scaler,

                    p_diff=global_p_diff,
                    p_semantic=global_p_semantic,
                    enable_diff=global_enable_diff,
                    enable_semantic=global_enable_semantic,

                    semantic_components=global_semantic_components,

                    p_neutral=global_p_neutral,
                    min_target_age=min_target_age,
                    max_target_age=max_target_age,

                    p_double_diff=global_p_double_diff,

                    paired_train_loader=global_paired_train_loader,
                    paired_loss_fn=global_paired_loss_fn,
                    paired_every_n_steps=global_paired_every_n_steps,
                    paired_weight=global_paired_weight,

                    grad_accum_steps=global_grad_accum_steps,
                    grad_clip=global_grad_clip,

                    step_scheduler=True,

                    max_batches=global_max_batches,
                    start_global_step=global_global_step,
                    start_optimizer_step=global_optimizer_step,

                    print_every=inner_print_every,
                    print_first_batch=print_first_batch,
                    verbose=inner_verbose,
                )

                elapsed = time.time() - start

                global_global_step = global_result["global_step"]
                global_optimizer_step = global_result["optimizer_step"]

                global_metrics = global_result["epoch_metrics"]
                global_metric_value = get_monitor_metric(
                    metrics=global_metrics,
                    monitor_key=global_monitor_key,
                )

                improved = False
                best_paths = {}

                if save_best and global_metric_value is not None:
                    improved, best_paths = global_ckpt_manager.save_best_if_improved(
                        bundle=mixed_global_bundle,
                        metric_value=global_metric_value,
                        epoch=epoch,
                        global_step=global_global_step,
                        optimizer_step=global_optimizer_step,
                        loss_value=global_metric_value,
                        scaler=scaler,
                        extra_metadata={
                            "run_name": run_name,
                            "branch": "global",
                            "epoch": epoch,
                            "monitor_key": global_monitor_key,
                            "mode_config": global_config,
                            "semantic_components": global_semantic_components,
                            "target_age_range": (min_target_age, max_target_age),
                            "grad_accum_steps": global_grad_accum_steps,
                            "branch_num_epochs": global_num_epochs,
                        },
                        save_inference_copy=save_inference_copy,
                    )

                latest_paths = {}

                if save_latest:
                    latest_paths = global_ckpt_manager.save_latest(
                        bundle=mixed_global_bundle,
                        epoch=epoch,
                        global_step=global_global_step,
                        optimizer_step=global_optimizer_step,
                        loss_value=global_metric_value,
                        metric_value=global_metric_value,
                        scaler=scaler,
                        extra_metadata={
                            "run_name": run_name,
                            "branch": "global",
                            "epoch": epoch,
                            "monitor_key": global_monitor_key,
                            "mode_config": global_config,
                            "semantic_components": global_semantic_components,
                            "target_age_range": (min_target_age, max_target_age),
                            "grad_accum_steps": global_grad_accum_steps,
                            "branch_num_epochs": global_num_epochs,
                        },
                        save_inference_copy=save_inference_copy,
                    )

                epoch_paths = {}

                if save_epoch_checkpoints:
                    epoch_paths = global_ckpt_manager.save_epoch(
                        bundle=mixed_global_bundle,
                        epoch=epoch,
                        global_step=global_global_step,
                        optimizer_step=global_optimizer_step,
                        loss_value=global_metric_value,
                        metric_value=global_metric_value,
                        scaler=scaler,
                        extra_metadata={
                            "run_name": run_name,
                            "branch": "global",
                            "epoch": epoch,
                            "monitor_key": global_monitor_key,
                            "mode_config": global_config,
                            "semantic_components": global_semantic_components,
                            "target_age_range": (min_target_age, max_target_age),
                            "grad_accum_steps": global_grad_accum_steps,
                            "branch_num_epochs": global_num_epochs,
                        },
                        save_inference_copy=save_inference_copy,
                    )

                if verbose:
                    print_branch_summary(
                        branch_name="global",
                        result=global_result,
                        elapsed=elapsed,
                        bundle=mixed_global_bundle,
                        monitor_key=global_monitor_key,
                        metric_value=global_metric_value,
                        best_metric=global_ckpt_manager.best_metric,
                        improved=improved,
                    )

                    print_checkpoint_report(
                        branch_name="global",
                        latest_paths={**latest_paths, **epoch_paths},
                        best_paths=best_paths,
                        improved=improved,
                    )

                history["global"].append({
                    "epoch": epoch,
                    "result": global_result,
                    "monitor_value": global_metric_value,
                    "improved": improved,
                    "elapsed": elapsed,
                })

                if offload_after_each_branch:
                    offload_branch_after_training(
                        bundle=mixed_global_bundle,
                        aux_objects=global_aux_objects,
                        branch_name="global",
                        print_memory=print_memory,
                    )

            else:
                raise ValueError(
                    f"Unknown branch in train_order: {branch}. "
                    "Expected 'local' or 'global'."
                )

        # ========================================================
        # Deterministic reconstruction sampling.
        # ========================================================

        should_sample = False

        if sampling_enabled:
            epoch_number = epoch + 1

            if epoch == start_epoch and sample_after_epoch_zero:
                should_sample = True

            if int(sample_every_epochs) > 0 and epoch_number % int(sample_every_epochs) == 0:
                should_sample = True

            # Also sample at the final loop epoch.
            if epoch == total_loop_epochs - 1:
                should_sample = True

        if should_sample:
            try:
                sample_result = run_deterministic_training_reconstruction_sample(
                    mixed_global_bundle=mixed_global_bundle,
                    mixed_local_bundle=mixed_local_bundle,

                    sampling_loader_global=sampling_loader_global,
                    sampling_loader_local=sampling_loader_local,

                    device=device,
                    run_name=run_name,
                    epoch=epoch,
                    output_dir=sampling_output_dir,

                    sample_global_forward_fn=sample_global_forward_fn,
                    sample_local_forward_fn=sample_local_forward_fn,

                    sample_global_strength=sample_global_strength,
                    sample_global_guidance_scale=sample_global_guidance_scale,
                    sample_global_num_inference_steps=sample_global_num_inference_steps,
                    sample_global_negative_prompt=sample_global_negative_prompt,

                    sample_local_strength=sample_local_strength,
                    sample_local_guidance_scale=sample_local_guidance_scale,
                    sample_local_num_inference_steps=sample_local_num_inference_steps,
                    sample_local_negative_prompt=sample_local_negative_prompt,
                    sample_local_recycle_passes=sample_local_recycle_passes,
                    sample_local_recycle_strength=sample_local_recycle_strength,
                    sample_local_recycle_guidance_scale=sample_local_recycle_guidance_scale,
                    sample_local_recycle_num_inference_steps=sample_local_recycle_num_inference_steps,

                    residual_alpha=sample_residual_alpha,
                    residual_sigma=sample_residual_sigma,
                    use_face_mask=sample_use_face_mask,
                    face_mask_blur_sigma=sample_face_mask_blur_sigma,
                    local_insert_alpha=sample_local_insert_alpha,
                    local_mask_blur_sigma=sample_local_mask_blur_sigma,
                    color_match=sample_color_match,
                    color_match_strength=sample_color_match_strength,

                    seed=sample_seed,
                    save_grid=sample_save_grid,
                    verbose=verbose,
                )

                history["samples"].append(sample_result)

            except Exception as e:
                print_section("Sampling failed")
                print(f"Sampling failed at epoch={epoch}: {repr(e)}")
                print(
                    "Training continues. If bundles do not expose img2img pipelines, "
                    "pass sample_global_forward_fn and sample_local_forward_fn."
                )

                hard_cuda_cleanup(label=f"after failed sampling epoch={epoch}", reset_peak=True)

        hard_cuda_cleanup(label=f"end of epoch {epoch}", reset_peak=True)

    # ============================================================
    # End.
    # ============================================================

    total_elapsed = time.time() - total_start_time

    print_box_title("Training finished")
    print(f"Run name          : {run_name}")
    print(f"Total time        : {format_seconds(total_elapsed)}")
    print(f"Loop epochs       : {total_loop_epochs}")
    print(f"Local epochs      : {local_num_epochs}")
    print(f"Global epochs     : {global_num_epochs}")
    print(f"Local best        : {local_ckpt_manager.best_metric}")
    print(f"Global best       : {global_ckpt_manager.best_metric}")
    print(f"Checkpoint root   : {checkpoint_root}")
    print(f"Sampling output   : {sampling_output_dir if sampling_enabled else 'disabled'}")

    return {
        "run_name": run_name,
        "history": history,

        "num_epochs": int(num_epochs),
        "total_loop_epochs": int(total_loop_epochs),
        "local_num_epochs": int(local_num_epochs),
        "global_num_epochs": int(global_num_epochs),

        "local_global_step": int(local_global_step),
        "local_optimizer_step": int(local_optimizer_step),
        "global_global_step": int(global_global_step),
        "global_optimizer_step": int(global_optimizer_step),

        "local_best_metric": local_ckpt_manager.best_metric,
        "global_best_metric": global_ckpt_manager.best_metric,

        "checkpoint_root": str(checkpoint_root),
        "sampling_output_dir": str(sampling_output_dir) if sampling_enabled else None,
        "checkpoint_managers": checkpoint_managers,
    }
