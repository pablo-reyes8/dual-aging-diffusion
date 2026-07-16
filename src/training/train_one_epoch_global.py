# ============================================================
# TRAIN ONE EPOCH - GLOBAL BRANCH
#
# Includes optional FADING-style explicit double-prompt training.
#
# Global loss modes:
#   diff:
#       pure diffusion reconstruction loss.
#       uses source prompts.
#       can use explicit double-prompt occasionally:
#           source prompt + neutral prompt.
#
#   semantic:
#       target prompt + target age.
#       may include VAE decode + age estimator + identity encoder.
#       NO double-prompt here by default.
#
# Memory-safe double-prompt:
#   forward source -> backward
#   forward neutral -> backward
#
# NOT:
#   loss = loss_source + loss_neutral
#   loss.backward()
# ============================================================

import random
from typing import Dict, Any, Optional, List, Tuple

import torch

from src.training.train_one_epoch_local import *
# ============================================================
# Global loss mode sampling
# ============================================================

def normalize_global_loss_mode_probs(
    p_diff: float = 0.70,
    p_semantic: float = 0.30,
    enable_diff: bool = True,
    enable_semantic: bool = True,
) -> Dict[str, float]:
    """
    Normalize probabilities for global loss modes.

    Modes:
        diff:
            diffusion reconstruction loss.
            cheap relative to semantic because it does not require VAE decode
            or auxiliary models.

        semantic:
            target aging semantic loss.
            more expensive because it usually uses one-step x0 estimate,
            VAE decode, age estimator, identity encoder, etc.
    """
    probs = {
        "diff": float(p_diff) if enable_diff else 0.0,
        "semantic": float(p_semantic) if enable_semantic else 0.0,
    }

    for k, v in probs.items():
        if v < 0:
            raise ValueError(f"Probability for mode={k} must be >= 0, got {v}")

    total = sum(probs.values())

    if total <= 0:
        raise ValueError(
            "At least one global loss mode must be enabled with positive probability."
        )

    return {k: v / total for k, v in probs.items()}


def sample_global_loss_mode(
    p_diff: float = 0.70,
    p_semantic: float = 0.30,
    enable_diff: bool = True,
    enable_semantic: bool = True,
) -> str:
    """
    Samples global loss mode.

    Equivalent idea to local p_full/p_score/p_zone, but for global:

        p_diff:
            probability of pure diffusion reconstruction mode.

        p_semantic:
            probability of semantic target-aging mode.
    """
    probs = normalize_global_loss_mode_probs(
        p_diff=p_diff,
        p_semantic=p_semantic,
        enable_diff=enable_diff,
        enable_semantic=enable_semantic,
    )

    u = random.random()

    if u < probs["diff"]:
        return "diff"

    return "semantic"


# ============================================================
# Bundle mode utilities
# ============================================================

def set_global_bundle_train_mode(bundle: Dict[str, Any]) -> None:
    """
    Sets global branch modules in expected train/eval modes.

    UNet:
        train, because LoRA adapters are trained.

    VAE and text encoder:
        eval, because they are frozen.

    Auxiliary models:
        usually live inside global loss bundle/object, not necessarily here.
        They should already be frozen/eval in the loss implementation.
    """
    if "unet" in bundle and bundle["unet"] is not None:
        bundle["unet"].train()

    if "vae" in bundle and bundle["vae"] is not None:
        bundle["vae"].eval()

    if "text_encoder" in bundle and bundle["text_encoder"] is not None:
        bundle["text_encoder"].eval()


def get_global_trainable_parameters_from_bundle(
    bundle: Dict[str, Any],
) -> List[torch.nn.Parameter]:
    """
    Returns trainable UNet parameters from global bundle.
    """
    if "unet" not in bundle:
        raise KeyError("bundle must contain key 'unet'.")

    return [p for p in bundle["unet"].parameters() if p.requires_grad]


def maybe_move_global_bundle_to_device(
    bundle: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Optional helper to move global bundle modules to device.
    Useful if training branches stage-wise.
    """
    for key in ["unet", "vae", "text_encoder"]:
        if key in bundle and bundle[key] is not None:
            bundle[key].to(device)

    return bundle


# ============================================================
# Global loss call adapter
# ============================================================

def call_global_loss_fn(
    global_loss_fn,
    batch: Dict[str, Any],
    loss_mode: str,
    prompt_pack: Dict[str, Any],
    device: torch.device,
    semantic_components: Tuple[str, ...] = ("age", "delta_age", "id"),
    source_prompts_override: Optional[List[str]] = None,
    target_prompts_override: Optional[List[str]] = None,
    target_ages_override: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Calls global loss function.

    Expected global_loss_fn signature conceptually:

        global_loss_fn(
            batch=batch,
            loss_mode=loss_mode,
            semantic_components=("age", "delta_age", "id"),
            source_prompts=...,
            target_prompts=...,
            target_ages=...,
        )

    This version allows prompt overrides for double-prompt training.

    Normal single-prompt diff:
        source_prompts_override=None
        -> uses prompt_pack["selected_source_prompts"]

    Double-prompt source diff:
        source_prompts_override=prompt_pack["source_prompts"]

    Double-prompt neutral diff:
        source_prompts_override=prompt_pack["neutral_prompts"]

    Semantic:
        uses target_prompts and target_ages.
    """
    source_prompts = (
        source_prompts_override
        if source_prompts_override is not None
        else prompt_pack["selected_source_prompts"]
    )

    target_prompts = (
        target_prompts_override
        if target_prompts_override is not None
        else prompt_pack["target_prompts"]
    )

    target_ages = (
        target_ages_override
        if target_ages_override is not None
        else prompt_pack["target_ages"]
    )

    target_ages = target_ages.to(device)

    loss_out = global_loss_fn(
        batch=batch,
        loss_mode=loss_mode,
        semantic_components=semantic_components,
        source_prompts=source_prompts,
        target_prompts=target_prompts,
        target_ages=target_ages,
    )

    if not isinstance(loss_out, dict):
        raise TypeError(
            "global_loss_fn must return a dictionary containing at least key 'loss'. "
            f"Got type: {type(loss_out)}"
        )

    if "loss" not in loss_out:
        raise KeyError("global_loss_fn output must contain key 'loss'.")

    return loss_out


# ============================================================
# Main train-one-epoch global
# ============================================================

def train_one_epoch_global(
    global_bundle: Dict[str, Any],
    global_loss_fn,
    train_loader,
    device: torch.device,
    epoch: int = 0,

    # Precision.
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    scaler=None,

    # Loss mode probabilities.
    p_diff: float = 0.70,
    p_semantic: float = 0.30,
    enable_diff: bool = True,
    enable_semantic: bool = True,

    # Semantic components.
    semantic_components: Tuple[str, ...] = ("age", "delta_age", "id"),

    # Prompt construction.
    p_neutral: float = 0.05,
    min_target_age: int = 18,
    max_target_age: int = 85,

    # Explicit double-prompt.
    p_double_diff: float = 0.10,

    # Optional real longitudinal supervision (FG-NET/AgeDB bundle).
    paired_train_loader=None,
    paired_loss_fn=None,
    paired_every_n_steps: int = 0,
    paired_weight: float = 0.0,

    # Optimization.
    grad_accum_steps: int = 1,
    grad_clip: Optional[float] = 1.0,
    zero_grad_set_to_none: bool = True,

    # Scheduler.
    step_scheduler: bool = True,

    # Loop control.
    max_batches: Optional[int] = None,
    start_global_step: int = 0,
    start_optimizer_step: int = 0,

    # Logging.
    print_every: int = 50,
    print_first_batch: bool = True,
    verbose: bool = True,

    # Misc.
    skip_nonfinite_loss: bool = True,
) -> Dict[str, Any]:
    """
    Train global branch for one epoch.

    Supports two regimes:

    1. Standard CFG-style prompt dropout:
        - one selected source prompt per diffusion forward.
        - selected source prompt is source or neutral according to p_neutral.

    2. Explicit double-prompt:
        - only when loss_mode == "diff".
        - with probability p_double_diff.
        - performs:
            source forward/backward
            neutral forward/backward.
        - avoids holding two UNet graphs simultaneously.

    Args:
        p_diff:
            Probability of sampling loss_mode="diff".

            This is the global equivalent of local p_full.

        p_semantic:
            Probability of sampling loss_mode="semantic".

        p_double_diff:
            Conditional probability of explicit double-prompt
            given that loss_mode == "diff".

            Effective total frequency:
                p_diff * p_double_diff

            Example:
                p_diff=0.55, p_double_diff=0.10
                -> about 5.5% of all micro-batches use double-prompt.
    """
    if "optimizer" not in global_bundle:
        raise KeyError(
            "global_bundle must contain key 'optimizer'. "
            "It should come from build_mixed_lora_dora_training_setup(...)."
        )

    if "unet" not in global_bundle:
        raise KeyError("global_bundle must contain key 'unet'.")

    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be > 0, got {grad_accum_steps}")

    if p_double_diff is None:
        p_double_diff = 0.0

    if p_double_diff < 0 or p_double_diff > 1:
        raise ValueError(f"p_double_diff must be in [0, 1], got {p_double_diff}")

    optimizer = global_bundle["optimizer"]
    scheduler = global_bundle.get("scheduler", None)

    set_global_bundle_train_mode(global_bundle)

    trainable_params = get_global_trainable_parameters_from_bundle(global_bundle)
    ensure_trainable_parameters_fp32(trainable_params, verbose=verbose)

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found in global_bundle['unet'].")

    mode_probs = normalize_global_loss_mode_probs(
        p_diff=p_diff,
        p_semantic=p_semantic,
        enable_diff=enable_diff,
        enable_semantic=enable_semantic,
    )

    if verbose:
        print("\n========== TRAIN ONE EPOCH GLOBAL ==========")
        print("Epoch:                 ", epoch)
        print("Device:                ", device)
        print("AMP enabled:           ", amp_enabled)
        print("AMP dtype:             ", amp_dtype)
        print("Grad accumulation:     ", grad_accum_steps)
        print("Grad clip:             ", grad_clip)
        print("p_neutral:             ", p_neutral)
        print("p_diff:                ", p_diff)
        print("p_semantic:            ", p_semantic)
        print("p_double_diff:         ", p_double_diff)
        print("Approx DP frequency:   ", mode_probs["diff"] * p_double_diff)
        print("Mode probs:            ", mode_probs)
        print("Semantic components:   ", semantic_components)
        print("Target age range:      ", (min_target_age, max_target_age))
        print("Trainable tensors:     ", len(trainable_params))
        print("Trainable params:      ", sum(p.numel() for p in trainable_params))

        if scheduler is not None and hasattr(scheduler, "get_lr"):
            print("Initial LR:            ", scheduler.get_lr())
        elif hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0:
            print("Initial LR:            ", optimizer.param_groups[0]["lr"])

    tracker = MetricsTracker()

    global_step = int(start_global_step)
    optimizer_step = int(start_optimizer_step)

    n_micro_steps = 0
    n_optimizer_steps_epoch = 0
    skipped_steps = 0
    double_prompt_steps = 0
    paired_enabled = paired_supervision_enabled(
        paired_train_loader, paired_loss_fn, paired_every_n_steps, paired_weight
    )
    paired_iter = iter(paired_train_loader) if paired_enabled else None
    paired_loss_steps = 0

    optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

    num_batches = len(train_loader)
    if max_batches is not None:
        num_batches = min(num_batches, int(max_batches))

    for batch_idx, batch in enumerate(train_loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        is_last_batch = batch_idx == (num_batches - 1)

        # --------------------------------------------------------
        # Move tensor values to device.
        # Prompts and metadata remain untouched.
        # --------------------------------------------------------
        batch = move_batch_to_device(batch, device)

        # --------------------------------------------------------
        # Build source / neutral / target global prompts.
        # --------------------------------------------------------
        global_prompt_pack = build_global_prompt_pack_from_loader_prompts(
            loader_prompts=batch["prompt"],
            source_ages=batch["age"],
            p_neutral=p_neutral,
            min_target_age=min_target_age,
            max_target_age=max_target_age,
        )

        # --------------------------------------------------------
        # Sample global loss mode.
        # --------------------------------------------------------
        loss_mode = sample_global_loss_mode(
            p_diff=p_diff,
            p_semantic=p_semantic,
            enable_diff=enable_diff,
            enable_semantic=enable_semantic,
        )

        # --------------------------------------------------------
        # Explicit double-prompt only for diff mode.
        # --------------------------------------------------------
        use_double_prompt = (
            loss_mode == "diff"
            and p_double_diff > 0
            and random.random() < float(p_double_diff)
        )

        # --------------------------------------------------------
        # Forward / backward.
        # --------------------------------------------------------
        if use_double_prompt:
            double_prompt_steps += 1

            # ====================================================
            # Explicit double-prompt, memory-safe.
            #
            # 1. source prompt diff forward/backward
            # 2. neutral prompt diff forward/backward
            #
            # Divide by 2 to keep the micro-step loss scale
            # comparable to a single-prompt step.
            # ====================================================

            # -----------------------------
            # Source prompt.
            # -----------------------------
            with autocast_ctx(
                device=device,
                enabled=amp_enabled,
                amp_dtype=amp_dtype,
                cache_enabled=True,
            ):
                loss_out_source = call_global_loss_fn(
                    global_loss_fn=global_loss_fn,
                    batch=batch,
                    loss_mode="diff",
                    prompt_pack=global_prompt_pack,
                    device=device,
                    semantic_components=semantic_components,
                    source_prompts_override=global_prompt_pack["source_prompts"],
                )

                raw_loss_source = loss_out_source["loss"]

                if not torch.is_tensor(raw_loss_source):
                    raise TypeError(
                        "loss_out_source['loss'] must be a torch.Tensor, "
                        f"got {type(raw_loss_source)}"
                    )

                loss_source = raw_loss_source / float(grad_accum_steps * 2)

            if skip_nonfinite_loss and not torch.isfinite(raw_loss_source.detach()).all():
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

                print(
                    "[WARN] Non-finite GLOBAL source loss. "
                    "Skipping accumulated gradients. "
                    f"{training_batch_diagnostic_context(branch='global', batch_idx=batch_idx, global_step=global_step, loss_mode='diff/source', batch=batch, prompt_pack=global_prompt_pack, use_double_prompt=use_double_prompt)} | "
                    f"Components: {nonfinite_loss_details(loss_out_source)}"
                )

                continue

            backward_with_optional_scaler(
                loss=loss_source,
                optimizer=optimizer,
                scaler=scaler,
                retain_graph=False,
            )

            # -----------------------------
            # Neutral prompt.
            # -----------------------------
            with autocast_ctx(
                device=device,
                enabled=amp_enabled,
                amp_dtype=amp_dtype,
                cache_enabled=True):
                loss_out_neutral = call_global_loss_fn(
                    global_loss_fn=global_loss_fn,
                    batch=batch,
                    loss_mode="diff",
                    prompt_pack=global_prompt_pack,
                    device=device,
                    semantic_components=semantic_components,
                    source_prompts_override=global_prompt_pack["neutral_prompts"],
                )

                raw_loss_neutral = loss_out_neutral["loss"]

                if not torch.is_tensor(raw_loss_neutral):
                    raise TypeError(
                        "loss_out_neutral['loss'] must be a torch.Tensor, "
                        f"got {type(raw_loss_neutral)}"
                    )

                loss_neutral = raw_loss_neutral / float(grad_accum_steps * 2)

            if skip_nonfinite_loss and not torch.isfinite(raw_loss_neutral.detach()).all():
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

                print(
                    "[WARN] Non-finite GLOBAL neutral loss. "
                    "Skipping accumulated gradients. "
                    f"{training_batch_diagnostic_context(branch='global', batch_idx=batch_idx, global_step=global_step, loss_mode='diff/neutral', batch=batch, prompt_pack=global_prompt_pack, use_double_prompt=use_double_prompt)} | "
                    f"Components: {nonfinite_loss_details(loss_out_neutral)}"
                )

                continue

            backward_with_optional_scaler(
                loss=loss_neutral,
                optimizer=optimizer,
                scaler=scaler,
                retain_graph=False,
            )

            # Metric-only combined loss.
            raw_loss = 0.5 * (
                raw_loss_source.detach().float()
                + raw_loss_neutral.detach().float()
            )

            loss_out = dict(loss_out_source)
            loss_out["loss"] = raw_loss
            loss_out["double_prompt/source_loss"] = raw_loss_source.detach()
            loss_out["double_prompt/neutral_loss"] = raw_loss_neutral.detach()
            loss_out["double_prompt/used"] = torch.tensor(1.0)

        else:
            # ====================================================
            # Standard single-prompt path.
            #
            # diff:
            #   selected_source_prompts is source or neutral
            #   according to p_neutral.
            #
            # semantic:
            #   target_prompts + target_ages.
            # ====================================================
            with autocast_ctx(
                device=device,
                enabled=amp_enabled,
                amp_dtype=amp_dtype,
                cache_enabled=True,
            ):
                loss_out = call_global_loss_fn(
                    global_loss_fn=global_loss_fn,
                    batch=batch,
                    loss_mode=loss_mode,
                    prompt_pack=global_prompt_pack,
                    device=device,
                    semantic_components=semantic_components,
                    source_prompts_override=None,
                )

                raw_loss = loss_out["loss"]

                if not torch.is_tensor(raw_loss):
                    raise TypeError(
                        "loss_out['loss'] must be a torch.Tensor, "
                        f"got {type(raw_loss)}"
                    )

                loss = raw_loss / float(grad_accum_steps)

            if skip_nonfinite_loss and not torch.isfinite(raw_loss.detach()).all():
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

                print(
                    "[WARN] Non-finite GLOBAL loss. "
                    "Skipping accumulated gradients. "
                    f"{training_batch_diagnostic_context(branch='global', batch_idx=batch_idx, global_step=global_step, loss_mode=loss_mode, batch=batch, prompt_pack=global_prompt_pack, use_double_prompt=use_double_prompt)} | "
                    f"Components: {nonfinite_loss_details(loss_out)}"
                )

                continue

            backward_with_optional_scaler(
                loss=loss,
                optimizer=optimizer,
                scaler=scaler,
                retain_graph=False,
            )

            loss_out["double_prompt/used"] = torch.tensor(0.0)

        # Separate GT forward: exact-age denoising on a real longitudinal pair.
        paired_out = None
        run_paired = paired_enabled and should_run_paired_supervision(
            batch_idx, paired_every_n_steps
        )
        if run_paired:
            paired_batch, paired_iter = next_cycling_batch(
                paired_train_loader, paired_iter
            )
            paired_batch = move_batch_to_device(paired_batch, device)
            with autocast_ctx(
                device=device,
                enabled=amp_enabled,
                amp_dtype=amp_dtype,
                cache_enabled=True,
            ):
                paired_out = call_paired_supervision_loss(paired_loss_fn, paired_batch)
                raw_paired_loss = paired_out["loss"]
                paired_loss = (
                    float(paired_weight) * raw_paired_loss / float(grad_accum_steps)
                )
            if skip_nonfinite_loss and not torch.isfinite(raw_paired_loss.detach()).all():
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
                print("[WARN] Non-finite GLOBAL paired supervision loss; skipping gradients.")
                continue
            backward_with_optional_scaler(
                loss=paired_loss,
                optimizer=optimizer,
                scaler=scaler,
                retain_graph=False,
            )
            paired_loss_steps += 1
            loss_out["loss"] = (
                loss_out["loss"].detach()
                + float(paired_weight) * raw_paired_loss.detach()
            )
            loss_out["paired/used"] = torch.tensor(1.0)
        else:
            loss_out["paired/used"] = torch.tensor(0.0)

        # --------------------------------------------------------
        # Counters.
        # Double-prompt still counts as one micro-step for
        # gradient accumulation.
        # --------------------------------------------------------
        n_micro_steps += 1
        global_step += 1

        # --------------------------------------------------------
        # Metrics.
        # --------------------------------------------------------
        batch_metrics = compute_global_training_metrics(
            batch=batch,
            prompt_pack=global_prompt_pack,
            loss_out=loss_out,
            loss_mode=loss_mode,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        batch_metrics["double_prompt/used"] = 1.0 if use_double_prompt else 0.0
        batch_metrics["paired/used"] = 1.0 if run_paired else 0.0
        if paired_out is not None:
            batch_metrics["paired/loss"] = float(
                paired_out["loss"].detach().float().cpu().item()
            )
            if "age_gap_mean" in paired_out:
                batch_metrics["paired/age_gap_mean"] = float(
                    paired_out["age_gap_mean"].detach().float().cpu().item()
                )

        if use_double_prompt:
            batch_metrics["double_prompt/source_loss"] = float(
                raw_loss_source.detach().float().cpu().item()
            )
            batch_metrics["double_prompt/neutral_loss"] = float(
                raw_loss_neutral.detach().float().cpu().item()
            )

        batch_size = int(batch["pixel_values"].shape[0]) if "pixel_values" in batch else 1
        tracker.update(batch_metrics, n=batch_size)

        # --------------------------------------------------------
        # Optimizer step.
        # --------------------------------------------------------
        should_step = (
            (n_micro_steps % grad_accum_steps == 0)
            or is_last_batch
        )

        if should_step:
            step_applied = optimizer_step_with_optional_scaler(
                optimizer=optimizer,
                scaler=scaler,
                grad_clip=grad_clip,
                parameters=trainable_params,
            )

            if not step_applied:
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
                print(
                    "[WARN] Non-finite GLOBAL gradients. "
                    "Skipping optimizer step. "
                    f"{training_batch_diagnostic_context(branch='global', batch_idx=batch_idx, global_step=global_step, loss_mode=loss_mode, batch=batch, prompt_pack=global_prompt_pack, use_double_prompt=use_double_prompt)} | "
                    f"Gradients: {nonfinite_gradient_details(trainable_params)} | "
                    f"Parameters: {nonfinite_parameter_details(trainable_params)}"
                )
                continue

            if step_scheduler and scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

            optimizer_step += 1
            n_optimizer_steps_epoch += 1

        # --------------------------------------------------------
        # Printing.
        # --------------------------------------------------------
        should_print = (
            (print_every is not None and print_every > 0 and batch_idx % print_every == 0)
            or (print_first_batch and batch_idx == 0)
        )

        if should_print:
            prefix = f"[GLOBAL train e{epoch} b{batch_idx}/{num_batches}]"
            print_global_metrics(
                batch_metrics,
                prefix=prefix,
                step=global_step,
            )

    # ============================================================
    # Epoch summary.
    # ============================================================

    epoch_metrics = tracker.compute()

    if verbose:
        print_global_metrics(
            epoch_metrics,
            prefix=f"[GLOBAL epoch {epoch} summary]",
            step=global_step,
        )

        print("\n[GLOBAL epoch counters]")
        print("Micro steps epoch:        ", n_micro_steps)
        print("Optimizer steps epoch:    ", n_optimizer_steps_epoch)
        print("Double-prompt microsteps: ", double_prompt_steps)
        print("Double-prompt fraction:   ", double_prompt_steps / max(1, n_micro_steps))
        print("Global step:              ", global_step)
        print("Optimizer step:           ", optimizer_step)
        print("Skipped nonfinite steps:  ", skipped_steps)

        if scheduler is not None:
            if hasattr(scheduler, "get_lr_dict"):
                print("Scheduler LR dict:        ", scheduler.get_lr_dict())
            elif hasattr(scheduler, "get_lr"):
                print("Scheduler LR:             ", scheduler.get_lr())

    return {
        "epoch": int(epoch),
        "epoch_metrics": epoch_metrics,
        "global_step": int(global_step),
        "optimizer_step": int(optimizer_step),
        "n_micro_steps": int(n_micro_steps),
        "n_optimizer_steps_epoch": int(n_optimizer_steps_epoch),
        "paired_loss_steps": int(paired_loss_steps),
        "double_prompt_steps": int(double_prompt_steps),
        "double_prompt_fraction": float(double_prompt_steps / max(1, n_micro_steps)),
        "skipped_steps": int(skipped_steps)}
