# ============================================================
# TRAIN ONE EPOCH - LOCAL BRANCH
#
# Responsibilities:
#   - sample local loss mode: full / zone / score
#   - build source / neutral / target prompts
#   - move batch tensors to device
#   - mixed precision forward
#   - gradient accumulation
#   - optional grad clipping
#   - optimizer step
#   - scheduler step if bundle["scheduler"] exists
#   - cheap training metrics
#
# ============================================================

import math
import random
from typing import Dict, Any, Optional, Tuple, List

import torch

from src.training.mixed_precision import * 
from src.training.target_prompt_building import *  
from src.training.metrics import * 


# ============================================================
# Loss mode sampling
# ============================================================

def normalize_loss_mode_probs(
    p_full: float = 0.50,
    p_score: float = 0.35,
    p_zone: float = 0.15,
    p_direction: float = 0.0,
    enable_full: bool = True,
    enable_score: bool = True,
    enable_zone: bool = True,
    enable_direction: bool = False) -> Dict[str, float]:

    """
    Normalize probabilities after applying mode flags.

    If a mode is disabled, its probability is set to zero.
    Remaining probabilities are renormalized.

    Returns:
        dict with keys: full, score, zone, direction.
    """
    probs = {
        "full": float(p_full) if enable_full else 0.0,
        "score": float(p_score) if enable_score else 0.0,
        "zone": float(p_zone) if enable_zone else 0.0,
        "direction": float(p_direction) if enable_direction else 0.0,
    }

    for k, v in probs.items():
        if v < 0:
            raise ValueError(f"Probability for mode={k} must be >= 0, got {v}")

    total = sum(probs.values())

    if total <= 0:
        raise ValueError(
            "At least one local loss mode must be enabled with positive probability."
        )

    return {k: v / total for k, v in probs.items()}


def sample_local_loss_mode(
    p_full: float = 0.50,
    p_score: float = 0.35,
    p_zone: float = 0.15,
    p_direction: float = 0.0,
    enable_full: bool = True,
    enable_score: bool = True,
    enable_zone: bool = True,
    enable_direction: bool = False) -> str:
    """
    Samples local loss mode.

    Modes:
        full:
            standard diffusion loss using selected source prompt.

        zone:
            zone-prompt diffusion loss.

        score:
            target prompt + target score loss.

        direction:
            relative/contrastive score loss (high vs low score edits).
    """
    probs = normalize_loss_mode_probs(
        p_full=p_full,
        p_score=p_score,
        p_zone=p_zone,
        p_direction=p_direction,
        enable_full=enable_full,
        enable_score=enable_score,
        enable_zone=enable_zone,
        enable_direction=enable_direction,
    )

    u = random.random()

    cum = probs["full"]
    if u < cum:
        return "full"

    cum += probs["score"]
    if u < cum:
        return "score"

    cum += probs["zone"]
    if u < cum:
        return "zone"

    return "direction"


# ============================================================
# Bundle mode utilities
# ============================================================

def set_local_bundle_train_mode(bundle: Dict[str, Any]) -> None:
    """
    Sets the local branch in the expected training/eval modes.

    UNet:
        train, because adapters are trained.

    VAE and text encoder:
        eval, because they are frozen.
    """
    if "unet" in bundle and bundle["unet"] is not None:
        bundle["unet"].train()

    if "vae" in bundle and bundle["vae"] is not None:
        bundle["vae"].eval()

    if "text_encoder" in bundle and bundle["text_encoder"] is not None:
        bundle["text_encoder"].eval()


def get_trainable_parameters_from_bundle(bundle: Dict[str, Any]) -> List[torch.nn.Parameter]:
    """
    Returns trainable UNet parameters from a bundle.
    """
    if "unet" not in bundle:
        raise KeyError("bundle must contain key 'unet'.")

    return [p for p in bundle["unet"].parameters() if p.requires_grad]


def maybe_move_local_bundle_to_device(
    bundle: Dict[str, Any],
    device: torch.device) -> Dict[str, Any]:

    """
    Optional helper.

    Moves main modules in bundle to device. Usually not necessary if
    you already loaded the bundle on GPU, but useful for stage-wise training.
    """
    for key in ["unet", "vae", "text_encoder"]:
        if key in bundle and bundle[key] is not None:
            bundle[key].to(device)

    return bundle


# ============================================================
# Loss call adapter
# ============================================================

def call_local_loss_fn(
    local_loss_fn,
    batch: Dict[str, Any],
    loss_mode: str,
    prompt_pack: Dict[str, Any],
    device: torch.device,
    source_prompts_override: Optional[List[str]] = None,
    target_prompts_override: Optional[List[str]] = None,
    zone_prompts_override: Optional[List[str]] = None,
    zone_prompt_key: str = "zone_prompt",
    pass_target_scores_for_all_modes: bool = True,
    direction_high_prompts: Optional[List[str]] = None,
    direction_low_prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calls local loss function.

    This version allows prompt overrides, which lets us implement
    FADING-style explicit double-prompt training without touching
    the loss class itself.

    Normal single-prompt case:
        source_prompts_override=None
        -> uses prompt_pack["selected_source_prompts"]

    Double-prompt source pass:
        source_prompts_override=prompt_pack["source_prompts"]

    Double-prompt neutral pass:
        source_prompts_override=prompt_pack["neutral_prompts"]

    Important:
        This function performs only ONE loss call / ONE UNet forward.
        Double-prompt is implemented by calling this function twice
        sequentially in train_one_epoch_local.
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

    if zone_prompts_override is not None:
        zone_prompts = zone_prompts_override
    elif zone_prompt_key in batch:
        zone_prompts = batch[zone_prompt_key]
    else:
        zone_prompts = prompt_pack["neutral_prompts"]

    target_scores = prompt_pack["target_scores"].to(device)

    if (not pass_target_scores_for_all_modes) and loss_mode != "score":
        target_scores_arg = None
    else:
        target_scores_arg = target_scores

    loss_out = local_loss_fn(
        batch=batch,
        loss_mode=loss_mode,
        source_prompts=source_prompts,
        zone_prompts=zone_prompts,
        target_prompts=target_prompts,
        target_scores=target_scores_arg,
        direction_high_prompts=direction_high_prompts,
        direction_low_prompts=direction_low_prompts,
    )

    if not isinstance(loss_out, dict):
        raise TypeError(
            "local_loss_fn must return a dictionary containing at least key 'loss'. "
            f"Got type: {type(loss_out)}"
        )

    if "loss" not in loss_out:
        raise KeyError("local_loss_fn output must contain key 'loss'.")

    return loss_out


def nonfinite_loss_details(loss_out: Dict[str, Any], max_items: int = 8) -> str:
    """
    Short diagnostic string for loss dictionaries with NaN/Inf tensors.
    """
    details = []
    for key, value in loss_out.items():
        if not torch.is_tensor(value):
            continue

        detached = value.detach()
        if detached.numel() == 0 or torch.isfinite(detached).all():
            continue

        nonfinite = (~torch.isfinite(detached)).sum().item()
        details.append(f"{key}: shape={tuple(detached.shape)} nonfinite={int(nonfinite)}")

        if len(details) >= int(max_items):
            break

    return "; ".join(details) if details else "no non-finite tensor components found"


def _tensor_range_text(name: str, value: Any) -> Optional[str]:
    if not torch.is_tensor(value):
        return None
    if value.numel() == 0:
        return f"{name}=empty"

    x = value.detach().float()
    finite = torch.isfinite(x)
    finite_count = int(finite.sum().item())
    nonfinite_count = int((~finite).sum().item())

    if finite_count == 0:
        return f"{name}: shape={tuple(value.shape)} all_nonfinite={nonfinite_count}"

    finite_x = x[finite]
    return (
        f"{name}: shape={tuple(value.shape)} "
        f"min={finite_x.min().item():.6g} "
        f"max={finite_x.max().item():.6g} "
        f"mean={finite_x.mean().item():.6g} "
        f"nonfinite={nonfinite_count}"
    )


def training_batch_diagnostic_context(
    *,
    branch: str,
    batch_idx: int,
    global_step: int,
    loss_mode: str,
    batch: Dict[str, Any],
    prompt_pack: Optional[Dict[str, Any]] = None,
    use_double_prompt: Optional[bool] = None,
) -> str:
    """
    Compact always-on context for NaN/Inf diagnostics.
    """
    parts = [
        f"branch={branch}",
        f"batch_idx={batch_idx}",
        f"global_step={global_step}",
        f"loss_mode={loss_mode}",
    ]

    if use_double_prompt is not None:
        parts.append(f"double_prompt={bool(use_double_prompt)}")

    for key in ["pixel_values", "score", "score_raw", "age", "age_norm"]:
        text = _tensor_range_text(key, batch.get(key, None))
        if text is not None:
            parts.append(text)

    if prompt_pack is not None:
        for key in ["target_scores", "target_scores_raw", "target_ages"]:
            text = _tensor_range_text(f"prompt_pack.{key}", prompt_pack.get(key, None))
            if text is not None:
                parts.append(text)

    return " | ".join(parts)


def _flatten_nested_prompts(prompts: List[List[str]], valid_mask: torch.Tensor) -> List[str]:
    out: List[str] = []
    valid_cpu = valid_mask.detach().cpu()
    for b, per_image in enumerate(prompts):
        for k, prompt in enumerate(per_image):
            if k < valid_cpu.shape[1] and bool(valid_cpu[b, k].item()):
                out.append(prompt)
    return out


def generate_local_aged_crops_for_fused_loss(
    local_loss_fn,
    fused_batch: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Generates target-conditioned local crops with gradients to the local UNet.
    """
    pixel_values = fused_batch["pixel_values"].to(device)
    valid_mask = fused_batch["valid_mask"].to(device=device, dtype=torch.bool)
    B, K = valid_mask.shape

    flat_pixels = pixel_values[valid_mask]
    target_prompts = _flatten_nested_prompts(fused_batch["prompt"], valid_mask)
    if flat_pixels.shape[0] == 0:
        raise ValueError("Fused batch has no valid local crops.")

    z0 = local_loss_fn.encode_images_to_latents(flat_pixels)
    noise = torch.randn_like(z0)

    t_min = int(getattr(local_loss_fn, "score_timestep_min", 20))
    t_max = int(getattr(local_loss_fn, "score_timestep_max", 400))
    timesteps = torch.randint(
        low=t_min,
        high=t_max + 1,
        size=(z0.shape[0],),
        device=device,
    ).long()

    zt = local_loss_fn.add_noise(z0, noise, timesteps)
    target_hidden = local_loss_fn.encode_prompts(target_prompts)
    noise_pred = local_loss_fn.predict_noise(
        noisy_latents=zt,
        timesteps=timesteps,
        encoder_hidden_states=target_hidden,
    )
    x0_hat = local_loss_fn.predict_x0_from_noise(
        noisy_latents=zt,
        timesteps=timesteps,
        noise_pred=noise_pred,
    )
    aged_flat = local_loss_fn.decode_latents_to_images(x0_hat)

    aged_crops = torch.zeros(
        B,
        K,
        aged_flat.shape[1],
        aged_flat.shape[2],
        aged_flat.shape[3],
        device=device,
        dtype=aged_flat.dtype,
    )
    aged_crops[valid_mask] = aged_flat
    return aged_crops


def should_run_fused_loss(
    use_fused_loss: bool,
    epoch: int,
    fused_loss_epoch: int,
    fused_loss_every_n_steps: int,
    global_step: int,
) -> bool:
    if not use_fused_loss:
        return False
    if int(epoch) < int(fused_loss_epoch):
        return False
    every = max(1, int(fused_loss_every_n_steps))
    return int(global_step) % every == 0

# ============================================================
# Main train-one-epoch function
# ============================================================

def train_one_epoch_local(
    local_bundle: Dict[str, Any],
    local_loss_fn,
    train_loader,
    device: torch.device,
    epoch: int = 0,

    # Precision.
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    scaler=None,

    # Loss mode probabilities.
    p_full: float = 0.50,
    p_score: float = 0.35,
    p_zone: float = 0.15,
    p_direction: float = 0.0,
    enable_full: bool = True,
    enable_score: bool = True,
    enable_zone: bool = True,
    enable_direction: bool = False,

    # Prompt construction.
    p_neutral: float = 0.10,

    # True classifier-free-guidance dropout: with this probability the source
    # prompt is replaced by the empty string "" (a real unconditional sample),
    # enabling CFG at inference. Off by default (baseline).
    p_uncond: float = 0.0,

    # Explicit double-prompt.
    p_double_full: float = 0.15,

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

    # Optional aligned fused local loss.
    fused_train_loader=None,
    local_fused_loss_fn=None,
    use_fused_loss: bool = False,
    fused_loss_epoch: int = 15,
    fused_loss_every_n_steps: int = 1,
    fused_global_forward_fn=None,

    # Logging.
    print_every: int = 50,
    print_first_batch: bool = True,
    verbose: bool = True,

    # Misc.
    zone_prompt_key: str = "zone_prompt",
    skip_nonfinite_loss: bool = True,
) -> Dict[str, Any]:
    """
    Train local branch for one epoch.

    This function supports two prompt-training regimes:

    1. Standard CFG-style prompt dropout:
        - one selected prompt per forward
        - selected prompt is source or neutral according to p_neutral

    2. Explicit double-prompt, used occasionally:
        - only when loss_mode == "full"
        - with probability p_double_full
        - performs:
            source forward/backward
            neutral forward/backward
        - avoids holding two UNet graphs simultaneously.

    Args:
        local_bundle:
            Must contain:
                local_bundle["unet"]
                local_bundle["optimizer"]
            May contain:
                local_bundle["scheduler"]

        local_loss_fn:
            Local loss object/function.

        train_loader:
            Local dataloader.

        device:
            torch.device.

        p_double_full:
            Probability of doing explicit source+neutral double-prompt
            when loss_mode == "full".

            Effective global frequency approximately:
                p_full * p_double_full

            Example:
                p_full=0.50, p_double_full=0.15
                -> ~7.5% of micro-batches use double-prompt.

    Returns:
        Dictionary with epoch metrics and updated counters.
    """
    if "optimizer" not in local_bundle:
        raise KeyError(
            "local_bundle must contain key 'optimizer'. "
            "It should come from build_mixed_lora_dora_training_setup(...)."
        )

    if "unet" not in local_bundle:
        raise KeyError("local_bundle must contain key 'unet'.")

    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be > 0, got {grad_accum_steps}")

    if p_double_full is None:
        p_double_full = 0.0

    if p_double_full < 0 or p_double_full > 1:
        raise ValueError(f"p_double_full must be in [0, 1], got {p_double_full}")

    optimizer = local_bundle["optimizer"]
    scheduler = local_bundle.get("scheduler", None)

    set_local_bundle_train_mode(local_bundle)

    trainable_params = get_trainable_parameters_from_bundle(local_bundle)
    ensure_trainable_parameters_fp32(trainable_params, verbose=verbose)

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found in local_bundle['unet'].")

    # The directional mode requires an activated directional loss on the loss fn.
    directional_available = (
        getattr(local_loss_fn, "directional_loss", None) is not None
    )
    enable_direction_effective = bool(enable_direction and directional_available)
    if enable_direction and not directional_available and verbose:
        print(
            "[WARN] enable_direction=True but local_loss_fn has no active "
            "directional_loss (use_directional_score / lambda_direction). "
            "Directional mode disabled."
        )

    mode_probs = normalize_loss_mode_probs(
        p_full=p_full,
        p_score=p_score,
        p_zone=p_zone,
        p_direction=p_direction,
        enable_full=enable_full,
        enable_score=enable_score,
        enable_zone=enable_zone,
        enable_direction=enable_direction_effective,
    )

    if verbose:
        print("\n========== TRAIN ONE EPOCH LOCAL ==========")
        print("Epoch:                 ", epoch)
        print("Device:                ", device)
        print("AMP enabled:           ", amp_enabled)
        print("AMP dtype:             ", amp_dtype)
        print("Grad accumulation:     ", grad_accum_steps)
        print("Grad clip:             ", grad_clip)
        print("p_neutral:             ", p_neutral)
        print("p_uncond (CFG drop):   ", p_uncond)
        print("p_direction:           ", p_direction, "| enabled:", enable_direction_effective)
        print("p_double_full:         ", p_double_full)
        print("Approx DP frequency:   ", mode_probs["full"] * p_double_full)
        print("Mode probs:            ", mode_probs)
        print("Fused loss enabled:    ", bool(use_fused_loss))
        if use_fused_loss:
            print("Fused loss epoch:      ", fused_loss_epoch)
            print("Fused every n steps:   ", fused_loss_every_n_steps)
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
    fused_loss_steps = 0

    fused_iter = None
    warned_missing_fused_global = False
    if use_fused_loss:
        if fused_train_loader is None:
            raise ValueError("use_fused_loss=True requires fused_train_loader.")
        if local_fused_loss_fn is None:
            raise ValueError("use_fused_loss=True requires local_fused_loss_fn.")
        fused_iter = iter(fused_train_loader)

    optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

    num_batches = len(train_loader)
    if max_batches is not None:
        num_batches = min(num_batches, int(max_batches))

    for batch_idx, batch in enumerate(train_loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        is_last_batch = batch_idx == (num_batches - 1)

        # --------------------------------------------------------
        # Move tensors to device.
        # Strings / metadata remain untouched.
        # --------------------------------------------------------
        batch = move_batch_to_device(batch, device)

        # --------------------------------------------------------
        # Build source / neutral / target prompts.
        # --------------------------------------------------------
        local_prompt_pack = build_local_prompt_pack_from_loader_prompts(
            loader_prompts=batch["prompt"],
            source_scores=batch["score_raw"],
            p_neutral=p_neutral,
        )

        # --------------------------------------------------------
        # True CFG dropout: replace the selected source prompt by "" with
        # probability p_uncond (per sample). Only affects the single-prompt
        # full/zone path; the explicit double-prompt path keeps source/neutral.
        # --------------------------------------------------------
        if p_uncond and float(p_uncond) > 0.0:
            local_prompt_pack["selected_source_prompts"] = [
                "" if random.random() < float(p_uncond) else sp
                for sp in local_prompt_pack["selected_source_prompts"]
            ]

        # --------------------------------------------------------
        # Build directional high/low prompts (Error 3) from the base prompts.
        # Only the score token differs; wording style is shared ("aging").
        # --------------------------------------------------------
        direction_high_prompts = None
        direction_low_prompts = None
        if enable_direction_effective:
            dl = local_loss_fn.directional_loss
            high_raw = int(round(float(dl.high_score) * 100))
            low_raw = int(round(float(dl.low_score) * 100))
            base_prompts = local_prompt_pack["source_prompts"]
            direction_high_prompts = [
                build_local_target_prompt_from_base(p, high_raw, "aging")
                for p in base_prompts
            ]
            direction_low_prompts = [
                build_local_target_prompt_from_base(p, low_raw, "aging")
                for p in base_prompts
            ]

        # --------------------------------------------------------
        # Sample loss mode.
        # --------------------------------------------------------
        loss_mode = sample_local_loss_mode(
            p_full=p_full,
            p_score=p_score,
            p_zone=p_zone,
            p_direction=p_direction,
            enable_full=enable_full,
            enable_score=enable_score,
            enable_zone=enable_zone,
            enable_direction=enable_direction_effective,
        )

        # --------------------------------------------------------
        # Decide explicit double-prompt.
        # Only for full diffusion reconstruction loss.
        # --------------------------------------------------------
        use_double_prompt = (
            loss_mode == "full"
            and p_double_full > 0
            and random.random() < float(p_double_full)
        )

        # --------------------------------------------------------
        # Forward / backward.
        # --------------------------------------------------------
        if use_double_prompt:
            double_prompt_steps += 1

            # ====================================================
            # Explicit double-prompt, memory-safe.
            #
            # We do two sequential loss calls and two sequential
            # backward calls.
            #
            # The division by 2 keeps the total magnitude of this
            # micro-step comparable to a single-prompt micro-step.
            # ====================================================

            # -----------------------------
            # 1) Source/conditioned prompt.
            # -----------------------------
            with autocast_ctx(
                device=device,
                enabled=amp_enabled,
                amp_dtype=amp_dtype,
                cache_enabled=True,
            ):
                loss_out_source = call_local_loss_fn(
                    local_loss_fn=local_loss_fn,
                    batch=batch,
                    loss_mode="full",
                    prompt_pack=local_prompt_pack,
                    device=device,
                    source_prompts_override=local_prompt_pack["source_prompts"],
                    zone_prompt_key=zone_prompt_key,
                    pass_target_scores_for_all_modes=True,
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
                    "[WARN] Non-finite LOCAL source loss. "
                    "Skipping accumulated gradients. "
                    f"{training_batch_diagnostic_context(branch='local', batch_idx=batch_idx, global_step=global_step, loss_mode='full/source', batch=batch, prompt_pack=local_prompt_pack, use_double_prompt=use_double_prompt)} | "
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
            # 2) Neutral prompt.
            # -----------------------------
            with autocast_ctx(
                device=device,
                enabled=amp_enabled,
                amp_dtype=amp_dtype,
                cache_enabled=True,
            ):
                loss_out_neutral = call_local_loss_fn(
                    local_loss_fn=local_loss_fn,
                    batch=batch,
                    loss_mode="full",
                    prompt_pack=local_prompt_pack,
                    device=device,
                    source_prompts_override=local_prompt_pack["neutral_prompts"],
                    zone_prompt_key=zone_prompt_key,
                    pass_target_scores_for_all_modes=True,
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
                    "[WARN] Non-finite LOCAL neutral loss. "
                    "Skipping accumulated gradients. "
                    f"{training_batch_diagnostic_context(branch='local', batch_idx=batch_idx, global_step=global_step, loss_mode='full/neutral', batch=batch, prompt_pack=local_prompt_pack, use_double_prompt=use_double_prompt)} | "
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

            # Use source loss_out as base for metrics.
            loss_out = dict(loss_out_source)
            loss_out["loss"] = raw_loss
            loss_out["double_prompt/source_loss"] = raw_loss_source.detach()
            loss_out["double_prompt/neutral_loss"] = raw_loss_neutral.detach()
            loss_out["double_prompt/used"] = torch.tensor(1.0)

        else:
            # ====================================================
            # Standard single-prompt path.
            #
            # This uses selected_source_prompts:
            #   - source prompt with prob 1 - p_neutral
            #   - neutral prompt with prob p_neutral
            # ====================================================
            with autocast_ctx(
                device=device,
                enabled=amp_enabled,
                amp_dtype=amp_dtype,
                cache_enabled=True,
            ):
                loss_out = call_local_loss_fn(
                    local_loss_fn=local_loss_fn,
                    batch=batch,
                    loss_mode=loss_mode,
                    prompt_pack=local_prompt_pack,
                    device=device,
                    source_prompts_override=None,
                    zone_prompt_key=zone_prompt_key,
                    pass_target_scores_for_all_modes=True,
                    direction_high_prompts=direction_high_prompts,
                    direction_low_prompts=direction_low_prompts,
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
                    "[WARN] Non-finite LOCAL loss. "
                    "Skipping accumulated gradients. "
                    f"{training_batch_diagnostic_context(branch='local', batch_idx=batch_idx, global_step=global_step, loss_mode=loss_mode, batch=batch, prompt_pack=local_prompt_pack, use_double_prompt=use_double_prompt)} | "
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

        # --------------------------------------------------------
        # Optional aligned fused loss.
        #
        # The base local loss has already been backpropagated above. This
        # keeps the base crop graph and fused graph from being alive together.
        # --------------------------------------------------------
        fused_out = None
        run_fused = should_run_fused_loss(
            use_fused_loss=use_fused_loss,
            epoch=epoch,
            fused_loss_epoch=fused_loss_epoch,
            fused_loss_every_n_steps=fused_loss_every_n_steps,
            global_step=global_step,
        )

        if run_fused:
            try:
                fused_batch = next(fused_iter)
            except StopIteration:
                fused_iter = iter(fused_train_loader)
                fused_batch = next(fused_iter)

            fused_batch = move_batch_to_device(fused_batch, device)

            with torch.no_grad():
                if fused_global_forward_fn is not None:
                    x_global = fused_global_forward_fn(fused_batch=fused_batch, device=device)
                else:
                    if verbose and not warned_missing_fused_global:
                        print(
                            "[WARN] use_fused_loss=True but fused_global_forward_fn=None. "
                            "Using x_global = x_orig fallback. This is okay for smoke tests, "
                            "but real training should pass a frozen global forward function."
                        )
                        warned_missing_fused_global = True
                    x_global = fused_batch["full_pixel_values"]
                x_global = x_global.detach()

            with autocast_ctx(
                device=device,
                enabled=amp_enabled,
                amp_dtype=amp_dtype,
                cache_enabled=True,
            ):
                aged_crops = generate_local_aged_crops_for_fused_loss(
                    local_loss_fn=local_loss_fn,
                    fused_batch=fused_batch,
                    device=device,
                )
                fused_out = local_fused_loss_fn(
                    x_orig=fused_batch["full_pixel_values"],
                    x_global=x_global,
                    aged_crops=aged_crops,
                    boxes=fused_batch["boxes"],
                    masks=fused_batch["masks"],
                    target_scores=fused_batch["target_scores"],
                    valid_mask=fused_batch["valid_mask"],
                )
                raw_fused_loss = fused_out["loss"]
                fused_loss = raw_fused_loss / float(grad_accum_steps)

            if skip_nonfinite_loss and not torch.isfinite(raw_fused_loss.detach()).all():
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

                print(
                    "[WARN] Non-finite LOCAL fused loss. "
                    "Skipping accumulated gradients. "
                    f"{training_batch_diagnostic_context(branch='local_fused', batch_idx=batch_idx, global_step=global_step, loss_mode='fused', batch=fused_batch, prompt_pack=None, use_double_prompt=False)} | "
                    f"Components: {nonfinite_loss_details(fused_out)}"
                )

                continue

            backward_with_optional_scaler(
                loss=fused_loss,
                optimizer=optimizer,
                scaler=scaler,
                retain_graph=False,
            )

            fused_loss_steps += 1
            loss_out["loss"] = loss_out["loss"].detach() + raw_fused_loss.detach()
            loss_out["loss_fuse_score"] = fused_out["loss_fuse_score"]
            loss_out["loss_fuse_seam"] = fused_out["loss_fuse_seam"]
            if "fused_score_pred_mean" in fused_out:
                loss_out["fused_score_pred_mean"] = fused_out["fused_score_pred_mean"]
            if "fused_score_target_mean" in fused_out:
                loss_out["fused_score_target_mean"] = fused_out["fused_score_target_mean"]
            if "fused_score_mae" in fused_out:
                loss_out["fused_score_mae"] = fused_out["fused_score_mae"]
            loss_out["fused/used"] = torch.tensor(1.0)
        else:
            loss_out["fused/used"] = torch.tensor(0.0)

        # --------------------------------------------------------
        # Counters.
        # Important:
        #   A double-prompt step still counts as one micro-step
        #   for gradient accumulation.
        # --------------------------------------------------------
        n_micro_steps += 1
        global_step += 1

        # --------------------------------------------------------
        # Metrics.
        # --------------------------------------------------------
        batch_metrics = compute_local_training_metrics(
            batch=batch,
            prompt_pack=local_prompt_pack,
            loss_out=loss_out,
            loss_mode=loss_mode,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        batch_metrics["double_prompt/used"] = 1.0 if use_double_prompt else 0.0
        batch_metrics["fused/used"] = 1.0 if run_fused else 0.0

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
                    "[WARN] Non-finite LOCAL gradients. "
                    "Skipping optimizer step. "
                    f"{training_batch_diagnostic_context(branch='local', batch_idx=batch_idx, global_step=global_step, loss_mode=loss_mode, batch=batch, prompt_pack=local_prompt_pack, use_double_prompt=use_double_prompt)} | "
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
            prefix = f"[LOCAL train e{epoch} b{batch_idx}/{num_batches}]"
            print_local_metrics(
                batch_metrics,
                prefix=prefix,
                step=global_step,
            )

    # ============================================================
    # Epoch summary.
    # ============================================================

    epoch_metrics = tracker.compute()

    if verbose:
        print_local_metrics(
            epoch_metrics,
            prefix=f"[LOCAL epoch {epoch} summary]",
            step=global_step,
        )

        print("\n[LOCAL epoch counters]")
        print("Micro steps epoch:        ", n_micro_steps)
        print("Optimizer steps epoch:    ", n_optimizer_steps_epoch)
        print("Double-prompt microsteps: ", double_prompt_steps)
        print("Double-prompt fraction:   ", double_prompt_steps / max(1, n_micro_steps))
        print("Fused-loss microsteps:    ", fused_loss_steps)
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
        "double_prompt_steps": int(double_prompt_steps),
        "double_prompt_fraction": float(double_prompt_steps / max(1, n_micro_steps)),
        "fused_loss_steps": int(fused_loss_steps),
        "skipped_steps": int(skipped_steps),
    }
