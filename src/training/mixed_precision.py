# ============================================================
# DEVICE + MIXED PRECISION UTILS
#
# Designed for diffusion training:
#   - Prefer bf16 when available.
#   - Fallback to fp16 on CUDA when bf16 is not supported.
#   - Avoid fp32 training unless explicitly requested.
#   - Use GradScaler only for fp16.
#   - Keep prompts / metadata untouched when moving batches.
# ============================================================

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Optional, Union, Tuple, List

import torch


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "none": torch.float32,
}


# ============================================================
# Device resolution
# ============================================================

def resolve_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """
    Resolves device.

    Priority when device='auto':
        CUDA -> MPS -> CPU

    For this project, CUDA is strongly preferred.
    """
    if isinstance(device, torch.device):
        requested = device

    elif device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    else:
        requested = torch.device(device)

    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but CUDA is not available.")

    if requested.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but MPS is not available.")

    return requested


def normalize_device_type(device: Union[str, torch.device] = "cuda") -> str:
    """
    Returns:
        'cuda', 'cpu', 'mps', etc.
    """
    return torch.device(device).type


# ============================================================
# Precision resolution
# ============================================================

def resolve_amp_dtype(
    amp_dtype: str = "bf16",
    device: Union[str, torch.device] = "cuda") -> torch.dtype:

    """
    Resolves string dtype into torch dtype.
    """
    amp_dtype = str(amp_dtype).lower().strip()

    if amp_dtype not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported amp_dtype={amp_dtype}. "
            f"Expected one of {sorted(DTYPE_MAP.keys())}."
        )

    return DTYPE_MAP[amp_dtype]


def cuda_supports_bf16() -> bool:
    """
    Checks whether current CUDA device supports bf16.

    Ampere and newer usually support bf16.
    """
    if not torch.cuda.is_available():
        return False

    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            pass

    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def get_effective_amp_dtype(
    amp_dtype: str = "bf16",
    device: Union[str, torch.device] = "cuda",
    fallback_bf16_to_fp16: bool = True) -> Optional[torch.dtype]:

    """
    Returns effective autocast dtype.

    Returns None when autocast should be disabled.

    Rules:
        - fp32/none -> None
        - CUDA bf16 -> bf16 if supported, else fp16 if fallback enabled
        - CUDA fp16 -> fp16
        - CPU bf16 -> bf16
        - MPS -> None for now, to avoid unstable mixed precision behavior
    """
    device_type = normalize_device_type(device)
    requested_dtype = resolve_amp_dtype(amp_dtype, device=device)

    if requested_dtype == torch.float32:
        return None

    if device_type == "cuda":
        if not torch.cuda.is_available():
            return None

        if requested_dtype == torch.bfloat16:
            if cuda_supports_bf16():
                return torch.bfloat16
            return torch.float16 if fallback_bf16_to_fp16 else None

        if requested_dtype == torch.float16:
            return torch.float16

        return None

    if device_type == "cpu":
        if requested_dtype == torch.bfloat16:
            return torch.bfloat16
        return None

    # Keep MPS conservative.
    return None


def should_use_grad_scaler(
    device: Union[str, torch.device] = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    fallback_bf16_to_fp16: bool = True) -> bool:

    """
    GradScaler is useful for fp16 CUDA training.
    It is usually unnecessary for bf16.
    """
    if not amp_enabled:
        return False

    if normalize_device_type(device) != "cuda":
        return False

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=device,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    return effective_dtype == torch.float16


def make_grad_scaler(
    device: Union[str, torch.device] = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    fallback_bf16_to_fp16: bool = True):

    """
    Builds GradScaler only when needed.

    Returns:
        scaler or None.
    """
    enabled = should_use_grad_scaler(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    if not enabled:
        return None

    device_type = normalize_device_type(device)

    # New PyTorch API.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            sig = inspect.signature(torch.amp.GradScaler)

            if "device" in sig.parameters:
                return torch.amp.GradScaler(device=device_type, enabled=True)

            return torch.amp.GradScaler(device_type, enabled=True)

        except Exception:
            pass

    # Older CUDA AMP API.
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=True)

    return None


# ============================================================
# Autocast context
# ============================================================

@contextmanager
def autocast_ctx(
    device: Union[str, torch.device] = "cuda",
    enabled: bool = True,
    amp_dtype: str = "bf16",
    cache_enabled: bool = True,
    fallback_bf16_to_fp16: bool = True):

    """
    Safe autocast context.

    Use:
        with autocast_ctx(device=device, enabled=amp_enabled, amp_dtype="bf16"):
            loss_out = loss_fn(...)

    Notes:
        - If bf16 is requested but unavailable on CUDA, fp16 is used if fallback is True.
        - If effective dtype is None, this becomes a nullcontext.
        - Do not wrap yield in try/except; it can break contextlib error propagation.
    """
    if not enabled:
        with nullcontext():
            yield
        return

    device_type = normalize_device_type(device)

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=device,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    if effective_dtype is None:
        with nullcontext():
            yield
        return

    if not hasattr(torch, "amp") or not hasattr(torch.amp, "autocast"):
        with nullcontext():
            yield
        return

    if device_type in {"cuda", "cpu"}:
        ctx = torch.amp.autocast(
            device_type=device_type,
            dtype=effective_dtype,
            cache_enabled=cache_enabled,
        )

        with ctx:
            yield

        return

    with nullcontext():
        yield


# ============================================================
# Setup object
# ============================================================

def setup_device_and_precision(
    device: Union[str, torch.device] = "auto",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    cache_enabled: bool = True,
    fallback_bf16_to_fp16: bool = True,
    forbid_fp32_training: bool = True) -> Dict[str, Any]:

    """
    Central setup for device and precision.

    Args:
        device:
            'auto', 'cuda', 'cpu', torch.device(...).

        amp_enabled:
            Whether to use autocast if possible.

        amp_dtype:
            Preferred AMP dtype: 'bf16' or 'fp16'.
            Recommended: 'bf16'.

        cache_enabled:
            Passed to autocast.

        fallback_bf16_to_fp16:
            If bf16 is requested but unavailable on CUDA, fallback to fp16.

        forbid_fp32_training:
            If True, raises when training would run in pure fp32 on CUDA.
            This protects us from accidentally training diffusion in fp32.

    Returns:
        Dictionary with device, amp dtype, scaler, etc.
    """
    resolved_device = resolve_device(device)
    device_type = normalize_device_type(resolved_device)

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=resolved_device,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16)


    final_amp_enabled = bool(amp_enabled and effective_dtype is not None)

    if (
        forbid_fp32_training
        and device_type == "cuda"
        and amp_enabled
        and effective_dtype is None):
        raise RuntimeError(
            "AMP was requested on CUDA, but no effective mixed precision dtype was resolved. "
            "This would likely train diffusion in fp32. Use amp_dtype='bf16' or 'fp16', "
            "or set forbid_fp32_training=False intentionally."
        )

    scaler = make_grad_scaler(
        device=resolved_device,
        amp_enabled=final_amp_enabled,
        amp_dtype=amp_dtype,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    precision = {
        "device": resolved_device,
        "device_type": device_type,
        "amp_enabled": final_amp_enabled,
        "amp_dtype_requested": amp_dtype,
        "amp_dtype_effective": effective_dtype,
        "use_grad_scaler": scaler is not None,
        "scaler": scaler,
        "cache_enabled": cache_enabled,
        "fallback_bf16_to_fp16": fallback_bf16_to_fp16,
    }

    print_precision_report(precision)

    return precision


def print_precision_report(precision: Dict[str, Any]) -> None:
    """
    Prints precision setup summary.
    """
    device = precision["device"]
    effective_dtype = precision["amp_dtype_effective"]

    if effective_dtype is None:
        dtype_name = "fp32 / autocast disabled"
    elif effective_dtype == torch.bfloat16:
        dtype_name = "bf16"
    elif effective_dtype == torch.float16:
        dtype_name = "fp16"
    else:
        dtype_name = str(effective_dtype)

    print("\n========== DEVICE / PRECISION ==========")
    print("Device:              ", device)
    print("Device type:         ", precision["device_type"])
    print("AMP enabled:         ", precision["amp_enabled"])
    print("Requested AMP dtype: ", precision["amp_dtype_requested"])
    print("Effective AMP dtype: ", dtype_name)
    print("GradScaler:          ", precision["use_grad_scaler"])
    print("Autocast cache:      ", precision["cache_enabled"])

    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        name = torch.cuda.get_device_name(device)
        cap = torch.cuda.get_device_capability(device)
        print("CUDA device name:    ", name)
        print("CUDA capability:     ", cap)
        print("CUDA bf16 support:   ", cuda_supports_bf16())


# ============================================================
# Batch movement
# ============================================================

def move_batch_to_device(
    batch: Union[Dict[str, Any], Tuple[Any, ...], List[Any], torch.Tensor],
    device: torch.device,
    non_blocking: bool = True) -> Union[Dict[str, Any], Tuple[Any, ...], List[Any], torch.Tensor]:
    """
    Recursively moves tensors to device.

    Important for this project:
        - prompt strings remain untouched.
        - region keys, image ids, json paths remain untouched.
        - tensors like pixel_values, score, age move to GPU.
    """
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=non_blocking)

    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device, non_blocking=non_blocking)
            for key, value in batch.items()
        }

    if isinstance(batch, tuple):
        return tuple(
            move_batch_to_device(value, device, non_blocking=non_blocking)
            for value in batch
        )

    if isinstance(batch, list):
        return [
            move_batch_to_device(value, device, non_blocking=non_blocking)
            for value in batch
        ]

    return batch


# ============================================================
# Backward helper
# ============================================================

def backward_with_optional_scaler(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler=None,
    retain_graph: bool = False) -> None:

    """
    Backward helper.

    This does NOT call optimizer.step().
    Use this during gradient accumulation.

    Example:
        loss = loss / grad_accum_steps
        backward_with_optional_scaler(loss, optimizer, scaler=scaler)
    """
    if scaler is not None:
        scaler.scale(loss).backward(retain_graph=retain_graph)
    else:
        loss.backward(retain_graph=retain_graph)


def nonfinite_gradient_details(parameters, max_items: int = 8) -> str:
    """
    Returns a compact diagnostic for parameters with NaN/Inf gradients.
    """
    if parameters is None:
        return "parameters=None"

    details = []
    for idx, p in enumerate(parameters):
        if p.grad is None:
            continue

        grad = p.grad.detach()
        if grad.numel() == 0 or torch.isfinite(grad).all():
            continue

        nonfinite = (~torch.isfinite(grad)).sum().item()
        details.append(
            f"param_idx={idx} shape={tuple(p.shape)} grad_nonfinite={int(nonfinite)}"
        )

        if len(details) >= int(max_items):
            break

    return "; ".join(details) if details else "no non-finite gradients found"


def nonfinite_parameter_details(parameters, max_items: int = 8) -> str:
    """
    Returns a compact diagnostic for parameters containing NaN/Inf values.
    """
    if parameters is None:
        return "parameters=None"

    details = []
    for idx, p in enumerate(parameters):
        data = p.detach()
        if data.numel() == 0 or torch.isfinite(data).all():
            continue

        nonfinite = (~torch.isfinite(data)).sum().item()
        details.append(
            f"param_idx={idx} shape={tuple(p.shape)} dtype={p.dtype} param_nonfinite={int(nonfinite)}"
        )

        if len(details) >= int(max_items):
            break

    return "; ".join(details) if details else "no non-finite parameters found"


def ensure_trainable_parameters_fp32(parameters, verbose: bool = False) -> int:
    """
    Casts trainable optimizer parameters to fp32 in-place.

    This is important for LoRA/DoRA adapters trained with AdamW under AMP:
    compute can run in bf16/fp16, but optimizer-owned trainable weights should
    be stored in fp32.
    """
    if parameters is None:
        return 0

    converted = 0
    for p in parameters:
        if p.dtype == torch.float32:
            continue
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()
        converted += 1

    if verbose and converted > 0:
        print(f"[AMP safety] Converted {converted} trainable tensors to fp32.")

    return converted


def optimizer_step_with_optional_scaler(
    optimizer: torch.optim.Optimizer,
    scaler=None,
    grad_clip: Optional[float] = None,
    parameters=None,
    skip_nonfinite_grad: bool = True) -> bool:

    """
    Optimizer step helper with optional GradScaler and optional gradient clipping.

    For fp16:
        scaler.unscale_(optimizer) before clipping.

    For bf16:
        normal clipping and optimizer.step().
    Returns:
        True if optimizer.step() was applied, False if non-finite gradients
        were detected and the step was skipped.
    """
    params = None
    if parameters is not None:
        params = list(parameters)

    def _has_nonfinite_grad() -> bool:
        if params is None:
            return False
        for p in params:
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad.detach()).all():
                return True
        return False

    def _has_nonfinite_param() -> bool:
        if params is None:
            return False
        for p in params:
            if not torch.isfinite(p.detach()).all():
                return True
        return False

    def _snapshot_params():
        if params is None:
            return None
        return [p.detach().clone() for p in params]

    def _restore_params_and_clear_state(snapshot) -> None:
        if params is None or snapshot is None:
            return
        for p, old_value in zip(params, snapshot):
            p.data.copy_(old_value.to(device=p.device, dtype=p.dtype))
            if p in optimizer.state:
                optimizer.state[p].clear()

    if scaler is not None:
        if params is not None:
            scaler.unscale_(optimizer)

        if skip_nonfinite_grad and _has_nonfinite_grad():
            scaler.update()
            return False

        if grad_clip is not None and params is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            if skip_nonfinite_grad and not torch.isfinite(total_norm.detach()):
                scaler.update()
                return False

        snapshot = _snapshot_params() if skip_nonfinite_grad else None
        scaler.step(optimizer)
        scaler.update()
        if skip_nonfinite_grad and _has_nonfinite_param():
            _restore_params_and_clear_state(snapshot)
            return False
        return True

    else:
        if skip_nonfinite_grad and _has_nonfinite_grad():
            return False

        if grad_clip is not None and params is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            if skip_nonfinite_grad and not torch.isfinite(total_norm.detach()):
                return False

        snapshot = _snapshot_params() if skip_nonfinite_grad else None
        optimizer.step()
        if skip_nonfinite_grad and _has_nonfinite_param():
            _restore_params_and_clear_state(snapshot)
            return False
        return True
