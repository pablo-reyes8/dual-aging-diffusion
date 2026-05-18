from __future__ import annotations

import gc
from typing import Any, Dict, Optional, Sequence

import torch

from src.training.training_display_helpers import print_section

# ============================================================
# Memory utilities
# ============================================================

def cuda_memory_report(label: str = "", device: Optional[torch.device] = None) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}

    if device is None:
        device = torch.device("cuda")

    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3

    report = {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
    }

    if label:
        print(
            f"[CUDA memory | {label}] "
            f"allocated={allocated:.3f} GB | "
            f"reserved={reserved:.3f} GB | "
            f"max_allocated={max_allocated:.3f} GB"
        )

    return report


def hard_cuda_cleanup(label: str = "", reset_peak: bool = True) -> None:
    """
    Aggressive cleanup between branches.

    Important:
        This does not delete model objects.
        It releases unreferenced CUDA tensors and clears allocator cache.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if reset_peak:
            torch.cuda.reset_peak_memory_stats()

        if label:
            cuda_memory_report(label=label)


def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    """
    Moves optimizer state tensors to device.

    Critical:
        model.to("cpu") does NOT move optimizer state.
        AdamW moments can keep large tensors on GPU unless moved explicitly.
    """
    if optimizer is None:
        return

    if hasattr(optimizer, "state"):
        for state in optimizer.state.values():
            if isinstance(state, dict):
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device, non_blocking=True)

    # Hybrid optimizer support.
    for attr in ["adamw", "muon"]:
        if hasattr(optimizer, attr):
            subopt = getattr(optimizer, attr)
            move_optimizer_state_to_device(subopt, device)


def move_any_to_device(obj: Any, device: torch.device) -> Any:
    """
    Generic object mover.

    Supports:
        - nn.Module
        - objects with .move_to(device)
        - dict/list/tuple containers
    """
    if obj is None:
        return None

    if hasattr(obj, "move_to") and callable(getattr(obj, "move_to")):
        obj.move_to(device)
        return obj

    if isinstance(obj, torch.nn.Module):
        obj.to(device)
        return obj

    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = move_any_to_device(v, device)
        return obj

    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = move_any_to_device(obj[i], device)
        return obj

    if isinstance(obj, tuple):
        return tuple(move_any_to_device(v, device) for v in obj)

    return obj


def move_bundle_modules_to_device(bundle: Dict[str, Any], device: torch.device) -> None:
    """
    Moves main bundle modules to target device.
    """
    for key in ["unet", "vae", "text_encoder", "tokenizer"]:
        if key in bundle:
            if key == "tokenizer":
                continue
            move_any_to_device(bundle[key], device)

    if "optimizer" in bundle:
        move_optimizer_state_to_device(bundle["optimizer"], device)


def enable_unet_gradient_checkpointing(bundle: Dict[str, Any], branch_name: str = "") -> None:
    """
    Enables gradient checkpointing on Diffusers UNet if available.
    """
    unet = bundle.get("unet", None)

    if unet is None:
        return

    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
        print(f"[{branch_name}] gradient checkpointing enabled on UNet.")
    else:
        print(f"[{branch_name}] UNet has no enable_gradient_checkpointing().")


def prepare_branch_for_training(
    active_bundle: Dict[str, Any],
    active_aux_objects: Optional[Sequence[Any]],
    inactive_bundle: Optional[Dict[str, Any]],
    inactive_aux_objects: Optional[Sequence[Any]],
    device: torch.device,
    branch_name: str,
    enable_gradient_checkpointing_flag: bool = True,
    print_memory: bool = True,
) -> None:
    """
    Moves inactive branch to CPU, active branch to GPU, and cleans memory.
    """
    print_section(f"Preparing {branch_name.upper()} branch")

    cpu = torch.device("cpu")

    if inactive_bundle is not None:
        print(f"Moving inactive branch to CPU.")
        move_bundle_modules_to_device(inactive_bundle, cpu)

    if inactive_aux_objects is not None:
        for obj in inactive_aux_objects:
            move_any_to_device(obj, cpu)

    hard_cuda_cleanup(label=f"after offloading inactive before {branch_name}", reset_peak=True)

    print(f"Moving active {branch_name} branch to {device}.")
    move_bundle_modules_to_device(active_bundle, device)

    if active_aux_objects is not None:
        for obj in active_aux_objects:
            move_any_to_device(obj, device)

    if enable_gradient_checkpointing_flag:
        enable_unet_gradient_checkpointing(active_bundle, branch_name=branch_name)

    hard_cuda_cleanup(label=f"after loading active {branch_name}", reset_peak=True)

    if print_memory:
        cuda_memory_report(label=f"{branch_name} ready", device=device)


def offload_branch_after_training(
    bundle: Dict[str, Any],
    aux_objects: Optional[Sequence[Any]],
    branch_name: str,
    print_memory: bool = True,
) -> None:
    """
    Moves trained branch to CPU and clears CUDA memory.
    """
    print_section(f"Offloading {branch_name.upper()} branch")

    cpu = torch.device("cpu")

    move_bundle_modules_to_device(bundle, cpu)

    if aux_objects is not None:
        for obj in aux_objects:
            move_any_to_device(obj, cpu)

    hard_cuda_cleanup(label=f"after offloading {branch_name}", reset_peak=True)

    if print_memory and torch.cuda.is_available():
        cuda_memory_report(label=f"{branch_name} offloaded")


# ============================================================
# Lightweight module-only movement for sampling
# ============================================================

def move_bundle_modules_only_to_device(
    bundle: Dict[str, Any],
    device: torch.device,
    eval_mode: bool = True,
) -> None:
    """
    Moves model modules to device WITHOUT moving optimizer state.

    Important:
        During training we move optimizer state because we need AdamW states.
        During sampling we should NOT move optimizer state to GPU.
        This keeps reconstruction monitoring much cheaper in VRAM.
    """
    if bundle is None:
        return

    for key in ["unet", "vae", "text_encoder"]:
        if key in bundle and bundle[key] is not None:
            module = bundle[key]
            if hasattr(module, "to"):
                module.to(device)
            if eval_mode and hasattr(module, "eval"):
                module.eval()

    # If bundle has an already-built diffusers pipeline.
    for key in ["pipe", "pipeline", "img2img_pipeline"]:
        if key in bundle and bundle[key] is not None:
            pipe = bundle[key]
            if hasattr(pipe, "to"):
                pipe.to(device)
            for attr in ["unet", "vae", "text_encoder", "text_encoder_2"]:
                module = getattr(pipe, attr, None)
                if module is not None and hasattr(module, "eval"):
                    module.eval()


def offload_bundle_modules_only(
    bundle: Dict[str, Any],
    label: str = "",
    reset_peak: bool = True,
) -> None:
    """
    Offloads modules/pipeline only. Does not touch optimizer state.
    """
    if bundle is None:
        return

    cpu = torch.device("cpu")
    move_bundle_modules_only_to_device(bundle, cpu, eval_mode=True)

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()

    if label:
        print(f"└─ [OFFLOAD] {label} modules moved to CPU")
