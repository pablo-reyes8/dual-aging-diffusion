# ============================================================
# CHECKPOINT UTILS FOR GLOBAL / LOCAL DIFFUSION BUNDLES
#
# Project design:
#   - Do NOT save full UNet/base model weights.
#   - Save only LoRA/DoRA adapter weights.
#   - Save optimizer/scheduler/scaler only for resume checkpoints.
#   - Save minimal metadata for inference checkpoints.
#
# Expected bundle structure:
#   bundle["unet"]
#   bundle["optimizer"]
#   bundle["scheduler"]                 optional
#   bundle["adapter_type"]              "lora" or "dora"
#   bundle["adapter_config"]            rank/alpha/dropout/target_suffixes
#   bundle["model_id"]
#   bundle["vae_id"]                    optional
#   bundle["name"]
#
# Two checkpoint types:
#   1. training checkpoint:
#       for resuming training.
#
#   2. inference checkpoint:
#       for loading adapters later for inference only.
# ============================================================

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch


# ============================================================
# Adapter state dict extraction
# ============================================================

def is_adapter_parameter_name(name: str) -> bool:
    """
    Detects trainable adapter parameters.

    LoRA:
        lora_down.*
        lora_up.*

    DoRA:
        lora_down.*
        lora_up.*
        magnitude
    """
    return (
        "lora_down" in name
        or "lora_up" in name
        or "magnitude" in name)


def get_adapter_state_dict(unet: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA/DoRA adapter weights from a manually injected UNet.

    Important:
        Do NOT use unet.state_dict() directly for checkpoints,
        because that would include the full frozen UNet weights.
    """
    adapter_state = {}

    for name, tensor in unet.state_dict().items():
        if is_adapter_parameter_name(name):
            adapter_state[name] = tensor.detach().cpu()

    if len(adapter_state) == 0:
        raise ValueError(
            "No adapter parameters found in UNet state_dict. "
            "Make sure LoRA/DoRA has been injected before saving."
        )

    return adapter_state


def load_adapter_state_dict(
    unet: torch.nn.Module,
    adapter_state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Loads adapter-only weights into a UNet that already has LoRA/DoRA injected.

    Important:
        You must first:
            1. load base UNet
            2. inject LoRA/DoRA with the same config
            3. call this function
    """
    current_state = unet.state_dict()

    missing_in_model = []
    loaded_keys = []

    for key, value in adapter_state_dict.items():
        if key not in current_state:
            missing_in_model.append(key)
            continue

        current_state[key].copy_(value.to(
            device=current_state[key].device,
            dtype=current_state[key].dtype,
        ))
        loaded_keys.append(key)

    missing_expected = [
        key for key in current_state.keys()
        if is_adapter_parameter_name(key) and key not in adapter_state_dict
    ]

    if strict and (missing_in_model or missing_expected):
        raise RuntimeError(
            "Adapter state_dict mismatch.\n"
            f"Keys in checkpoint but not model: {missing_in_model[:20]}\n"
            f"Adapter keys in model but not checkpoint: {missing_expected[:20]}"
        )

    return {
        "loaded_keys": loaded_keys,
        "n_loaded": len(loaded_keys),
        "missing_in_model": missing_in_model,
        "missing_expected": missing_expected,
    }


# ============================================================
# Generic safe serialization helpers
# ============================================================

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_torch_save(obj: Any, path: Path) -> None:
    """
    Saves to temporary file then renames.

    Helps avoid corrupted checkpoints if the notebook crashes while saving.
    """
    path = Path(path)
    ensure_dir(path.parent)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    os.replace(tmp_path, path)


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def torch_load_cpu(path: Path) -> Any:
    """
    Loads checkpoint safely on CPU first.
    """
    return torch.load(Path(path), map_location="cpu")


def get_optimizer_state_dict(optimizer) -> Optional[Dict[str, Any]]:
    if optimizer is None:
        return None
    return optimizer.state_dict()


def get_scheduler_state_dict(scheduler) -> Optional[Dict[str, Any]]:
    if scheduler is None:
        return None

    if hasattr(scheduler, "state_dict"):
        return scheduler.state_dict()

    return None


def get_scaler_state_dict(scaler) -> Optional[Dict[str, Any]]:
    if scaler is None:
        return None

    if hasattr(scaler, "state_dict"):
        return scaler.state_dict()

    return None


def load_optimizer_state_dict(optimizer, state_dict: Optional[Dict[str, Any]]) -> None:
    if optimizer is None or state_dict is None:
        return
    optimizer.load_state_dict(state_dict)


def load_scheduler_state_dict(scheduler, state_dict: Optional[Dict[str, Any]]) -> None:
    if scheduler is None or state_dict is None:
        return

    if hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state_dict)


def load_scaler_state_dict(scaler, state_dict: Optional[Dict[str, Any]]) -> None:
    if scaler is None or state_dict is None:
        return

    if hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(state_dict)


# ============================================================
# Bundle metadata
# ============================================================

def get_bundle_checkpoint_metadata(
    bundle: Dict[str, Any],
    branch_name: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Metadata needed to reconstruct adapter architecture.

    This does not include full base model weights.
    """
    metadata = {
        "branch_name": branch_name,
        "bundle_name": bundle.get("name", None),

        # Base model reconstruction.
        "model_id": bundle.get("model_id", None),
        "vae_id": bundle.get("vae_id", None),

        # Adapter reconstruction.
        "adapter_type": bundle.get("adapter_type", None),
        "adapter_config": bundle.get("adapter_config", None),

        # Useful diagnostics.
        "param_stats": bundle.get("param_stats", None),
        "scheduler_config": bundle.get("scheduler_config", None),
    }

    if extra_metadata is not None:
        metadata["extra_metadata"] = extra_metadata

    return metadata


# ============================================================
# Inference checkpoint
# ============================================================

def save_inference_checkpoint_for_bundle(
    bundle: Dict[str, Any],
    output_dir: Path,
    branch_name: str,
    filename: str = "adapter_inference.pt",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Saves lightweight inference checkpoint.

    Contains:
        - adapter weights only
        - metadata needed to reconstruct base + adapter

    Does NOT contain:
        - optimizer state
        - scheduler state
        - scaler state
        - full UNet/base model weights
    """
    output_dir = ensure_dir(Path(output_dir))

    if "unet" not in bundle:
        raise KeyError("bundle must contain key 'unet'.")

    adapter_state = get_adapter_state_dict(bundle["unet"])

    metadata = get_bundle_checkpoint_metadata(
        bundle=bundle,
        branch_name=branch_name,
        extra_metadata=extra_metadata,
    )

    checkpoint = {
        "checkpoint_type": "inference",
        "branch_name": branch_name,
        "adapter_state_dict": adapter_state,
        "metadata": metadata,
    }

    ckpt_path = output_dir / filename
    atomic_torch_save(checkpoint, ckpt_path)

    # Also save readable metadata.
    save_json(metadata, output_dir / "metadata.json")

    print("\n[Inference checkpoint saved]")
    print("Branch:      ", branch_name)
    print("Path:        ", ckpt_path)
    print("Adapter keys:", len(adapter_state))
    print("Size MB:     ", ckpt_path.stat().st_size / 1024**2)

    return ckpt_path


def load_inference_checkpoint(path: Path) -> Dict[str, Any]:
    """
    Loads lightweight inference checkpoint on CPU.
    """
    ckpt = torch_load_cpu(path)

    if ckpt.get("checkpoint_type", None) != "inference":
        raise ValueError(
            f"Expected checkpoint_type='inference', got {ckpt.get('checkpoint_type')}"
        )

    return ckpt


# ============================================================
# Training/resume checkpoint
# ============================================================

def save_training_checkpoint_for_bundle(
    bundle: Dict[str, Any],
    output_dir: Path,
    branch_name: str,
    epoch: int,
    global_step: int,
    optimizer_step: int,
    loss_value: Optional[float] = None,
    best_metric: Optional[float] = None,
    scaler=None,
    filename: str = "training_resume.pt",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Saves resume checkpoint.

    Contains:
        - adapter weights only
        - optimizer state
        - scheduler state if bundle["scheduler"] exists
        - scaler state if provided
        - epoch / global_step / optimizer_step
        - metadata

    Does NOT contain:
        - full frozen UNet/base model weights
    """
    output_dir = ensure_dir(Path(output_dir))

    if "unet" not in bundle:
        raise KeyError("bundle must contain key 'unet'.")

    if "optimizer" not in bundle:
        raise KeyError(
            "bundle must contain key 'optimizer' to save training checkpoint."
        )

    adapter_state = get_adapter_state_dict(bundle["unet"])

    optimizer = bundle.get("optimizer", None)
    scheduler = bundle.get("scheduler", None)

    metadata = get_bundle_checkpoint_metadata(
        bundle=bundle,
        branch_name=branch_name,
        extra_metadata=extra_metadata,
    )

    checkpoint = {
        "checkpoint_type": "training_resume",
        "branch_name": branch_name,

        # Model/adapters.
        "adapter_state_dict": adapter_state,
        "metadata": metadata,

        # Training state.
        "optimizer_state_dict": get_optimizer_state_dict(optimizer),
        "scheduler_state_dict": get_scheduler_state_dict(scheduler),
        "scaler_state_dict": get_scaler_state_dict(scaler),

        # Counters.
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer_step": int(optimizer_step),

        # Metrics.
        "loss_value": None if loss_value is None else float(loss_value),
        "best_metric": None if best_metric is None else float(best_metric),
    }

    ckpt_path = output_dir / filename
    atomic_torch_save(checkpoint, ckpt_path)

    save_json(metadata, output_dir / "metadata.json")

    print("\n[Training checkpoint saved]")
    print("Branch:        ", branch_name)
    print("Path:          ", ckpt_path)
    print("Epoch:         ", epoch)
    print("Global step:   ", global_step)
    print("Optimizer step:", optimizer_step)
    print("Adapter keys:  ", len(adapter_state))
    print("Size MB:       ", ckpt_path.stat().st_size / 1024**2)

    return ckpt_path


def load_training_checkpoint(
    path: Path,
) -> Dict[str, Any]:
    """
    Loads resume checkpoint on CPU.
    """
    ckpt = torch_load_cpu(path)

    if ckpt.get("checkpoint_type", None) != "training_resume":
        raise ValueError(
            f"Expected checkpoint_type='training_resume', "
            f"got {ckpt.get('checkpoint_type')}"
        )

    return ckpt


def restore_training_checkpoint_into_bundle(
    bundle: Dict[str, Any],
    checkpoint_path: Path,
    scaler=None,
    strict_adapter: bool = True,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    load_scaler: bool = True,
) -> Dict[str, Any]:
    """
    Restores a training checkpoint into an already reconstructed bundle.

    Required before calling:
        - base model loaded
        - adapter injected with the same adapter_config
        - optimizer created
        - scheduler optionally created

    This function loads:
        - adapter weights into bundle["unet"]
        - optimizer state into bundle["optimizer"]
        - scheduler state into bundle["scheduler"], if present
        - scaler state into scaler, if provided
    """
    ckpt = load_training_checkpoint(checkpoint_path)

    adapter_report = load_adapter_state_dict(
        unet=bundle["unet"],
        adapter_state_dict=ckpt["adapter_state_dict"],
        strict=strict_adapter,
    )

    if load_optimizer:
        load_optimizer_state_dict(
            optimizer=bundle.get("optimizer", None),
            state_dict=ckpt.get("optimizer_state_dict", None),
        )

    if load_scheduler:
        load_scheduler_state_dict(
            scheduler=bundle.get("scheduler", None),
            state_dict=ckpt.get("scheduler_state_dict", None),
        )

    if load_scaler:
        load_scaler_state_dict(
            scaler=scaler,
            state_dict=ckpt.get("scaler_state_dict", None),
        )

    report = {
        "checkpoint_path": str(checkpoint_path),
        "branch_name": ckpt.get("branch_name", None),
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "optimizer_step": ckpt.get("optimizer_step", 0),
        "loss_value": ckpt.get("loss_value", None),
        "best_metric": ckpt.get("best_metric", None),
        "adapter_report": adapter_report,
        "metadata": ckpt.get("metadata", {}),
    }

    print("\n[Training checkpoint restored]")
    print("Path:          ", checkpoint_path)
    print("Branch:        ", report["branch_name"])
    print("Epoch:         ", report["epoch"])
    print("Global step:   ", report["global_step"])
    print("Optimizer step:", report["optimizer_step"])
    print("Adapter loaded:", adapter_report["n_loaded"])

    return report


def restore_inference_checkpoint_into_bundle(
    bundle: Dict[str, Any],
    checkpoint_path: Path,
    strict_adapter: bool = True,
) -> Dict[str, Any]:
    """
    Loads inference adapter checkpoint into already reconstructed bundle.

    Required before calling:
        - base model loaded
        - adapter injected with same adapter_config
    """
    ckpt = load_inference_checkpoint(checkpoint_path)

    adapter_report = load_adapter_state_dict(
        unet=bundle["unet"],
        adapter_state_dict=ckpt["adapter_state_dict"],
        strict=strict_adapter,
    )

    report = {
        "checkpoint_path": str(checkpoint_path),
        "branch_name": ckpt.get("branch_name", None),
        "adapter_report": adapter_report,
        "metadata": ckpt.get("metadata", {}),
    }

    print("\n[Inference checkpoint restored]")
    print("Path:          ", checkpoint_path)
    print("Branch:        ", report["branch_name"])
    print("Adapter loaded:", adapter_report["n_loaded"])

    return report


# ============================================================
# Checkpoint manager
# ============================================================

class BranchCheckpointManager:
    """
    Small manager for one branch: global OR local.

    It keeps:
        - latest training checkpoint
        - best training checkpoint
        - latest inference checkpoint
        - best inference checkpoint

    It does NOT mix global and local in the same file.
    """

    def __init__(
        self,
        root_dir: Path,
        branch_name: str,
        metric_mode: str = "min",
    ):
        self.root_dir = ensure_dir(Path(root_dir))
        self.branch_name = str(branch_name)
        self.metric_mode = str(metric_mode)

        if self.metric_mode not in {"min", "max"}:
            raise ValueError("metric_mode must be 'min' or 'max'.")

        self.branch_dir = ensure_dir(self.root_dir / self.branch_name)

        self.latest_train_path = self.branch_dir / "latest_training_resume.pt"
        self.best_train_path = self.branch_dir / "best_training_resume.pt"

        self.latest_infer_path = self.branch_dir / "latest_adapter_inference.pt"
        self.best_infer_path = self.branch_dir / "best_adapter_inference.pt"

        self.state_path = self.branch_dir / "checkpoint_state.json"

        self.best_metric = None

    def is_better(self, metric: Optional[float]) -> bool:
        if metric is None:
            return False

        metric = float(metric)

        if self.best_metric is None:
            return True

        if self.metric_mode == "min":
            return metric < self.best_metric

        return metric > self.best_metric

    def save_state_json(self) -> None:
        state = {
            "branch_name": self.branch_name,
            "metric_mode": self.metric_mode,
            "best_metric": self.best_metric,
            "latest_train_path": str(self.latest_train_path),
            "best_train_path": str(self.best_train_path),
            "latest_infer_path": str(self.latest_infer_path),
            "best_infer_path": str(self.best_infer_path),
        }

        save_json(state, self.state_path)

    def save_latest(
        self,
        bundle: Dict[str, Any],
        epoch: int,
        global_step: int,
        optimizer_step: int,
        loss_value: Optional[float] = None,
        metric_value: Optional[float] = None,
        scaler=None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        save_inference_copy: bool = True,
    ) -> Dict[str, Path]:
        """
        Always overwrites latest checkpoints.
        """
        train_path = save_training_checkpoint_for_bundle(
            bundle=bundle,
            output_dir=self.branch_dir,
            branch_name=self.branch_name,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            loss_value=loss_value,
            best_metric=self.best_metric,
            scaler=scaler,
            filename=self.latest_train_path.name,
            extra_metadata=extra_metadata,
        )

        paths = {"latest_training": train_path}

        if save_inference_copy:
            infer_path = save_inference_checkpoint_for_bundle(
                bundle=bundle,
                output_dir=self.branch_dir,
                branch_name=self.branch_name,
                filename=self.latest_infer_path.name,
                extra_metadata=extra_metadata,
            )
            paths["latest_inference"] = infer_path

        self.save_state_json()

        return paths

    def save_best_if_improved(
        self,
        bundle: Dict[str, Any],
        metric_value: float,
        epoch: int,
        global_step: int,
        optimizer_step: int,
        loss_value: Optional[float] = None,
        scaler=None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        save_inference_copy: bool = True,
    ) -> Tuple[bool, Dict[str, Path]]:
        """
        Saves best checkpoints only if metric improves.

        For loss/val_loss:
            metric_mode='min'

        For score/identity:
            metric_mode='max'
        """
        if not self.is_better(metric_value):
            return False, {}

        self.best_metric = float(metric_value)

        train_path = save_training_checkpoint_for_bundle(
            bundle=bundle,
            output_dir=self.branch_dir,
            branch_name=self.branch_name,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            loss_value=loss_value,
            best_metric=self.best_metric,
            scaler=scaler,
            filename=self.best_train_path.name,
            extra_metadata=extra_metadata,
        )

        paths = {"best_training": train_path}

        if save_inference_copy:
            infer_path = save_inference_checkpoint_for_bundle(
                bundle=bundle,
                output_dir=self.branch_dir,
                branch_name=self.branch_name,
                filename=self.best_infer_path.name,
                extra_metadata=extra_metadata,
            )
            paths["best_inference"] = infer_path

        self.save_state_json()

        print("\n[Best checkpoint updated]")
        print("Branch:     ", self.branch_name)
        print("Best metric:", self.best_metric)

        return True, paths


# ============================================================
# Convenience builders for this project
# ============================================================

def build_global_local_checkpoint_managers(
    root_dir,
    global_metric_mode: str = "min",
    local_metric_mode: str = "min") -> Dict[str, BranchCheckpointManager]:
    """
    Creates separate checkpoint managers.

    Output structure:
        root_dir/
            global/
                latest_training_resume.pt
                best_training_resume.pt
                latest_adapter_inference.pt
                best_adapter_inference.pt
                metadata.json
                checkpoint_state.json

            local/
                latest_training_resume.pt
                best_training_resume.pt
                latest_adapter_inference.pt
                best_adapter_inference.pt
                metadata.json
                checkpoint_state.json
    """
    root_dir = ensure_dir(Path(root_dir))

    managers = {
        "global": BranchCheckpointManager(
            root_dir=root_dir,
            branch_name="global",
            metric_mode=global_metric_mode,
        ),
        "local": BranchCheckpointManager(
            root_dir=root_dir,
            branch_name="local",
            metric_mode=local_metric_mode,
        ),
    }

    print("\n[Checkpoint managers created]")
    print("Root dir:", root_dir)
    print("Global dir:", managers["global"].branch_dir)
    print("Local dir: ", managers["local"].branch_dir)

    return managers