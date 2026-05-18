# ============================================================
# WARMUP + COSINE LR SCHEDULER
#
# Designed for this project:
#   - Optimizers are already created inside the bundles:
#       mixed_global_bundle["optimizer"]
#       mixed_local_bundle["optimizer"]
#
#   - This scheduler only attaches LR scheduling.
#   - Compatible with standard torch optimizers such as AdamW.
#   - Also supports hybrid optimizers with:
#       optimizer.muon
#       optimizer.adamw
#
# Usage:
#
#   global_scheduler = build_warmup_cosine_scheduler_for_bundle(
#       bundle=mixed_global_bundle,
#       total_steps=global_total_optimizer_steps,
#       warmup_steps=global_warmup_steps,
#       min_lr=1e-6,
#   )
#
#   local_scheduler = build_warmup_cosine_scheduler_for_bundle(
#       bundle=mixed_local_bundle,
#       total_steps=local_total_optimizer_steps,
#       warmup_steps=local_warmup_steps,
#       min_lr=1e-6,
#   )
#
# In training loop:
#       optimizer.step()
#       scheduler.step()
# ============================================================

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class WarmupCosineLR:
    """
    Step-based linear warmup + cosine decay scheduler.

    Behavior:
        - step 0:
            LR is initialized to 0 if warmup_steps > 0.
        - steps 1..warmup_steps:
            LR increases linearly from 0 to base_lr.
        - after warmup:
            cosine decay from base_lr to min_lr.
        - after total_steps:
            LR stays at min_lr.

    Supports:
        1. Standard torch optimizer, e.g. AdamW.
        2. Hybrid optimizer with:
            optimizer.muon
            optimizer.adamw

    Important:
        Call scheduler.step() after optimizer.step().
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 1e-6,
        min_muon_lr: Optional[float] = None,
        start_at_zero: bool = True,
    ):
        if optimizer is None:
            raise ValueError("optimizer cannot be None.")

        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")

        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps must be < total_steps. "
                f"Got warmup_steps={warmup_steps}, total_steps={total_steps}."
            )

        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {min_lr}")

        if min_muon_lr is not None and min_muon_lr < 0:
            raise ValueError(f"min_muon_lr must be >= 0, got {min_muon_lr}")

        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.min_muon_lr = float(min_muon_lr) if min_muon_lr is not None else None
        self.start_at_zero = bool(start_at_zero)

        self.step_num = 0

        self.is_hybrid = (
            hasattr(optimizer, "muon")
            and hasattr(optimizer, "adamw")
        )

        if self.is_hybrid:
            self.base_adamw_lrs = [
                float(group["lr"])
                for group in optimizer.adamw.param_groups
            ]

            self.base_muon_lrs = [
                float(group["lr"])
                for group in optimizer.muon.param_groups
            ]

            self.base_lrs = self.base_muon_lrs + self.base_adamw_lrs

        else:
            if not hasattr(optimizer, "param_groups"):
                raise TypeError(
                    "optimizer must be a torch optimizer with param_groups "
                    "or a hybrid optimizer with .muon and .adamw."
                )

            self.base_lrs = [
                float(group["lr"])
                for group in optimizer.param_groups
            ]

            self.base_adamw_lrs = None
            self.base_muon_lrs = None

        if len(self.base_lrs) == 0:
            raise ValueError("optimizer has no parameter groups.")

        if self.start_at_zero and self.warmup_steps > 0:
            self._set_lr(step=0)

    def _compute_lr(
        self,
        base_lr: float,
        min_lr: float,
        step: int,
    ) -> float:
        """
        Compute LR for a given base_lr and current step.
        """
        step = int(step)

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return float(base_lr * step / max(1, self.warmup_steps))

        t = min(max(step, self.warmup_steps), self.total_steps)

        denom = max(1, self.total_steps - self.warmup_steps)
        progress = (t - self.warmup_steps) / denom
        progress = min(1.0, max(0.0, progress))

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine

        return float(lr)

    def _set_lr_standard(self, step: int) -> None:
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            group["lr"] = self._compute_lr(
                base_lr=base_lr,
                min_lr=self.min_lr,
                step=step,
            )

    def _set_lr_hybrid(self, step: int) -> None:
        muon_min_lr = self.min_lr if self.min_muon_lr is None else self.min_muon_lr

        for i, group in enumerate(self.optimizer.adamw.param_groups):
            base_lr = self.base_adamw_lrs[i]
            group["lr"] = self._compute_lr(
                base_lr=base_lr,
                min_lr=self.min_lr,
                step=step,
            )

        for i, group in enumerate(self.optimizer.muon.param_groups):
            base_lr = self.base_muon_lrs[i]
            group["lr"] = self._compute_lr(
                base_lr=base_lr,
                min_lr=muon_min_lr,
                step=step,
            )

    def _set_lr(self, step: int) -> None:
        if self.is_hybrid:
            self._set_lr_hybrid(step)
        else:
            self._set_lr_standard(step)

    def step(self) -> None:
        """
        Advance scheduler by one optimizer step.

        Recommended order:
            optimizer.step()
            scheduler.step()
        """
        self.step_num += 1
        self._set_lr(self.step_num)

    def set_step(self, step: int) -> None:
        """
        Explicitly set scheduler step.

        Useful when resuming from checkpoint.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        self.step_num = int(step)
        self._set_lr(self.step_num)

    def get_last_lr(self) -> List[float]:
        """
        Return current learning rates.
        """
        if self.is_hybrid:
            return (
                [float(group["lr"]) for group in self.optimizer.muon.param_groups]
                + [float(group["lr"]) for group in self.optimizer.adamw.param_groups]
            )

        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def get_lr(self) -> float:
        """
        Return first LR for compact logging.
        """
        lrs = self.get_last_lr()
        return float(lrs[0]) if len(lrs) > 0 else 0.0

    def get_lr_dict(self) -> Dict[str, Any]:
        """
        Logging-friendly LR dictionary.
        """
        if self.is_hybrid:
            muon_lrs = [float(group["lr"]) for group in self.optimizer.muon.param_groups]
            adamw_lrs = [float(group["lr"]) for group in self.optimizer.adamw.param_groups]

            return {
                "step": int(self.step_num),
                "muon_lr": float(muon_lrs[0]) if muon_lrs else None,
                "adamw_lr": float(adamw_lrs[0]) if adamw_lrs else None,
                "muon_lrs": muon_lrs,
                "adamw_lrs": adamw_lrs,
            }

        lrs = [float(group["lr"]) for group in self.optimizer.param_groups]

        return {
            "step": int(self.step_num),
            "lr": float(lrs[0]) if lrs else None,
            "lrs": lrs,
        }

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "step_num": int(self.step_num),
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps),
            "min_lr": float(self.min_lr),
            "min_muon_lr": self.min_muon_lr,
            "start_at_zero": bool(self.start_at_zero),
            "is_hybrid": bool(self.is_hybrid),
            "base_lrs": list(self.base_lrs),
        }

        if self.is_hybrid:
            state["base_adamw_lrs"] = list(self.base_adamw_lrs)
            state["base_muon_lrs"] = list(self.base_muon_lrs)

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return

        self.step_num = int(state_dict.get("step_num", 0))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self.start_at_zero = bool(state_dict.get("start_at_zero", self.start_at_zero))

        loaded_min_muon_lr = state_dict.get("min_muon_lr", self.min_muon_lr)
        self.min_muon_lr = (
            float(loaded_min_muon_lr)
            if loaded_min_muon_lr is not None
            else None
        )

        if self.is_hybrid:
            loaded_adamw = state_dict.get("base_adamw_lrs", None)
            loaded_muon = state_dict.get("base_muon_lrs", None)

            if (
                isinstance(loaded_adamw, (list, tuple))
                and len(loaded_adamw) == len(self.optimizer.adamw.param_groups)
            ):
                self.base_adamw_lrs = [float(x) for x in loaded_adamw]

            if (
                isinstance(loaded_muon, (list, tuple))
                and len(loaded_muon) == len(self.optimizer.muon.param_groups)
            ):
                self.base_muon_lrs = [float(x) for x in loaded_muon]

            self.base_lrs = self.base_muon_lrs + self.base_adamw_lrs

        else:
            loaded_base_lrs = state_dict.get("base_lrs", None)

            if (
                isinstance(loaded_base_lrs, (list, tuple))
                and len(loaded_base_lrs) == len(self.optimizer.param_groups)
            ):
                self.base_lrs = [float(x) for x in loaded_base_lrs]

        self._set_lr(self.step_num)


# ============================================================
# Builder helpers
# ============================================================

def build_warmup_cosine_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 1e-6,
    min_muon_lr: Optional[float] = None,
    start_at_zero: bool = True,
) -> WarmupCosineLR:
    """
    Build scheduler compatible with AdamW or hybrid Muon/AdamW.
    """
    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        min_muon_lr=min_muon_lr,
        start_at_zero=start_at_zero,
    )

    return scheduler


def build_warmup_cosine_scheduler_for_bundle(
    bundle: Dict[str, Any],
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 1e-6,
    min_muon_lr: Optional[float] = None,
    start_at_zero: bool = True,
    scheduler_key: str = "scheduler",
) -> WarmupCosineLR:
    """
    Builds and stores a warmup cosine scheduler inside an existing bundle.

    Expected:
        bundle["optimizer"] already exists.

    Example:
        global_scheduler = build_warmup_cosine_scheduler_for_bundle(
            bundle=mixed_global_bundle,
            total_steps=1000,
            warmup_steps=100,
            min_lr=1e-6,
        )

    Then:
        mixed_global_bundle["scheduler"] is available.
    """
    if "optimizer" not in bundle:
        raise KeyError(
            "bundle must contain key 'optimizer'. "
            "In this project it should come from build_mixed_lora_dora_training_setup(...)."
        )

    scheduler = build_warmup_cosine_scheduler(
        optimizer=bundle["optimizer"],
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        min_muon_lr=min_muon_lr,
        start_at_zero=start_at_zero,
    )

    bundle[scheduler_key] = scheduler

    print("\n[Scheduler created]")
    print("Bundle:       ", bundle.get("name", "Unnamed bundle"))
    print("Total steps:  ", total_steps)
    print("Warmup steps: ", warmup_steps)
    print("Min LR:       ", min_lr)
    print("Start at zero:", start_at_zero)
    print("Initial LR(s):", scheduler.get_last_lr())

    return scheduler


def estimate_optimizer_steps(
    num_batches_per_epoch: int,
    num_epochs: int,
    grad_accum_steps: int = 1,
    drop_last_accum: bool = False) -> int:

    """
    Estimates number of optimizer steps.

    Args:
        num_batches_per_epoch:
            len(train_loader)

        num_epochs:
            Number of epochs.

        grad_accum_steps:
            Gradient accumulation steps.

        drop_last_accum:
            If True:
                only full accumulation windows count.
            If False:
                counts a final partial optimizer step at epoch end.

    Returns:
        total optimizer steps, not dataloader iterations.
    """
    if num_batches_per_epoch <= 0:
        raise ValueError("num_batches_per_epoch must be > 0.")

    if num_epochs <= 0:
        raise ValueError("num_epochs must be > 0.")

    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0.")

    total_micro_steps = int(num_batches_per_epoch) * int(num_epochs)

    if drop_last_accum:
        return total_micro_steps // int(grad_accum_steps)

    return math.ceil(total_micro_steps / int(grad_accum_steps))


def compute_warmup_steps(
    total_steps: int,
    warmup_ratio: float = 0.05,
    min_warmup_steps: int = 10,
    max_warmup_steps: Optional[int] = None,) -> int:

    """
    Computes warmup steps from total steps.

    For small overfitting/debug runs, avoid too much warmup.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0.")

    if warmup_ratio < 0:
        raise ValueError("warmup_ratio must be >= 0.")

    warmup_steps = int(round(total_steps * warmup_ratio))

    if total_steps <= min_warmup_steps:
        warmup_steps = max(0, total_steps // 5)
    else:
        warmup_steps = max(int(min_warmup_steps), warmup_steps)

    if max_warmup_steps is not None:
        warmup_steps = min(warmup_steps, int(max_warmup_steps))

    # Must be strictly less than total_steps.
    warmup_steps = min(warmup_steps, total_steps - 1)

    return int(max(0, warmup_steps))


def build_bundle_scheduler_from_loader(
    bundle: Dict[str, Any],
    train_loader,
    num_epochs: int,
    grad_accum_steps: int = 1,
    warmup_ratio: float = 0.05,
    min_lr: float = 1e-6,
    min_warmup_steps: int = 10,
    max_warmup_steps: Optional[int] = None,
    scheduler_key: str = "scheduler") -> WarmupCosineLR:

    """
    Convenience builder using len(train_loader), num_epochs and grad accumulation.

    This is usually what we want before training each branch.
    """
    total_steps = estimate_optimizer_steps(
        num_batches_per_epoch=len(train_loader),
        num_epochs=num_epochs,
        grad_accum_steps=grad_accum_steps,
        drop_last_accum=False)

    warmup_steps = compute_warmup_steps(
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
        min_warmup_steps=min_warmup_steps,
        max_warmup_steps=max_warmup_steps,)

    scheduler = build_warmup_cosine_scheduler_for_bundle(
        bundle=bundle,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        start_at_zero=True,
        scheduler_key=scheduler_key)

    bundle["scheduler_config"] = {
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "warmup_ratio": warmup_ratio,
        "min_lr": min_lr,
        "num_epochs": num_epochs,
        "grad_accum_steps": grad_accum_steps}

    return scheduler