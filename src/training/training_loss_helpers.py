from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

# ============================================================
# Loss instantiation helpers
# ============================================================

def maybe_build_loss(
    existing_loss,
    loss_factory: Optional[Callable],
    loss_kwargs: Optional[Dict[str, Any]],
    name: str,
):
    """
    Uses existing loss if provided. Otherwise builds it from factory.

    This keeps the wrapper flexible because your current notebook may already
    have local_loss_fn/global_aging_loss instantiated.
    """
    if existing_loss is not None:
        return existing_loss

    if loss_factory is None:
        raise ValueError(
            f"{name} loss is None and no {name}_loss_factory was provided."
        )

    if loss_kwargs is None:
        loss_kwargs = {}

    return loss_factory(**loss_kwargs)


def get_monitor_metric(
    metrics: Dict[str, float],
    monitor_key: str = "loss/total",
    fallback_keys: Sequence[str] = ("loss/loss", "loss/full_loss", "loss/diff_loss"),
) -> Optional[float]:
    if monitor_key in metrics:
        return float(metrics[monitor_key])

    for k in fallback_keys:
        if k in metrics:
            return float(metrics[k])

    return None


# ============================================================
# Optional paired-supervision helpers
# ============================================================

def paired_supervision_enabled(
    loader,
    loss_fn,
    every_n_steps: int,
    weight: float,
) -> bool:
    provided = (loader is not None, loss_fn is not None)
    if any(provided) and not all(provided):
        raise ValueError("paired_train_loader and paired_loss_fn must be passed together")
    if every_n_steps < 0:
        raise ValueError("paired_every_n_steps must be >= 0")
    if weight < 0:
        raise ValueError("paired_weight must be >= 0")
    return all(provided) and int(every_n_steps) > 0 and float(weight) > 0.0


def should_run_paired_supervision(batch_idx: int, every_n_steps: int) -> bool:
    return int(every_n_steps) > 0 and (int(batch_idx) + 1) % int(every_n_steps) == 0


def next_cycling_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def call_paired_supervision_loss(loss_fn, batch: Dict[str, Any]) -> Dict[str, Any]:
    output = loss_fn(batch)
    if not isinstance(output, dict) or "loss" not in output:
        raise TypeError("paired_loss_fn(batch) must return a dict containing key 'loss'")
    return output
