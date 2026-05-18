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
