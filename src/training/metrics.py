# ============================================================
# TRAINING METRICS UTILS
# Cheap metrics for Global + Local branches
#
# Design:
#   - No extra UNet forward.
#   - No extra VAE decode.
#   - No extra ScoreNet / AgeNet / ArcFace call.
#   - Only uses:
#       batch
#       prompt_pack
#       loss_out
#       optimizer/scheduler state
#
# Expected prompt packs:
#   local_pack:
#       target_scores_raw
#       target_scores
#       target_modes
#
#   global_pack:
#       target_ages
#       target_modes
#
# Expected loss_out:
#   flexible dictionary. This code reads keys if present.
# ============================================================

from __future__ import annotations

from collections import defaultdict, Counter
from typing import Dict, Any, Optional, List, Union

import math
import torch


# ============================================================
# Small conversion helpers
# ============================================================

def metric_to_float(x, default: Optional[float] = None) -> Optional[float]:
    """
    Converts scalar tensor / int / float to Python float.
    Returns default if conversion is not possible.
    """
    if x is None:
        return default

    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return default
            return float(x.detach().float().mean().cpu().item())

        return float(x)

    except Exception:
        return default


def tensor_mean_float(x, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default

    try:
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        if x.numel() == 0:
            return default

        return float(x.detach().float().mean().cpu().item())

    except Exception:
        return default


def tensor_std_float(x, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default

    try:
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        if x.numel() <= 1:
            return 0.0

        return float(x.detach().float().std(unbiased=False).cpu().item())

    except Exception:
        return default


def tensor_fraction(x, condition_fn, default: Optional[float] = None) -> Optional[float]:
    """
    Fraction of elements satisfying condition_fn.
    Returns fraction in [0, 1].
    """
    if x is None:
        return default

    try:
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.detach().float().cpu()

        if x.numel() == 0:
            return default

        mask = condition_fn(x)
        return float(mask.float().mean().item())

    except Exception:
        return default


def safe_get_lr(optimizer=None, scheduler=None) -> Optional[float]:
    """
    Reads current LR from scheduler if possible, otherwise from optimizer.
    """
    if scheduler is not None:
        if hasattr(scheduler, "get_lr"):
            try:
                return float(scheduler.get_lr())
            except Exception:
                pass

        if hasattr(scheduler, "get_last_lr"):
            try:
                lrs = scheduler.get_last_lr()
                if len(lrs) > 0:
                    return float(lrs[0])
            except Exception:
                pass

    if optimizer is not None and hasattr(optimizer, "param_groups"):
        try:
            if len(optimizer.param_groups) > 0:
                return float(optimizer.param_groups[0]["lr"])
        except Exception:
            pass

    return None


def count_modes(modes: Optional[List[str]]) -> Dict[str, float]:
    """
    Returns mode fractions.
    """
    if modes is None:
        return {}

    if len(modes) == 0:
        return {}

    c = Counter([str(m) for m in modes])
    total = max(1, sum(c.values()))

    return {
        f"mode_frac/{k}": float(v) / total
        for k, v in c.items()
    }


# ============================================================
# Running meters
# ============================================================

class AverageMeter:
    """
    Tracks weighted average.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: Optional[float], n: int = 1):
        if value is None:
            return

        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return

        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.sum / self.count


class MetricsTracker:
    """
    Generic running metrics container.
    """

    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def reset(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, metrics: Dict[str, Any], n: int = 1):
        for k, v in metrics.items():
            value = metric_to_float(v, default=None)
            if value is not None:
                self.meters[k].update(value, n=n)

    def compute(self) -> Dict[str, float]:
        out = {}
        for k, meter in self.meters.items():
            if meter.avg is not None:
                out[k] = meter.avg
        return out

    def format(
        self,
        prefix: str = "",
        keys: Optional[List[str]] = None,
        precision: int = 4,
        max_items: Optional[int] = None,
    ) -> str:
        metrics = self.compute()

        if keys is None:
            keys = sorted(metrics.keys())

        parts = []
        for k in keys:
            if k not in metrics:
                continue

            name = f"{prefix}{k}" if prefix else k
            parts.append(f"{name}={metrics[k]:.{precision}f}")

        if max_items is not None:
            parts = parts[:max_items]

        return " | ".join(parts)


# ============================================================
# Loss output extraction
# ============================================================

def extract_loss_metrics(loss_out: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts scalar losses from loss_out without assuming exact names.

    It will collect:
        - loss
        - keys starting with 'loss'
        - keys starting with 'L_'
        - common semantic keys if present
    """
    if loss_out is None:
        return {}

    metrics = {}

    candidate_keys = []

    for k in loss_out.keys():
        ks = str(k)

        if ks == "loss":
            candidate_keys.append(k)
        elif ks.startswith("loss"):
            candidate_keys.append(k)
        elif ks.startswith("L_"):
            candidate_keys.append(k)
        elif ks in [
            "diff_loss",
            "full_loss",
            "zone_loss",
            "score_loss",
            "cycle_loss",
            "age_loss",
            "delta_age_loss",
            "id_loss",
            "perc_loss",
        ]:
            candidate_keys.append(k)

    for k in candidate_keys:
        value = metric_to_float(loss_out.get(k), default=None)
        if value is not None:
            metrics[f"loss/{k}"] = value

    # Normalize main loss name.
    if "loss" in loss_out:
        value = metric_to_float(loss_out["loss"], default=None)
        if value is not None:
            metrics["loss/total"] = value

    return metrics


def extract_prediction_metrics_from_loss_out(loss_out: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts cheap prediction metrics if they already exist in loss_out.

    Does not compute expensive models.
    """
    if loss_out is None:
        return {}

    metrics = {}

    # Local score predictions, if returned by L_score.
    score_pred_keys = ["score_pred", "pred_score", "score_hat", "pred_scores"]
    score_target_keys = ["target_scores", "score_target", "target_score"]

    score_pred = None
    score_target = None

    for k in score_pred_keys:
        if k in loss_out:
            score_pred = loss_out[k]
            break

    for k in score_target_keys:
        if k in loss_out:
            score_target = loss_out[k]
            break

    if score_pred is not None:
        metrics["pred/score_mean"] = tensor_mean_float(score_pred)
        metrics["pred/score_std"] = tensor_std_float(score_pred)

    if score_pred is not None and score_target is not None:
        try:
            sp = score_pred.detach().float().view(-1).cpu()
            st = score_target.detach().float().view(-1).cpu()

            if sp.numel() == st.numel():
                metrics["pred/score_mae"] = float((sp - st).abs().mean().item())
                metrics["pred/score_bias"] = float((sp - st).mean().item())
        except Exception:
            pass

    # Global age predictions, if returned by semantic loss.
    age_pred_keys = ["age_pred", "pred_age", "age_hat", "pred_ages"]
    age_target_keys = ["target_ages", "age_target", "target_age"]

    age_pred = None
    age_target = None

    for k in age_pred_keys:
        if k in loss_out:
            age_pred = loss_out[k]
            break

    for k in age_target_keys:
        if k in loss_out:
            age_target = loss_out[k]
            break

    if age_pred is not None:
        metrics["pred/age_mean"] = tensor_mean_float(age_pred)
        metrics["pred/age_std"] = tensor_std_float(age_pred)

    if age_pred is not None and age_target is not None:
        try:
            ap = age_pred.detach().float().view(-1).cpu()
            at = age_target.detach().float().view(-1).cpu()

            if ap.numel() == at.numel():
                metrics["pred/age_mae"] = float((ap - at).abs().mean().item())
                metrics["pred/age_bias"] = float((ap - at).mean().item())
        except Exception:
            pass

    # Identity similarity, if returned by semantic loss.
    for key in ["id_similarity", "identity_similarity", "arcface_similarity", "id_sim"]:
        if key in loss_out:
            value = metric_to_float(loss_out[key], default=None)
            if value is not None:
                metrics["pred/id_similarity"] = value
            break

    # Diffusion/noise quality if returned by loss.
    # These are cheap only if eps_pred/noise are already in loss_out.
    eps_pred = None
    eps_true = None

    for k in ["noise_pred", "eps_pred", "epsilon_pred"]:
        if k in loss_out:
            eps_pred = loss_out[k]
            break

    for k in ["noise", "eps", "epsilon"]:
        if k in loss_out:
            eps_true = loss_out[k]
            break

    if eps_pred is not None:
        metrics["diff/eps_pred_abs_mean"] = tensor_mean_float(eps_pred.detach().abs())
        metrics["diff/eps_pred_std"] = tensor_std_float(eps_pred)

    if eps_true is not None:
        metrics["diff/eps_true_abs_mean"] = tensor_mean_float(eps_true.detach().abs())
        metrics["diff/eps_true_std"] = tensor_std_float(eps_true)

    if eps_pred is not None and eps_true is not None:
        try:
            a = eps_pred.detach().float().view(eps_pred.shape[0], -1).cpu()
            b = eps_true.detach().float().view(eps_true.shape[0], -1).cpu()

            if a.shape == b.shape:
                cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
                metrics["diff/eps_cosine"] = float(cos.mean().item())
                metrics["diff/eps_mae"] = float((a - b).abs().mean().item())
        except Exception:
            pass

    return metrics


# ============================================================
# Local training metrics
# ============================================================

def compute_local_training_metrics(
    batch: Dict[str, Any],
    prompt_pack: Dict[str, Any],
    loss_out: Optional[Dict[str, Any]] = None,
    loss_mode: Optional[str] = None,
    optimizer=None,
    scheduler=None,
) -> Dict[str, float]:
    """
    Computes cheap local branch metrics.

    Does not call ScoreNet.
    Does not decode images.
    """
    metrics = {}

    # ------------------------------------------------------------
    # Losses and loss-mode
    # ------------------------------------------------------------
    if loss_out is not None:
        metrics.update(extract_loss_metrics(loss_out))
        metrics.update(extract_prediction_metrics_from_loss_out(loss_out))

    if loss_mode is not None:
        metrics[f"loss_mode/{loss_mode}"] = 1.0

    # ------------------------------------------------------------
    # LR
    # ------------------------------------------------------------
    lr = safe_get_lr(optimizer=optimizer, scheduler=scheduler)
    if lr is not None:
        metrics["optim/lr"] = lr

    # ------------------------------------------------------------
    # Source scores
    # ------------------------------------------------------------
    source_scores = batch.get("score_raw", None)

    if source_scores is None and "score" in batch:
        source_scores = batch["score"]
        try:
            if torch.is_tensor(source_scores) and source_scores.detach().float().max().item() <= 1.0:
                source_scores = source_scores * 100.0
        except Exception:
            pass

    target_scores_raw = prompt_pack.get("target_scores_raw", None)

    if source_scores is not None:
        metrics["local/source_score_mean"] = tensor_mean_float(source_scores)
        metrics["local/source_score_std"] = tensor_std_float(source_scores)
        metrics["local/source_score_ge_65"] = tensor_fraction(source_scores, lambda x: x >= 65)
        metrics["local/source_score_ge_85"] = tensor_fraction(source_scores, lambda x: x >= 85)
        metrics["local/source_score_ge_95"] = tensor_fraction(source_scores, lambda x: x >= 95)

    if target_scores_raw is not None:
        metrics["local/target_score_mean"] = tensor_mean_float(target_scores_raw)
        metrics["local/target_score_std"] = tensor_std_float(target_scores_raw)
        metrics["local/target_score_ge_65"] = tensor_fraction(target_scores_raw, lambda x: x >= 65)
        metrics["local/target_score_ge_85"] = tensor_fraction(target_scores_raw, lambda x: x >= 85)
        metrics["local/target_score_ge_95"] = tensor_fraction(target_scores_raw, lambda x: x >= 95)

    if source_scores is not None and target_scores_raw is not None:
        try:
            s = source_scores.detach().float().view(-1).cpu()
            t = target_scores_raw.detach().float().view(-1).cpu()

            if s.numel() == t.numel():
                delta = t - s
                metrics["local/score_delta_mean"] = float(delta.mean().item())
                metrics["local/score_delta_abs_mean"] = float(delta.abs().mean().item())
                metrics["local/score_delta_positive_frac"] = float((delta > 0).float().mean().item())
                metrics["local/score_delta_anchor_frac"] = float((delta.abs() <= 3).float().mean().item())
                metrics["local/score_delta_negative_frac"] = float((delta < 0).float().mean().item())
        except Exception:
            pass

    # ------------------------------------------------------------
    # Target mode distribution
    # ------------------------------------------------------------
    target_modes = prompt_pack.get("target_modes", None)
    metrics.update(count_modes(target_modes))

    # ------------------------------------------------------------
    # Pixel stats: cheap sanity checks
    # ------------------------------------------------------------
    pixel_values = batch.get("pixel_values", None)

    if pixel_values is not None:
        metrics["image/pixel_mean"] = tensor_mean_float(pixel_values)
        metrics["image/pixel_std"] = tensor_std_float(pixel_values)
        metrics["image/pixel_abs_mean"] = tensor_mean_float(pixel_values.detach().abs())

    return {k: v for k, v in metrics.items() if v is not None}


# ============================================================
# Global training metrics
# ============================================================

def compute_global_training_metrics(
    batch: Dict[str, Any],
    prompt_pack: Dict[str, Any],
    loss_out: Optional[Dict[str, Any]] = None,
    loss_mode: Optional[str] = None,
    optimizer=None,
    scheduler=None,
) -> Dict[str, float]:
    """
    Computes cheap global branch metrics.

    Does not call AgeNet.
    Does not call ArcFace.
    Does not decode images.
    """
    metrics = {}

    # ------------------------------------------------------------
    # Losses and loss-mode
    # ------------------------------------------------------------
    if loss_out is not None:
        metrics.update(extract_loss_metrics(loss_out))
        metrics.update(extract_prediction_metrics_from_loss_out(loss_out))

    if loss_mode is not None:
        metrics[f"loss_mode/{loss_mode}"] = 1.0

    # ------------------------------------------------------------
    # LR
    # ------------------------------------------------------------
    lr = safe_get_lr(optimizer=optimizer, scheduler=scheduler)
    if lr is not None:
        metrics["optim/lr"] = lr

    # ------------------------------------------------------------
    # Source / target ages
    # ------------------------------------------------------------
    source_ages = batch.get("age", None)
    target_ages = prompt_pack.get("target_ages", None)

    if source_ages is not None:
        metrics["global/source_age_mean"] = tensor_mean_float(source_ages)
        metrics["global/source_age_std"] = tensor_std_float(source_ages)
        metrics["global/source_age_ge_60"] = tensor_fraction(source_ages, lambda x: x >= 60)
        metrics["global/source_age_ge_70"] = tensor_fraction(source_ages, lambda x: x >= 70)
        metrics["global/source_age_ge_80"] = tensor_fraction(source_ages, lambda x: x >= 80)

    if target_ages is not None:
        metrics["global/target_age_mean"] = tensor_mean_float(target_ages)
        metrics["global/target_age_std"] = tensor_std_float(target_ages)
        metrics["global/target_age_ge_60"] = tensor_fraction(target_ages, lambda x: x >= 60)
        metrics["global/target_age_ge_70"] = tensor_fraction(target_ages, lambda x: x >= 70)
        metrics["global/target_age_ge_80"] = tensor_fraction(target_ages, lambda x: x >= 80)

    if source_ages is not None and target_ages is not None:
        try:
            s = source_ages.detach().float().view(-1).cpu()
            t = target_ages.detach().float().view(-1).cpu()

            if s.numel() == t.numel():
                delta = t - s
                metrics["global/age_delta_mean"] = float(delta.mean().item())
                metrics["global/age_delta_abs_mean"] = float(delta.abs().mean().item())
                metrics["global/age_delta_positive_frac"] = float((delta > 0).float().mean().item())
                metrics["global/age_delta_anchor_frac"] = float((delta.abs() <= 3).float().mean().item())
                metrics["global/age_delta_negative_frac"] = float((delta < 0).float().mean().item())
        except Exception:
            pass

    # ------------------------------------------------------------
    # Target mode distribution
    # ------------------------------------------------------------
    target_modes = prompt_pack.get("target_modes", None)
    metrics.update(count_modes(target_modes))

    # ------------------------------------------------------------
    # Pixel stats: cheap sanity checks
    # ------------------------------------------------------------
    pixel_values = batch.get("pixel_values", None)

    if pixel_values is not None:
        metrics["image/pixel_mean"] = tensor_mean_float(pixel_values)
        metrics["image/pixel_std"] = tensor_std_float(pixel_values)
        metrics["image/pixel_abs_mean"] = tensor_mean_float(pixel_values.detach().abs())

    return {k: v for k, v in metrics.items() if v is not None}


# ============================================================
# Pretty printing
# ============================================================

LOCAL_PRINT_KEYS = [
    "loss/total",
    "loss/loss",
    "loss/full_loss",
    "loss/zone_loss",
    "loss/score_loss",
    "pred/score_mae",
    "pred/score_bias",
    "local/source_score_mean",
    "local/target_score_mean",
    "local/score_delta_mean",
    "local/score_delta_positive_frac",
    "local/score_delta_anchor_frac",
    "local/source_score_ge_95",
    "local/target_score_ge_95",
    "mode_frac/aging",
    "mode_frac/anchor",
    "mode_frac/contrast",
    "double_prompt/used",
    "double_prompt/source_loss",
    "double_prompt/neutral_loss",
    "optim/lr",
]

GLOBAL_PRINT_KEYS = [
    "loss/total",
    "loss/loss",
    "loss/diff_loss",
    "loss/age_loss",
    "loss/delta_age_loss",
    "loss/id_loss",
    "pred/age_mae",
    "pred/age_bias",
    "pred/id_similarity",
    "global/source_age_mean",
    "global/target_age_mean",
    "global/age_delta_mean",
    "global/age_delta_positive_frac",
    "global/age_delta_anchor_frac",
    "global/target_age_ge_70",
    "global/target_age_ge_80",
    "mode_frac/aging",
    "mode_frac/anchor",
    "mode_frac/mild_younger",
    "optim/lr"]


def format_metrics(
    metrics: Dict[str, float],
    keys: Optional[List[str]] = None,
    precision: int = 4,
    max_items: Optional[int] = None) -> str:

    if keys is None:
        keys = sorted(metrics.keys())

    parts = []

    for k in keys:
        if k not in metrics:
            continue

        v = metrics[k]

        if v is None:
            continue

        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue

        parts.append(f"{k}={v:.{precision}f}")

    if max_items is not None:
        parts = parts[:max_items]

    return " | ".join(parts)


def print_local_metrics(
    metrics: Dict[str, float],
    prefix: str = "[LOCAL]",
    step: Optional[int] = None,) -> None:

    step_text = f" step={step}" if step is not None else ""
    msg = format_metrics(metrics, keys=LOCAL_PRINT_KEYS, precision=4)
    print(f"{prefix}{step_text} {msg}")


def print_global_metrics(
    metrics: Dict[str, float],
    prefix: str = "[GLOBAL]",
    step: Optional[int] = None) -> None:

    step_text = f" step={step}" if step is not None else ""
    msg = format_metrics(metrics, keys=GLOBAL_PRINT_KEYS, precision=4)
    print(f"{prefix}{step_text} {msg}")

