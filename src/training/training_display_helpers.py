from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

# ============================================================
# Pretty printing
# ============================================================

BOX = "═"
LINE = "─"
THIN = "┄"


def print_box_title(title: str, width: int = 110) -> None:
    print("\n" + BOX * width)
    print(title)
    print(BOX * width)


def print_section(title: str, width: int = 110) -> None:
    print("\n" + LINE * width)
    print(title)
    print(LINE * width)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fmt_float(x, precision: int = 5, sci: bool = False) -> str:
    if x is None:
        return "None"
    try:
        x = float(x)
    except Exception:
        return str(x)

    if sci:
        return f"{x:.{precision}e}"
    return f"{x:.{precision}f}"


def get_first_lr(bundle: Dict[str, Any]) -> Optional[float]:
    scheduler = bundle.get("scheduler", None)
    optimizer = bundle.get("optimizer", None)

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
            return float(optimizer.param_groups[0]["lr"])
        except Exception:
            pass

    return None


def print_run_header(
    run_name: str,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: str,
    num_epochs: int,
    start_epoch: int,
    local_grad_accum_steps: int,
    global_grad_accum_steps: int,
    local_config: Dict[str, Any],
    global_config: Dict[str, Any],
    checkpoint_root: Path,
) -> None:
    print_box_title(f"Face Aging Global-Local Diffusion run: {run_name}")

    print(
        f"Device    : {device} | AMP: {amp_enabled} ({amp_dtype})\n"
        f"Schedule  : epochs={num_epochs} | start_epoch={start_epoch}\n"
        f"Local     : grad_accum={local_grad_accum_steps} | "
        f"p_full={local_config.get('p_full')} | "
        f"p_score={local_config.get('p_score')} | "
        f"p_zone={local_config.get('p_zone')} | "
        f"p_double_full={local_config.get('p_double_full')}\n"
        f"Global    : grad_accum={global_grad_accum_steps} | "
        f"p_diff={global_config.get('p_diff')} | "
        f"p_semantic={global_config.get('p_semantic')} | "
        f"p_double_diff={global_config.get('p_double_diff')}\n"
        f"Monitor   : train loss per branch (min) | checkpoints: LAST + BEST only\n"
        f"Checkpoints: {checkpoint_root}"
    )
    print(LINE * 110)


def print_epoch_header(epoch: int, num_epochs: int) -> None:
    print_section(f"Epoch {epoch:03d}/{num_epochs - 1:03d}")


def print_branch_summary(
    branch_name: str,
    result: Dict[str, Any],
    elapsed: float,
    bundle: Dict[str, Any],
    monitor_key: str,
    metric_value: Optional[float],
    best_metric: Optional[float],
    improved: bool,
) -> None:
    metrics = result.get("epoch_metrics", {})
    lr = get_first_lr(bundle)

    print_section(f"{branch_name.upper()} epoch summary")

    print(
        f"step={result.get('global_step')} | "
        f"optim_step={result.get('optimizer_step')} | "
        f"time={format_seconds(elapsed)} | "
        f"lr={fmt_float(lr, precision=3, sci=True)}"
    )

    main_parts = []

    for k in [
        "loss/total",
        "loss/loss",
        "loss/full_loss",
        "loss/diff_loss",
        "loss/score_loss",
        "loss/zone_loss",
        "loss/age_loss",
        "loss/delta_age_loss",
        "loss/id_loss",
        "pred/score_mae",
        "pred/age_mae",
        "pred/id_similarity",
    ]:
        if k in metrics:
            main_parts.append(f"{k}={fmt_float(metrics[k], precision=5)}")

    if len(main_parts) > 0:
        print("train -> " + " | ".join(main_parts))

    control_parts = []

    for k in [
        "local/source_score_mean",
        "local/target_score_mean",
        "local/score_delta_mean",
        "global/source_age_mean",
        "global/target_age_mean",
        "global/age_delta_mean",
        "mode_frac/aging",
        "mode_frac/anchor",
        "mode_frac/contrast",
        "mode_frac/mild_younger",
        "double_prompt/used",
    ]:
        if k in metrics:
            control_parts.append(f"{k}={fmt_float(metrics[k], precision=4)}")

    if len(control_parts) > 0:
        print("ctrl  -> " + " | ".join(control_parts))

    print(
        f"monitor -> {monitor_key}={fmt_float(metric_value, precision=6)} | "
        f"best={fmt_float(best_metric, precision=6)} | "
        f"improved={improved}"
    )

    print(
        f"steps -> micro={result.get('n_micro_steps')} | "
        f"optim_epoch={result.get('n_optimizer_steps_epoch')} | "
        f"double_prompt={result.get('double_prompt_steps', 0)} "
        f"({fmt_float(result.get('double_prompt_fraction', 0.0), precision=4)}) | "
        f"skipped={result.get('skipped_steps', 0)}"
    )


def print_checkpoint_report(branch_name: str, latest_paths: Dict[str, Path], best_paths: Dict[str, Path], improved: bool) -> None:
    print_section(f"{branch_name.upper()} checkpointing")

    for _, path in latest_paths.items():
        print(f"└─ [LAST] saved → {path}")

    if improved:
        for _, path in best_paths.items():
            print(f"└─ [BEST] saved → {path}")
    else:
        print("└─ [BEST] not updated")


# ============================================================
# Updated run header
# ============================================================

def print_run_header_v2(
    run_name: str,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: str,
    total_epochs: int,
    local_num_epochs: int,
    global_num_epochs: int,
    start_epoch: int,
    local_grad_accum_steps: int,
    global_grad_accum_steps: int,
    local_config: Dict[str, Any],
    global_config: Dict[str, Any],
    checkpoint_root: Path,
    sampling_enabled: bool,
    sample_every_epochs: int,
) -> None:
    print_box_title(f"Face Aging Global-Local Diffusion run: {run_name}")

    print(
        f"Device    : {device} | AMP: {amp_enabled} ({amp_dtype})\n"
        f"Schedule  : total_loop_epochs={total_epochs} | start_epoch={start_epoch} | "
        f"local_epochs={local_num_epochs} | global_epochs={global_num_epochs}\n"
        f"Local     : grad_accum={local_grad_accum_steps} | "
        f"p_full={local_config.get('p_full')} | "
        f"p_score={local_config.get('p_score')} | "
        f"p_zone={local_config.get('p_zone')} | "
        f"p_double_full={local_config.get('p_double_full')}\n"
        f"Global    : grad_accum={global_grad_accum_steps} | "
        f"p_diff={global_config.get('p_diff')} | "
        f"p_semantic={global_config.get('p_semantic')} | "
        f"p_double_diff={global_config.get('p_double_diff')}\n"
        f"Sampling  : enabled={sampling_enabled} | every={sample_every_epochs} epochs | deterministic fusion only\n"
        f"Monitor   : train loss per branch (min) | checkpoints: LAST + BEST only\n"
        f"Checkpoints: {checkpoint_root}"
    )
    print("─" * 110)
