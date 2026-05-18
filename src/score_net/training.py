# ============================================================
# BASIC BUT SOLID SCORENET TRAINING LOOP
# Training by epochs, printing/evaluating by optimizer steps.
#
# Assumes already defined:
#   - score_net
#   - train_loader
#   - val_loader
#   - device
#   - print_gpu_mem
# ============================================================

import math
import torch
import torch.nn.functional as F
from src.utils.cuda_utils import * 
from pathlib import Path

# ============================================================
# Helpers
# ============================================================

def batch_corrcoef(pred, target, eps=1e-8):
    pred = pred.view(-1).float()
    target = target.view(-1).float()

    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()

    cov = (pred_centered * target_centered).mean()
    pred_std = pred_centered.std(unbiased=False)
    target_std = target_centered.std(unbiased=False)

    return cov / (pred_std * target_std + eps)


def batch_r2_score(pred, target, eps=1e-8):
    pred = pred.view(-1).float()
    target = target.view(-1).float()

    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()

    return 1.0 - ss_res / (ss_tot + eps)


def scorenet_extreme_aware_loss(
    pred,
    y,
    mae_weight=0.25,
    std_weight=0.05,
    high_tail_weight=4.0,
    high_under_weight=3.0,
    high_threshold=0.70,
    high_power=2.0,
    eps=1e-8,
):
    """
    Extreme-aware regression loss for local aging scores in [0, 1].

    Components:
    1. Weighted MSE:
       Gives larger weight to high aging scores.

    2. MAE:
       Stabilizes absolute calibration.

    3. STD loss:
       Penalizes range compression, encouraging pred std to match target std.

    4. High-underprediction loss:
       Strongly penalizes cases where y is high but pred is too low.
       This is useful because the model was regressing high scores toward the mean.
    """

    pred = pred.view(-1).float()
    y = y.view(-1).float()

    # ------------------------------------------------------------
    # Weighted MSE: high targets matter more.
    # ------------------------------------------------------------
    # Smooth weighting across the whole [0, 1] range.
    # y=0   -> weight approx 1
    # y=1   -> weight approx 1 + high_tail_weight
    target_weight = 1.0 + high_tail_weight * torch.clamp(y, 0.0, 1.0).pow(high_power)

    weighted_mse = (target_weight * (pred - y).pow(2)).mean()

    # ------------------------------------------------------------
    #  MAE: improves calibration.
    # ------------------------------------------------------------
    mae = F.l1_loss(pred, y)

    # ------------------------------------------------------------
    # STD loss: prevents prediction collapse toward the mean.
    # ------------------------------------------------------------
    pred_std = pred.std(unbiased=False)
    y_std = y.std(unbiased=False)
    std_loss = torch.abs(pred_std - y_std)

    # ------------------------------------------------------------
    # High-underprediction penalty.
    # ------------------------------------------------------------
    # Only activates strongly for y >= high_threshold.
    # Penalizes pred < y, but does not punish overprediction here.
    high_mask = (y >= high_threshold).float()

    under_error = F.relu(y - pred).pow(2)

    high_under_loss = (high_mask * under_error).sum() / (high_mask.sum() + eps)

    # ------------------------------------------------------------
    # Total loss.
    # ------------------------------------------------------------
    loss = (
        weighted_mse
        + mae_weight * mae
        + std_weight * std_loss
        + high_under_weight * high_under_loss
    )

    loss_parts = {
        "loss": loss.detach(),
        "weighted_mse": weighted_mse.detach(),
        "mae": mae.detach(),
        "std_loss": std_loss.detach(),
        "high_under_loss": high_under_loss.detach(),
        "pred_std": pred_std.detach(),
        "target_std": y_std.detach(),
        "mean_weight": target_weight.mean().detach(),
        "max_weight": target_weight.max().detach(),
        "n_high": high_mask.sum().detach(),
    }

    return loss, loss_parts


@torch.no_grad()
def evaluate_score_net(
    score_net,
    loader,
    device,
    max_batches=None,
):
    score_net.eval()

    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        x = batch["pixel_values"].to(device=device, dtype=torch.float32)
        y = batch["score"].to(device=device, dtype=torch.float32).view(-1)

        pred = score_net(x).view(-1)

        all_preds.append(pred.detach().cpu())
        all_targets.append(y.detach().cpu())

    pred = torch.cat(all_preds, dim=0).float()
    y = torch.cat(all_targets, dim=0).float()

    mse = F.mse_loss(pred, y)
    rmse = torch.sqrt(mse)
    mae = F.l1_loss(pred, y)
    corr = batch_corrcoef(pred, y)
    r2 = batch_r2_score(pred, y)

    high_mask = y >= 0.70

    if high_mask.any():
        high_mae = F.l1_loss(pred[high_mask], y[high_mask])
        high_mse = F.mse_loss(pred[high_mask], y[high_mask])
        high_under_mae = F.relu(y[high_mask] - pred[high_mask]).mean()
    else:
        high_mae = torch.tensor(float("nan"))
        high_mse = torch.tensor(float("nan"))
        high_under_mae = torch.tensor(float("nan"))

    metrics = {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "corr": corr.item(),
        "r2": r2.item(),

        "target_min": y.min().item(),
        "target_mean": y.mean().item(),
        "target_max": y.max().item(),
        "target_std": y.std(unbiased=False).item(),

        "pred_min": pred.min().item(),
        "pred_mean": pred.mean().item(),
        "pred_max": pred.max().item(),
        "pred_std": pred.std(unbiased=False).item(),

        "std_ratio": (pred.std(unbiased=False) / (y.std(unbiased=False) + 1e-8)).item(),

        "high_mae": high_mae.item(),
        "high_mse": high_mse.item(),
        "high_under_mae": high_under_mae.item(),
        "n_high": int(high_mask.sum().item()),
    }

    return metrics


def print_score_metrics(prefix, metrics):
    print(f"{prefix}_mse:        {metrics['mse']:.6f}")
    print(f"{prefix}_rmse:       {metrics['rmse']:.6f}")
    print(f"{prefix}_mae:        {metrics['mae']:.6f}")
    print(f"{prefix}_corr:       {metrics['corr']:.4f}")
    print(f"{prefix}_r2:         {metrics['r2']:.4f}")

    print(
        f"{prefix}_target min/mean/max/std: "
        f"{metrics['target_min']:.4f} / "
        f"{metrics['target_mean']:.4f} / "
        f"{metrics['target_max']:.4f} / "
        f"{metrics['target_std']:.4f}"
    )

    print(
        f"{prefix}_pred   min/mean/max/std: "
        f"{metrics['pred_min']:.4f} / "
        f"{metrics['pred_mean']:.4f} / "
        f"{metrics['pred_max']:.4f} / "
        f"{metrics['pred_std']:.4f}"
    )

    print(f"{prefix}_std_ratio pred/target: {metrics['std_ratio']:.4f}")

    print(
        f"{prefix}_high_targets y>=0.70 | "
        f"n={metrics['n_high']} | "
        f"high_mae={metrics['high_mae']:.6f} | "
        f"high_mse={metrics['high_mse']:.6f} | "
        f"high_under_mae={metrics['high_under_mae']:.6f}"
    )

def compute_score_selection_metric(metrics):
    """
    Composite metric for selecting the best ScoreNet checkpoint.

    Higher is better.

    We care about:
        - high correlation
        - high R2
        - low MAE
        - low high-score underprediction
        - reasonable prediction variance

    This is not a pure validation-generalization criterion.
    It is designed for selecting a useful auxiliary signal model.
    """
    corr = float(metrics["corr"])
    r2 = float(metrics["r2"])
    mae = float(metrics["mae"])
    high_under_mae = float(metrics["high_under_mae"])
    std_ratio = float(metrics["std_ratio"])

    # Penalize strong variance collapse or excessive variance.
    std_penalty = abs(1.0 - std_ratio)

    score = (
        1.00 * corr
        + 0.50 * r2
        - 0.75 * mae
        - 0.75 * high_under_mae
        - 0.25 * std_penalty
    )

    return float(score)


def merge_train_val_selection_score(train_metrics, val_metrics):
    """
    Best-overall criterion.

    Since this ScoreNet is an auxiliary teacher/signal model, we do not want
    to select only by validation. We want a model that is strong on train while
    not being terrible on validation.

    Higher is better.
    """
    train_score = compute_score_selection_metric(train_metrics)

    if val_metrics is not None:
        val_score = compute_score_selection_metric(val_metrics)

        # Weighted toward train because the model is not intended as a
        # standalone general-purpose inference model.
        overall_score = 0.65 * train_score + 0.35 * val_score
    else:
        val_score = None
        overall_score = train_score

    return {
        "overall_score": float(overall_score),
        "train_score": float(train_score),
        "val_score": None if val_score is None else float(val_score),
    }


def save_score_net_weights_only(
    path,
    score_net,
    epoch,
    global_step,
    train_metrics=None,
    val_metrics=None,
    selection_scores=None,
):
    """
    Saves only what we need for forward later:
        - model state_dict
        - minimal metadata

    No optimizer.
    No scheduler.
    No giant training state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": score_net.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "selection_scores": selection_scores,
    }

    torch.save(payload, path)



def train_score_net_epochs(
    score_net,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=20,
    print_every=25,
    eval_every=100,
    grad_clip=1.0,

    # Loss hyperparameters.
    mae_weight=0.25,
    std_weight=0.05,
    high_tail_weight=4.0,
    high_under_weight=3.0,
    high_threshold=0.70,
    high_power=2.0,

    # Checkpointing.
    checkpoint_dir="/content/score_net_checkpoints",
    best_filename="score_net_best_overall.pt",
    last_filename="score_net_last.pt",

    # For best-overall selection.
    # If None, evaluates the whole train_loader.
    # If train_loader is too slow, use e.g. 30 or 50.
    train_eval_max_batches=None,
):
    score_net.to(device=device, dtype=torch.float32)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir / best_filename
    last_path = checkpoint_dir / last_filename

    global_step = 0
    best_overall_score = -float("inf")

    running_loss = 0.0
    running_weighted_mse = 0.0
    running_mae_loss = 0.0
    running_std_loss = 0.0
    running_high_under_loss = 0.0

    running_mae = 0.0
    running_corr = 0.0
    running_r2 = 0.0
    running_batches = 0

    print("\n========== START SCORENET TRAINING ==========")
    print("Training mode: epochs")
    print("num_epochs:", num_epochs)
    print("print_every steps:", print_every)
    print("eval_every steps:", eval_every)
    print("device:", device)

    print("\n[Checkpointing]")
    print("checkpoint_dir:", str(checkpoint_dir))
    print("best_overall:", str(best_path))
    print("last:", str(last_path))
    print("train_eval_max_batches:", train_eval_max_batches)

    print("\n[Loss config]")
    print(f"mae_weight:        {mae_weight}")
    print(f"std_weight:        {std_weight}")
    print(f"high_tail_weight:  {high_tail_weight}")
    print(f"high_under_weight: {high_under_weight}")
    print(f"high_threshold:    {high_threshold}")
    print(f"high_power:        {high_power}")

    for epoch in range(1, num_epochs + 1):
        score_net.train()

        print("\n==================================================")
        print(f"EPOCH {epoch}/{num_epochs}")
        print("==================================================")

        for batch_idx, batch in enumerate(train_loader, start=1):
            global_step += 1

            x = batch["pixel_values"].to(device=device, dtype=torch.float32)
            y = batch["score"].to(device=device, dtype=torch.float32).view(-1)

            optimizer.zero_grad(set_to_none=True)

            pred = score_net(x).view(-1)

            loss, loss_parts = scorenet_extreme_aware_loss(
                pred=pred,
                y=y,
                mae_weight=mae_weight,
                std_weight=std_weight,
                high_tail_weight=high_tail_weight,
                high_under_weight=high_under_weight,
                high_threshold=high_threshold,
                high_power=high_power,
            )

            # Diagnostics.
            plain_mse = F.mse_loss(pred.detach(), y.detach())
            mae = F.l1_loss(pred.detach(), y.detach())
            corr = batch_corrcoef(pred.detach(), y.detach())
            r2 = batch_r2_score(pred.detach(), y.detach())

            loss.backward()

            if grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    score_net.parameters(),
                    max_norm=grad_clip,
                )
            else:
                grad_norm = torch.tensor(float("nan"))

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            running_weighted_mse += loss_parts["weighted_mse"].item()
            running_mae_loss += loss_parts["mae"].item()
            running_std_loss += loss_parts["std_loss"].item()
            running_high_under_loss += loss_parts["high_under_loss"].item()

            running_mae += mae.item()
            running_corr += corr.item()
            running_r2 += r2.item()
            running_batches += 1

            if global_step == 1 or global_step % print_every == 0:
                lr = optimizer.param_groups[0]["lr"]

                avg_loss = running_loss / max(running_batches, 1)
                avg_weighted_mse = running_weighted_mse / max(running_batches, 1)
                avg_mae_loss = running_mae_loss / max(running_batches, 1)
                avg_std_loss = running_std_loss / max(running_batches, 1)
                avg_high_under_loss = running_high_under_loss / max(running_batches, 1)

                avg_mae = running_mae / max(running_batches, 1)
                avg_corr = running_corr / max(running_batches, 1)
                avg_r2 = running_r2 / max(running_batches, 1)

                print("\n--------------------------------------------------")
                print(f"[Step {global_step:05d} | Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)}]")
                print(f"lr:                    {lr:.8f}")

                print("\n[Train objective]")
                print(f"avg_total_loss:        {avg_loss:.6f}")
                print(f"avg_weighted_mse:      {avg_weighted_mse:.6f}")
                print(f"avg_mae_loss:          {avg_mae_loss:.6f}")
                print(f"avg_std_loss:          {avg_std_loss:.6f}")
                print(f"avg_high_under_loss:   {avg_high_under_loss:.6f}")

                print("\n[Train diagnostics]")
                print(f"last_plain_mse:        {plain_mse.item():.6f}")
                print(f"last_plain_rmse:       {math.sqrt(plain_mse.item()):.6f}")
                print(f"avg_mae:               {avg_mae:.6f}")
                print(f"avg_corr:              {avg_corr:.4f}")
                print(f"avg_r2:                {avg_r2:.4f}")
                print(f"last_mae:              {mae.item():.6f}")
                print(f"last_corr:             {corr.item():.4f}")
                print(f"last_r2:               {r2.item():.4f}")
                print(f"grad_norm:             {float(grad_norm):.4f}")

                print("\n[Loss internals - current batch]")
                print(f"mean_weight:           {loss_parts['mean_weight'].item():.4f}")
                print(f"max_weight:            {loss_parts['max_weight'].item():.4f}")
                print(f"n_high y>={high_threshold}:       {int(loss_parts['n_high'].item())}")

                print("\n[Current batch ranges]")
                print(
                    f"target min/mean/max/std: "
                    f"{y.min().item():.4f} / "
                    f"{y.mean().item():.4f} / "
                    f"{y.max().item():.4f} / "
                    f"{y.std(unbiased=False).item():.4f}"
                )
                print(
                    f"pred   min/mean/max/std: "
                    f"{pred.min().item():.4f} / "
                    f"{pred.mean().item():.4f} / "
                    f"{pred.max().item():.4f} / "
                    f"{pred.std(unbiased=False).item():.4f}"
                )

                std_ratio = pred.detach().std(unbiased=False) / (y.detach().std(unbiased=False) + 1e-8)
                print(f"std_ratio pred/target: {std_ratio.item():.4f}")

                high_mask = y >= high_threshold
                if high_mask.any():
                    high_mae = F.l1_loss(pred.detach()[high_mask], y.detach()[high_mask])
                    high_under_mae = F.relu(y.detach()[high_mask] - pred.detach()[high_mask]).mean()
                    print(
                        f"high targets y>={high_threshold} | "
                        f"n={int(high_mask.sum().item())} | "
                        f"high_mae={high_mae.item():.6f} | "
                        f"high_under_mae={high_under_mae.item():.6f}"
                    )

                print("\n[First predictions]")
                n_show = min(8, y.shape[0])
                for i in range(n_show):
                    print(f"  y={y[i].item():.3f} | pred={pred[i].item():.3f}")

                running_loss = 0.0
                running_weighted_mse = 0.0
                running_mae_loss = 0.0
                running_std_loss = 0.0
                running_high_under_loss = 0.0

                running_mae = 0.0
                running_corr = 0.0
                running_r2 = 0.0
                running_batches = 0

            if val_loader is not None and (global_step == 1 or global_step % eval_every == 0):
                print("\n================ VALIDATION ================")
                val_metrics = evaluate_score_net(
                    score_net=score_net,
                    loader=val_loader,
                    device=device,
                    max_batches=None,
                )
                print(f"[Validation at step {global_step:05d}]")
                print_score_metrics("val", val_metrics)

                score_net.train()

        # ========================================================
        # End-of-epoch evaluation + checkpointing
        # ========================================================

        print("\n================ END OF EPOCH TRAIN EVAL ================")
        train_metrics_epoch = evaluate_score_net(
            score_net=score_net,
            loader=train_loader,
            device=device,
            max_batches=train_eval_max_batches,
        )
        print(f"[Train eval epoch {epoch}/{num_epochs} | step {global_step:05d}]")
        print_score_metrics("train_eval", train_metrics_epoch)

        if val_loader is not None:
            print("\n================ END OF EPOCH VALIDATION ================")
            val_metrics_epoch = evaluate_score_net(
                score_net=score_net,
                loader=val_loader,
                device=device,
                max_batches=None,
            )
            print(f"[Val eval epoch {epoch}/{num_epochs} | step {global_step:05d}]")
            print_score_metrics("val", val_metrics_epoch)
        else:
            val_metrics_epoch = None

        selection_scores = merge_train_val_selection_score(
            train_metrics=train_metrics_epoch,
            val_metrics=val_metrics_epoch,
        )

        overall_score = selection_scores["overall_score"]

        print("\n[Checkpoint selection scores]")
        print(f"overall_score: {selection_scores['overall_score']:.6f}")
        print(f"train_score:   {selection_scores['train_score']:.6f}")
        if selection_scores["val_score"] is not None:
            print(f"val_score:     {selection_scores['val_score']:.6f}")
        print(f"best_so_far:   {best_overall_score:.6f}")

        # Always overwrite last.
        save_score_net_weights_only(
            path=last_path,
            score_net=score_net,
            epoch=epoch,
            global_step=global_step,
            train_metrics=train_metrics_epoch,
            val_metrics=val_metrics_epoch,
            selection_scores=selection_scores,
        )
        print(f"[Saved last] {last_path}")

        # Overwrite best only if better.
        if overall_score > best_overall_score:
            best_overall_score = overall_score

            save_score_net_weights_only(
                path=best_path,
                score_net=score_net,
                epoch=epoch,
                global_step=global_step,
                train_metrics=train_metrics_epoch,
                val_metrics=val_metrics_epoch,
                selection_scores=selection_scores,
            )
            print(f"[Saved BEST overall] {best_path}")
        else:
            print("[BEST unchanged]")

        score_net.train()

    print("\n========== DONE SCORENET TRAINING ==========")
    print("Final global_step:", global_step)
    print("Best overall score:", best_overall_score)
    print("Best checkpoint:", str(best_path))
    print("Last checkpoint:", str(last_path))

    try:
        print_gpu_mem("[After ScoreNet training] ")
    except NameError:
        pass