from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * float(sigma) ** 2))
    g = g / g.sum().clamp_min(1e-8)
    kernel = torch.outer(g, g)
    return kernel.view(1, 1, kernel_size, kernel_size)


def gaussian_blur(x: torch.Tensor, kernel_size: int = 21, sigma: float = 5.0) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=x.shape[1])


def differentiable_fusion_train(
    x_orig: torch.Tensor,
    x_global: torch.Tensor,
    aged_crops: torch.Tensor,
    boxes: torch.Tensor,
    masks: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    alpha_global: float = 0.35,
    residual_sigma: float = 9.0,
) -> Dict[str, torch.Tensor]:
    """
    Differentiable tensor-only global/local fusion path for training.

    Inputs are expected in [-1, 1]. Boxes use full-image coordinates:
    [left, top, right, bottom].
    """
    if x_orig.ndim != 4 or x_global.ndim != 4:
        raise ValueError("x_orig and x_global must be [B, 3, H, W].")
    if aged_crops.ndim != 5:
        raise ValueError("aged_crops must be [B, K, 3, h, w].")

    x_global = x_global.detach()
    residual_low = gaussian_blur(
        x_global - x_orig,
        kernel_size=max(3, int(round(residual_sigma * 4)) | 1),
        sigma=float(residual_sigma),
    )
    x_coarse = (x_orig + float(alpha_global) * residual_low).clamp(-1.0, 1.0)
    x_final = x_coarse.clone()
    edge_masks = torch.zeros(x_orig.shape[0], 1, x_orig.shape[2], x_orig.shape[3], device=x_orig.device, dtype=x_orig.dtype)

    B, K = aged_crops.shape[:2]
    if valid_mask is None:
        valid_mask = torch.ones(B, K, device=x_orig.device, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=x_orig.device, dtype=torch.bool)

    H, W = x_orig.shape[-2:]

    for b in range(B):
        for k in range(K):
            if not bool(valid_mask[b, k].item()):
                continue

            left, top, right, bottom = [int(v) for v in boxes[b, k].detach().cpu().tolist()]
            left = max(0, min(left, W - 1))
            right = max(left + 1, min(right, W))
            top = max(0, min(top, H - 1))
            bottom = max(top + 1, min(bottom, H))

            patch_h = bottom - top
            patch_w = right - left

            crop = F.interpolate(
                aged_crops[b, k].unsqueeze(0),
                size=(patch_h, patch_w),
                mode="bilinear",
                align_corners=False,
            )
            mask = F.interpolate(
                masks[b, k].unsqueeze(0).to(device=x_orig.device, dtype=x_orig.dtype),
                size=(patch_h, patch_w),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)

            base_patch = x_final[b:b + 1, :, top:bottom, left:right]
            blended = mask * crop + (1.0 - mask) * base_patch
            x_final[b:b + 1, :, top:bottom, left:right] = blended

            large = gaussian_blur(mask, kernel_size=21, sigma=5.0)
            small = gaussian_blur(mask, kernel_size=7, sigma=1.5)
            edge = (large - small).abs()
            edge = edge / edge.amax().clamp_min(1e-8)
            current = edge_masks[b:b + 1, :, top:bottom, left:right]
            edge_masks[b:b + 1, :, top:bottom, left:right] = torch.maximum(current, edge)

    return {
        "x_coarse": x_coarse,
        "x_final": x_final.clamp(-1.0, 1.0),
        "edge_masks": edge_masks,
    }


def crop_regions_from_full_image(
    x: torch.Tensor,
    boxes: torch.Tensor,
    valid_mask: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    crops = []
    H, W = x.shape[-2:]
    for b in range(x.shape[0]):
        for k in range(boxes.shape[1]):
            if not bool(valid_mask[b, k].item()):
                continue
            left, top, right, bottom = [int(v) for v in boxes[b, k].detach().cpu().tolist()]
            left = max(0, min(left, W - 1))
            right = max(left + 1, min(right, W))
            top = max(0, min(top, H - 1))
            bottom = max(top + 1, min(bottom, H))
            crop = x[b:b + 1, :, top:bottom, left:right]
            crop = F.interpolate(crop, size=(output_size, output_size), mode="bilinear", align_corners=False)
            crops.append(crop.squeeze(0))

    if not crops:
        raise ValueError("No valid crops were available for fused score loss.")
    return torch.stack(crops, dim=0)


class LocalFusedFusionLoss(nn.Module):
    def __init__(
        self,
        score_net,
        local_resolution: int = 256,
        lambda_fuse_score: float = 0.03,
        lambda_fuse_seam: float = 0.01,
        alpha_global: float = 0.35,
        residual_sigma: float = 9.0,
    ):
        super().__init__()
        self.score_net = score_net
        self.local_resolution = int(local_resolution)
        self.lambda_fuse_score = float(lambda_fuse_score)
        self.lambda_fuse_seam = float(lambda_fuse_seam)
        self.alpha_global = float(alpha_global)
        self.residual_sigma = float(residual_sigma)

        if self.lambda_fuse_score > 0.0 and self.score_net is None:
            raise ValueError("lambda_fuse_score > 0 requires a frozen score_net.")

        if self.score_net is not None:
            self.score_net.eval()
            for p in self.score_net.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        x_orig: torch.Tensor,
        x_global: torch.Tensor,
        aged_crops: torch.Tensor,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        target_scores: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        fused = differentiable_fusion_train(
            x_orig=x_orig,
            x_global=x_global.detach(),
            aged_crops=aged_crops,
            boxes=boxes,
            masks=masks,
            valid_mask=valid_mask,
            alpha_global=self.alpha_global,
            residual_sigma=self.residual_sigma,
        )

        if valid_mask is None:
            valid_mask = torch.ones_like(target_scores, dtype=torch.bool)
        valid_mask = valid_mask.to(device=x_orig.device, dtype=torch.bool)
        target = target_scores.to(device=x_orig.device, dtype=torch.float32)[valid_mask].view(-1)

        loss_fuse_score = x_orig.new_zeros(())
        score_pred = None
        if self.lambda_fuse_score > 0.0:
            fused_crops = crop_regions_from_full_image(
                fused["x_final"],
                boxes=boxes,
                valid_mask=valid_mask,
                output_size=self.local_resolution,
            )
            score_pred = self.score_net(fused_crops.float()).view(-1)
            loss_fuse_score = F.mse_loss(score_pred.float(), target.float())

        edge = fused["edge_masks"].detach()
        loss_fuse_seam = (edge * (fused["x_final"] - fused["x_coarse"]).abs()).mean()

        total = self.lambda_fuse_score * loss_fuse_score + self.lambda_fuse_seam * loss_fuse_seam

        out = {
            "loss": total,
            "loss_fuse_score": loss_fuse_score.detach(),
            "loss_fuse_seam": loss_fuse_seam.detach(),
            "score_pred": score_pred.detach() if score_pred is not None else None,
            "target_scores": target.detach(),
            "x_final": fused["x_final"].detach(),
            "x_coarse": fused["x_coarse"].detach(),
        }

        if score_pred is not None:
            out["fused_score_pred_mean"] = score_pred.detach().float().mean()
            out["fused_score_target_mean"] = target.detach().float().mean()
            out["fused_score_mae"] = (score_pred.detach().float() - target.detach().float()).abs().mean()

        return out
