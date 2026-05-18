import torch

from src.loss.local_fused_loss import LocalFusedFusionLoss, differentiable_fusion_train


class TinyScoreNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.proj(x).mean(dim=(1, 2, 3))


def test_differentiable_fusion_preserves_gradient_to_aged_crops():
    x_orig = torch.zeros(1, 3, 64, 64)
    x_global = torch.zeros(1, 3, 64, 64, requires_grad=True)
    aged_crops = torch.randn(1, 2, 3, 16, 16, requires_grad=True)
    boxes = torch.tensor([[[8, 8, 24, 24], [30, 30, 46, 46]]])
    masks = torch.ones(1, 2, 1, 16, 16)
    valid_mask = torch.tensor([[True, True]])

    out = differentiable_fusion_train(
        x_orig=x_orig,
        x_global=x_global.detach(),
        aged_crops=aged_crops,
        boxes=boxes,
        masks=masks,
        valid_mask=valid_mask,
    )
    loss = out["x_final"].mean()
    loss.backward()

    assert out["x_final"].shape == x_orig.shape
    assert aged_crops.grad is not None
    assert torch.isfinite(aged_crops.grad).all()
    assert x_global.grad is None


def test_local_fused_loss_forward_freezes_scorenet_and_keeps_loss_attached():
    score_net = TinyScoreNet()
    loss_fn = LocalFusedFusionLoss(
        score_net=score_net,
        local_resolution=16,
        lambda_fuse_score=0.03,
        lambda_fuse_seam=0.01,
    )

    x_orig = torch.zeros(1, 3, 64, 64)
    x_global = torch.zeros(1, 3, 64, 64)
    aged_crops = torch.randn(1, 1, 3, 16, 16, requires_grad=True)
    boxes = torch.tensor([[[8, 8, 24, 24]]])
    masks = torch.ones(1, 1, 1, 16, 16)
    target_scores = torch.tensor([[0.75]])
    valid_mask = torch.tensor([[True]])

    out = loss_fn(
        x_orig=x_orig,
        x_global=x_global,
        aged_crops=aged_crops,
        boxes=boxes,
        masks=masks,
        target_scores=target_scores,
        valid_mask=valid_mask,
    )

    assert out["loss"].requires_grad is True
    assert torch.isfinite(out["loss"]).all()
    assert torch.isfinite(out["loss_fuse_score"]).all()
    assert torch.isfinite(out["loss_fuse_seam"]).all()
    assert all(not p.requires_grad for p in score_net.parameters())

    out["loss"].backward()
    assert aged_crops.grad is not None
    assert torch.isfinite(aged_crops.grad).all()
