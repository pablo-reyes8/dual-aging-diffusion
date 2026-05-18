import torch

from src.score_net.arquitecture import DepthwiseSeparableConv, LocalScoreNet, SqueezeExcitation


def test_squeeze_excitation_and_depthwise_blocks_preserve_expected_shapes():
    x = torch.randn(2, 8, 16, 16)
    se = SqueezeExcitation(8)
    assert se(x).shape == x.shape

    residual_block = DepthwiseSeparableConv(8, 8, stride=1)
    assert residual_block(x).shape == x.shape

    downsample_block = DepthwiseSeparableConv(8, 16, stride=2)
    assert downsample_block(x).shape == (2, 16, 8, 8)


def test_local_scorenet_outputs_batch_scores_in_unit_interval():
    model = LocalScoreNet(base_channels=8, dropout=0.0)
    model.eval()
    x = torch.randn(2, 3, 64, 64).clamp(-1, 1)

    with torch.no_grad():
        scores = model(x)

    assert scores.shape == (2,)
    assert torch.all(scores >= 0)
    assert torch.all(scores <= 1)
