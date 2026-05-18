# ============================================================
# LOCAL SCORENET / MINI EFFICIENTNET-LIKE CNN
# Predicts local aging score from crop image
#
# Input:
#   x: [B, 3, 256, 256] in [-1, 1]
#
# Output:
#   score_pred: [B] in [0, 1]
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(8, channels // reduction)

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.net(x)
        return x * scale


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expansion=2,
        use_se=True,
        dropout=0.0,
    ):
        super().__init__()

        hidden_channels = in_channels * expansion
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        self.se = SqueezeExcitation(hidden_channels) if use_se else nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        out = self.dropout(out)

        if self.use_residual:
            out = out + x

        return out
    

class LocalScoreNet(nn.Module):
    """
    Lightweight CNN regressor for local aging score.

    Input:
        crop image [B, 3, 256, 256] in [-1, 1]

    Output:
        score_pred [B] in [0, 1]
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        dropout=0.15,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            # 256 -> 128
            DepthwiseSeparableConv(base_channels, 48, stride=1, expansion=2, dropout=0.00),

            # 128 -> 64
            DepthwiseSeparableConv(48, 64, stride=2, expansion=2, dropout=0.02),
            DepthwiseSeparableConv(64, 64, stride=1, expansion=2, dropout=0.02),

            # 64 -> 32
            DepthwiseSeparableConv(64, 96, stride=2, expansion=2, dropout=0.04),
            DepthwiseSeparableConv(96, 96, stride=1, expansion=2, dropout=0.04),

            # 32 -> 16
            DepthwiseSeparableConv(96, 128, stride=2, expansion=2, dropout=0.06),
            DepthwiseSeparableConv(128, 128, stride=1, expansion=2, dropout=0.06),

            # 16 -> 8
            DepthwiseSeparableConv(128, 192, stride=2, expansion=2, dropout=0.08),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(192),
            nn.Dropout(dropout),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x comes in [-1, 1].
        # CNNs are okay with this, but mapping to [0, 1] can stabilize early training.
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)

        x = self.stem(x)
        x = self.blocks(x)
        logits = self.head(x).squeeze(-1)

        # score in [0, 1]
        score_pred = torch.sigmoid(logits)

        return score_pred