"""Coarse mask head: Detectron2 PointRend's ConvFCHead.

A light head that turns ROIAlign features (C, 2S, 2S) into low-resolution
(S x S, 7 x 7 by default) mask logits, which the point head then refines.
It is FC-based, like the box head, so every output cell sees the whole box.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CoarseMaskHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_resolution: int = 14,
        num_mask_classes: int = 1,
        conv_dim: int = 256,
        fc_dim: int = 1024,
        num_fc: int = 2,
        out_resolution: int = 7,
    ):
        super().__init__()
        if in_resolution % 2:
            raise ValueError(f"in_resolution must be even (it is halved by a stride-2 conv), got {in_resolution}")
        self.num_mask_classes = num_mask_classes
        self.out_resolution = out_resolution

        self.reduce_channel_dim_conv = (
            nn.Conv2d(in_channels, conv_dim, kernel_size=1) if in_channels != conv_dim else None
        )
        self.reduce_spatial_dim_conv = nn.Conv2d(conv_dim, conv_dim, kernel_size=2, stride=2)
        fcs = []
        dim = conv_dim * (in_resolution // 2) ** 2
        for _ in range(num_fc):
            fcs.append(nn.Linear(dim, fc_dim))
            dim = fc_dim
        self.fcs = nn.ModuleList(fcs)
        self.prediction = nn.Linear(dim, num_mask_classes * out_resolution**2)

        for conv in (self.reduce_channel_dim_conv, self.reduce_spatial_dim_conv):
            if conv is not None:
                nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(conv.bias)
        for fc in self.fcs:
            nn.init.kaiming_uniform_(fc.weight, a=1)
            nn.init.zeros_(fc.bias)
        nn.init.normal_(self.prediction.weight, std=0.001)
        nn.init.zeros_(self.prediction.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(R, C, 2S, 2S) ROI features -> (R, num_mask_classes, S, S) logits."""
        if self.reduce_channel_dim_conv is not None:
            x = F.relu(self.reduce_channel_dim_conv(x))
        x = F.relu(self.reduce_spatial_dim_conv(x))
        x = torch.flatten(x, 1)
        for fc in self.fcs:
            x = F.relu(fc(x))
        S = self.out_resolution
        return self.prediction(x).view(-1, self.num_mask_classes, S, S)
