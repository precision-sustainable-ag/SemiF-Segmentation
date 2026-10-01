"""Point head: Detectron2 PointRend's StandardPointHead.

A per-point MLP (1x1 Conv1d) over fine-grained FPN features concatenated
with the coarse mask logits at the same points. The coarse logits are
re-appended to each hidden layer's output (coarse_pred_each_layer).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PointHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_mask_classes: int = 1,
        fc_dim: int = 256,
        num_fc: int = 3,
        coarse_pred_each_layer: bool = True,
    ):
        super().__init__()
        self.coarse_pred_each_layer = coarse_pred_each_layer
        fcs = []
        dim = in_channels + num_mask_classes
        for _ in range(num_fc):
            fc = nn.Conv1d(dim, fc_dim, kernel_size=1)
            nn.init.kaiming_normal_(fc.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(fc.bias)
            fcs.append(fc)
            dim = fc_dim + (num_mask_classes if coarse_pred_each_layer else 0)
        self.fcs = nn.ModuleList(fcs)
        self.predictor = nn.Conv1d(dim, num_mask_classes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, fine_grained_features: torch.Tensor, coarse_features: torch.Tensor) -> torch.Tensor:
        """(R, C_fine, P) and (R, num_mask_classes, P) -> (R, num_mask_classes, P) point logits."""
        x = torch.cat([fine_grained_features, coarse_features], dim=1)
        for fc in self.fcs:
            x = F.relu(fc(x))
            if self.coarse_pred_each_layer:
                x = torch.cat([x, coarse_features], dim=1)
        return self.predictor(x)
