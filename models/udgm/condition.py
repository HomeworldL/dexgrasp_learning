from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from models.udgm.utils import build_mlp


class UDGMConditionAdapter(nn.Module):
    """把 backbone 全局特征映射到 flow 条件空间。"""

    def __init__(
        self,
        point_feat_dim: int,
        condition_dim: int,
        hidden_dims: Iterable[int] | None = None,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.point_feat_dim = int(point_feat_dim)
        self.condition_dim = int(condition_dim)
        self.projection = build_mlp(
            input_dim=self.point_feat_dim,
            hidden_dims=list(hidden_dims or [self.point_feat_dim]),
            output_dim=self.condition_dim,
            activation=activation,
        )

    def forward(self, global_feature: torch.Tensor) -> torch.Tensor:
        if global_feature.ndim != 2:
            raise ValueError(
                "UDGMConditionAdapter expects global_feature with shape [B, C], "
                f"got {tuple(global_feature.shape)}."
            )
        if global_feature.shape[-1] != self.point_feat_dim:
            raise ValueError(
                f"Expected point feature dim {self.point_feat_dim}, got {global_feature.shape[-1]}."
            )
        return self.projection(global_feature)
