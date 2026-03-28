from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def _build_activation(name: str) -> nn.Module:
    activation_name = str(name).strip().lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    raise ValueError(f"Unsupported PointNet activation: {name}")


class PointNet(nn.Module):
    """最小 PointNet 编码器，只输出全局特征。"""

    def __init__(
        self,
        point_feature_dim: int,
        local_conv_hidden_dims: Iterable[int],
        global_mlp_hidden_dims: Iterable[int],
        output_dim: int,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        local_dims = [int(point_feature_dim), *[int(dim) for dim in local_conv_hidden_dims]]
        if len(local_dims) < 2:
            raise ValueError("local_conv_hidden_dims must contain at least one layer.")

        conv_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(local_dims[:-1], local_dims[1:]):
            conv_layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1))
            conv_layers.append(_build_activation(activation))
        self.local_encoder = nn.Sequential(*conv_layers)

        global_dims = [local_dims[-1], *[int(dim) for dim in global_mlp_hidden_dims], int(output_dim)]
        mlp_layers: list[nn.Module] = []
        for layer_index, (in_dim, out_dim) in enumerate(zip(global_dims[:-1], global_dims[1:])):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if layer_index + 1 < len(global_dims) - 1:
                mlp_layers.append(_build_activation(activation))
        self.global_encoder = nn.Sequential(*mlp_layers)

    def forward(self, point_cloud: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if point_cloud.ndim != 3:
            raise ValueError(
                f"PointNet expects point_cloud with shape [B, N, C], got {point_cloud.shape}."
            )
        features = point_cloud.transpose(1, 2).contiguous()
        local_feature = self.local_encoder(features)
        pooled = local_feature.max(dim=2)[0]
        global_feature = self.global_encoder(pooled)
        return global_feature, local_feature
