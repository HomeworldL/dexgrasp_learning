from __future__ import annotations

import torch
from torch import nn

from models.basic_mlp import BasicMLP


class UDGMStagedHead(nn.Module):
    """基于 squeeze_pose 与全局条件回归 init_pose 和 squeeze_joint。"""

    def __init__(
        self,
        condition_dim: int,
        squeeze_pose_dim: int,
        init_pose_dim: int,
        joint_dim: int,
        *,
        hidden_dims: list[int] | None = None,
        activation: str = "leaky_relu",
        network_type: str = "residual",
        residual_num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.squeeze_pose_dim = int(squeeze_pose_dim)
        self.init_pose_dim = int(init_pose_dim)
        self.joint_dim = int(joint_dim)
        self.output_dim = self.init_pose_dim + self.joint_dim
        self.projection = BasicMLP(
            input_dim=self.condition_dim + self.squeeze_pose_dim,
            output_dim=self.output_dim,
            hidden_dims=list(hidden_dims or [256, 256]),
            activation=activation,
            network_type=network_type,
            residual_num_blocks=int(residual_num_blocks),
        )

    def forward(
        self,
        condition: torch.Tensor,
        squeeze_pose: torch.Tensor,
    ) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError(
                f"UDGMStagedHead expects condition [B, C], got {tuple(condition.shape)}."
            )
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected condition dim {self.condition_dim}, got {condition.shape[-1]}."
            )
        if squeeze_pose.shape[-1] != self.squeeze_pose_dim:
            raise ValueError(
                f"Expected squeeze pose dim {self.squeeze_pose_dim}, got {squeeze_pose.shape[-1]}."
            )
        if squeeze_pose.ndim == 2:
            return self.projection(torch.cat([condition, squeeze_pose], dim=-1))
        if squeeze_pose.ndim != 3:
            raise ValueError(
                "UDGMStagedHead expects squeeze_pose with shape [B, D] or [B, K, D], "
                f"got {tuple(squeeze_pose.shape)}."
            )
        batch_size, num_samples, _ = squeeze_pose.shape
        repeated_condition = (
            condition[:, None, :]
            .expand(batch_size, num_samples, self.condition_dim)
            .reshape(batch_size * num_samples, self.condition_dim)
        )
        flat_pose = squeeze_pose.reshape(batch_size * num_samples, self.squeeze_pose_dim)
        prediction = self.projection(torch.cat([repeated_condition, flat_pose], dim=-1))
        return prediction.reshape(batch_size, num_samples, self.output_dim)
