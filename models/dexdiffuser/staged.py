from __future__ import annotations

import torch
from torch import nn

from models.basic_mlp import BasicMLP


class DexDiffuserStagedHead(nn.Module):
    """基于 predicted squeeze_pose 回归 init_pose 与 squeeze_joint。"""

    def __init__(
        self,
        point_feat_dim: int,
        squeeze_pose_dim: int,
        init_pose_dim: int,
        joint_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "leaky_relu",
        network_type: str = "residual",
        residual_num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.point_feat_dim = int(point_feat_dim)
        self.squeeze_pose_dim = int(squeeze_pose_dim)
        self.init_pose_dim = int(init_pose_dim)
        self.joint_dim = int(joint_dim)
        self.output_dim = self.init_pose_dim + self.joint_dim
        self.projection = BasicMLP(
            input_dim=self.point_feat_dim + self.squeeze_pose_dim,
            output_dim=self.output_dim,
            hidden_dims=list(hidden_dims or [256, 256]),
            activation=activation,
            network_type=network_type,
            residual_num_blocks=int(residual_num_blocks),
        )

    def forward(
        self,
        global_feature: torch.Tensor,
        squeeze_pose: torch.Tensor,
    ) -> torch.Tensor:
        if global_feature.ndim != 2:
            raise ValueError(
                "DexDiffuserStagedHead expects global_feature with shape [B, C], "
                f"got {tuple(global_feature.shape)}."
            )
        if global_feature.shape[-1] != self.point_feat_dim:
            raise ValueError(
                f"Expected global feature dim {self.point_feat_dim}, got {global_feature.shape[-1]}."
            )
        if squeeze_pose.shape[-1] != self.squeeze_pose_dim:
            raise ValueError(
                f"Expected squeeze pose dim {self.squeeze_pose_dim}, got {squeeze_pose.shape[-1]}."
            )

        if squeeze_pose.ndim == 2:
            fused_feature = torch.cat([global_feature, squeeze_pose], dim=-1)
            return self.projection(fused_feature)

        if squeeze_pose.ndim != 3:
            raise ValueError(
                "DexDiffuserStagedHead expects squeeze_pose with shape [B, D] or [B, K, D], "
                f"got {tuple(squeeze_pose.shape)}."
            )

        batch_size, num_samples, _ = squeeze_pose.shape
        repeated_feature = (
            global_feature[:, None, :]
            .expand(batch_size, num_samples, self.point_feat_dim)
            .reshape(batch_size * num_samples, self.point_feat_dim)
        )
        flat_pose = squeeze_pose.reshape(batch_size * num_samples, self.squeeze_pose_dim)
        prediction = self.projection(torch.cat([repeated_feature, flat_pose], dim=-1))
        return prediction.reshape(batch_size, num_samples, self.output_dim)
