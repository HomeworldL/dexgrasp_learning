from __future__ import annotations

import torch
import torch.nn as nn

from models.basic_mlp import BasicMLP


class MLPRTJ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.joint_mlp = BasicMLP(
            cfg.in_feat_dim + 9 + 3,
            (cfg.joint_num + 12) * cfg.traj_length,
            hidden_dims=[64, 64],
            network_type="residual",
            residual_num_blocks=2,
        )
        self.joint_loss = torch.nn.SmoothL1Loss(reduction="none")
        return

    def forward(self, data, global_feature):
        return

    def sample(self, data, global_feature, sample_num):
        return
