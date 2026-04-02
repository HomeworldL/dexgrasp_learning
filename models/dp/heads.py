from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from nflows.nn.nets.resnet import ResidualNet

from models.dp.diffusion import GaussianDiffusion1D, MLPWrapper


class RunningMeanStd(nn.Module):
    def __init__(self, shape: int) -> None:
        super().__init__()
        self.register_buffer("n", torch.zeros(1))
        self.register_buffer("mean", torch.zeros((1, int(shape))))
        self.register_buffer("S", torch.ones((1, int(shape))) * 1e-4)
        self.register_buffer("std", torch.sqrt(self.S))

    def update(self, x: torch.Tensor) -> None:
        self.n += 1
        old_mean = self.mean.clone()
        new_mean = x.mean(dim=0, keepdim=True)
        self.mean = old_mean + (new_mean - old_mean) / self.n
        self.S = (
            self.S
            + (x - new_mean).pow(2).mean(dim=0, keepdim=True)
            + (old_mean - new_mean).pow(2) * (self.n - 1) / self.n
        )
        self.std = torch.sqrt(self.S / self.n)


class Normalization(nn.Module):
    def __init__(self, shape: int, max_update: int = 2000) -> None:
        super().__init__()
        self.running_ms = RunningMeanStd(shape=shape)
        self.register_buffer("max_update", torch.tensor(int(max_update)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and int(self.max_update.item()) > 0:
            self.running_ms.update(x)
            self.max_update -= 1
        return (x - self.running_ms.mean) / self.running_ms.std

    def inv(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.running_ms.std + self.running_ms.mean


class BasicMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_features: int = 64) -> None:
        super().__init__()
        self.net = ResidualNet(
            int(input_dim),
            int(output_dim),
            hidden_features=int(hidden_features),
            num_blocks=2,
            dropout_probability=0.0,
            use_batch_norm=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DPDiffusionHead(nn.Module):
    """DexLearn DiffusionRTJ 的当前仓库适配版。"""

    def __init__(
        self,
        condition_dim: int,
        target_dim: int,
        diffusion_config: dict[str, Any],
        *,
        rms_enabled: bool = True,
        rms_max_update: int = 2000,
    ) -> None:
        super().__init__()
        self.target_dim = int(target_dim)
        self.condition_dim = int(condition_dim)
        policy = MLPWrapper(
            channels=self.target_dim,
            feature_dim=self.condition_dim,
            hidden_layers_dim=[512, 256],
            output_dim=self.target_dim,
            act="mish",
        )
        self.diffusion = GaussianDiffusion1D(policy, diffusion_config)
        self.rms_enabled = bool(rms_enabled)
        self.rms = Normalization(self.target_dim, max_update=rms_max_update) if self.rms_enabled else None

    def forward(self, target: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.rms is not None:
            target = self.rms(target)
        loss_diffusion = self.diffusion(target, condition)
        return {
            "loss_diffusion": loss_diffusion,
            "loss": loss_diffusion,
        }

    def sample(self, condition: torch.Tensor, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        repeated_condition = condition.repeat_interleave(num_samples, dim=0)
        prediction, log_prob = self.diffusion.sample(cond=repeated_condition)
        if self.rms is not None:
            prediction = self.rms.inv(prediction)
        batch_size = condition.shape[0]
        return (
            prediction.reshape(batch_size, num_samples, self.target_dim),
            log_prob.reshape(batch_size, num_samples),
        )


class DPDiffusionRTHead(nn.Module):
    """DexLearn DiffusionRT_MLPRTJ 的当前仓库适配版。"""

    def __init__(
        self,
        condition_dim: int,
        squeeze_pose_dim: int,
        init_pose_dim: int,
        joint_dim: int,
        diffusion_config: dict[str, Any],
        *,
        mlp_hidden_features: int = 64,
        rms_enabled: bool = True,
        rms_max_update: int = 2000,
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.squeeze_pose_dim = int(squeeze_pose_dim)
        self.init_pose_dim = int(init_pose_dim)
        self.joint_dim = int(joint_dim)
        self.regression_dim = self.init_pose_dim + self.joint_dim
        policy = MLPWrapper(
            channels=self.squeeze_pose_dim,
            feature_dim=self.condition_dim,
            hidden_layers_dim=[512, 256],
            output_dim=self.squeeze_pose_dim,
            act="mish",
        )
        self.diffusion = GaussianDiffusion1D(policy, diffusion_config)
        self.rms_enabled = bool(rms_enabled)
        self.rms = (
            Normalization(self.squeeze_pose_dim, max_update=rms_max_update)
            if self.rms_enabled
            else None
        )
        self.regression_head = BasicMLP(
            self.condition_dim + self.squeeze_pose_dim,
            self.regression_dim,
            hidden_features=mlp_hidden_features,
        )
        self.pose_loss = nn.SmoothL1Loss()
        self.joint_loss = nn.SmoothL1Loss()

    def forward(
        self,
        squeeze_pose: torch.Tensor,
        init_pose: torch.Tensor,
        squeeze_joint: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        squeeze_pose_target = self.rms(squeeze_pose) if self.rms is not None else squeeze_pose
        loss_diffusion = self.diffusion(squeeze_pose_target, condition)
        predicted_squeeze_pose = self.diffusion.predict_x0(squeeze_pose_target, condition)
        if self.rms is not None:
            predicted_squeeze_pose = self.rms.inv(predicted_squeeze_pose)

        regression_prediction = self.regression_head(
            torch.cat([condition, predicted_squeeze_pose], dim=-1)
        )
        pred_init_pose = regression_prediction[:, : self.init_pose_dim]
        pred_squeeze_joint = regression_prediction[:, self.init_pose_dim :]
        loss_init_pose = self.pose_loss(pred_init_pose, init_pose)
        loss_joint = self.joint_loss(pred_squeeze_joint, squeeze_joint)
        total_loss = loss_diffusion + loss_init_pose + loss_joint
        return {
            "loss_diffusion": loss_diffusion,
            "loss_init_pose": loss_init_pose,
            "loss_joint": loss_joint,
            "loss": total_loss,
        }

    def sample(
        self,
        condition: torch.Tensor,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        repeated_condition = condition.repeat_interleave(num_samples, dim=0)
        squeeze_pose_prediction, log_prob = self.diffusion.sample(cond=repeated_condition)
        if self.rms is not None:
            squeeze_pose_prediction = self.rms.inv(squeeze_pose_prediction)
        regression_prediction = self.regression_head(
            torch.cat([repeated_condition, squeeze_pose_prediction], dim=-1)
        )
        pred_init_pose = regression_prediction[:, : self.init_pose_dim]
        pred_squeeze_joint = regression_prediction[:, self.init_pose_dim :]
        batch_size = condition.shape[0]
        return (
            pred_init_pose.reshape(batch_size, num_samples, self.init_pose_dim),
            squeeze_pose_prediction.reshape(batch_size, num_samples, self.squeeze_pose_dim),
            pred_squeeze_joint.reshape(batch_size, num_samples, self.joint_dim),
        ), log_prob.reshape(batch_size, num_samples)
