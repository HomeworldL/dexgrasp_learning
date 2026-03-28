from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base_sc import BaseSCModel
from models.cvae.cvae import VAE


class CVAESingleConditionModel(BaseSCModel):
    """单条件 PointNet + CVAE 生成器。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        if self.algorithm != "cvae":
            raise NotImplementedError(
                "Only model.algorithm=cvae is implemented in the current mainline."
            )
        self.beta_kld = float(config["train"].get("beta_kld", 1e-3))
        self.cvae = VAE(
            encoder_layer_sizes=[self.target_dim]
            + list(self.model_config.get("encoder_hidden_dims", [512, 256])),
            latent_size=int(self.model_config.get("latent_dim", 64)),
            decoder_layer_sizes=list(self.model_config.get("decoder_hidden_dims", [256, 256]))
            + [self.target_dim],
            conditional=True,
            condition_size=self.point_feat_dim,
        )
        self.pose_loss = nn.SmoothL1Loss()
        self.joint_loss = nn.SmoothL1Loss()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """训练阶段前向。"""
        condition = self.encode_condition(batch)
        target = self.target_vector(batch)
        reconstruction, means, log_var, _ = self.cvae(target, c=condition)
        prediction = self.split_prediction(reconstruction)
        loss_init_pose = self.pose_loss(prediction["pred_init_pose"], batch["init_pose"])
        loss_squeeze_pose = self.pose_loss(
            prediction["pred_squeeze_pose"], batch["squeeze_pose"]
        )
        loss_joint = self.joint_loss(
            prediction["pred_squeeze_joint"], batch["squeeze_joint"]
        )
        loss_kld = -0.5 * (
            1 + log_var - means.pow(2) - log_var.exp()
        ).sum(dim=-1).mean()
        loss = loss_init_pose + loss_squeeze_pose + loss_joint + self.beta_kld * loss_kld
        return {
            "loss_init_pose": loss_init_pose,
            "loss_squeeze_pose": loss_squeeze_pose,
            "loss_joint": loss_joint,
            "loss_kld": loss_kld,
            "loss": loss,
        }

    def infer(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """单样本推理。"""
        sampled = self.sample(batch=batch, num_samples=1)
        return {
            "pred_init_pose": sampled["pred_init_pose"][:, 0, :],
            "pred_squeeze_pose": sampled["pred_squeeze_pose"][:, 0, :],
            "pred_squeeze_joint": sampled["pred_squeeze_joint"][:, 0, :],
        }

    def sample(
        self,
        batch: dict[str, torch.Tensor],
        num_samples: int,
    ) -> dict[str, torch.Tensor]:
        """并行采样多个抓取候选。"""
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        condition = self.encode_condition(batch)
        batch_size = condition.shape[0]
        repeated_condition = (
            condition[:, None, :]
            .expand(batch_size, num_samples, condition.shape[-1])
            .reshape(batch_size * num_samples, condition.shape[-1])
        )
        reconstruction = self.cvae.inference(
            n=batch_size * num_samples,
            c=repeated_condition,
        )
        prediction = self.split_prediction(reconstruction)
        return {
            "pred_init_pose": prediction["pred_init_pose"].reshape(
                batch_size, num_samples, self.init_pose_dim
            ),
            "pred_squeeze_pose": prediction["pred_squeeze_pose"].reshape(
                batch_size, num_samples, self.squeeze_pose_dim
            ),
            "pred_squeeze_joint": prediction["pred_squeeze_joint"].reshape(
                batch_size, num_samples, self.joint_dim
            ),
        }
