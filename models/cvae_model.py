from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base_model import BaseModel
from models.cvae.cvae import VAE


class CVAEModel(BaseModel):
    """PointNet/BPS 条件下的 CVAE 抓取生成器。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        if self.algorithm != "cvae":
            raise NotImplementedError(
                "Only model.algorithm=cvae is implemented in the current mainline."
            )
        algorithm_config = dict(self.model_config)
        loss_weights = dict(algorithm_config.get("loss_weights", {}))
        self.loss_weights = {
            "init_pose": float(loss_weights.get("init_pose", 1.0)),
            "squeeze_pose": float(loss_weights.get("squeeze_pose", 1.0)),
            "joint": float(loss_weights.get("joint", 1.0)),
            "kld": float(loss_weights.get("kld", 1e-3)),
        }
        self.cvae = VAE(
            encoder_layer_sizes=[self.target_dim]
            + list(algorithm_config.get("encoder_hidden_dims", [512, 256])),
            latent_size=int(algorithm_config.get("latent_dim", 64)),
            decoder_layer_sizes=list(algorithm_config.get("decoder_hidden_dims", [256, 256]))
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
        loss = (
            self.loss_weights["init_pose"] * loss_init_pose
            + self.loss_weights["squeeze_pose"] * loss_squeeze_pose
            + self.loss_weights["joint"] * loss_joint
            + self.loss_weights["kld"] * loss_kld
        )
        return {
            "loss_init_pose": loss_init_pose,
            "loss_squeeze_pose": loss_squeeze_pose,
            "loss_joint": loss_joint,
            "loss_kld": loss_kld,
            "loss": loss,
        }

    def sample(
        self,
        batch: dict[str, torch.Tensor],
        num_samples: int,
    ) -> dict[str, torch.Tensor]:
        """并行采样多个抓取候选。"""
        self._validate_num_samples(num_samples)
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
