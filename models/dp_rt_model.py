from __future__ import annotations

from typing import Any

import torch

from models.base_model import BaseModel
from models.dp import DPDiffusionRTHead


class DPRTModel(BaseModel):
    """DexLearn DiffusionRT_MLPRTJ 风格的两阶段轻量 diffusion 生成器。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        if self.algorithm != "dp_rt":
            raise NotImplementedError("DPRTModel only supports model.algorithm=dp_rt.")
        algorithm_config = dict(self.model_config)
        rms_config = dict(algorithm_config.get("rms", {}))
        regression_config = dict(algorithm_config.get("regression", {}))
        algorithm_loss_weights = dict(algorithm_config.get("loss_weights", {}))
        self.loss_weights = {
            "diffusion": float(algorithm_loss_weights.get("diffusion", 1.0)),
            "init_pose": float(algorithm_loss_weights.get("init_pose", 1.0)),
            "joint": float(algorithm_loss_weights.get("joint", 5.0)),
        }
        self.head = DPDiffusionRTHead(
            condition_dim=self.point_feat_dim,
            squeeze_pose_dim=self.squeeze_pose_dim,
            init_pose_dim=self.init_pose_dim,
            joint_dim=self.joint_dim,
            diffusion_config=dict(algorithm_config.get("diffusion", {})),
            mlp_hidden_features=int(regression_config.get("hidden_features", 64)),
            rms_enabled=bool(rms_config.get("enabled", True)),
            rms_max_update=int(rms_config.get("max_update", 2000)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        condition = self.encode_condition(batch)
        outputs = self.head(
            squeeze_pose=batch["squeeze_pose"],
            init_pose=batch["init_pose"],
            squeeze_joint=batch["squeeze_joint"],
            condition=condition,
        )
        outputs["loss"] = (
            self.loss_weights["diffusion"] * outputs["loss_diffusion"]
            + self.loss_weights["init_pose"] * outputs["loss_init_pose"]
            + self.loss_weights["joint"] * outputs["loss_joint"]
        )
        return outputs

    def sample(
        self,
        batch: dict[str, torch.Tensor],
        num_samples: int,
    ) -> dict[str, torch.Tensor]:
        self._validate_num_samples(num_samples)
        condition = self.encode_condition(batch)
        prediction, _ = self.head.sample(condition=condition, num_samples=num_samples)
        pred_init_pose, pred_squeeze_pose, pred_squeeze_joint = prediction
        return {
            "pred_init_pose": pred_init_pose,
            "pred_squeeze_pose": pred_squeeze_pose,
            "pred_squeeze_joint": pred_squeeze_joint,
        }
