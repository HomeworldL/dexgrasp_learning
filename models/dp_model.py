from __future__ import annotations

from typing import Any

import torch

from models.base_model import BaseModel
from models.dp import DPDiffusionHead, DPStagedDiffusionHead


class DPModel(BaseModel):
    """DexLearn 风格轻量 diffusion 生成器。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        if self.algorithm != "dp":
            raise NotImplementedError("DPModel only supports model.algorithm=dp.")
        algorithm_config = dict(self.model_config)
        rms_config = dict(algorithm_config.get("rms", {}))
        self.loss_weights: dict[str, float] | None = None

        if self.prediction_structure_name == "flat":
            self.require_algorithm_config("diffusion.scheduler_type")
            self.require_algorithm_config("diffusion.scheduler.num_train_timesteps")
            self.require_algorithm_config("diffusion.num_inference_timesteps")
            self.require_algorithm_config("diffusion.loss_type")
            self.head = DPDiffusionHead(
                condition_dim=self.point_feat_dim,
                target_dim=self.target_dim,
                diffusion_config=dict(algorithm_config.get("diffusion", {})),
                rms_enabled=bool(rms_config.get("enabled", True)),
                rms_max_update=int(rms_config.get("max_update", 2000)),
            )
            return

        if self.prediction_structure_name != "staged":
            raise NotImplementedError(
                "DPModel only supports model.prediction_structure.name=flat or staged."
            )

        self.require_algorithm_config("diffusion.scheduler_type")
        self.require_algorithm_config("diffusion.scheduler.num_train_timesteps")
        self.require_algorithm_config("diffusion.num_inference_timesteps")
        self.require_algorithm_config("diffusion.loss_type")
        regression_config = self.get_staged_regression_config()
        algorithm_loss_weights = dict(algorithm_config.get("loss_weights", {}))
        self.loss_weights = {
            "diffusion": float(algorithm_loss_weights.get("diffusion", 1.0)),
            "init_pose": float(algorithm_loss_weights.get("init_pose", 1.0)),
            "joint": float(algorithm_loss_weights.get("joint", 5.0)),
        }
        self.head = DPStagedDiffusionHead(
            condition_dim=self.point_feat_dim,
            squeeze_pose_dim=self.squeeze_pose_dim,
            init_pose_dim=self.init_pose_dim,
            joint_dim=self.joint_dim,
            diffusion_config=dict(algorithm_config.get("diffusion", {})),
            regression_config=regression_config,
            rms_enabled=bool(rms_config.get("enabled", True)),
            rms_max_update=int(rms_config.get("max_update", 2000)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        condition = self.encode_condition(batch)
        if self.prediction_structure_name == "flat":
            target = self.target_vector(batch)
            return self.head(target=target, condition=condition)

        outputs = self.head(
            squeeze_pose=batch["squeeze_pose"],
            init_pose=batch["init_pose"],
            squeeze_joint=batch["squeeze_joint"],
            condition=condition,
        )
        if self.loss_weights is None:
            raise RuntimeError("Staged DP loss weights were not initialized.")
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
        if self.prediction_structure_name == "flat":
            prediction, _ = self.head.sample(condition=condition, num_samples=num_samples)
            split_prediction = self.split_prediction(prediction)
            return {
                "pred_init_pose": split_prediction["pred_init_pose"],
                "pred_squeeze_pose": split_prediction["pred_squeeze_pose"],
                "pred_squeeze_joint": split_prediction["pred_squeeze_joint"],
            }

        prediction, _ = self.head.sample(condition=condition, num_samples=num_samples)
        pred_init_pose, pred_squeeze_pose, pred_squeeze_joint = prediction
        return {
            "pred_init_pose": pred_init_pose,
            "pred_squeeze_pose": pred_squeeze_pose,
            "pred_squeeze_joint": pred_squeeze_joint,
        }
