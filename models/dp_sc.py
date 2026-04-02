from __future__ import annotations

from typing import Any

import torch

from models.base_sc import BaseSCModel
from models.dp import DPDiffusionHead


class DPScModel(BaseSCModel):
    """DexLearn DiffusionRTJ 风格的单阶段轻量 diffusion 生成器。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        if self.algorithm != "dp":
            raise NotImplementedError("DPScModel only supports model.algorithm=dp.")
        algorithm_config = dict(self.model_config)
        rms_config = dict(algorithm_config.get("rms", {}))
        self.head = DPDiffusionHead(
            condition_dim=self.point_feat_dim,
            target_dim=self.target_dim,
            diffusion_config=dict(algorithm_config.get("diffusion", {})),
            rms_enabled=bool(rms_config.get("enabled", True)),
            rms_max_update=int(rms_config.get("max_update", 2000)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        condition = self.encode_condition(batch)
        target = self.target_vector(batch)
        return self.head(target=target, condition=condition)

    def infer(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
        condition = self.encode_condition(batch)
        prediction, _ = self.head.sample(condition=condition, num_samples=num_samples)
        split_prediction = self.split_prediction(prediction)
        return {
            "pred_init_pose": split_prediction["pred_init_pose"],
            "pred_squeeze_pose": split_prediction["pred_squeeze_pose"],
            "pred_squeeze_joint": split_prediction["pred_squeeze_joint"],
        }
