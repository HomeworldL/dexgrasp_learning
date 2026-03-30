from __future__ import annotations

from typing import Any

import torch

from models.base_sc import BaseSCModel
from models.udgm import FlowTargetCodec, UDGMConditionAdapter, UDGMFlow


class UDGMScModel(BaseSCModel):
    """适配当前单条件主线的 UDGM 条件 flow 生成器。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        if self.algorithm != "udgm":
            raise NotImplementedError("UDGMScModel only supports model.algorithm=udgm.")

        algorithm_config = dict(self.model_config)
        self.codec = FlowTargetCodec(
            init_pose_dim=self.init_pose_dim,
            squeeze_pose_dim=self.squeeze_pose_dim,
            joint_dim=self.joint_dim,
            normalization_config=dict(algorithm_config.get("target_normalization", {})),
        )

        self.condition_dim = int(algorithm_config.get("condition_dim", self.point_feat_dim))
        condition_config = dict(algorithm_config.get("condition", {}))
        self.condition_adapter = UDGMConditionAdapter(
            point_feat_dim=self.point_feat_dim,
            condition_dim=self.condition_dim,
            hidden_dims=list(condition_config.get("hidden_dims", [self.point_feat_dim])),
            activation=str(condition_config.get("activation", "leaky_relu")),
        )

        flow_config = dict(algorithm_config.get("flow", {}))
        flow_activation = str(
            flow_config.get("activation", condition_config.get("activation", "leaky_relu"))
        )
        self.flow = UDGMFlow(
            target_dim=self.target_dim,
            condition_dim=self.condition_dim,
            hidden_dim=int(flow_config.get("hidden_dim", 256)),
            num_layers=int(flow_config.get("num_layers", 8)),
            num_blocks_per_layer=int(flow_config.get("num_blocks_per_layer", 2)),
            scale_clamp=float(flow_config.get("scale_clamp", 2.0)),
            activation=flow_activation,
        )

    def _build_condition(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        global_feature = self.encode_condition(batch)
        return self.condition_adapter(global_feature)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target = self.codec.build_from_batch(batch)
        condition = self._build_condition(batch)
        log_prob = self.flow.log_prob(target, condition)
        loss_nll = -log_prob.mean()
        return {
            "loss_nll": loss_nll,
            "mean_log_prob": log_prob.mean(),
            "loss": loss_nll,
        }

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
        condition = self._build_condition(batch)
        sampled_x, _ = self.flow.sample_and_log_prob(
            num_samples=num_samples,
            context=condition,
            sort_by_log_prob=True,
        )
        return self.codec.split(sampled_x)
