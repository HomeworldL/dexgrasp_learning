from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base_model import BaseModel
from models.udgm import (
    InitJointCodec,
    SqueezePoseCodec,
    UDGMConditionAdapter,
    UDGMFlow,
    UDGMRTHead,
)


class UDGMRTModel(BaseModel):
    """DexLearn flow+MLP 风格的两阶段 UDGM。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        if self.algorithm != "udgm_rt":
            raise NotImplementedError("UDGMRTModel only supports model.algorithm=udgm_rt.")

        algorithm_config = dict(self.model_config)
        normalization_config = dict(algorithm_config.get("target_normalization", {}))
        self.squeeze_pose_codec = SqueezePoseCodec(
            squeeze_pose_dim=self.squeeze_pose_dim,
            normalization_config=normalization_config,
        )
        self.init_joint_codec = InitJointCodec(
            init_pose_dim=self.init_pose_dim,
            joint_dim=self.joint_dim,
            normalization_config=normalization_config,
        )

        self.condition_dim = int(algorithm_config.get("condition_dim", self.point_feat_dim))
        condition_config = dict(algorithm_config.get("condition", {}))
        self.condition_adapter = UDGMConditionAdapter(
            point_feat_dim=self.point_feat_dim,
            condition_dim=self.condition_dim,
            hidden_dims=list(condition_config.get("hidden_dims", [self.point_feat_dim])),
            activation=str(condition_config.get("activation", "leaky_relu")),
            network_type=str(condition_config.get("network_type", "residual")),
            residual_num_blocks=int(condition_config.get("residual_num_blocks", 2)),
        )

        flow_config = dict(algorithm_config.get("flow", {}))
        flow_activation = str(
            flow_config.get("activation", condition_config.get("activation", "leaky_relu"))
        )
        self.flow = UDGMFlow(
            target_dim=self.squeeze_pose_dim,
            condition_dim=self.condition_dim,
            hidden_dim=int(flow_config.get("hidden_dim", 256)),
            num_layers=int(flow_config.get("num_layers", 8)),
            num_blocks_per_layer=int(flow_config.get("num_blocks_per_layer", 2)),
            scale_clamp=float(flow_config.get("scale_clamp", 2.0)),
            activation=flow_activation,
            use_actnorm=bool(flow_config.get("use_actnorm", True)),
            use_invertible_linear=bool(flow_config.get("use_invertible_linear", True)),
            conditioner_type=str(flow_config.get("conditioner_type", "residual")),
            residual_num_blocks=int(flow_config.get("residual_num_blocks", 2)),
        )
        self.loss_clamp_max = flow_config.get("loss_clamp_max")
        if self.loss_clamp_max is not None:
            self.loss_clamp_max = float(self.loss_clamp_max)
            if self.loss_clamp_max <= 0.0:
                raise ValueError(
                    "model.algorithms.udgm_rt.flow.loss_clamp_max must be positive, "
                    f"got {self.loss_clamp_max}."
                )
        self.training_pose_num_samples = int(flow_config.get("training_pose_num_samples", 8))
        if self.training_pose_num_samples <= 0:
            raise ValueError(
                "model.algorithms.udgm_rt.flow.training_pose_num_samples must be positive, "
                f"got {self.training_pose_num_samples}."
            )

        regression_config = dict(algorithm_config.get("regression", {}))
        self.regression_head = UDGMRTHead(
            condition_dim=self.condition_dim,
            squeeze_pose_dim=self.squeeze_pose_dim,
            init_pose_dim=self.init_pose_dim,
            joint_dim=self.joint_dim,
            hidden_dims=list(regression_config.get("hidden_dims", [256, 256])),
            activation=str(regression_config.get("activation", "leaky_relu")),
            network_type=str(regression_config.get("network_type", "residual")),
            residual_num_blocks=int(regression_config.get("residual_num_blocks", 2)),
        )
        self.pose_loss = nn.SmoothL1Loss()
        self.joint_loss = nn.SmoothL1Loss()
        algorithm_loss_weights = dict(algorithm_config.get("loss_weights", {}))
        self.loss_weights = {
            "flow": float(algorithm_loss_weights.get("flow", 1.0)),
            "init_pose": float(algorithm_loss_weights.get("init_pose", 1.0)),
            "joint": float(algorithm_loss_weights.get("joint", 5.0)),
        }

    def _build_condition(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        global_feature = self.encode_condition(batch)
        return self.condition_adapter(global_feature)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        condition = self._build_condition(batch)
        squeeze_pose_target = self.squeeze_pose_codec.build_from_batch(batch)
        log_prob = self.flow.log_prob(squeeze_pose_target, condition)
        raw_nll = -log_prob
        nll = raw_nll
        if self.loss_clamp_max is not None:
            nll = nll.clamp_max(self.loss_clamp_max)
        loss_flow = nll.mean()

        sampled_squeeze_pose, _ = self.flow.sample_and_log_prob(
            num_samples=self.training_pose_num_samples,
            context=condition,
            sort_by_log_prob=True,
        )
        sampled_squeeze_pose = self.squeeze_pose_codec.split(sampled_squeeze_pose)["pred_squeeze_pose"][:, 0, :]
        regression_prediction = self.regression_head(condition, sampled_squeeze_pose)
        split_prediction = self.init_joint_codec.split(regression_prediction)
        loss_init_pose = self.pose_loss(split_prediction["pred_init_pose"], batch["init_pose"])
        loss_joint = self.joint_loss(split_prediction["pred_squeeze_joint"], batch["squeeze_joint"])
        loss = (
            self.loss_weights["flow"] * loss_flow
            + self.loss_weights["init_pose"] * loss_init_pose
            + self.loss_weights["joint"] * loss_joint
        )
        outputs: dict[str, torch.Tensor] = {
            "loss_flow": loss_flow,
            "loss_init_pose": loss_init_pose,
            "loss_joint": loss_joint,
            "mean_log_prob": log_prob.mean(),
            "raw_nll": raw_nll.mean(),
            "loss": loss,
        }
        if self.loss_clamp_max is not None:
            outputs["clip_fraction"] = (raw_nll > self.loss_clamp_max).float().mean()
        return outputs

    def sample(
        self,
        batch: dict[str, torch.Tensor],
        num_samples: int,
    ) -> dict[str, torch.Tensor]:
        self._validate_num_samples(num_samples)
        condition = self._build_condition(batch)
        sampled_squeeze_pose, _ = self.flow.sample_and_log_prob(
            num_samples=num_samples,
            context=condition,
            sort_by_log_prob=True,
        )
        decoded_squeeze_pose = self.squeeze_pose_codec.split(sampled_squeeze_pose)["pred_squeeze_pose"]
        init_joint_prediction = self.regression_head(condition, decoded_squeeze_pose)
        return self.init_joint_codec.merge(
            squeeze_pose=decoded_squeeze_pose,
            init_pose_and_joint=init_joint_prediction,
        )
