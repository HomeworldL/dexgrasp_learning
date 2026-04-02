from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from nflows.nn.nets.resnet import ResidualNet
from torch import nn

from models.base_model import build_input_encoder, materialize_model_config


class GraspEvaluator(nn.Module):
    """点云条件下的组式抓取打分网络。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = materialize_model_config(config)
        point_feat_dim = int(model_config.get("point_feat_dim", 128))
        joint_dim = int(model_config.get("joint_dim", 20))
        self.joint_dim = joint_dim
        self.pose_dim = 6
        self.point_feat_dim = point_feat_dim

        evaluator_model_config = dict(config.get("evaluator", {}).get("model", {}))
        evaluator_train_config = dict(config.get("evaluator", {}).get("train", {}))
        self.grasp_feat_dim = int(
            evaluator_model_config.get("grasp_feat_dim", point_feat_dim)
        )
        hidden_features = int(evaluator_model_config.get("hidden_features", 256))
        num_blocks = int(evaluator_model_config.get("num_blocks", 2))
        dropout_probability = float(
            evaluator_model_config.get("dropout_probability", 0.0)
        )
        use_batch_norm = bool(evaluator_model_config.get("use_batch_norm", False))
        loss_weights = dict(evaluator_train_config.get("loss_weights", {}))
        self.rank_loss_weight = float(loss_weights.get("rank", 1.0))
        self.reg_loss_weight = float(loss_weights.get("reg", 0.25))

        self.input_encoder = build_input_encoder(model_config)

        self.grasp_encoder = ResidualNet(
            in_features=self.pose_dim + self.joint_dim,
            out_features=self.grasp_feat_dim,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self.classifier = ResidualNet(
            in_features=self.point_feat_dim + self.grasp_feat_dim,
            out_features=1,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self.regression_loss = nn.SmoothL1Loss()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """训练前向，输出组内排序损失与打分指标。"""
        logits = self._compute_logits(batch)
        score = torch.sigmoid(logits)
        stage_label = batch["stage_label"]
        target_score = batch["target_score"]

        loss_rank = _pairwise_ranking_loss(logits=logits, stage_label=stage_label)
        loss_reg = self.regression_loss(score, target_score)
        loss = self.rank_loss_weight * loss_rank + self.reg_loss_weight * loss_reg

        pairwise_accuracy = _pairwise_accuracy(score=score, stage_label=stage_label)
        top1_success_rate = _top1_success_rate(score=score, stage_label=stage_label)
        return {
            "logits": logits,
            "score": score,
            "loss_rank": loss_rank,
            "loss_reg": loss_reg,
            "loss": loss,
            "pairwise_accuracy": pairwise_accuracy,
            "top1_success_rate": top1_success_rate,
        }

    def score(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """推理阶段输出 0 到 1 的抓取可行性分数。"""
        logits = self._compute_logits(batch)
        return torch.sigmoid(logits)

    def _compute_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """兼容 `[B, K, ...]` 训练输入和 `[M, ...]` 推理输入。"""
        point_feature, _ = self.input_encoder(batch["point_cloud"])
        grasp_pose = batch["grasp_pose"]
        grasp_joint = batch["grasp_joint"]

        if grasp_pose.ndim == 2:
            grasp_input = torch.cat([grasp_pose, grasp_joint], dim=-1)
            grasp_feature = self.grasp_encoder(grasp_input)
            fused_feature = torch.cat([point_feature, grasp_feature], dim=-1)
            return self.classifier(fused_feature).squeeze(-1)

        if grasp_pose.ndim != 3:
            raise ValueError(
                "GraspEvaluator expects grasp tensors with ndim 2 or 3, "
                f"got grasp_pose shape {tuple(grasp_pose.shape)}."
            )

        batch_size, num_grasps, _ = grasp_pose.shape
        if point_feature.shape[0] != batch_size:
            raise ValueError(
                "Point feature batch size must match grouped grasp batch size: "
                f"{point_feature.shape[0]} vs {batch_size}."
            )
        grasp_input = torch.cat([grasp_pose, grasp_joint], dim=-1).reshape(
            batch_size * num_grasps, -1
        )
        grasp_feature = self.grasp_encoder(grasp_input)
        repeated_condition = (
            point_feature[:, None, :]
            .expand(batch_size, num_grasps, point_feature.shape[-1])
            .reshape(batch_size * num_grasps, point_feature.shape[-1])
        )
        fused_feature = torch.cat([repeated_condition, grasp_feature], dim=-1)
        return self.classifier(fused_feature).reshape(batch_size, num_grasps)


def _pairwise_ranking_loss(
    logits: torch.Tensor,
    stage_label: torch.Tensor,
) -> torch.Tensor:
    stage_values = stage_label.to(logits.dtype)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        stage_values = stage_values.unsqueeze(0)
    logit_diff = logits.unsqueeze(2) - logits.unsqueeze(1)
    stage_diff = stage_values.unsqueeze(2) - stage_values.unsqueeze(1)
    valid_mask = stage_diff > 0
    if not bool(valid_mask.any()):
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    weights = stage_diff.clamp_min(0.0)
    loss = F.softplus(-logit_diff) * weights
    return loss[valid_mask].mean()


def _pairwise_accuracy(
    score: torch.Tensor,
    stage_label: torch.Tensor,
) -> torch.Tensor:
    stage_values = stage_label.to(score.dtype)
    if score.ndim == 1:
        score = score.unsqueeze(0)
        stage_values = stage_values.unsqueeze(0)
    score_diff = score.unsqueeze(2) - score.unsqueeze(1)
    stage_diff = stage_values.unsqueeze(2) - stage_values.unsqueeze(1)
    valid_mask = stage_diff > 0
    if not bool(valid_mask.any()):
        return torch.tensor(0.0, device=score.device, dtype=score.dtype)
    return (score_diff[valid_mask] > 0.0).to(torch.float32).mean()


def _top1_success_rate(
    score: torch.Tensor,
    stage_label: torch.Tensor,
) -> torch.Tensor:
    if score.ndim == 1:
        top_stage = stage_label[torch.argmax(score)]
        return (top_stage == 3).to(torch.float32)
    top_index = torch.argmax(score, dim=-1, keepdim=True)
    top_stage = torch.gather(stage_label, dim=1, index=top_index).squeeze(1)
    return (top_stage == 3).to(torch.float32).mean()
