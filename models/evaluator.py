from __future__ import annotations

from typing import Any

import torch
from nflows.nn.nets.resnet import ResidualNet
from torch import nn

from models.base_sc import build_input_encoder, materialize_model_config


class GraspEvaluator(nn.Module):
    """点云条件下的抓取可行性二分类网络。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = materialize_model_config(config)
        input_encoder_name = str(
            model_config.get("input_encoder", {}).get("name", "pointnet")
        ).strip().lower()

        point_feat_dim = int(model_config.get("point_feat_dim", 128))
        joint_dim = int(model_config.get("joint_dim", 20))
        self.joint_dim = joint_dim
        self.pose_dim = 6
        self.point_feat_dim = point_feat_dim

        evaluator_model_config = dict(config.get("evaluator", {}).get("model", {}))
        self.grasp_feat_dim = int(
            evaluator_model_config.get("grasp_feat_dim", point_feat_dim)
        )
        hidden_features = int(evaluator_model_config.get("hidden_features", 256))
        num_blocks = int(evaluator_model_config.get("num_blocks", 2))
        dropout_probability = float(
            evaluator_model_config.get("dropout_probability", 0.0)
        )
        use_batch_norm = bool(evaluator_model_config.get("use_batch_norm", False))

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
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """训练前向，输出 logits、score 与 loss。"""
        point_feature, _ = self.input_encoder(batch["point_cloud"])
        grasp_input = torch.cat([batch["grasp_pose"], batch["grasp_joint"]], dim=-1)
        grasp_feature = self.grasp_encoder(grasp_input)
        fused_feature = torch.cat([point_feature, grasp_feature], dim=-1)
        logits = self.classifier(fused_feature).squeeze(-1)
        score = torch.sigmoid(logits)
        label = batch["label"].reshape(-1)
        loss = self.loss_fn(logits, label)
        prediction = (score >= 0.5).to(label.dtype)
        accuracy = (prediction == label).to(torch.float32).mean()

        positive_mask = label > 0.5
        negative_mask = ~positive_mask
        positive_accuracy = _masked_accuracy(prediction, label, positive_mask)
        negative_accuracy = _masked_accuracy(prediction, label, negative_mask)
        return {
            "logits": logits,
            "score": score,
            "loss": loss,
            "accuracy": accuracy,
            "positive_accuracy": positive_accuracy,
            "negative_accuracy": negative_accuracy,
        }

    def score(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """推理阶段输出 0 到 1 的抓取可行性分数。"""
        point_feature, _ = self.input_encoder(batch["point_cloud"])
        grasp_input = torch.cat([batch["grasp_pose"], batch["grasp_joint"]], dim=-1)
        grasp_feature = self.grasp_encoder(grasp_input)
        fused_feature = torch.cat([point_feature, grasp_feature], dim=-1)
        logits = self.classifier(fused_feature).squeeze(-1)
        return torch.sigmoid(logits)


def _masked_accuracy(
    prediction: torch.Tensor,
    label: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not bool(mask.any()):
        return torch.tensor(0.0, device=prediction.device, dtype=torch.float32)
    return (prediction[mask] == label[mask]).to(torch.float32).mean()
