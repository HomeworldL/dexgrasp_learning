from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from models.backbones.bps import BPSEncoder
from models.backbones.pointnet import PointNet


def materialize_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """展开统一配置中的 algorithm / encoder 选择。"""
    model_config = deepcopy(dict(config.get("model", {})))
    algorithm = str(model_config.get("algorithm", "cvae")).strip().lower()
    input_encoder_selector = deepcopy(dict(model_config.get("input_encoder", {})))
    encoder_name = str(input_encoder_selector.get("name", "pointnet")).strip().lower()

    effective = deepcopy(dict(model_config.get("common", {})))
    effective.update(deepcopy(dict(model_config.get("algorithms", {}).get(algorithm, {}))))

    input_encoder_config = deepcopy(
        dict(model_config.get("input_encoders", {}).get(encoder_name, {}))
    )
    input_encoder_config.update(input_encoder_selector)
    input_encoder_config["name"] = encoder_name
    effective["input_encoder"] = input_encoder_config
    effective["algorithm"] = algorithm
    return effective


def build_input_encoder(model_config: dict[str, Any]) -> nn.Module:
    """按统一配置构建点云编码器。"""
    point_feat_dim = int(model_config.get("point_feat_dim", 128))
    input_encoder_config = dict(model_config.get("input_encoder", {}))
    input_encoder_name = str(input_encoder_config.get("name", "pointnet")).strip().lower()

    if input_encoder_name == "pointnet":
        return PointNet(
            point_feature_dim=int(input_encoder_config.get("point_feature_dim", 3)),
            local_conv_hidden_dims=list(
                input_encoder_config.get("local_conv_hidden_dims", [64, 128, 256])
            ),
            global_mlp_hidden_dims=list(
                input_encoder_config.get("global_mlp_hidden_dims", [256])
            ),
            output_dim=point_feat_dim,
            activation=str(input_encoder_config.get("activation", "leaky_relu")),
        )

    if input_encoder_name == "bps":
        return BPSEncoder(
            output_dim=point_feat_dim,
            basis_path=input_encoder_config.get("basis_path"),
            feature_types=list(input_encoder_config.get("feature_types", ["dists"])),
            mlp_hidden_dims=list(input_encoder_config.get("mlp_hidden_dims", [512, 256])),
            activation=str(input_encoder_config.get("activation", "leaky_relu")),
            bps_type=str(input_encoder_config.get("bps_type", "random_uniform")),
            n_bps_points=int(input_encoder_config.get("n_bps_points", 4096)),
            radius=float(input_encoder_config.get("radius", 1.0)),
            n_dims=int(input_encoder_config.get("n_dims", 3)),
        )

    raise NotImplementedError(
        f"model.input_encoder.name={input_encoder_name} is reserved for future work. "
        "The current mainline implements pointnet and bps."
    )


class BaseSCModel(nn.Module):
    """单条件模型的最小公共骨架。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.model_config = materialize_model_config(config)
        self.algorithm = str(self.model_config.get("algorithm", "cvae")).strip().lower()
        self.input_encoder_name = str(
            self.model_config.get("input_encoder", {}).get("name", "pointnet")
        ).strip().lower()

        self.point_feat_dim = int(self.model_config.get("point_feat_dim", 128))
        self.init_pose_dim = int(self.model_config.get("init_pose_dim", 6))
        self.squeeze_pose_dim = int(self.model_config.get("squeeze_pose_dim", 6))
        self.joint_dim = int(self.model_config.get("joint_dim", 20))
        self.target_dim = self.init_pose_dim + self.squeeze_pose_dim + self.joint_dim

        self.input_encoder = build_input_encoder(self.model_config)

    def encode_condition(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """提取点云条件特征。"""
        condition, _ = self.input_encoder(batch["point_cloud"])
        return condition

    def target_vector(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """拼接训练目标向量。"""
        return torch.cat(
            [batch["init_pose"], batch["squeeze_pose"], batch["squeeze_joint"]],
            dim=-1,
        )

    def split_prediction(self, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        """把网络输出拆成结构化字段。"""
        init_end = self.init_pose_dim
        squeeze_end = init_end + self.squeeze_pose_dim
        return {
            "pred_init_pose": prediction[:, :init_end],
            "pred_squeeze_pose": prediction[:, init_end:squeeze_end],
            "pred_squeeze_joint": prediction[:, squeeze_end : squeeze_end + self.joint_dim],
        }
