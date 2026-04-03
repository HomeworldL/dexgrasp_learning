from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from models.backbones.bps import BPSEncoder
from models.backbones.pointnet import PointNet


SUPPORTED_INPUT_ENCODERS = frozenset({"pointnet", "bps"})


def normalize_algorithm_name(raw_value: Any) -> str:
    return str(raw_value).strip().lower()


def normalize_prediction_structure_name(raw_value: Any) -> str:
    return str(raw_value).strip().lower()


def normalize_input_encoder_name(raw_value: Any) -> str:
    return str(raw_value).strip().lower()


def get_model_required(config: dict[str, Any], dotted_key: str) -> Any:
    """读取模型侧必填配置项，缺失则立即报错。"""
    current: Any = config
    traversed: list[str] = []
    for key in dotted_key.split("."):
        traversed.append(key)
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing required config key: {'.'.join(traversed)}")
        current = current[key]
    return current


def materialize_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """展开统一配置中的 algorithm / prediction_structure / encoder 选择。"""
    model_config = deepcopy(dict(config.get("model", {})))
    algorithm = normalize_algorithm_name(model_config.get("algorithm", "cvae"))
    prediction_structure_raw = dict(model_config.get("prediction_structure", {})).get("name")
    if prediction_structure_raw is None:
        raise KeyError("Missing required config key: model.prediction_structure.name")
    prediction_structure_name = normalize_prediction_structure_name(
        prediction_structure_raw
    )
    input_encoder_name = normalize_input_encoder_name(
        dict(model_config.get("input_encoder", {})).get("name", "pointnet")
    )

    effective = deepcopy(dict(model_config.get("common", {})))
    algorithm_config = deepcopy(dict(model_config.get("algorithms", {}).get(algorithm, {})))
    effective.update(deepcopy(dict(algorithm_config.get(prediction_structure_name, {}))))

    input_encoder_config = deepcopy(
        dict(model_config.get("input_encoders", {}).get(input_encoder_name, {}))
    )
    input_encoder_config.update(deepcopy(dict(model_config.get("input_encoder", {}))))
    input_encoder_config["name"] = input_encoder_name
    effective["input_encoder"] = input_encoder_config
    effective["algorithm"] = algorithm
    effective["prediction_structure_name"] = prediction_structure_name
    return effective


def build_input_encoder(model_config: dict[str, Any]) -> nn.Module:
    """按统一配置构建点云编码器。"""
    point_feat_dim = int(model_config.get("point_feat_dim", 128))
    input_encoder_config = dict(model_config.get("input_encoder", {}))
    input_encoder_name = normalize_input_encoder_name(
        input_encoder_config.get("name", "pointnet")
    )

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
        "The current mainline implements "
        f"{', '.join(sorted(SUPPORTED_INPUT_ENCODERS))}."
    )


class BaseModel(nn.Module):
    """点云条件抓取模型的最小公共骨架。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.model_config = materialize_model_config(config)
        self.algorithm = normalize_algorithm_name(
            self.model_config.get("algorithm", "cvae")
        )
        self.prediction_structure_name = normalize_prediction_structure_name(
            self.model_config["prediction_structure_name"]
        )
        self.input_encoder_name = normalize_input_encoder_name(
            self.model_config.get("input_encoder", {}).get("name", "pointnet")
        )

        self.point_feat_dim = int(self.model_config.get("point_feat_dim", 128))
        self.init_pose_dim = int(self.model_config.get("init_pose_dim", 6))
        self.squeeze_pose_dim = int(self.model_config.get("squeeze_pose_dim", 6))
        self.joint_dim = int(self.model_config.get("joint_dim", 20))
        self.target_dim = self.init_pose_dim + self.squeeze_pose_dim + self.joint_dim

        self.input_encoder = build_input_encoder(self.model_config)

    def encode_condition_features(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回全局条件特征及 backbone 原始特征。"""
        return self.input_encoder(batch["point_cloud"])

    def encode_condition(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """提取点云条件特征。"""
        condition, _ = self.encode_condition_features(batch)
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
            "pred_init_pose": prediction[..., :init_end],
            "pred_squeeze_pose": prediction[..., init_end:squeeze_end],
            "pred_squeeze_joint": prediction[
                ..., squeeze_end : squeeze_end + self.joint_dim
            ],
        }

    def require_algorithm_config(self, relative_key: str) -> Any:
        """读取当前算法当前结构下的必填配置。"""
        return get_model_required(
            self.config,
            "model.algorithms."
            f"{self.algorithm}.{self.prediction_structure_name}.{relative_key}",
        )

    def get_staged_regression_config(self) -> dict[str, Any]:
        """读取 staged 回归头统一配置。"""
        if self.prediction_structure_name != "staged":
            raise RuntimeError(
                "get_staged_regression_config is only valid for staged prediction structures."
            )
        self.require_algorithm_config("regression.hidden_dims")
        self.require_algorithm_config("regression.activation")
        self.require_algorithm_config("regression.network_type")
        self.require_algorithm_config("regression.residual_num_blocks")
        return dict(self.model_config.get("regression", {}))

    @staticmethod
    def _validate_num_samples(num_samples: int) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")

    @staticmethod
    def _squeeze_single_sample_dim(
        sampled: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        squeezed: dict[str, torch.Tensor] = {}
        for key, value in sampled.items():
            if value.ndim < 2:
                raise ValueError(
                    f"Expected sampled tensor '{key}' to have at least 2 dims, got {value.shape}."
                )
            if value.shape[1] != 1:
                raise ValueError(
                    f"Expected sampled tensor '{key}' to have sample dim 1, got {value.shape}."
                )
            squeezed[key] = value[:, 0, ...]
        return squeezed

    def infer(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """默认单样本推理路径。"""
        return self._squeeze_single_sample_dim(self.sample(batch=batch, num_samples=1))

    def sample(
        self,
        batch: dict[str, torch.Tensor],
        num_samples: int,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError
