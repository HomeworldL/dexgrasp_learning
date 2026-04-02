from __future__ import annotations

from typing import Any

import torch


class DiffusionTargetCodec:
    """结构化抓取字段与扩散目标向量之间的编解码器。"""

    def __init__(
        self,
        init_pose_dim: int,
        squeeze_pose_dim: int,
        joint_dim: int,
        normalization_config: dict[str, Any] | None = None,
    ) -> None:
        self.init_pose_dim = int(init_pose_dim)
        self.squeeze_pose_dim = int(squeeze_pose_dim)
        self.joint_dim = int(joint_dim)
        self.target_dim = self.init_pose_dim + self.squeeze_pose_dim + self.joint_dim
        normalization = dict(normalization_config or {})
        self.normalization_enabled = bool(normalization.get("enabled", False))
        pose_scale = float(normalization.get("pose_scale", 1.0))
        joint_scale = float(normalization.get("joint_scale", 1.0))
        if pose_scale <= 0.0:
            raise ValueError(f"target_normalization.pose_scale must be positive, got {pose_scale}.")
        if joint_scale <= 0.0:
            raise ValueError(
                f"target_normalization.joint_scale must be positive, got {joint_scale}."
            )
        self.pose_scale = pose_scale
        self.joint_scale = joint_scale

    def build_from_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = torch.cat(
            [batch["init_pose"], batch["squeeze_pose"], batch["squeeze_joint"]],
            dim=-1,
        )
        return self.normalize(target)

    def normalize(self, target: torch.Tensor) -> torch.Tensor:
        if not self.normalization_enabled:
            return target
        pieces = self._split_raw(target)
        return torch.cat(
            [
                pieces["pred_init_pose"] / self.pose_scale,
                pieces["pred_squeeze_pose"] / self.pose_scale,
                pieces["pred_squeeze_joint"] / self.joint_scale,
            ],
            dim=-1,
        )

    def denormalize(self, prediction: torch.Tensor) -> torch.Tensor:
        if not self.normalization_enabled:
            return prediction
        pieces = self._split_raw(prediction)
        return torch.cat(
            [
                pieces["pred_init_pose"] * self.pose_scale,
                pieces["pred_squeeze_pose"] * self.pose_scale,
                pieces["pred_squeeze_joint"] * self.joint_scale,
            ],
            dim=-1,
        )

    def split(self, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        denormalized = self.denormalize(prediction)
        return self._split_raw(denormalized)

    def _split_raw(self, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        init_end = self.init_pose_dim
        squeeze_end = init_end + self.squeeze_pose_dim
        if prediction.shape[-1] != self.target_dim:
            raise ValueError(
                "Unexpected diffusion target size: "
                f"expected {self.target_dim}, got {prediction.shape[-1]}."
            )
        return {
            "pred_init_pose": prediction[..., :init_end],
            "pred_squeeze_pose": prediction[..., init_end:squeeze_end],
            "pred_squeeze_joint": prediction[..., squeeze_end : squeeze_end + self.joint_dim],
        }


class SqueezePoseCodec:
    """仅针对 squeeze_pose 的扩散编解码器。"""

    def __init__(
        self,
        squeeze_pose_dim: int,
        normalization_config: dict[str, Any] | None = None,
    ) -> None:
        self.squeeze_pose_dim = int(squeeze_pose_dim)
        normalization = dict(normalization_config or {})
        self.normalization_enabled = bool(normalization.get("enabled", False))
        pose_scale = float(normalization.get("pose_scale", 1.0))
        if pose_scale <= 0.0:
            raise ValueError(f"target_normalization.pose_scale must be positive, got {pose_scale}.")
        self.pose_scale = pose_scale

    @property
    def target_dim(self) -> int:
        return self.squeeze_pose_dim

    def build_from_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.normalize(batch["squeeze_pose"])

    def normalize(self, pose: torch.Tensor) -> torch.Tensor:
        if not self.normalization_enabled:
            return pose
        return pose / self.pose_scale

    def denormalize(self, pose: torch.Tensor) -> torch.Tensor:
        if not self.normalization_enabled:
            return pose
        return pose * self.pose_scale

    def split(self, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        if prediction.shape[-1] != self.squeeze_pose_dim:
            raise ValueError(
                f"Unexpected squeeze pose size: expected {self.squeeze_pose_dim}, "
                f"got {prediction.shape[-1]}."
            )
        return {
            "pred_squeeze_pose": self.denormalize(prediction),
        }


class InitJointCodec:
    """针对 init_pose 与 squeeze_joint 的回归编解码器。"""

    def __init__(
        self,
        init_pose_dim: int,
        joint_dim: int,
        normalization_config: dict[str, Any] | None = None,
    ) -> None:
        self.init_pose_dim = int(init_pose_dim)
        self.joint_dim = int(joint_dim)
        self.target_dim = self.init_pose_dim + self.joint_dim
        normalization = dict(normalization_config or {})
        self.normalization_enabled = bool(normalization.get("enabled", False))
        pose_scale = float(normalization.get("pose_scale", 1.0))
        joint_scale = float(normalization.get("joint_scale", 1.0))
        if pose_scale <= 0.0:
            raise ValueError(f"target_normalization.pose_scale must be positive, got {pose_scale}.")
        if joint_scale <= 0.0:
            raise ValueError(
                f"target_normalization.joint_scale must be positive, got {joint_scale}."
            )
        self.pose_scale = pose_scale
        self.joint_scale = joint_scale

    def build_from_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = torch.cat([batch["init_pose"], batch["squeeze_joint"]], dim=-1)
        return self.normalize(target)

    def normalize(self, target: torch.Tensor) -> torch.Tensor:
        if not self.normalization_enabled:
            return target
        pieces = self._split_raw(target)
        return torch.cat(
            [
                pieces["pred_init_pose"] / self.pose_scale,
                pieces["pred_squeeze_joint"] / self.joint_scale,
            ],
            dim=-1,
        )

    def denormalize(self, prediction: torch.Tensor) -> torch.Tensor:
        if not self.normalization_enabled:
            return prediction
        pieces = self._split_raw(prediction)
        return torch.cat(
            [
                pieces["pred_init_pose"] * self.pose_scale,
                pieces["pred_squeeze_joint"] * self.joint_scale,
            ],
            dim=-1,
        )

    def split(self, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._split_raw(self.denormalize(prediction))

    def merge(
        self,
        squeeze_pose: torch.Tensor,
        init_pose_and_joint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        merged = self.split(init_pose_and_joint)
        merged["pred_squeeze_pose"] = squeeze_pose
        return merged

    def _split_raw(self, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        if prediction.shape[-1] != self.target_dim:
            raise ValueError(
                f"Unexpected init/joint size: expected {self.target_dim}, "
                f"got {prediction.shape[-1]}."
            )
        init_end = self.init_pose_dim
        return {
            "pred_init_pose": prediction[..., :init_end],
            "pred_squeeze_joint": prediction[..., init_end : init_end + self.joint_dim],
        }
