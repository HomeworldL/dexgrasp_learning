from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import normalize_cloud_type, normalize_frame
from src.grasp_dataset_sc import DistinctObjectBatchSampler, load_conditioning_point_cloud
from src.manifest import ManifestItem, load_manifest
from src.transforms import matrix_to_se3_log, qpos_to_matrix, world_to_camera_pose


@dataclass(frozen=True)
class EvaluatorSampleInfo:
    """缓存每个 object-scale 的正负样本数量。"""

    positive_count: int
    negative_count: int


class EvaluatorDataset(Dataset):
    """抓取评估网络数据集，使用正负抓取做二分类。"""

    def __init__(
        self,
        manifest_path: str,
        split: str,
        cloud_type: str,
        frame: str,
        n_points: int,
        joint_dim: int,
        seed: int,
        positive_probability: float = 0.5,
    ) -> None:
        self.dataset_root, self.items = load_manifest(manifest_path, split=split)
        self.split = split
        self.cloud_type = normalize_cloud_type(cloud_type)
        self.frame = normalize_frame(frame)
        self.n_points = int(n_points)
        self.joint_dim = int(joint_dim)
        self.seed = int(seed)
        self.positive_probability = float(positive_probability)
        if not 0.0 < self.positive_probability < 1.0:
            raise ValueError(
                "positive_probability must be in (0, 1), "
                f"got {self.positive_probability}."
            )
        self.object_name_to_indices = self._build_object_index()
        self.sample_info = self._build_sample_info()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        item_index, sample_seed = _unpack_sample_index(index=index, base_seed=self.seed)
        rng = np.random.default_rng(sample_seed)
        item = self.items[item_index]
        sample_info = self.sample_info[item_index]
        point_cloud_sample = load_conditioning_point_cloud(
            dataset_root=self.dataset_root,
            item=item,
            cloud_type=self.cloud_type,
            frame=self.frame,
            n_points=self.n_points,
            rng=rng,
        )

        use_positive = bool(rng.random() < self.positive_probability)
        if use_positive:
            qpos, grasp_index = _load_positive_qpos(
                dataset_root=self.dataset_root,
                item=item,
                joint_dim=self.joint_dim,
                sample_count=sample_info.positive_count,
                rng=rng,
            )
            label = 1.0
            failure_stage = "positive"
        else:
            qpos, grasp_index, failure_stage = _load_negative_qpos(
                dataset_root=self.dataset_root,
                item=item,
                joint_dim=self.joint_dim,
                sample_count=sample_info.negative_count,
                rng=rng,
            )
            label = 0.0

        grasp_matrix = qpos_to_matrix(qpos)
        if self.frame == "camera":
            if point_cloud_sample.cam_extrinsic is None:
                raise ValueError(
                    f"{item.object_scale_key} is missing camera extrinsic for camera mode."
                )
            grasp_matrix = world_to_camera_pose(grasp_matrix, point_cloud_sample.cam_extrinsic)

        return {
            "point_cloud": torch.tensor(point_cloud_sample.point_cloud, dtype=torch.float32),
            "grasp_pose": torch.tensor(matrix_to_se3_log(grasp_matrix), dtype=torch.float32),
            "grasp_joint": torch.tensor(
                qpos[7 : 7 + self.joint_dim], dtype=torch.float32
            ),
            "label": torch.tensor(label, dtype=torch.float32),
            "meta": {
                "object_scale_key": item.object_scale_key,
                "object_name": item.object_name,
                "cloud_type": self.cloud_type,
                "frame": self.frame,
                "view_index": (
                    -1 if point_cloud_sample.view_index is None else point_cloud_sample.view_index
                ),
                "grasp_index": grasp_index,
                "label": int(label),
                "failure_stage": failure_stage,
            },
        }

    def _build_object_index(self) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = {}
        for index, item in enumerate(self.items):
            mapping.setdefault(item.object_name, []).append(index)
        return mapping

    def _build_sample_info(self) -> list[EvaluatorSampleInfo]:
        sample_info: list[EvaluatorSampleInfo] = []
        for item in self.items:
            sample_info.append(
                _inspect_sample_info(
                    dataset_root=self.dataset_root,
                    item=item,
                    joint_dim=self.joint_dim,
                )
            )
        return sample_info


def _inspect_sample_info(
    dataset_root: Path,
    item: ManifestItem,
    joint_dim: int,
) -> EvaluatorSampleInfo:
    positive_path = dataset_root / item.grasp_h5_path
    if not positive_path.exists():
        raise FileNotFoundError(f"Positive grasp file not found: {positive_path}")
    if item.grasp_h5_fail_path is None:
        raise KeyError(f"{item.object_scale_key} is missing grasp_h5_fail_path.")
    negative_path = dataset_root / item.grasp_h5_fail_path
    if not negative_path.exists():
        raise FileNotFoundError(f"Negative grasp file not found: {negative_path}")

    with h5py.File(positive_path, "r") as handle:
        qpos_squeeze = handle["qpos_squeeze"][:].astype(np.float32)
    with h5py.File(negative_path, "r") as handle:
        qpos_fail = handle["qpos_fail"][:].astype(np.float32)

    _validate_qpos_array(qpos_squeeze, joint_dim=joint_dim, name="qpos_squeeze")
    _validate_qpos_array(qpos_fail, joint_dim=joint_dim, name="qpos_fail")
    if qpos_squeeze.shape[0] <= 0:
        raise ValueError(f"{item.object_scale_key} has no positive evaluator samples.")
    if qpos_fail.shape[0] <= 0:
        raise ValueError(f"{item.object_scale_key} has no negative evaluator samples.")
    return EvaluatorSampleInfo(
        positive_count=int(qpos_squeeze.shape[0]),
        negative_count=int(qpos_fail.shape[0]),
    )


def _load_positive_qpos(
    dataset_root: Path,
    item: ManifestItem,
    joint_dim: int,
    sample_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    grasp_index = int(rng.integers(0, sample_count))
    with h5py.File(dataset_root / item.grasp_h5_path, "r") as handle:
        qpos = handle["qpos_squeeze"][grasp_index].astype(np.float32)
    _validate_qpos_vector(qpos, joint_dim=joint_dim, name="qpos_squeeze")
    return qpos, grasp_index


def _load_negative_qpos(
    dataset_root: Path,
    item: ManifestItem,
    joint_dim: int,
    sample_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, str]:
    if item.grasp_h5_fail_path is None:
        raise KeyError(f"{item.object_scale_key} is missing grasp_h5_fail_path.")
    grasp_index = int(rng.integers(0, sample_count))
    with h5py.File(dataset_root / item.grasp_h5_fail_path, "r") as handle:
        qpos = handle["qpos_fail"][grasp_index].astype(np.float32)
        failure_stage = _decode_failure_stage(handle["failure_stage"][grasp_index])
    _validate_qpos_vector(qpos, joint_dim=joint_dim, name="qpos_fail")
    return qpos, grasp_index, failure_stage


def _validate_qpos_array(array: np.ndarray, joint_dim: int, name: str) -> None:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape [B, D], got {values.shape}.")
    if values.shape[1] < 7 + int(joint_dim):
        raise ValueError(
            f"{name} must have at least {7 + int(joint_dim)} dims, got {values.shape[1]}."
        )


def _validate_qpos_vector(values: np.ndarray, joint_dim: int, name: str) -> None:
    qpos = np.asarray(values, dtype=np.float32).reshape(-1)
    if qpos.shape[0] < 7 + int(joint_dim):
        raise ValueError(
            f"{name} must have at least {7 + int(joint_dim)} dims, got {qpos.shape[0]}."
        )


def _decode_failure_stage(raw_value: Any) -> str:
    if isinstance(raw_value, bytes):
        return raw_value.decode("utf-8")
    if hasattr(raw_value, "decode"):
        return raw_value.decode("utf-8")
    return str(raw_value)


def _unpack_sample_index(index: int | tuple[int, int], base_seed: int) -> tuple[int, int]:
    if isinstance(index, tuple):
        if len(index) != 2:
            raise ValueError(f"Sample index tuple must have length 2, got {index}")
        item_index, sample_seed = index
        return int(item_index), int(sample_seed)
    return int(index), int(base_seed + int(index))


__all__ = ["DistinctObjectBatchSampler", "EvaluatorDataset"]
