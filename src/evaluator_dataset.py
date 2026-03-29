from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import normalize_cloud_type, normalize_frame, normalize_point_sampling
from src.grasp_dataset_sc import DistinctObjectBatchSampler, load_conditioning_point_cloud
from src.manifest import ManifestItem, load_manifest
from src.transforms import matrix_to_se3_log, qpos_to_matrix, world_to_camera_pose


NEGATIVE_STAGE_ORDER = (
    "prepared_contact",
    "insufficient_contact",
    "extforce_failure",
)
STAGE_TO_LABEL = {
    "prepared_contact": 0,
    "insufficient_contact": 1,
    "extforce_failure": 2,
    "positive": 3,
}
STAGE_TO_SCORE = {
    "prepared_contact": 0.0,
    "insufficient_contact": 0.08,
    "extforce_failure": 0.25,
    "positive": 1.0,
}


@dataclass(frozen=True)
class EvaluatorSampleInfo:
    """缓存每个 object-scale 的正样本数量与负样本分档索引。"""

    positive_count: int
    negative_indices_by_stage: dict[str, np.ndarray]


class EvaluatorDataset(Dataset):
    """抓取评估网络数据集，输出一个 observation 对应的 K 个抓取及其档位标签。"""

    def __init__(
        self,
        manifest_path: str,
        split: str,
        cloud_type: str,
        frame: str,
        n_points: int,
        joint_dim: int,
        grasps_per_object: int,
        seed: int,
        positive_ratio: float = 0.5,
        point_sampling: str = "random",
    ) -> None:
        self.dataset_root, self.items = load_manifest(manifest_path, split=split)
        self.split = split
        self.cloud_type = normalize_cloud_type(cloud_type)
        self.frame = normalize_frame(frame)
        self.point_sampling = normalize_point_sampling(point_sampling)
        self.n_points = int(n_points)
        self.joint_dim = int(joint_dim)
        self.grasps_per_object = int(grasps_per_object)
        self.seed = int(seed)
        self.positive_ratio = float(positive_ratio)
        if self.grasps_per_object < 2:
            raise ValueError(
                f"grasps_per_object must be at least 2 for evaluator ranking, got {self.grasps_per_object}."
            )
        if not 0.0 < self.positive_ratio < 1.0:
            raise ValueError(f"positive_ratio must be in (0, 1), got {self.positive_ratio}.")
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
            point_sampling=self.point_sampling,
        )

        num_positive, num_negative = _split_group_counts(
            grasps_per_object=self.grasps_per_object,
            positive_ratio=self.positive_ratio,
        )
        positive_rows = _sample_positive_rows(
            dataset_root=self.dataset_root,
            item=item,
            joint_dim=self.joint_dim,
            sample_count=sample_info.positive_count,
            count=num_positive,
            rng=rng,
        )
        negative_rows = _sample_negative_rows(
            dataset_root=self.dataset_root,
            item=item,
            joint_dim=self.joint_dim,
            negative_indices_by_stage=sample_info.negative_indices_by_stage,
            count=num_negative,
            rng=rng,
        )
        rows = positive_rows + negative_rows
        permutation = rng.permutation(len(rows))
        rows = [rows[int(index)] for index in permutation.tolist()]

        grasp_pose_list: list[np.ndarray] = []
        grasp_joint_list: list[np.ndarray] = []
        stage_labels: list[int] = []
        target_scores: list[float] = []
        grasp_indices: list[int] = []

        for row in rows:
            grasp_matrix = qpos_to_matrix(row.qpos)
            if self.frame == "camera":
                if point_cloud_sample.cam_extrinsic is None:
                    raise ValueError(
                        f"{item.object_scale_key} is missing camera extrinsic for camera mode."
                    )
                grasp_matrix = world_to_camera_pose(
                    grasp_matrix, point_cloud_sample.cam_extrinsic
                )
            grasp_pose_list.append(matrix_to_se3_log(grasp_matrix))
            grasp_joint_list.append(row.qpos[7 : 7 + self.joint_dim].astype(np.float32))
            stage_labels.append(STAGE_TO_LABEL[row.stage_name])
            target_scores.append(STAGE_TO_SCORE[row.stage_name])
            grasp_indices.append(row.grasp_index)

        return {
            "point_cloud": torch.tensor(point_cloud_sample.point_cloud, dtype=torch.float32),
            "grasp_pose": torch.tensor(np.stack(grasp_pose_list, axis=0), dtype=torch.float32),
            "grasp_joint": torch.tensor(np.stack(grasp_joint_list, axis=0), dtype=torch.float32),
            "stage_label": torch.tensor(stage_labels, dtype=torch.long),
            "target_score": torch.tensor(target_scores, dtype=torch.float32),
            "grasp_indices": torch.tensor(grasp_indices, dtype=torch.long),
            "meta": {
                "object_scale_key": item.object_scale_key,
                "object_name": item.object_name,
                "cloud_type": self.cloud_type,
                "frame": self.frame,
                "point_sampling": self.point_sampling,
                "view_index": (
                    -1 if point_cloud_sample.view_index is None else point_cloud_sample.view_index
                ),
                "num_grasps": self.grasps_per_object,
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


@dataclass(frozen=True)
class EvaluatorRow:
    """组内一个抓取候选。"""

    qpos: np.ndarray
    grasp_index: int
    stage_name: str


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
        failure_stage = [_decode_failure_stage(raw) for raw in handle["failure_stage"][:]]

    _validate_qpos_array(qpos_squeeze, joint_dim=joint_dim, name="qpos_squeeze")
    _validate_qpos_array(qpos_fail, joint_dim=joint_dim, name="qpos_fail")
    if qpos_squeeze.shape[0] <= 0:
        raise ValueError(f"{item.object_scale_key} has no positive evaluator samples.")
    if qpos_fail.shape[0] <= 0:
        raise ValueError(f"{item.object_scale_key} has no negative evaluator samples.")

    negative_indices_by_stage: dict[str, np.ndarray] = {}
    for stage_name in NEGATIVE_STAGE_ORDER:
        indices = [index for index, value in enumerate(failure_stage) if value == stage_name]
        negative_indices_by_stage[stage_name] = np.asarray(indices, dtype=np.int64)
    if sum(int(values.shape[0]) for values in negative_indices_by_stage.values()) <= 0:
        raise ValueError(f"{item.object_scale_key} has no recognized negative failure stages.")

    return EvaluatorSampleInfo(
        positive_count=int(qpos_squeeze.shape[0]),
        negative_indices_by_stage=negative_indices_by_stage,
    )


def _split_group_counts(
    grasps_per_object: int,
    positive_ratio: float,
) -> tuple[int, int]:
    num_positive = int(round(grasps_per_object * positive_ratio))
    num_positive = min(max(num_positive, 1), grasps_per_object - 1)
    num_negative = grasps_per_object - num_positive
    return num_positive, num_negative


def _sample_positive_rows(
    dataset_root: Path,
    item: ManifestItem,
    joint_dim: int,
    sample_count: int,
    count: int,
    rng: np.random.Generator,
) -> list[EvaluatorRow]:
    grasp_indices = rng.choice(sample_count, size=count, replace=sample_count < count).astype(np.int64)
    rows: list[EvaluatorRow] = []
    with h5py.File(dataset_root / item.grasp_h5_path, "r") as handle:
        for grasp_index in grasp_indices.tolist():
            qpos = handle["qpos_squeeze"][int(grasp_index)].astype(np.float32)
            _validate_qpos_vector(qpos, joint_dim=joint_dim, name="qpos_squeeze")
            rows.append(
                EvaluatorRow(
                    qpos=qpos,
                    grasp_index=int(grasp_index),
                    stage_name="positive",
                )
            )
    return rows


def _sample_negative_rows(
    dataset_root: Path,
    item: ManifestItem,
    joint_dim: int,
    negative_indices_by_stage: dict[str, np.ndarray],
    count: int,
    rng: np.random.Generator,
) -> list[EvaluatorRow]:
    if item.grasp_h5_fail_path is None:
        raise KeyError(f"{item.object_scale_key} is missing grasp_h5_fail_path.")
    available_stages = [
        stage_name
        for stage_name in NEGATIVE_STAGE_ORDER
        if negative_indices_by_stage[stage_name].shape[0] > 0
    ]
    if not available_stages:
        raise ValueError(f"{item.object_scale_key} has no negative stages for evaluator.")

    selected_stages: list[str] = []
    stage_cycle = available_stages.copy()
    while len(selected_stages) < count:
        rng.shuffle(stage_cycle)
        for stage_name in stage_cycle:
            selected_stages.append(stage_name)
            if len(selected_stages) == count:
                break

    rows: list[EvaluatorRow] = []
    with h5py.File(dataset_root / item.grasp_h5_fail_path, "r") as handle:
        for stage_name in selected_stages:
            candidate_indices = negative_indices_by_stage[stage_name]
            chosen_position = int(rng.integers(0, candidate_indices.shape[0]))
            grasp_index = int(candidate_indices[chosen_position])
            qpos = handle["qpos_fail"][grasp_index].astype(np.float32)
            _validate_qpos_vector(qpos, joint_dim=joint_dim, name="qpos_fail")
            rows.append(
                EvaluatorRow(
                    qpos=qpos,
                    grasp_index=grasp_index,
                    stage_name=stage_name,
                )
            )
    return rows


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
