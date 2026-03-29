from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.config import normalize_cloud_type, normalize_frame
from src.manifest import ManifestItem, load_manifest
from src.transforms import (
    matrix_to_se3_log,
    qpos_to_matrix,
    sample_point_cloud,
    world_to_camera_points,
    world_to_camera_pose,
)


@dataclass(frozen=True)
class PointCloudSample:
    """单次采样得到的点云条件。"""

    point_cloud: np.ndarray
    view_index: int | None
    cam_extrinsic: np.ndarray | None


class GraspDatasetSC(Dataset):
    """单条件抓取数据集。"""

    def __init__(
        self,
        manifest_path: str,
        split: str,
        cloud_type: str,
        frame: str,
        n_points: int,
        joint_dim: int,
        seed: int,
    ) -> None:
        self.dataset_root, self.items = load_manifest(manifest_path, split=split)
        self.split = split
        self.cloud_type = normalize_cloud_type(cloud_type)
        self.frame = normalize_frame(frame)
        self.n_points = int(n_points)
        self.joint_dim = int(joint_dim)
        self.seed = int(seed)
        self.object_name_to_indices = self._build_object_index()
        self._validate_manifest_items()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        item_index, sample_seed = _unpack_sample_index(index=index, base_seed=self.seed)
        rng = np.random.default_rng(sample_seed)
        item = self.items[item_index]
        qpos_init, qpos_squeeze = _load_grasp_poses(
            dataset_root=self.dataset_root,
            item=item,
        )
        grasp_count = qpos_squeeze.shape[0]
        if grasp_count <= 0:
            raise ValueError(f"{item.object_scale_key} has no squeeze grasps.")
        grasp_index = int(rng.integers(0, grasp_count))
        point_cloud_sample = load_conditioning_point_cloud(
            dataset_root=self.dataset_root,
            item=item,
            cloud_type=self.cloud_type,
            frame=self.frame,
            n_points=self.n_points,
            rng=rng,
        )

        init_matrix = qpos_to_matrix(qpos_init[grasp_index])
        squeeze_matrix = qpos_to_matrix(qpos_squeeze[grasp_index])
        if self.frame == "camera":
            if point_cloud_sample.cam_extrinsic is None:
                raise ValueError(
                    f"{item.object_scale_key} is missing camera extrinsic for camera mode."
                )
            init_matrix = world_to_camera_pose(init_matrix, point_cloud_sample.cam_extrinsic)
            squeeze_matrix = world_to_camera_pose(
                squeeze_matrix, point_cloud_sample.cam_extrinsic
            )

        return {
            "point_cloud": torch.tensor(point_cloud_sample.point_cloud, dtype=torch.float32),
            "init_pose": torch.tensor(matrix_to_se3_log(init_matrix), dtype=torch.float32),
            "squeeze_pose": torch.tensor(
                matrix_to_se3_log(squeeze_matrix), dtype=torch.float32
            ),
            "squeeze_joint": torch.tensor(
                qpos_squeeze[grasp_index, 7 : 7 + self.joint_dim], dtype=torch.float32
            ),
            "meta": {
                "object_scale_key": item.object_scale_key,
                "object_name": item.object_name,
                "cloud_type": self.cloud_type,
                "frame": self.frame,
                "view_index": (
                    -1 if point_cloud_sample.view_index is None else point_cloud_sample.view_index
                ),
                "grasp_index": grasp_index,
            },
        }

    def _build_object_index(self) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = {}
        for index, item in enumerate(self.items):
            mapping.setdefault(item.object_name, []).append(index)
        return mapping

    def _validate_manifest_items(self) -> None:
        for item in self.items:
            grasp_file = self.dataset_root / item.grasp_h5_path
            if not grasp_file.exists():
                raise FileNotFoundError(f"grasp.h5 not found: {grasp_file}")
            if self.cloud_type == "global" and item.global_pc_path is None:
                raise KeyError(
                    f"{item.object_scale_key} is missing global_pc_path for global mode."
                )
            if self.cloud_type == "partial" and self.frame == "world" and not item.partial_pc_path:
                raise KeyError(
                    f"{item.object_scale_key} is missing partial_pc_path for partial mode."
                )
            if (
                self.cloud_type == "partial"
                and self.frame == "camera"
                and not item.partial_pc_cam_path
            ):
                raise KeyError(
                    f"{item.object_scale_key} is missing partial_pc_cam_path for camera partial mode."
                )
            if self.frame == "camera" and not item.cam_ex_path:
                raise KeyError(
                    f"{item.object_scale_key} is missing cam_ex_path for camera mode."
                )


class DistinctObjectBatchSampler(Sampler[list[tuple[int, int]]]):
    """按 step 生成 batch，并保证 batch 内 object 不重复。"""

    def __init__(
        self,
        dataset: GraspDatasetSC,
        batch_size: int,
        num_steps: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_steps = int(num_steps)
        self.seed = int(seed)
        self.object_names = sorted(dataset.object_name_to_indices)
        if self.batch_size > len(self.object_names):
            raise ValueError(
                "batch_size cannot exceed the number of unique objects. "
                f"batch_size={self.batch_size}, unique_objects={len(self.object_names)}"
            )

    def __iter__(self) -> Iterable[list[tuple[int, int]]]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_steps):
            selected_objects = rng.choice(
                self.object_names, size=self.batch_size, replace=False
            )
            batch: list[tuple[int, int]] = []
            for object_name in selected_objects.tolist():
                candidates = self.dataset.object_name_to_indices[object_name]
                item_index = int(candidates[int(rng.integers(0, len(candidates)))])
                sample_seed = int(rng.integers(0, 2**31 - 1))
                batch.append((item_index, sample_seed))
            yield batch

    def __len__(self) -> int:
        return self.num_steps


def load_conditioning_point_cloud(
    dataset_root: Path,
    item: ManifestItem,
    cloud_type: str,
    frame: str,
    n_points: int,
    rng: np.random.Generator,
) -> PointCloudSample:
    """根据模式读取点云条件。"""
    cloud_mode = normalize_cloud_type(cloud_type)
    frame_name = normalize_frame(frame)
    if cloud_mode == "partial":
        view_index = _sample_view_index(item=item, frame=frame_name, rng=rng)
        cam_extrinsic = _load_cam_extrinsic(
            dataset_root=dataset_root,
            item=item,
            view_index=view_index,
            frame=frame_name,
        )
        point_cloud = _load_partial_point_cloud(
            dataset_root=dataset_root,
            item=item,
            view_index=view_index,
            frame=frame_name,
        )
        return PointCloudSample(
            point_cloud=sample_point_cloud(point_cloud, n_points=n_points, rng=rng),
            view_index=view_index,
            cam_extrinsic=cam_extrinsic,
        )

    if item.global_pc_path is None:
        raise KeyError(f"{item.object_scale_key} is missing global_pc_path.")
    point_cloud_world = _load_npy_point_cloud(dataset_root / item.global_pc_path)
    view_index = None
    cam_extrinsic = None
    if frame_name == "camera":
        view_index = _sample_view_index(item=item, frame=frame_name, rng=rng)
        cam_extrinsic = _load_cam_extrinsic(
            dataset_root=dataset_root,
            item=item,
            view_index=view_index,
            frame=frame_name,
        )
        point_cloud_world = world_to_camera_points(point_cloud_world, cam_extrinsic)
    return PointCloudSample(
        point_cloud=sample_point_cloud(point_cloud_world, n_points=n_points, rng=rng),
        view_index=view_index,
        cam_extrinsic=cam_extrinsic,
    )


def _load_grasp_poses(
    dataset_root: Path,
    item: ManifestItem,
) -> tuple[np.ndarray, np.ndarray]:
    grasp_h5_path = dataset_root / item.grasp_h5_path
    with h5py.File(grasp_h5_path, "r") as handle:
        qpos_init = handle["qpos_init"][:].astype(np.float32)
        qpos_squeeze = handle["qpos_squeeze"][:].astype(np.float32)
    if qpos_init.shape[0] != qpos_squeeze.shape[0]:
        raise ValueError(
            f"{item.object_scale_key} has mismatched init/squeeze grasp counts: "
            f"{qpos_init.shape[0]} vs {qpos_squeeze.shape[0]}"
        )
    return qpos_init, qpos_squeeze


def _load_partial_point_cloud(
    dataset_root: Path,
    item: ManifestItem,
    view_index: int,
    frame: str,
) -> np.ndarray:
    if frame == "world":
        return _load_npy_point_cloud(dataset_root / item.partial_pc_path[view_index])
    return _load_npy_point_cloud(dataset_root / item.partial_pc_cam_path[view_index])


def _load_cam_extrinsic(
    dataset_root: Path,
    item: ManifestItem,
    view_index: int,
    frame: str,
) -> np.ndarray | None:
    if frame != "camera":
        return None
    if not item.cam_ex_path:
        raise KeyError(f"{item.object_scale_key} is missing cam_ex_path.")
    cam_extrinsic = np.load(dataset_root / item.cam_ex_path[view_index]).astype(np.float32)
    if cam_extrinsic.shape != (4, 4):
        raise ValueError(
            f"Camera extrinsic must have shape [4, 4], got {cam_extrinsic.shape}."
        )
    return cam_extrinsic


def _sample_view_index(
    item: ManifestItem,
    frame: str,
    rng: np.random.Generator,
) -> int:
    if frame == "world":
        candidates = item.partial_pc_path
    else:
        candidates = item.partial_pc_cam_path
    if not candidates:
        raise ValueError(f"{item.object_scale_key} has no {frame} point cloud views.")
    return int(rng.integers(0, len(candidates)))


def _load_npy_point_cloud(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")
    points = np.load(path).astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud must have shape [N, 3], got {points.shape}.")
    return points


def _unpack_sample_index(index: int | tuple[int, int], base_seed: int) -> tuple[int, int]:
    if isinstance(index, tuple):
        if len(index) != 2:
            raise ValueError(f"Sample index tuple must have length 2, got {index}")
        item_index, sample_seed = index
        return int(item_index), int(sample_seed)
    return int(index), int(base_seed + int(index))
