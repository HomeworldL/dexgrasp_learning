from __future__ import annotations

from typing import Final

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import se3_exp_map, se3_log_map
from scipy.spatial.transform import Rotation


IDENTITY_4X4: Final[np.ndarray] = np.eye(4, dtype=np.float32)


def qpos_to_matrix(qpos: np.ndarray) -> np.ndarray:
    """把 `[x,y,z,qw,qx,qy,qz,...]` 转为 4x4 齐次矩阵。"""
    values = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if values.shape[0] < 7:
        raise ValueError(f"qpos must have at least 7 values, got {values.shape[0]}.")
    transform = IDENTITY_4X4.copy()
    transform[:3, :3] = Rotation.from_quat(_xyzw_from_wxyz(values[3:7])).as_matrix()
    transform[:3, 3] = values[:3]
    return transform


def matrix_to_qpos(transform: np.ndarray, joint_values: np.ndarray) -> np.ndarray:
    """把齐次矩阵和关节向量还原为 qpos。"""
    matrix = np.asarray(transform, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {matrix.shape}.")
    quat_xyzw = Rotation.from_matrix(project_rotation_matrix(matrix[:3, :3])).as_quat()
    quat_wxyz = _wxyz_from_xyzw(quat_xyzw)
    base = np.concatenate([matrix[:3, 3].astype(np.float32), quat_wxyz], axis=0)
    return np.concatenate([base, np.asarray(joint_values, dtype=np.float32)], axis=0)


def matrix_to_se3_log(transform: np.ndarray) -> np.ndarray:
    """把齐次矩阵转为 PyTorch3D 的 6 维 SE(3) log 表示。"""
    matrix = np.asarray(transform, dtype=np.float32)
    tensor = torch.from_numpy(matrix.T.copy()).unsqueeze(0)
    return se3_log_map(tensor).squeeze(0).cpu().numpy().astype(np.float32)


def se3_log_to_matrix(se3_log: np.ndarray) -> np.ndarray:
    """把 6 维 SE(3) log 表示还原为齐次矩阵。"""
    tensor = torch.from_numpy(np.asarray(se3_log, dtype=np.float32)).unsqueeze(0)
    transform = se3_exp_map(tensor).squeeze(0).cpu().numpy().T.astype(np.float32)
    transform[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    transform[:3, :3] = project_rotation_matrix(transform[:3, :3])
    transform[:3, 3] = np.nan_to_num(
        transform[:3, 3], nan=0.0, posinf=0.0, neginf=0.0
    )
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """计算 4x4 刚体变换的逆。"""
    matrix = np.asarray(transform, dtype=np.float32)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = IDENTITY_4X4.copy()
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """对点云施加 4x4 变换。"""
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {xyz.shape}.")
    rotation = np.asarray(transform, dtype=np.float32)[:3, :3]
    translation = np.asarray(transform, dtype=np.float32)[:3, 3]
    return (xyz @ rotation.T + translation[None, :]).astype(np.float32)


def world_to_camera_pose(world_pose: np.ndarray, cam_extrinsic: np.ndarray) -> np.ndarray:
    """把 world 系位姿转到相机系。"""
    return invert_transform(cam_extrinsic) @ world_pose


def camera_to_world_pose(camera_pose: np.ndarray, cam_extrinsic: np.ndarray) -> np.ndarray:
    """把相机系位姿转回 world 系。"""
    return np.asarray(cam_extrinsic, dtype=np.float32) @ np.asarray(
        camera_pose, dtype=np.float32
    )


def world_to_camera_points(points: np.ndarray, cam_extrinsic: np.ndarray) -> np.ndarray:
    """把 world 系点云转到相机系。"""
    return transform_points(points, invert_transform(cam_extrinsic))


def sample_point_cloud(
    points: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """按规则做 FPS 下采样或随机重复补齐。"""
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"point cloud must have shape [N, 3], got {xyz.shape}.")
    total_points = xyz.shape[0]
    if total_points <= 0:
        raise ValueError("point cloud is empty.")
    if total_points == n_points:
        return xyz.copy()
    if total_points > n_points:
        sampled, _ = sample_farthest_points(
            torch.from_numpy(xyz).unsqueeze(0),
            K=int(n_points),
            random_start_point=False,
        )
        return sampled.squeeze(0).cpu().numpy().astype(np.float32)

    pad_indices = rng.integers(0, total_points, size=n_points - total_points)
    padded = np.concatenate([xyz, xyz[pad_indices]], axis=0)
    rng.shuffle(padded)
    return padded.astype(np.float32)


def project_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """把近似旋转矩阵投影回 SO(3)。"""
    matrix = np.nan_to_num(
        np.asarray(rotation, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    u, _, vh = np.linalg.svd(matrix)
    projected = u @ vh
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected.astype(np.float32)


def _xyzw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float32)
    return np.array([values[1], values[2], values[3], values[0]], dtype=np.float32)


def _wxyz_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float32)
    return np.array([values[3], values[0], values[1], values[2]], dtype=np.float32)
