from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict[str, Any]:
    """加载 YAML 配置文件。"""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """应用 `--set a.b=value` 风格的 CLI 覆盖。"""
    merged = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}', expected KEY=VALUE.")
        key_path, raw_value = override.split("=", 1)
        keys = [part.strip() for part in key_path.split(".") if part.strip()]
        if not keys:
            raise ValueError(f"Invalid override key path: '{override}'")
        value = yaml.safe_load(raw_value)
        target = merged
        for key in keys[:-1]:
            current = target.get(key)
            if current is None:
                current = {}
                target[key] = current
            if not isinstance(current, dict):
                raise ValueError(f"Cannot override nested key under non-mapping: '{key_path}'")
            target = current
        target[keys[-1]] = value
    return merged


def set_random_seed(seed: int) -> None:
    """统一设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_frame(frame: str) -> str:
    frame_name = str(frame).strip().lower()
    if frame_name == "cam":
        frame_name = "camera"
    if frame_name not in {"world", "camera"}:
        raise ValueError(f"Unsupported data.frame='{frame}'. Expected world or camera.")
    return frame_name


def normalize_cloud_type(cloud_type: str) -> str:
    cloud_name = str(cloud_type).strip().lower()
    if cloud_name in {"full", "complete"}:
        cloud_name = "global"
    if cloud_name not in {"global", "partial"}:
        raise ValueError(
            f"Unsupported data.cloud_type='{cloud_type}'. Expected global or partial."
        )
    return cloud_name


def get_required(config: dict[str, Any], dotted_key: str) -> Any:
    """读取必填配置项，缺失则立即报错。"""
    current: Any = config
    traversed: list[str] = []
    for key in dotted_key.split("."):
        traversed.append(key)
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing required config key: {'.'.join(traversed)}")
        current = current[key]
    return current


def validate_common_config(config: dict[str, Any]) -> None:
    """校验 train / sim 共用的必要字段。"""
    get_required(config, "seed")
    get_required(config, "data.manifest_path")
    normalize_cloud_type(get_required(config, "data.cloud_type"))
    normalize_frame(get_required(config, "data.frame"))
    get_required(config, "data.n_points")
    get_required(config, "model.algorithm")
    get_required(config, "model.input_encoder.name")
    get_required(config, "model.common.point_feat_dim")
    get_required(config, "model.common.joint_dim")
    get_required(config, "model.algorithms.cvae.latent_dim")
    get_required(config, "hand.xml_path")
    get_required(config, "hand.prepared_joints")
    get_required(config, "hand.target_body_params")


def validate_train_config(config: dict[str, Any]) -> None:
    """校验训练所需配置。"""
    validate_common_config(config)
    get_required(config, "train.batch_size")
    get_required(config, "train.max_steps")
    get_required(config, "train.lr")
    get_required(config, "train.output_dir")


def validate_sim_config(config: dict[str, Any]) -> None:
    """校验仿真所需配置。"""
    validate_common_config(config)
    get_required(config, "sim.samples_per_object_scale")
    get_required(config, "sim.num_grasp_samples")
    get_required(config, "sim.extforce.duration")
    get_required(config, "sim.extforce.trans_thresh")
    get_required(config, "sim.extforce.angle_thresh")
    get_required(config, "sim.extforce.force_mag")
    get_required(config, "sim.extforce.check_step")
