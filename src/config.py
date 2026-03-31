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


def normalize_point_sampling(point_sampling: str) -> str:
    sampling_name = str(point_sampling).strip().lower()
    if sampling_name not in {"fps", "random"}:
        raise ValueError(
            "Unsupported data.point_sampling="
            f"'{point_sampling}'. Expected fps or random."
        )
    return sampling_name


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


def _validate_extforce_mapping(extforce: dict[str, Any], key_prefix: str) -> None:
    for key in ("duration", "trans_thresh", "angle_thresh", "force_mag", "check_steps", "close_steps"):
        if key not in extforce:
            raise KeyError(f"Missing required config key: {key_prefix}.{key}")
    if float(extforce["duration"]) <= 0.0:
        raise ValueError(f"{key_prefix}.duration must be positive.")
    if float(extforce["trans_thresh"]) < 0.0:
        raise ValueError(f"{key_prefix}.trans_thresh must be non-negative.")
    if float(extforce["angle_thresh"]) < 0.0:
        raise ValueError(f"{key_prefix}.angle_thresh must be non-negative.")
    if float(extforce["force_mag"]) < 0.0:
        raise ValueError(f"{key_prefix}.force_mag must be non-negative.")
    if int(extforce["check_steps"]) <= 0:
        raise ValueError(f"{key_prefix}.check_steps must be positive.")
    if int(extforce["close_steps"]) <= 0:
        raise ValueError(f"{key_prefix}.close_steps must be positive.")


def _validate_sim_runtime_config(sim_config: dict[str, Any]) -> None:
    friction = sim_config.get("friction")
    if friction is None:
        return
    friction_values = np.asarray(friction, dtype=np.float32).reshape(-1)
    if friction_values.size not in {1, 2, 3}:
        raise ValueError(
            "sim.friction must have length 1, 2, or 3. "
            f"Got {int(friction_values.size)}."
        )
    if np.any(friction_values < 0.0):
        raise ValueError("sim.friction must be non-negative.")


def validate_common_config(config: dict[str, Any]) -> None:
    """校验 train / sim 共用的必要字段。"""
    get_required(config, "seed")
    get_required(config, "data.manifest_path")
    normalize_cloud_type(get_required(config, "data.cloud_type"))
    normalize_frame(get_required(config, "data.frame"))
    normalize_point_sampling(config.get("data", {}).get("point_sampling", "random"))
    get_required(config, "data.n_points")
    get_required(config, "model.algorithm")
    get_required(config, "model.input_encoder.name")
    get_required(config, "model.common.point_feat_dim")
    get_required(config, "model.common.joint_dim")
    _validate_algorithm_config(config)
    get_required(config, "hand.xml_path")
    get_required(config, "hand.prepared_joints")
    get_required(config, "hand.target_body_params")


def _validate_algorithm_config(config: dict[str, Any]) -> None:
    algorithm = str(get_required(config, "model.algorithm")).strip().lower()
    if algorithm == "cvae":
        get_required(config, "model.algorithms.cvae.latent_dim")
        return
    if algorithm == "dexdiffuser":
        get_required(config, "model.algorithms.dexdiffuser.condition.context_dim")
        get_required(config, "model.algorithms.dexdiffuser.unet.d_model")
        get_required(config, "model.algorithms.dexdiffuser.diffusion.steps")
        get_required(config, "model.algorithms.dexdiffuser.diffusion.schedule.beta")
        get_required(config, "model.algorithms.dexdiffuser.diffusion.schedule.beta_schedule")
        input_encoder_name = str(get_required(config, "model.input_encoder.name")).strip().lower()
        if input_encoder_name == "pointnet":
            get_required(
                config,
                "model.algorithms.dexdiffuser.condition.pointnet.num_condition_tokens",
            )
            return
        if input_encoder_name == "bps":
            get_required(
                config,
                "model.algorithms.dexdiffuser.condition.bps.num_condition_tokens",
            )
            return
        raise NotImplementedError(
            f"DexDiffuser currently supports pointnet and bps, got {input_encoder_name}."
        )
    if algorithm == "udgm":
        get_required(config, "model.algorithms.udgm.condition_dim")
        get_required(config, "model.algorithms.udgm.flow.hidden_dim")
        get_required(config, "model.algorithms.udgm.flow.num_layers")
        get_required(config, "model.algorithms.udgm.flow.num_blocks_per_layer")
        return
    raise NotImplementedError(
        f"model.algorithm={algorithm} is reserved for future work. "
        "The current mainline implements cvae, dexdiffuser, and udgm."
    )


def validate_train_config(config: dict[str, Any]) -> None:
    """校验训练所需配置。"""
    validate_common_config(config)
    get_required(config, "train.batch_size")
    get_required(config, "train.max_steps")
    get_required(config, "train.lr")
    get_required(config, "train.output_dir")
    init_ckpt_path = config.get("train", {}).get("init_ckpt_path")
    if init_ckpt_path is not None:
        checkpoint_path = Path(str(init_ckpt_path)).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"train.init_ckpt_path not found: {checkpoint_path}")
    initial_step = config.get("train", {}).get("initial_step")
    if initial_step is not None and int(initial_step) < 0:
        raise ValueError("train.initial_step must be non-negative when provided.")
    loss_weights = config.get("train", {}).get("loss_weights", {})
    if loss_weights:
        if not isinstance(loss_weights, dict):
            raise ValueError("train.loss_weights must be a mapping when provided.")
        for key in ("init_pose", "squeeze_pose", "joint", "kld"):
            if key not in loss_weights:
                continue
            value = float(loss_weights[key])
            if value < 0.0:
                raise ValueError(f"train.loss_weights.{key} must be non-negative, got {value}.")


def validate_sim_config(config: dict[str, Any]) -> None:
    """校验仿真所需配置。"""
    validate_common_config(config)
    get_required(config, "sim.num_grasp_samples")
    sim_config = get_required(config, "sim")
    extforce = get_required(config, "sim.extforce")
    if not isinstance(sim_config, dict) or not isinstance(extforce, dict):
        raise ValueError("sim and sim.extforce must be mappings.")
    _validate_extforce_mapping(extforce, key_prefix="sim.extforce")
    _validate_sim_runtime_config(sim_config)
    if bool(config.get("evaluator", {}).get("enabled", False)):
        get_required(config, "evaluator.ckpt_path")
        get_required(config, "evaluator.topk")


def validate_evaluator_train_config(config: dict[str, Any]) -> None:
    """校验评估网络训练所需配置。"""
    get_required(config, "seed")
    get_required(config, "data.manifest_path")
    normalize_cloud_type(get_required(config, "data.cloud_type"))
    normalize_frame(get_required(config, "data.frame"))
    normalize_point_sampling(config.get("data", {}).get("point_sampling", "random"))
    get_required(config, "data.n_points")
    get_required(config, "model.input_encoder.name")
    get_required(config, "model.common.point_feat_dim")
    get_required(config, "model.common.joint_dim")
    get_required(config, "evaluator.model.grasp_feat_dim")
    get_required(config, "evaluator.model.hidden_features")
    get_required(config, "evaluator.model.num_blocks")
    get_required(config, "evaluator.train.batch_size")
    get_required(config, "evaluator.train.grasps_per_object")
    get_required(config, "evaluator.train.max_steps")
    get_required(config, "evaluator.train.lr")
    get_required(config, "evaluator.train.output_dir")
