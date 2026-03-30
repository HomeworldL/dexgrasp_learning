#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from models import build_model
from models.evaluator import GraspEvaluator
from src.config import (
    apply_overrides,
    load_config,
    normalize_cloud_type,
    normalize_frame,
    set_random_seed,
    validate_sim_config,
)
from src.grasp_dataset_sc import load_conditioning_point_cloud
from src.manifest import load_manifest
from src.mj_ho import MjHO
from src.transforms import camera_to_world_pose, matrix_to_qpos, se3_log_to_matrix


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the unified single-condition grasp generator.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set sim.num_grasp_samples=32",
    )
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def build_device(device_name: str) -> torch.device:
    requested = str(device_name).strip()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def load_generator(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    payload = torch.load(checkpoint_file, map_location="cpu")
    checkpoint_config = dict(payload["config"])
    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, checkpoint_config


def load_evaluator(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GraspEvaluator, dict[str, Any]]:
    """加载抓取评估网络 checkpoint。"""
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Evaluator checkpoint not found: {checkpoint_file}")
    payload = torch.load(checkpoint_file, map_location="cpu")
    checkpoint_config = dict(payload["config"])
    model = GraspEvaluator(checkpoint_config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, checkpoint_config


def build_sim_output_dir(checkpoint_path: str) -> Path:
    """仿真输出固定落到训练 run 目录下的 sim/。"""
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    run_dir = checkpoint_file.parent
    sim_dir = run_dir / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    return sim_dir


def build_sim_runtime_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[float] | None]:
    """构建 sim_sc 运行时使用的仿真参数。"""
    sim_config = dict(config.get("sim", {}))
    extforce_config = dict(sim_config.get("extforce", {}))
    friction = sim_config.get("friction")
    if friction is None:
        return extforce_config, None
    friction_values = np.asarray(friction, dtype=np.float32).reshape(-1)
    return extforce_config, friction_values.tolist()


def validate_evaluator_runtime(
    sim_config: dict[str, Any],
    generator_config: dict[str, Any],
    evaluator_config: dict[str, Any],
) -> None:
    """检查 evaluator 与当前仿真设置是否一致。"""
    sim_frame = normalize_frame(str(sim_config["data"]["frame"]))
    sim_cloud_type = normalize_cloud_type(str(sim_config["data"]["cloud_type"]))
    evaluator_frame = normalize_frame(str(evaluator_config["data"]["frame"]))
    evaluator_cloud_type = normalize_cloud_type(str(evaluator_config["data"]["cloud_type"]))
    if evaluator_frame != sim_frame:
        raise ValueError(
            f"Evaluator frame mismatch: evaluator={evaluator_frame}, sim={sim_frame}."
        )
    if evaluator_cloud_type != sim_cloud_type:
        raise ValueError(
            "Evaluator cloud_type mismatch: "
            f"evaluator={evaluator_cloud_type}, sim={sim_cloud_type}."
        )

    generator_encoder = str(
        generator_config["model"]["input_encoder"]["name"]
    ).strip().lower()
    evaluator_encoder = str(
        evaluator_config["model"]["input_encoder"]["name"]
    ).strip().lower()
    if evaluator_encoder != generator_encoder:
        raise ValueError(
            "Evaluator input encoder mismatch: "
            f"evaluator={evaluator_encoder}, generator={generator_encoder}."
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    config = apply_overrides(config, args.set)
    validate_sim_config(config)

    seed = int(config["seed"])
    set_random_seed(seed)

    device = build_device(args.device or str(config["sim"].get("device", "cpu")))
    checkpoint_path = args.ckpt_path or config["sim"].get("ckpt_path")
    if not checkpoint_path:
        raise ValueError("sim.ckpt_path or --ckpt-path is required.")
    model, checkpoint_config = load_generator(str(checkpoint_path), device=device)
    evaluator_enabled = bool(config.get("evaluator", {}).get("enabled", False))
    evaluator_model: GraspEvaluator | None = None
    evaluator_topk = int(config.get("evaluator", {}).get("topk", 0))
    if evaluator_enabled:
        evaluator_ckpt_path = str(config.get("evaluator", {}).get("ckpt_path") or "").strip()
        if not evaluator_ckpt_path:
            raise ValueError("evaluator.ckpt_path is required when evaluator.enabled=true.")
        if evaluator_topk <= 0:
            raise ValueError("evaluator.topk must be positive when evaluator.enabled=true.")
        evaluator_model, evaluator_checkpoint_config = load_evaluator(
            evaluator_ckpt_path,
            device=device,
        )
        validate_evaluator_runtime(
            sim_config=config,
            generator_config=checkpoint_config,
            evaluator_config=evaluator_checkpoint_config,
        )

    dataset_root, manifest_items = load_manifest(
        str(config["data"]["manifest_path"]),
        split=str(config["sim"].get("split", "train")),
    )
    cloud_type = normalize_cloud_type(str(config["data"]["cloud_type"]))
    frame = normalize_frame(str(config["data"]["frame"]))
    n_points = int(config["data"]["n_points"])
    point_sampling = str(config["data"].get("point_sampling", "random"))
    joint_dim = int(checkpoint_config["model"]["common"]["joint_dim"])
    prepared_joints = np.asarray(config["hand"]["prepared_joints"], dtype=np.float32)
    if prepared_joints.shape[0] != joint_dim:
        raise ValueError(
            f"hand.prepared_joints length {prepared_joints.shape[0]} != joint_dim {joint_dim}"
        )

    num_grasp_samples = int(config["sim"]["num_grasp_samples"])
    if evaluator_enabled and evaluator_topk > num_grasp_samples:
        raise ValueError(
            "evaluator.topk cannot exceed sim.num_grasp_samples. "
            f"topk={evaluator_topk}, num_grasp_samples={num_grasp_samples}"
        )
    visualize = bool(args.visualize or config["sim"].get("visualize", False))
    extforce_config, sim_friction = build_sim_runtime_config(config)

    summary_items: list[dict[str, Any]] = []
    total_generated_candidates = 0
    total_simulated_candidates = 0
    successful_candidates = 0
    total_sampling_time_sec = 0.0
    total_scoring_time_sec = 0.0
    total_simulation_time_sec = 0.0

    for item_index, item in enumerate(manifest_items):
        rng = np.random.default_rng(seed + item_index * 100003)
        mjcf_path = (dataset_root / item.mjcf_path).resolve()
        mjho_valid = MjHO(
            obj_info={"name": item.object_name, "xml_abs": str(mjcf_path)},
            hand_xml_path=str(Path(config["hand"]["xml_path"]).expanduser().resolve()),
            target_body_params=dict(config["hand"]["target_body_params"]),
            hand_profile=dict(config["hand"].get("profile", {})),
            friction_coef=(0.2, 0.2) if sim_friction is None else sim_friction,
            object_fixed=False,
            visualize=visualize,
        )

        attempt_records: list[dict[str, Any]] = []
        object_success = False
        object_successful_candidates = 0
        object_total_candidates = 0
        try:
            point_cloud_sample = load_conditioning_point_cloud(
                dataset_root=dataset_root,
                item=item,
                cloud_type=cloud_type,
                frame=frame,
                n_points=n_points,
                rng=rng,
                point_sampling=point_sampling,
            )
            batch = {
                "point_cloud": torch.tensor(
                    point_cloud_sample.point_cloud[None, ...],
                    dtype=torch.float32,
                    device=device,
                )
            }
            sampling_start = perf_counter()
            with torch.no_grad():
                predictions = model.sample(batch=batch, num_samples=num_grasp_samples)
            sampling_time_sec = perf_counter() - sampling_start
            total_sampling_time_sec += sampling_time_sec
            total_generated_candidates += num_grasp_samples

            selected_candidate_indices = np.arange(num_grasp_samples, dtype=np.int64)
            candidate_scores: np.ndarray | None = None
            scoring_time_sec = 0.0
            if evaluator_model is not None:
                scoring_start = perf_counter()
                evaluator_batch = {
                    "point_cloud": batch["point_cloud"].repeat(num_grasp_samples, 1, 1),
                    "grasp_pose": predictions["pred_squeeze_pose"][0],
                    "grasp_joint": predictions["pred_squeeze_joint"][0],
                }
                with torch.no_grad():
                    candidate_scores = (
                        evaluator_model.score(evaluator_batch)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                scoring_time_sec = perf_counter() - scoring_start
                total_scoring_time_sec += scoring_time_sec
                selected_candidate_indices = np.argsort(-candidate_scores)[:evaluator_topk]

            candidate_records: list[dict[str, Any]] = []
            simulation_start = perf_counter()
            for rank_index, candidate_index in enumerate(selected_candidate_indices.tolist()):
                squeeze_pose = (
                    predictions["pred_squeeze_pose"][0, candidate_index]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                squeeze_joint = (
                    predictions["pred_squeeze_joint"][0, candidate_index]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                squeeze_matrix = se3_log_to_matrix(squeeze_pose)
                if frame == "camera":
                    if point_cloud_sample.cam_extrinsic is None:
                        raise ValueError(
                            f"{item.object_scale_key} is missing camera extrinsic in camera mode."
                        )
                    squeeze_matrix = camera_to_world_pose(
                        squeeze_matrix, point_cloud_sample.cam_extrinsic
                    )
                qpos_squeeze = matrix_to_qpos(squeeze_matrix, squeeze_joint)
                qpos_prepared = matrix_to_qpos(squeeze_matrix, prepared_joints)
                success, pos_delta, angle_delta = mjho_valid.sim_under_extforce(
                    qpos_target=qpos_squeeze.copy(),
                    qpos_prepared=qpos_prepared.copy(),
                    visualize=visualize,
                    **extforce_config,
                )
                total_simulated_candidates += 1
                successful_candidates += int(bool(success))
                object_successful_candidates += int(bool(success))
                evaluator_score = None
                if candidate_scores is not None:
                    evaluator_score = float(candidate_scores[candidate_index])
                candidate_records.append(
                    {
                        "rank_index": rank_index,
                        "candidate_index": candidate_index,
                        "evaluator_score": evaluator_score,
                        "success": bool(success),
                        "pos_delta": float(pos_delta),
                        "angle_delta": float(angle_delta),
                    }
                )
            simulation_time_sec = perf_counter() - simulation_start
            total_simulation_time_sec += simulation_time_sec

            object_total_candidates = len(selected_candidate_indices)
            object_success = object_successful_candidates > 0
            attempt_records.append(
                {
                    "attempt_index": 0,
                    "view_index": point_cloud_sample.view_index,
                    "success": object_success,
                    "success_count": object_successful_candidates,
                    "generated_num_grasp_samples": num_grasp_samples,
                    "simulated_num_grasp_samples": object_total_candidates,
                    "grasp_success_str": (
                        f"{object_successful_candidates}/{object_total_candidates}"
                    ),
                    "grasp_success_rate": float(
                        object_successful_candidates / max(1, object_total_candidates)
                    ),
                    "sampling_time_sec": float(sampling_time_sec),
                    "scoring_time_sec": float(scoring_time_sec),
                    "simulation_time_sec": float(simulation_time_sec),
                    "candidates": candidate_records,
                }
            )
        finally:
            viewer = getattr(mjho_valid, "viewer", None)
            if viewer is not None:
                try:
                    viewer.close()
                except Exception:
                    pass

        summary_items.append(
            {
                "object_scale_key": item.object_scale_key,
                "object_name": item.object_name,
                "success": object_success,
                "success_count": object_successful_candidates,
                "generated_num_grasp_samples": num_grasp_samples,
                "simulated_num_grasp_samples": object_total_candidates,
                "grasp_success_str": (
                    f"{object_successful_candidates}/{object_total_candidates}"
                ),
                "grasp_success_rate": float(
                    object_successful_candidates / max(1, object_total_candidates)
                ),
                "attempts": attempt_records,
            }
        )
        LOGGER.info(
            "[sim_sc] item=%d/%d object_scale_key=%s grasp=%s success=%s",
            item_index + 1,
            len(manifest_items),
            item.object_scale_key,
            f"{object_successful_candidates}/{object_total_candidates}",
            object_success,
        )
        LOGGER.info(
            "[sim_sc] generated=%d selected=%d sampling_time=%.4fs scoring_time=%.4fs simulation_time=%.4fs",
            num_grasp_samples,
            object_total_candidates,
            attempt_records[0]["sampling_time_sec"],
            attempt_records[0]["scoring_time_sec"],
            attempt_records[0]["simulation_time_sec"],
        )

    run_dir = build_sim_output_dir(str(checkpoint_path))
    successful_objects = int(sum(1 for item in summary_items if item["success"]))
    summary = {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "manifest_path": str(Path(config["data"]["manifest_path"]).expanduser().resolve()),
        "cloud_type": cloud_type,
        "frame": frame,
        "point_sampling": point_sampling,
        "num_grasp_samples": num_grasp_samples,
        "sim_friction": sim_friction,
        "extforce": extforce_config,
        "evaluator_enabled": evaluator_enabled,
        "simulated_topk": (
            evaluator_topk if evaluator_enabled else num_grasp_samples
        ),
        "total_items": len(manifest_items),
        "total_generated_candidates": total_generated_candidates,
        "total_candidates": total_simulated_candidates,
        "successful_candidates": successful_candidates,
        "GSR": float(successful_candidates / max(1, total_simulated_candidates)),
        "successful_objects": successful_objects,
        "OSR": float(successful_objects / max(1, len(summary_items))),
        "total_sampling_time_sec": float(total_sampling_time_sec),
        "avg_sampling_time_sec_per_object": float(
            total_sampling_time_sec / max(1, len(summary_items))
        ),
        "avg_sampling_time_sec_per_generated_candidate": float(
            total_sampling_time_sec / max(1, total_generated_candidates)
        ),
        "total_scoring_time_sec": float(total_scoring_time_sec),
        "avg_scoring_time_sec_per_object": float(
            total_scoring_time_sec / max(1, len(summary_items))
        ),
        "total_simulation_time_sec": float(total_simulation_time_sec),
        "avg_simulation_time_sec_per_object": float(
            total_simulation_time_sec / max(1, len(summary_items))
        ),
        "items": summary_items,
    }
    (run_dir / "resolved_sim_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path = run_dir / "sim_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info(
        "[sim_sc] finished GSR=%.6f OSR=%.6f summary=%s",
        summary["GSR"],
        summary["OSR"],
        summary_path,
    )


if __name__ == "__main__":
    main()
