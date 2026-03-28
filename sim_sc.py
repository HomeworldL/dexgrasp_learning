#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models import build_model
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
    parser = argparse.ArgumentParser(description="Simulate the single-condition PointNet + CVAE model.")
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    config = apply_overrides(config, args.set)
    validate_sim_config(config)

    if bool(config.get("evaluator", {}).get("enabled", False)):
        raise NotImplementedError(
            "Evaluator is reserved in the interface but not implemented in the current mainline."
        )

    seed = int(config["seed"])
    set_random_seed(seed)

    device = build_device(args.device or str(config["sim"].get("device", "cpu")))
    checkpoint_path = args.ckpt_path or config["sim"].get("ckpt_path")
    if not checkpoint_path:
        raise ValueError("sim.ckpt_path or --ckpt-path is required.")
    model, checkpoint_config = load_generator(str(checkpoint_path), device=device)

    dataset_root, manifest_items = load_manifest(
        str(config["data"]["manifest_path"]),
        split=str(config["sim"].get("split", "train")),
    )
    cloud_type = normalize_cloud_type(str(config["data"]["cloud_type"]))
    frame = normalize_frame(str(config["data"]["frame"]))
    n_points = int(config["data"]["n_points"])
    joint_dim = int(checkpoint_config["model"]["common"]["joint_dim"])
    prepared_joints = np.asarray(config["hand"]["prepared_joints"], dtype=np.float32)
    if prepared_joints.shape[0] != joint_dim:
        raise ValueError(
            f"hand.prepared_joints length {prepared_joints.shape[0]} != joint_dim {joint_dim}"
        )

    samples_per_object_scale = int(config["sim"]["samples_per_object_scale"])
    num_grasp_samples = int(config["sim"]["num_grasp_samples"])
    visualize = bool(args.visualize or config["sim"].get("visualize", False))
    extforce_config = dict(config["sim"]["extforce"])

    summary_items: list[dict[str, Any]] = []
    total_attempts = 0
    successful_attempts = 0
    total_candidates = 0
    successful_candidates = 0

    for item_index, item in enumerate(manifest_items):
        rng = np.random.default_rng(seed + item_index * 100003)
        mjcf_path = (dataset_root / item.mjcf_path).resolve()
        mjho_valid = MjHO(
            obj_info={"name": item.object_name, "xml_abs": str(mjcf_path)},
            hand_xml_path=str(Path(config["hand"]["xml_path"]).expanduser().resolve()),
            target_body_params=dict(config["hand"]["target_body_params"]),
            hand_profile=dict(config["hand"].get("profile", {})),
            object_fixed=False,
            visualize=visualize,
        )

        attempt_records: list[dict[str, Any]] = []
        object_success = False
        try:
            for attempt_index in range(samples_per_object_scale):
                point_cloud_sample = load_conditioning_point_cloud(
                    dataset_root=dataset_root,
                    item=item,
                    cloud_type=cloud_type,
                    frame=frame,
                    n_points=n_points,
                    rng=rng,
                )
                batch = {
                    "point_cloud": torch.tensor(
                        point_cloud_sample.point_cloud[None, ...],
                        dtype=torch.float32,
                        device=device,
                    )
                }
                with torch.no_grad():
                    predictions = model.sample(batch=batch, num_samples=num_grasp_samples)

                candidate_records: list[dict[str, Any]] = []
                attempt_success = False
                for candidate_index in range(num_grasp_samples):
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
                    total_candidates += 1
                    successful_candidates += int(bool(success))
                    attempt_success = attempt_success or bool(success)
                    candidate_records.append(
                        {
                            "candidate_index": candidate_index,
                            "success": bool(success),
                            "pos_delta": float(pos_delta),
                            "angle_delta": float(angle_delta),
                        }
                    )

                total_attempts += 1
                successful_attempts += int(attempt_success)
                object_success = object_success or attempt_success
                attempt_records.append(
                    {
                        "attempt_index": attempt_index,
                        "view_index": point_cloud_sample.view_index,
                        "success": attempt_success,
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
                "attempts": attempt_records,
            }
        )
        LOGGER.info(
            "[sim_sc] item=%d/%d object_scale_key=%s success=%s",
            item_index + 1,
            len(manifest_items),
            item.object_scale_key,
            object_success,
        )

    output_root = Path(config["sim"].get("output_dir", "outputs/sim_sc")).expanduser().resolve()
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "manifest_path": str(Path(config["data"]["manifest_path"]).expanduser().resolve()),
        "cloud_type": cloud_type,
        "frame": frame,
        "samples_per_object_scale": samples_per_object_scale,
        "num_grasp_samples": num_grasp_samples,
        "total_items": len(manifest_items),
        "total_attempts": total_attempts,
        "successful_attempts": successful_attempts,
        "GSR": float(successful_attempts / max(1, total_attempts)),
        "total_candidates": total_candidates,
        "successful_candidates": successful_candidates,
        "candidate_success_rate": float(successful_candidates / max(1, total_candidates)),
        "successful_objects": int(sum(1 for item in summary_items if item["success"])),
        "OSR": float(
            sum(1 for item in summary_items if item["success"]) / max(1, len(summary_items))
        ),
        "items": summary_items,
    }
    summary_path = run_dir / "sim_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info(
        "[sim_sc] finished GSR=%.6f candidate_success_rate=%.6f summary=%s",
        summary["GSR"],
        summary["candidate_success_rate"],
        summary_path,
    )


if __name__ == "__main__":
    main()
