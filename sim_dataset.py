#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.config import load_config, set_random_seed
from src.manifest import load_manifest
from src.mj_ho import MjHO


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle simulation using grasps stored in the dataset manifest."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ycb_liberhand_sc.yaml",
    )
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def load_qpos_grasps(grasp_h5_path: Path) -> dict[str, np.ndarray]:
    if not grasp_h5_path.exists():
        raise FileNotFoundError(f"grasp.h5 not found: {grasp_h5_path}")
    with h5py.File(grasp_h5_path, "r") as handle:
        required = ["qpos_init", "qpos_prepared", "qpos_squeeze"]
        arrays = {key: handle[key][:].astype(np.float32) for key in required}
    return arrays


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    config = load_config(args.config)
    set_random_seed(int(config["seed"]))

    dataset_root, items = load_manifest(str(config["data"]["manifest_path"]), split=args.split)
    visualize = bool(args.visualize)

    summary_items: list[dict[str, Any]] = []
    total_success = 0
    total_attempts = 0
    skipped_items: list[dict[str, str]] = []

    for item in items:
        grasp_h5_path = dataset_root / item.grasp_h5_path
        mjcf_path = dataset_root / item.mjcf_path
        try:
            grasp_arrays = load_qpos_grasps(grasp_h5_path)
            qpos_squeeze = grasp_arrays["qpos_squeeze"]
            mjho_collision = MjHO(
                {"name": item.object_name, "xml_abs": str(mjcf_path.resolve())},
                str(Path(config["hand"]["xml_path"]).expanduser().resolve()),
                target_body_params=dict(config["hand"]["target_body_params"]),
                hand_profile=dict(config["hand"].get("profile", {})),
                object_fixed=True,
                visualize=visualize,
            )
            mjho_valid = MjHO(
                {"name": item.object_name, "xml_abs": str(mjcf_path.resolve())},
                str(Path(config["hand"]["xml_path"]).expanduser().resolve()),
                target_body_params=dict(config["hand"]["target_body_params"]),
                hand_profile=dict(config["hand"].get("profile", {})),
                object_fixed=False,
                visualize=visualize,
            )
        except Exception as exc:
            skipped_items.append(
                {"object_scale_key": item.object_scale_key, "reason": str(exc)}
            )
            LOGGER.warning("Skip %s: %s", item.object_scale_key, exc)
            continue

        success_count = 0
        attempts: list[dict[str, Any]] = []
        try:
            for grasp_index, qpos_target in enumerate(qpos_squeeze):
                prepared_joints = grasp_arrays["qpos_prepared"][grasp_index][7:].copy()
                qpos_prepared = mjho_collision.build_pregrasp_qpos(
                    qpos_target=qpos_target.copy(),
                    prepared_joints=prepared_joints,
                )
                success, pos_delta, angle_delta = mjho_valid.sim_under_extforce(
                    qpos_target=qpos_target.copy(),
                    qpos_prepared=qpos_prepared.copy(),
                    visualize=visualize,
                    **dict(config["sim"]["extforce"]),
                )
                total_attempts += 1
                success_count += int(bool(success))
                total_success += int(bool(success))
                attempts.append(
                    {
                        "grasp_index": grasp_index,
                        "success": bool(success),
                        "pos_delta": float(pos_delta),
                        "angle_delta": float(angle_delta),
                    }
                )
        finally:
            for instance in (mjho_collision, mjho_valid):
                viewer = getattr(instance, "viewer", None)
                if viewer is not None:
                    try:
                        viewer.close()
                    except Exception:
                        pass

        summary_items.append(
            {
                "object_scale_key": item.object_scale_key,
                "object_name": item.object_name,
                "grasp_count": int(qpos_squeeze.shape[0]),
                "success_count": success_count,
                "success_rate": float(success_count / max(1, qpos_squeeze.shape[0])),
                "attempts": attempts,
            }
        )

    summary = {
        "split": args.split,
        "manifest_path": str(
            Path(config["data"]["manifest_path"]).expanduser().resolve()
        ),
        "total_items": len(items),
        "evaluated_items": len(summary_items),
        "skipped_items": skipped_items,
        "total_success": total_success,
        "total_attempts": total_attempts,
        "success_rate": float(total_success / max(1, total_attempts)),
        "items": summary_items,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
