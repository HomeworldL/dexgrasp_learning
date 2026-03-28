from __future__ import annotations

import argparse
import json

import torch

from src.config import apply_overrides, load_config
from src.grasp_dataset_sc import GraspDatasetSC


def describe(value) -> str:
    if torch.is_tensor(value):
        return f"tensor shape={tuple(value.shape)} dtype={value.dtype}"
    return f"{type(value).__name__}: {value}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print one sample from GraspDatasetSC.")
    parser.add_argument("--config", type=str, default="configs/ycb_liberhand_sc.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set data.frame=camera",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval", "test"])
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.set)
    dataset = GraspDatasetSC(
        manifest_path=str(config["data"]["manifest_path"]),
        split=args.split,
        cloud_type=str(config["data"]["cloud_type"]),
        frame=str(config["data"]["frame"]),
        n_points=int(config["data"]["n_points"]),
        joint_dim=int(config["model"]["common"]["joint_dim"]),
        seed=int(config["seed"]),
    )
    sample = dataset[args.index]
    print(
        json.dumps(
            {"split": args.split, "index": args.index, "length": len(dataset)},
            indent=2,
            ensure_ascii=False,
        )
    )
    for key, value in sample.items():
        print(f"{key}: {describe(value)}")


if __name__ == "__main__":
    main()
