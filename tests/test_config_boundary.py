from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import build_model
from src.config import validate_train_config


def _base_config_without_algorithm_details() -> dict:
    return {
        "seed": 0,
        "data": {
            "manifest_path": "assets/datasets/graspdata_YCB_liberhand/train.json",
            "cloud_type": "global",
            "frame": "world",
            "n_points": 64,
            "point_sampling": "random",
        },
        "model": {
            "algorithm": "cvae",
            "prediction_structure": {"name": "flat"},
            "input_encoder": {"name": "pointnet"},
            "common": {
                "point_feat_dim": 64,
                "init_pose_dim": 6,
                "squeeze_pose_dim": 6,
                "joint_dim": 20,
            },
            "algorithms": {
                "cvae": {
                    "flat": {},
                },
            },
            "input_encoders": {
                "pointnet": {
                    "point_feature_dim": 3,
                    "local_conv_hidden_dims": [32, 64],
                    "global_mlp_hidden_dims": [64],
                    "activation": "leaky_relu",
                },
            },
        },
        "hand": {
            "xml_path": "/tmp/dummy.xml",
            "prepared_joints": [0.0] * 20,
            "target_body_params": {},
        },
        "train": {
            "batch_size": 2,
            "max_steps": 1,
            "lr": 1e-3,
            "output_dir": "outputs/test",
        },
    }


def test_train_config_validation_skips_algorithm_specific_keys() -> None:
    validate_train_config(_base_config_without_algorithm_details())


def test_build_model_validates_algorithm_specific_keys() -> None:
    with pytest.raises(KeyError, match="model\\.algorithms\\.cvae\\.flat\\.latent_dim"):
        build_model(_base_config_without_algorithm_details())
