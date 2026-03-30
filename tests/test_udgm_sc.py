from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import build_model


def _base_config() -> dict:
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
            "algorithm": "udgm",
            "input_encoder": {"name": "pointnet"},
            "common": {
                "point_feat_dim": 64,
                "init_pose_dim": 6,
                "squeeze_pose_dim": 6,
                "joint_dim": 20,
            },
            "algorithms": {
                "cvae": {
                    "latent_dim": 32,
                    "encoder_hidden_dims": [64, 32],
                    "decoder_hidden_dims": [32, 32],
                },
                "udgm": {
                    "condition_dim": 48,
                    "target_normalization": {"enabled": False},
                    "condition": {
                        "hidden_dims": [64],
                        "activation": "leaky_relu",
                    },
                    "flow": {
                        "hidden_dim": 64,
                        "num_layers": 4,
                        "num_blocks_per_layer": 2,
                        "scale_clamp": 2.0,
                        "activation": "leaky_relu",
                    },
                },
            },
            "input_encoders": {
                "pointnet": {
                    "point_feature_dim": 3,
                    "local_conv_hidden_dims": [32, 64],
                    "global_mlp_hidden_dims": [64],
                    "activation": "leaky_relu",
                },
                "bps": {
                    "basis_path": str(
                        REPO_ROOT
                        / "models_ref"
                        / "DexDiffuser"
                        / "models"
                        / "basis_point_set.npy"
                    ),
                    "feature_types": ["dists"],
                    "mlp_hidden_dims": [64],
                    "activation": "leaky_relu",
                    "bps_type": "random_uniform",
                    "n_bps_points": 4096,
                    "radius": 1.0,
                    "n_dims": 3,
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


def _dummy_batch(batch_size: int = 2, n_points: int = 64) -> dict[str, torch.Tensor]:
    return {
        "point_cloud": torch.randn(batch_size, n_points, 3),
        "init_pose": torch.randn(batch_size, 6),
        "squeeze_pose": torch.randn(batch_size, 6),
        "squeeze_joint": torch.randn(batch_size, 20),
    }


def test_udgm_pointnet_forward_and_sample() -> None:
    model = build_model(_base_config())
    batch = _dummy_batch()

    outputs = model(batch)
    sampled = model.sample(batch, num_samples=3)

    assert "loss_nll" in outputs
    assert outputs["loss"].ndim == 0
    assert sampled["pred_init_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_joint"].shape == (2, 3, 20)


def test_udgm_bps_forward_and_sample() -> None:
    config = deepcopy(_base_config())
    config["model"]["input_encoder"]["name"] = "bps"
    model = build_model(config)
    batch = _dummy_batch()

    outputs = model(batch)
    sampled = model.sample(batch, num_samples=2)

    assert "mean_log_prob" in outputs
    assert sampled["pred_init_pose"].shape == (2, 2, 6)
    assert sampled["pred_squeeze_pose"].shape == (2, 2, 6)
    assert sampled["pred_squeeze_joint"].shape == (2, 2, 20)
