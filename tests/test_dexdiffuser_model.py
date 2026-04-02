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
            "algorithm": "dexdiffuser",
            "input_encoder": {"name": "pointnet"},
            "common": {
                "point_feat_dim": 128,
                "init_pose_dim": 6,
                "squeeze_pose_dim": 6,
                "joint_dim": 20,
            },
            "algorithms": {
                "cvae": {
                    "latent_dim": 64,
                    "encoder_hidden_dims": [128, 64],
                    "decoder_hidden_dims": [64, 64],
                },
                "dexdiffuser": {
                    "target_normalization": {"enabled": False},
                    "condition": {
                        "context_dim": 64,
                        "append_global_token": True,
                        "token_sampling": "uniform_stride",
                        "pointnet": {"num_condition_tokens": 16},
                        "bps": {
                            "num_condition_tokens": 16,
                            "feature_types": ["dists"],
                        },
                    },
                    "unet": {
                        "d_model": 64,
                        "time_embed_mult": 2,
                        "nblocks": 2,
                        "resblock_dropout": 0.0,
                        "transformer_num_heads": 4,
                        "transformer_dim_head": 16,
                        "transformer_dropout": 0.0,
                        "transformer_depth": 1,
                        "transformer_mult_ff": 2,
                        "use_position_embedding": False,
                    },
                    "diffusion": {
                        "steps": 6,
                        "rand_t_type": "half",
                        "loss_type": "l2",
                        "schedule": {
                            "beta": [0.0001, 0.01],
                            "beta_schedule": "linear",
                            "s": 0.008,
                        },
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


def test_dexdiffuser_pointnet_forward_and_sample() -> None:
    config = _base_config()
    model = build_model(config)
    batch = _dummy_batch()

    outputs = model(batch)
    sampled = model.sample(batch, num_samples=3)

    assert "loss" in outputs
    assert outputs["loss"].ndim == 0
    assert sampled["pred_init_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_joint"].shape == (2, 3, 20)


def test_dexdiffuser_bps_forward_and_sample() -> None:
    config = deepcopy(_base_config())
    config["model"]["input_encoder"]["name"] = "bps"
    model = build_model(config)
    batch = _dummy_batch()

    outputs = model(batch)
    sampled = model.sample(batch, num_samples=2)

    assert "loss_diffusion" in outputs
    assert sampled["pred_init_pose"].shape == (2, 2, 6)
    assert sampled["pred_squeeze_pose"].shape == (2, 2, 6)
    assert sampled["pred_squeeze_joint"].shape == (2, 2, 20)
