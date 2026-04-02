from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import torch
from torch import nn

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
            "algorithm": "dexdiffuser_rt",
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
                "dexdiffuser_rt": {
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
                    "regression": {
                        "hidden_dims": [64, 64],
                        "activation": "leaky_relu",
                    },
                    "loss_weights": {
                        "diffusion": 1.0,
                        "init_pose": 1.0,
                        "joint": 1.0,
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


def test_dexdiffuser_rt_pointnet_forward_and_sample() -> None:
    config = _base_config()
    model = build_model(config)
    batch = _dummy_batch()

    outputs = model(batch)
    sampled = model.sample(batch, num_samples=3)

    assert "loss_diffusion" in outputs
    assert "loss_init_pose" in outputs
    assert "loss_joint" in outputs
    assert outputs["loss"].ndim == 0
    assert sampled["pred_init_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_joint"].shape == (2, 3, 20)


def test_dexdiffuser_rt_bps_forward_and_sample() -> None:
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


class _FakeDDPM(nn.Module):
    def __init__(self, predicted_pose: torch.Tensor) -> None:
        super().__init__()
        self.predicted_pose = predicted_pose

    def compute_loss_with_prediction(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "loss_diffusion": x0.new_tensor(0.5),
            "loss": x0.new_tensor(0.5),
            "pred_x0": self.predicted_pose.to(device=x0.device, dtype=x0.dtype),
        }

    def sample(
        self,
        context: torch.Tensor,
        num_samples: int,
        sample_shape: tuple[int, ...],
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        return self.predicted_pose[:batch_size, None, :].expand(batch_size, num_samples, sample_shape[0])


class _CaptureHead(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.last_squeeze_pose: torch.Tensor | None = None

    def forward(self, global_feature: torch.Tensor, squeeze_pose: torch.Tensor) -> torch.Tensor:
        self.last_squeeze_pose = squeeze_pose.detach().clone()
        if squeeze_pose.ndim == 2:
            return squeeze_pose.new_zeros(squeeze_pose.shape[0], self.output_dim)
        return squeeze_pose.new_zeros(squeeze_pose.shape[0], squeeze_pose.shape[1], self.output_dim)


def test_dexdiffuser_rt_forward_uses_predicted_squeeze_pose() -> None:
    config = _base_config()
    model = build_model(config)
    batch = _dummy_batch()
    predicted_pose = torch.full_like(batch["squeeze_pose"], 0.25)
    model.ddpm = _FakeDDPM(predicted_pose)
    capture_head = _CaptureHead(output_dim=model.init_joint_codec.target_dim)
    model.regression_head = capture_head

    _ = model(batch)

    assert capture_head.last_squeeze_pose is not None
    assert torch.allclose(capture_head.last_squeeze_pose, predicted_pose)
    assert not torch.allclose(capture_head.last_squeeze_pose, batch["squeeze_pose"])
