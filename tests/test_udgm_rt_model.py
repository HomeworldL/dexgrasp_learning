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
            "algorithm": "udgm_rt",
            "input_encoder": {"name": "pointnet"},
            "common": {
                "point_feat_dim": 64,
                "init_pose_dim": 6,
                "squeeze_pose_dim": 6,
                "joint_dim": 20,
            },
            "algorithms": {
                "udgm_rt": {
                    "condition_dim": 48,
                    "target_normalization": {"enabled": False},
                    "condition": {
                        "hidden_dims": [64],
                        "activation": "leaky_relu",
                        "network_type": "residual",
                        "residual_num_blocks": 2,
                    },
                    "flow": {
                        "hidden_dim": 64,
                        "num_layers": 4,
                        "num_blocks_per_layer": 2,
                        "scale_clamp": 2.0,
                        "activation": "leaky_relu",
                        "use_actnorm": True,
                        "use_invertible_linear": True,
                        "conditioner_type": "residual",
                        "residual_num_blocks": 2,
                        "loss_clamp_max": 80.0,
                    },
                    "regression": {
                        "hidden_dims": [64, 64],
                        "activation": "leaky_relu",
                        "network_type": "residual",
                        "residual_num_blocks": 2,
                    },
                    "loss_weights": {
                        "flow": 1.0,
                        "init_pose": 1.0,
                        "joint": 1.0,
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


def test_udgm_rt_pointnet_forward_and_sample() -> None:
    model = build_model(_base_config())
    batch = _dummy_batch()
    outputs = model(batch)
    sampled = model.sample(batch, num_samples=3)
    assert "loss_flow" in outputs
    assert "loss_init_pose" in outputs
    assert "loss_joint" in outputs
    assert sampled["pred_init_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_joint"].shape == (2, 3, 20)


def test_udgm_rt_bps_forward_and_sample() -> None:
    config = deepcopy(_base_config())
    config["model"]["input_encoder"]["name"] = "bps"
    model = build_model(config)
    batch = _dummy_batch()
    outputs = model(batch)
    sampled = model.sample(batch, num_samples=2)
    assert "raw_nll" in outputs
    assert sampled["pred_init_pose"].shape == (2, 2, 6)
    assert sampled["pred_squeeze_pose"].shape == (2, 2, 6)
    assert sampled["pred_squeeze_joint"].shape == (2, 2, 20)


class _FakeFlow(nn.Module):
    def __init__(self, sampled_pose: torch.Tensor) -> None:
        super().__init__()
        self.sampled_pose = sampled_pose

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0])

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: torch.Tensor,
        *,
        sort_by_log_prob: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]
        sampled = self.sampled_pose[:batch_size, None, :].expand(batch_size, num_samples, -1)
        log_prob = torch.zeros(batch_size, num_samples, device=context.device, dtype=context.dtype)
        return sampled, log_prob


class _CaptureHead(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.last_pose: torch.Tensor | None = None

    def forward(self, condition: torch.Tensor, squeeze_pose: torch.Tensor) -> torch.Tensor:
        self.last_pose = squeeze_pose.detach().clone()
        if squeeze_pose.ndim == 2:
            return squeeze_pose.new_zeros(squeeze_pose.shape[0], self.output_dim)
        return squeeze_pose.new_zeros(squeeze_pose.shape[0], squeeze_pose.shape[1], self.output_dim)


def test_udgm_rt_forward_uses_sampled_squeeze_pose() -> None:
    model = build_model(_base_config())
    batch = _dummy_batch()
    predicted_pose = torch.full_like(batch["squeeze_pose"], 0.25)
    model.flow = _FakeFlow(predicted_pose)
    capture = _CaptureHead(output_dim=model.init_joint_codec.target_dim)
    model.regression_head = capture
    _ = model(batch)
    assert capture.last_pose is not None
    assert torch.allclose(capture.last_pose, predicted_pose)
    assert not torch.allclose(capture.last_pose, batch["squeeze_pose"])
