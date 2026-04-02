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


def _base_model_common() -> dict:
    return {
        "common": {
            "point_feat_dim": 64,
            "init_pose_dim": 6,
            "squeeze_pose_dim": 6,
            "joint_dim": 20,
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
                    REPO_ROOT / "models_ref" / "DexDiffuser" / "models" / "basis_point_set.npy"
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
    }


def _base_config(algorithm: str) -> dict:
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
            "algorithm": algorithm,
            "input_encoder": {"name": "pointnet"},
            **_base_model_common(),
            "algorithms": {
                "dp": {
                    "rms": {"enabled": True, "max_update": 10},
                    "diffusion": {
                        "ode": True,
                        "scheduler_type": "DDIMScheduler",
                        "scheduler": {
                            "beta_schedule": "squaredcos_cap_v2",
                            "prediction_type": "v_prediction",
                            "num_train_timesteps": 16,
                            "clip_sample": False,
                        },
                        "num_inference_timesteps": 4,
                        "log_prob_type": None,
                        "loss_type": "l1",
                    },
                },
                "dp_rt": {
                    "rms": {"enabled": True, "max_update": 10},
                    "regression": {"hidden_features": 32},
                    "loss_weights": {
                        "diffusion": 1.0,
                        "init_pose": 1.0,
                        "joint": 1.0,
                    },
                    "diffusion": {
                        "ode": True,
                        "scheduler_type": "DDIMScheduler",
                        "scheduler": {
                            "beta_schedule": "squaredcos_cap_v2",
                            "prediction_type": "v_prediction",
                            "num_train_timesteps": 16,
                            "clip_sample": False,
                        },
                        "num_inference_timesteps": 4,
                        "log_prob_type": None,
                        "loss_type": "l1",
                    },
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


def test_dp_pointnet_forward_and_sample() -> None:
    config = _base_config("dp")
    model = build_model(config)
    batch = _dummy_batch()
    outputs = model(batch)
    sampled = model.sample(batch, num_samples=3)
    assert "loss_diffusion" in outputs
    assert sampled["pred_init_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_joint"].shape == (2, 3, 20)


def test_dp_rt_bps_forward_and_sample() -> None:
    config = _base_config("dp_rt")
    config["model"]["input_encoder"]["name"] = "bps"
    model = build_model(config)
    batch = _dummy_batch()
    outputs = model(batch)
    sampled = model.sample(batch, num_samples=2)
    assert "loss_diffusion" in outputs
    assert "loss_init_pose" in outputs
    assert "loss_joint" in outputs
    assert sampled["pred_init_pose"].shape == (2, 2, 6)
    assert sampled["pred_squeeze_pose"].shape == (2, 2, 6)
    assert sampled["pred_squeeze_joint"].shape == (2, 2, 20)


class _FakeDiffusion(nn.Module):
    def __init__(self, predicted_pose: torch.Tensor) -> None:
        super().__init__()
        self.predicted_pose = predicted_pose

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.5)

    def predict_x0(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.predicted_pose.to(device=x.device, dtype=x.dtype)

    def sample(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = cond.shape[0]
        return (
            self.predicted_pose[:batch_size].to(device=cond.device, dtype=cond.dtype),
            torch.zeros(batch_size, device=cond.device, dtype=cond.dtype),
        )


class _CaptureRegression(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return x.new_zeros(x.shape[0], self.output_dim)


def test_dp_rt_forward_uses_predicted_squeeze_pose() -> None:
    config = _base_config("dp_rt")
    config["model"]["algorithms"]["dp_rt"]["rms"]["enabled"] = False
    model = build_model(config)
    batch = _dummy_batch()
    predicted_pose = torch.full_like(batch["squeeze_pose"], 0.25)
    model.head.diffusion = _FakeDiffusion(predicted_pose)
    capture = _CaptureRegression(output_dim=model.init_pose_dim + model.joint_dim)
    model.head.regression_head = capture

    _ = model(batch)

    assert capture.last_input is not None
    captured_pose = capture.last_input[:, -model.squeeze_pose_dim :]
    assert torch.allclose(captured_pose, predicted_pose, atol=1e-6)
    assert not torch.allclose(captured_pose, batch["squeeze_pose"])


def test_dp_listed_in_registry() -> None:
    config = deepcopy(_base_config("dp"))
    model = build_model(config)
    assert model.algorithm == "dp"
