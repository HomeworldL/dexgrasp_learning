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


def _base_config(prediction_structure: str = "flat") -> dict:
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
            "prediction_structure": {"name": prediction_structure},
            "input_encoder": {"name": "pointnet"},
            "common": {
                "point_feat_dim": 64,
                "init_pose_dim": 6,
                "squeeze_pose_dim": 6,
                "joint_dim": 20,
            },
            "algorithms": {
                "cvae": {
                    "flat": {
                        "latent_dim": 32,
                        "encoder_hidden_dims": [64, 32],
                        "decoder_hidden_dims": [32, 32],
                        "loss_weights": {
                            "init_pose": 1.0,
                            "squeeze_pose": 1.0,
                            "joint": 1.0,
                            "kld": 0.001,
                        },
                    },
                    "staged": {
                        "latent_dim": 32,
                        "encoder_hidden_dims": [64, 32],
                        "decoder_hidden_dims": [32, 32],
                        "regression": {
                            "hidden_dims": [32, 32],
                            "activation": "leaky_relu",
                            "network_type": "residual",
                            "residual_num_blocks": 2,
                        },
                        "loss_weights": {
                            "init_pose": 1.0,
                            "squeeze_pose": 1.0,
                            "joint": 1.0,
                            "kld": 0.001,
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


def test_cvae_flat_forward_and_sample() -> None:
    model = build_model(_base_config("flat"))
    batch = _dummy_batch()

    outputs = model(batch)
    sampled = model.sample(batch, num_samples=3)

    assert "loss_kld" in outputs
    assert sampled["pred_init_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_pose"].shape == (2, 3, 6)
    assert sampled["pred_squeeze_joint"].shape == (2, 3, 20)


def test_cvae_staged_bps_forward_and_sample() -> None:
    config = deepcopy(_base_config("staged"))
    config["model"]["input_encoder"]["name"] = "bps"
    model = build_model(config)
    batch = _dummy_batch()

    outputs = model(batch)
    sampled = model.sample(batch, num_samples=2)

    assert "loss_kld" in outputs
    assert "loss_init_pose" in outputs
    assert "loss_squeeze_pose" in outputs
    assert "loss_joint" in outputs
    assert sampled["pred_init_pose"].shape == (2, 2, 6)
    assert sampled["pred_squeeze_pose"].shape == (2, 2, 6)
    assert sampled["pred_squeeze_joint"].shape == (2, 2, 20)


class _FakeCVAE(nn.Module):
    def __init__(self, reconstruction: torch.Tensor) -> None:
        super().__init__()
        self.reconstruction = reconstruction

    def forward(
        self,
        target: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = target.shape[0]
        reconstruction = self.reconstruction[:batch_size].to(device=target.device, dtype=target.dtype)
        zeros = torch.zeros(batch_size, 4, device=target.device, dtype=target.dtype)
        return reconstruction, zeros, zeros, zeros

    def inference(self, n: int, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is None:
            raise ValueError("condition is required")
        return self.reconstruction[:n].to(device=c.device, dtype=c.dtype)


class _CaptureRegression(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return x.new_zeros(x.shape[0], self.output_dim)


def test_cvae_staged_forward_uses_reconstructed_squeeze_pose() -> None:
    model = build_model(_base_config("staged"))
    batch = _dummy_batch()
    reconstructed_pose = torch.full_like(batch["squeeze_pose"], 0.25)
    model.cvae = _FakeCVAE(reconstructed_pose)
    capture = _CaptureRegression(output_dim=model.init_pose_dim + model.joint_dim)
    model.regression_head = capture

    _ = model(batch)

    assert capture.last_input is not None
    captured_pose = capture.last_input[:, -model.squeeze_pose_dim :]
    assert torch.allclose(captured_pose, reconstructed_pose, atol=1e-6)
    assert not torch.allclose(captured_pose, batch["squeeze_pose"])
