from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dexdiffuser.representation import DiffusionTargetCodec


def test_codec_round_trip_without_normalization() -> None:
    codec = DiffusionTargetCodec(
        init_pose_dim=6,
        squeeze_pose_dim=6,
        joint_dim=20,
        normalization_config={"enabled": False},
    )
    batch = {
        "init_pose": torch.randn(2, 6),
        "squeeze_pose": torch.randn(2, 6),
        "squeeze_joint": torch.randn(2, 20),
    }

    target = codec.build_from_batch(batch)
    split = codec.split(target)

    assert target.shape == (2, 32)
    assert torch.allclose(split["pred_init_pose"], batch["init_pose"])
    assert torch.allclose(split["pred_squeeze_pose"], batch["squeeze_pose"])
    assert torch.allclose(split["pred_squeeze_joint"], batch["squeeze_joint"])


def test_codec_supports_normalization_and_multisample_split() -> None:
    codec = DiffusionTargetCodec(
        init_pose_dim=6,
        squeeze_pose_dim=6,
        joint_dim=20,
        normalization_config={"enabled": True, "pose_scale": 2.0, "joint_scale": 4.0},
    )
    samples = torch.randn(3, 5, 32)

    split = codec.split(samples)

    assert split["pred_init_pose"].shape == (3, 5, 6)
    assert split["pred_squeeze_pose"].shape == (3, 5, 6)
    assert split["pred_squeeze_joint"].shape == (3, 5, 20)
