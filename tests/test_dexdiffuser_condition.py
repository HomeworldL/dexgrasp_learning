from __future__ import annotations

from pathlib import Path
import sys

import importlib.util
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dexdiffuser.condition import BPSConditionTokenizer, DexDiffuserConditionAdapter


def test_pointnet_condition_adapter_shapes() -> None:
    adapter = DexDiffuserConditionAdapter(
        point_feat_dim=128,
        input_encoder_name="pointnet",
        input_encoder_config={"local_conv_hidden_dims": [64, 128, 256]},
        condition_config={
            "context_dim": 64,
            "append_global_token": True,
            "token_sampling": "uniform_stride",
            "pointnet": {"num_condition_tokens": 32},
            "bps": {"num_condition_tokens": 16, "feature_types": ["dists"]},
        },
    )
    global_feature = torch.randn(2, 128)
    local_feature = torch.randn(2, 256, 128)

    tokens = adapter.from_pointnet(global_feature, local_feature)

    assert tokens.shape == (2, 33, 64)


def test_bps_condition_adapter_shapes() -> None:
    adapter = DexDiffuserConditionAdapter(
        point_feat_dim=128,
        input_encoder_name="bps",
        input_encoder_config={"feature_types": ["dists"], "local_conv_hidden_dims": [64, 128, 256]},
        condition_config={
            "context_dim": 32,
            "append_global_token": False,
            "token_sampling": "uniform_stride",
            "pointnet": {"num_condition_tokens": 16},
            "bps": {"num_condition_tokens": 20, "feature_types": ["dists", "deltas"]},
        },
    )
    global_feature = torch.randn(4, 128)
    bps_tokens = torch.randn(4, 128, 4)

    tokens = adapter.from_bps(global_feature, bps_tokens)

    assert tokens.shape == (4, 20, 32)


def test_bps_condition_tokenizer_shapes() -> None:
    if importlib.util.find_spec("bps_torch") is None:
        return
    tokenizer = BPSConditionTokenizer(
        basis_path=str(
            REPO_ROOT / "models_ref" / "DexDiffuser" / "models" / "basis_point_set.npy"
        ),
        feature_types=["dists"],
        n_bps_points=4096,
    )
    point_cloud = torch.randn(2, 64, 3)

    tokens = tokenizer(point_cloud)

    assert tokens.shape == (2, 4096, 1)
