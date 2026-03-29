from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim_sc import build_sim_runtime_config
from src.mj_ho import _normalize_friction_coef


def test_normalize_friction_coef_for_sliding_and_torsional() -> None:
    friction, condim = _normalize_friction_coef((0.2, 0.2))

    assert np.allclose(friction, np.array([0.2, 0.2], dtype=float))
    assert condim == 4


def test_normalize_friction_coef_for_rolling_support() -> None:
    friction, condim = _normalize_friction_coef((0.2, 0.2, 0.01))

    assert np.allclose(friction, np.array([0.2, 0.2, 0.01], dtype=float))
    assert condim == 6


def test_sim_runtime_config_reads_flat_sim_fields() -> None:
    config = {
        "sim": {
            "friction": [0.6, 0.2],
            "extforce": {
                "duration": 0.2,
                "trans_thresh": 0.05,
                "angle_thresh": 10.0,
                "force_mag": 1.0,
                "check_steps": 50,
                "close_steps": 100,
            },
        }
    }

    extforce_config, friction = build_sim_runtime_config(config)

    assert extforce_config == {
        "duration": 0.2,
        "trans_thresh": 0.05,
        "angle_thresh": 10.0,
        "force_mag": 1.0,
        "check_steps": 50,
        "close_steps": 100,
    }
    assert np.allclose(friction, np.array([0.6, 0.2], dtype=float))
