from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import normalize_point_sampling
from src.transforms import sample_point_cloud


def test_sample_point_cloud_defaults_to_random_sampling() -> None:
    points = np.arange(30, dtype=np.float32).reshape(10, 3)

    sampled_default = sample_point_cloud(
        points,
        n_points=4,
        rng=np.random.default_rng(0),
    )
    sampled_random = sample_point_cloud(
        points,
        n_points=4,
        rng=np.random.default_rng(0),
        point_sampling="random",
    )

    assert sampled_default.shape == (4, 3)
    assert np.array_equal(sampled_default, sampled_random)


def test_sample_point_cloud_supports_fps_sampling() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    sampled = sample_point_cloud(
        points,
        n_points=4,
        rng=np.random.default_rng(0),
        point_sampling="fps",
    )

    assert sampled.shape == (4, 3)
    assert sampled.dtype == np.float32
    for row in sampled:
        assert any(np.array_equal(row, source_row) for source_row in points)


def test_normalize_point_sampling_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Expected fps or random"):
        normalize_point_sampling("grid")
