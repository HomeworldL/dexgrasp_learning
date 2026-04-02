from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import initialize_model_from_checkpoint, resolve_initial_step


def test_resolve_initial_step_prefers_checkpoint_step_when_not_configured() -> None:
    assert resolve_initial_step(None, 30_000) == 30_000
    assert resolve_initial_step(None, None) == 0


def test_resolve_initial_step_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match checkpoint step"):
        resolve_initial_step(20_000, 30_000)


def test_initialize_model_from_checkpoint_loads_weights_and_step(tmp_path: Path) -> None:
    source_model = torch.nn.Linear(3, 2)
    target_model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        source_model.weight.fill_(1.25)
        source_model.bias.fill_(-0.5)
        target_model.weight.zero_()
        target_model.bias.zero_()

    checkpoint_path = tmp_path / "init.ckpt"
    torch.save(
        {
            "model": source_model.state_dict(),
            "step": 30_000,
        },
        checkpoint_path,
    )

    initial_step, resolved_path = initialize_model_from_checkpoint(
        model=target_model,
        init_ckpt_path=str(checkpoint_path),
        configured_initial_step=None,
    )

    assert initial_step == 30_000
    assert resolved_path == str(checkpoint_path.resolve())
    assert torch.allclose(target_model.weight, source_model.weight)
    assert torch.allclose(target_model.bias, source_model.bias)
