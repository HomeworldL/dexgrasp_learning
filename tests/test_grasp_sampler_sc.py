from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grasp_dataset_sc import DistinctObjectBatchSampler


class _DummyDataset:
    def __init__(self, num_items: int) -> None:
        self.items = [object() for _ in range(num_items)]

    def __len__(self) -> int:
        return len(self.items)


def test_sampler_uses_object_scale_item_count() -> None:
    dataset = _DummyDataset(num_items=4)
    sampler = DistinctObjectBatchSampler(dataset=dataset, batch_size=3, num_steps=1, seed=0)

    batch = next(iter(sampler))
    item_indices = [item_index for item_index, _ in batch]

    assert len(batch) == 3
    assert len(set(item_indices)) == 3


def test_sampler_rejects_batch_size_larger_than_num_items() -> None:
    dataset = _DummyDataset(num_items=4)

    try:
        DistinctObjectBatchSampler(dataset=dataset, batch_size=5, num_steps=1, seed=0)
    except ValueError as exc:
        assert "num_items=4" in str(exc)
    else:
        raise AssertionError("Expected ValueError when batch_size exceeds dataset size.")
