#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.evaluator import GraspEvaluator
from src.config import (
    apply_overrides,
    load_config,
    normalize_cloud_type,
    normalize_frame,
    set_random_seed,
    validate_evaluator_train_config,
)
from src.evaluator_dataset import DistinctObjectBatchSampler, EvaluatorDataset


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the point-cloud grasp evaluator.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set evaluator.train.batch_size=64",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_device(device_name: str) -> torch.device:
    requested = str(device_name).strip()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def build_output_dir(config: dict[str, Any]) -> Path:
    output_root = Path(config["evaluator"]["train"]["output_dir"]).expanduser().resolve()
    input_encoder_name = str(config["model"]["input_encoder"]["name"]).strip().lower()
    frame = normalize_frame(str(config["data"]["frame"]))
    cloud_type = normalize_cloud_type(str(config["data"]["cloud_type"]))
    experiment_tag = f"{input_encoder_name}_{frame}_{cloud_type}"
    run_dir = output_root / experiment_tag / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR | None,
    step: int,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "step": int(step),
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    config = apply_overrides(config, args.set)
    validate_evaluator_train_config(config)

    seed = int(config["seed"])
    set_random_seed(seed)

    run_dir = build_output_dir(config)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train_config = dict(config["evaluator"]["train"])
    max_steps = int(train_config["max_steps"])
    if args.smoke:
        max_steps = min(max_steps, int(train_config.get("smoke_steps", 2)))

    train_dataset = EvaluatorDataset(
        manifest_path=str(config["data"]["manifest_path"]),
        split="train",
        cloud_type=str(config["data"]["cloud_type"]),
        frame=str(config["data"]["frame"]),
        n_points=int(config["data"]["n_points"]),
        joint_dim=int(config["model"]["common"]["joint_dim"]),
        seed=seed,
        positive_probability=float(train_config.get("positive_probability", 0.5)),
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=DistinctObjectBatchSampler(
            dataset=train_dataset,
            batch_size=int(train_config["batch_size"]),
            num_steps=max_steps,
            seed=seed,
        ),
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=bool(train_config.get("pin_memory", False)),
        persistent_workers=bool(train_config.get("persistent_workers", False))
        if int(train_config.get("num_workers", 0)) > 0
        else False,
    )

    device = build_device(str(train_config.get("device", "cpu")))
    model = GraspEvaluator(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )

    scheduler: CosineAnnealingLR | None = None
    if "lr_min" in train_config:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=float(train_config["lr_min"]),
        )

    log_every = int(train_config.get("log_every", 50))
    save_every = int(train_config.get("save_every", 500))
    grad_clip = float(train_config.get("grad_clip", 10.0))
    last_checkpoint = run_dir / "last.ckpt"
    last_record: dict[str, float] = {}

    model.train()
    for step, batch in enumerate(train_loader, start=1):
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        last_record = {
            key: float(value.detach().cpu().item())
            for key, value in outputs.items()
            if torch.is_tensor(value) and key not in {"logits", "score"}
        }
        if step % log_every == 0 or step == max_steps:
            LOGGER.info(
                "[train_evaluator] step=%d loss=%.6f acc=%.4f pos_acc=%.4f neg_acc=%.4f",
                step,
                last_record["loss"],
                last_record["accuracy"],
                last_record["positive_accuracy"],
                last_record["negative_accuracy"],
            )
        if step % save_every == 0 or step == max_steps:
            save_checkpoint(
                path=last_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                config=config,
            )

    summary = {
        "run_dir": str(run_dir),
        "max_steps": max_steps,
        "last_checkpoint": str(last_checkpoint),
        "last_record": last_record,
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info(
        "[train_evaluator] finished steps=%d checkpoint=%s loss=%.6f acc=%.4f",
        max_steps,
        last_checkpoint,
        last_record.get("loss", float("nan")),
        last_record.get("accuracy", float("nan")),
    )


if __name__ == "__main__":
    main()
