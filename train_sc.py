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

from models import build_model
from src.config import (
    apply_overrides,
    load_config,
    normalize_cloud_type,
    normalize_frame,
    set_random_seed,
    validate_train_config,
)
from src.grasp_dataset_sc import DistinctObjectBatchSampler, GraspDatasetSC


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the unified single-condition grasp generator.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set data.frame=camera",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_device(device_name: str) -> torch.device:
    requested = str(device_name).strip()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def build_output_dir(config: dict[str, Any]) -> Path:
    output_root = Path(config["train"]["output_dir"]).expanduser().resolve()
    algorithm = str(config["model"]["algorithm"]).strip().lower()
    input_encoder_name = str(config["model"]["input_encoder"]["name"]).strip().lower()
    frame = normalize_frame(str(config["data"]["frame"]))
    cloud_type = normalize_cloud_type(str(config["data"]["cloud_type"]))
    experiment_tag = f"{algorithm}_{input_encoder_name}_{frame}_{cloud_type}"
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


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must be a mapping: {checkpoint_path}")
    return payload


def resolve_initial_step(
    configured_initial_step: int | None,
    checkpoint_step: int | None,
) -> int:
    if configured_initial_step is None:
        return 0 if checkpoint_step is None else int(checkpoint_step)
    resolved = int(configured_initial_step)
    if resolved < 0:
        raise ValueError(f"train.initial_step must be non-negative, got {resolved}.")
    if checkpoint_step is not None and resolved != int(checkpoint_step):
        raise ValueError(
            "train.initial_step does not match checkpoint step: "
            f"{resolved} != {int(checkpoint_step)}."
        )
    return resolved


def initialize_model_from_checkpoint(
    model: torch.nn.Module,
    init_ckpt_path: str | None,
    configured_initial_step: int | None,
) -> tuple[int, str | None]:
    if init_ckpt_path is None:
        return resolve_initial_step(configured_initial_step, checkpoint_step=None), None

    checkpoint_path = Path(init_ckpt_path).expanduser().resolve()
    payload = load_checkpoint_payload(checkpoint_path)
    state_dict = payload.get("model", payload)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint model state must be a mapping: {checkpoint_path}")
    model.load_state_dict(state_dict)
    initial_step = resolve_initial_step(configured_initial_step, payload.get("step"))
    return initial_step, str(checkpoint_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    config = apply_overrides(config, args.set)
    validate_train_config(config)

    seed = int(config["seed"])
    set_random_seed(seed)

    run_dir = build_output_dir(config)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    max_steps = int(config["train"]["max_steps"])
    if args.smoke:
        max_steps = min(max_steps, int(config["train"].get("smoke_steps", 2)))

    device = build_device(str(config["train"].get("device", "cpu")))
    model = build_model(config).to(device)
    initial_step, init_ckpt_path = initialize_model_from_checkpoint(
        model=model,
        init_ckpt_path=config["train"].get("init_ckpt_path"),
        configured_initial_step=config["train"].get("initial_step"),
    )
    if initial_step >= max_steps:
        raise ValueError(
            f"train.max_steps={max_steps} must be greater than initial_step={initial_step}."
        )
    remaining_steps = max_steps - initial_step
    if init_ckpt_path is not None:
        LOGGER.info(
            "[train_sc] initializing model from checkpoint=%s initial_step=%d target_step=%d",
            init_ckpt_path,
            initial_step,
            max_steps,
        )

    train_dataset = GraspDatasetSC(
        manifest_path=str(config["data"]["manifest_path"]),
        split="train",
        cloud_type=str(config["data"]["cloud_type"]),
        frame=str(config["data"]["frame"]),
        n_points=int(config["data"]["n_points"]),
        joint_dim=int(config["model"]["common"]["joint_dim"]),
        seed=seed,
        point_sampling=str(config["data"].get("point_sampling", "random")),
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=DistinctObjectBatchSampler(
            dataset=train_dataset,
            batch_size=int(config["train"]["batch_size"]),
            num_steps=remaining_steps,
            seed=seed,
        ),
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=bool(config["train"].get("pin_memory", False)),
        persistent_workers=bool(config["train"].get("persistent_workers", False))
        if int(config["train"].get("num_workers", 0)) > 0
        else False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    scheduler: CosineAnnealingLR | None = None
    if "lr_min" in config["train"]:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=remaining_steps,
            eta_min=float(config["train"]["lr_min"]),
        )

    log_every = int(config["train"].get("log_every", 50))
    save_every = int(config["train"].get("save_every", 500))
    grad_clip = float(config["train"].get("grad_clip", 10.0))
    last_checkpoint = run_dir / "last.ckpt"
    last_record: dict[str, float] = {}

    model.train()
    for step, batch in enumerate(train_loader, start=initial_step + 1):
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
            if torch.is_tensor(value)
        }
        if step % log_every == 0 or step == max_steps:
            metric_parts = [f"{key}={value:.6f}" for key, value in sorted(last_record.items())]
            LOGGER.info("[train_sc] step=%d %s", step, " ".join(metric_parts))
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
        "initial_step": initial_step,
        "init_ckpt_path": init_ckpt_path,
        "last_checkpoint": str(last_checkpoint),
        "last_record": last_record,
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info(
        "[train_sc] finished steps=%d checkpoint=%s loss=%.6f",
        max_steps,
        last_checkpoint,
        last_record.get("loss", float("nan")),
    )


if __name__ == "__main__":
    main()
