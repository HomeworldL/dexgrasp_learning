from __future__ import annotations

import math
from typing import Any

import torch


def make_schedule_ddpm(
    timesteps: int,
    beta: list[float],
    beta_schedule: str,
    s: float = 0.008,
) -> dict[str, torch.Tensor]:
    """构建 DDPM 所需的缓冲量。"""
    if len(beta) != 2:
        raise ValueError(f"schedule.beta must have length 2, got {len(beta)}.")
    beta_start = float(beta[0])
    beta_end = float(beta[1])
    if not (0.0 < beta_start < beta_end < 1.0):
        raise ValueError(
            "schedule.beta must satisfy 0 < beta[0] < beta[1] < 1, "
            f"got {beta}."
        )
    schedule_name = str(beta_schedule).strip().lower()
    if schedule_name == "linear":
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    elif schedule_name == "cosine":
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + float(s)) / (1.0 + float(s)) * math.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0, 0.999).to(torch.float32)
    elif schedule_name == "sqrt":
        betas = torch.sqrt(torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32))
    else:
        raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]],
        dim=0,
    )
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
        "sqrt_recipm1_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod - 1.0),
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": torch.log(posterior_variance.clamp(min=1e-20)),
        "posterior_mean_coef1": betas
        * torch.sqrt(alphas_cumprod_prev)
        / (1.0 - alphas_cumprod),
        "posterior_mean_coef2": (1.0 - alphas_cumprod_prev)
        * torch.sqrt(alphas)
        / (1.0 - alphas_cumprod),
    }


def parse_schedule_config(config: dict[str, Any]) -> dict[str, Any]:
    schedule = dict(config)
    if "beta" not in schedule:
        raise KeyError("Missing required config key: diffusion.schedule.beta")
    if "beta_schedule" not in schedule:
        raise KeyError("Missing required config key: diffusion.schedule.beta_schedule")
    return schedule
