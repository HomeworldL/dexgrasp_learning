from __future__ import annotations

import math
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn


class _DummyAccel:
    def __getattr__(self, name: str):  # pragma: no cover - compatibility shim
        if name == "device_count":
            return lambda *args, **kwargs: 0
        if name in {"is_available", "_is_in_bad_fork"}:
            return lambda *args, **kwargs: False
        return lambda *args, **kwargs: None


if not hasattr(torch, "xpu"):  # pragma: no cover - env-specific compatibility
    torch.xpu = _DummyAccel()
if not hasattr(torch, "mps"):  # pragma: no cover - env-specific compatibility
    torch.mps = _DummyAccel()

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (  # noqa: E402
    EulerAncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler  # noqa: E402


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.nn.functional.softplus(x))


def _to_namespace(config: dict[str, Any]) -> SimpleNamespace:
    namespace_data: dict[str, Any] = {}
    for key, value in dict(config).items():
        if isinstance(value, dict):
            namespace_data[key] = _to_namespace(value)
        else:
            namespace_data[key] = value
    return SimpleNamespace(**namespace_data)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: int = 10_000):
        super().__init__()
        self.dim = int(dim)
        self.theta = int(theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * self.theta
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers_dim: list[int],
        output_dim: int,
        act: str | None = None,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        activation_name = "leaky_relu" if act is None else str(act).strip().lower()
        act_fn = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "mish": Mish,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
        }[activation_name]
        hidden_dims = deepcopy(list(hidden_layers_dim))
        hidden_dims.insert(0, int(input_dim))
        self.mlp = nn.Sequential()
        for index in range(1, len(hidden_dims)):
            self.mlp.add_module(
                f"linear{index - 1}",
                nn.Linear(hidden_dims[index - 1], hidden_dims[index]),
            )
            if use_layer_norm:
                self.mlp.add_module(f"ln{index - 1}", nn.LayerNorm(hidden_dims[index]))
            self.mlp.add_module(f"act{index - 1}", act_fn())
        self.mlp.add_module(
            f"linear{len(hidden_dims) - 1}",
            nn.Linear(hidden_dims[-1], int(output_dim)),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MLPWrapper(MLP):
    def __init__(self, channels: int, feature_dim: int, *args: Any, **kwargs: Any) -> None:
        self.channels = int(channels)
        input_dim = self.channels + int(feature_dim)
        super().__init__(input_dim=input_dim, *args, **kwargs)
        self.embedding = SinusoidalPosEmb(int(feature_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_embedding = self.embedding(t)
        return super().forward(torch.cat([x, cond + t_embedding], dim=-1))


def jacobian_matrix(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    jacobian = torch.zeros((*f.shape, z.shape[-1]), device=f.device)
    for index in range(f.shape[-1]):
        jacobian[..., index, :] = torch.autograd.grad(
            f[..., index].sum(),
            z,
            retain_graph=(index != f.shape[-1] - 1),
            allow_unused=True,
        )[0]
    return jacobian.contiguous()


def approx_jacobian_trace(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    e = torch.normal(mean=0, std=1, size=f.shape, device=f.device, dtype=f.dtype)
    grad = torch.autograd.grad(f, z, grad_outputs=e)[0]
    return torch.einsum("nka,nka->nk", e, grad)


def jacobian_trace(log_prob_type: str | None, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    if log_prob_type == "accurate_cont":
        jacobian_mat = jacobian_matrix(dy, dx)
        return jacobian_mat.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    if log_prob_type == "estimate":
        return approx_jacobian_trace(dy, dx)
    return torch.zeros(dx.shape[0], device=dx.device, dtype=dx.dtype)


class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        cond_fn=lambda x, t, cond: cond,
    ) -> None:
        super().__init__()
        self.config = _to_namespace(config)
        self.model = model
        self.cond_fn = cond_fn
        scheduler_type = str(self.config.scheduler_type)
        scheduler_kwargs = dict(vars(self.config.scheduler))
        if scheduler_type == "DDPMScheduler":
            self.scheduler = DDPMScheduler(**scheduler_kwargs)
        elif scheduler_type == "DDIMScheduler":
            self.scheduler = DDIMScheduler(**scheduler_kwargs)
        elif scheduler_type == "EulerAncestralDiscreteScheduler":
            self.scheduler = EulerAncestralDiscreteScheduler(**scheduler_kwargs)
        elif scheduler_type == "EulerDiscreteScheduler":
            self.scheduler = EulerDiscreteScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")
        self.timesteps = int(self.config.scheduler.num_train_timesteps)
        self.inference_timesteps = int(self.config.num_inference_timesteps)
        self.prediction_type = str(self.config.scheduler.prediction_type)
        loss_type = str(self.config.loss_type).strip().lower()
        if loss_type == "l1":
            self.diff_loss = nn.SmoothL1Loss(reduction="mean")
        elif loss_type == "l2":
            self.diff_loss = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unsupported diffusion loss_type: {self.config.loss_type}")

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.calculate_loss(*args, **kwargs)

    def calculate_loss(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)
        noised_x = self.scheduler.add_noise(x, noise, t)
        conditioned = self.cond_fn(noised_x, t / self.timesteps, cond)
        pred = self.model(noised_x, t / self.timesteps, cond=conditioned)
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = x
        elif self.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(x, noise, t)
        else:
            raise NotImplementedError(f"Unsupported prediction_type: {self.prediction_type}")
        return self.diff_loss(pred, target)

    def predict_x0(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)
        noised_x = self.scheduler.add_noise(x, noise, t)
        conditioned = self.cond_fn(noised_x, t / self.timesteps, cond)
        model_output = self.model(noised_x, t / self.timesteps, cond=conditioned)
        alpha_prod = self.scheduler.alphas_cumprod.to(x.device)[t][:, None].to(noised_x.dtype)
        beta_prod = (1 - alpha_prod).to(noised_x.dtype)
        if self.prediction_type == "epsilon":
            pred_x0 = (noised_x - beta_prod.sqrt() * model_output) / alpha_prod.sqrt()
        elif self.prediction_type == "sample":
            pred_x0 = model_output
        elif self.prediction_type == "v_prediction":
            pred_x0 = alpha_prod.sqrt() * noised_x - beta_prod.sqrt() * model_output
        else:
            raise NotImplementedError(f"Unsupported prediction_type: {self.prediction_type}")
        return pred_x0.detach()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(cond.shape[0], self.model.channels, device=cond.device)
        log_prob = (-x.square() / 2 - np.log(2 * np.pi) / 2).sum(1)
        self.scheduler.set_timesteps(self.inference_timesteps, device=cond.device)
        need_log_prob = getattr(self.config, "log_prob_type", None) is not None
        last_t = self.timesteps
        with torch.set_grad_enabled(need_log_prob):
            for timestep in self.scheduler.timesteps:
                dx = torch.zeros_like(x)
                dx.requires_grad_(need_log_prob)
                x = x + dx
                dt = torch.full(
                    (x.shape[0], 1),
                    (last_t - timestep.item()) / self.timesteps,
                    device=x.device,
                    dtype=torch.float,
                )
                last_t = timestep.item()
                t_pad = torch.full(
                    (x.shape[0],),
                    timestep.item(),
                    device=x.device,
                    dtype=torch.long,
                )
                conditioned = self.cond_fn(x, t_pad / self.timesteps, cond)
                model_output = self.model(x, t_pad / self.timesteps, cond=conditioned)
                alpha_prod = self.scheduler.alphas_cumprod.to(x.device)[t_pad][:, None]
                betas = self.scheduler.betas.to(x.device)[t_pad][:, None]
                if self.prediction_type == "epsilon":
                    noise = model_output
                elif self.prediction_type == "v_prediction":
                    noise = model_output * alpha_prod.sqrt() + x * (1 - alpha_prod).sqrt()
                else:
                    noise = model_output
                score = -1 / (1 - alpha_prod).sqrt() * noise
                beta = betas * self.timesteps
                if bool(getattr(self.config, "ode", False)):
                    dy = (-0.5 * beta * x - score * beta / 2) * dt
                else:
                    dy = (
                        -0.5 * beta * x - score * beta
                    ) * dt + beta.sqrt() * torch.randn_like(x) * dt.sqrt()
                log_prob -= jacobian_trace(getattr(self.config, "log_prob_type", None), dx, -dy / dt) * dt[:, 0]
                x = x - dy
                x = x.detach()
                log_prob = log_prob.detach()
        if not need_log_prob:
            log_prob = torch.zeros_like(log_prob)
        return x, log_prob
