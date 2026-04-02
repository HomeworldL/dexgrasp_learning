from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from models.dexdiffuser.schedule import make_schedule_ddpm, parse_schedule_config


class DDPM(nn.Module):
    """简化后的 DexDiffuser DDPM 核心。"""

    def __init__(
        self,
        config: dict[str, Any],
        eps_model: nn.Module,
    ) -> None:
        super().__init__()
        self.eps_model = eps_model
        self.timesteps = int(config.get("steps", 100))
        self.rand_t_type = str(config.get("rand_t_type", "half")).strip().lower()
        self.loss_type = str(config.get("loss_type", "l2")).strip().lower()
        schedule_config = parse_schedule_config(dict(config.get("schedule", {})))
        for key, value in make_schedule_ddpm(self.timesteps, **schedule_config).items():
            self.register_buffer(key, value)

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x0.shape[0]
        reshape = (batch_size,) + (1,) * (x0.ndim - 1)
        return (
            self.sqrt_alphas_cumprod[t].reshape(reshape) * x0
            + self.sqrt_one_minus_alphas_cumprod[t].reshape(reshape) * noise
        )

    def compute_loss(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.compute_loss_with_prediction(x0=x0, context=context)
        return {
            "loss_noise": outputs["loss_noise"],
            "loss": outputs["loss_noise"],
        }

    def _sample_training_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.rand_t_type == "all":
            return torch.randint(
                0,
                self.timesteps,
                (batch_size,),
                device=device,
            ).long()
        if self.rand_t_type == "half":
            half = torch.randint(
                0,
                self.timesteps,
                ((batch_size + 1) // 2,),
                device=device,
            )
            if batch_size % 2 == 1:
                return torch.cat([half, self.timesteps - half[:-1] - 1], dim=0).long()
            return torch.cat([half, self.timesteps - half - 1], dim=0).long()
        raise ValueError(f"Unsupported rand_t_type: {self.rand_t_type}")

    def _noise_loss(
        self,
        pred_noise: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred_noise, noise)
        if self.loss_type == "l2":
            return F.mse_loss(pred_noise, noise)
        raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def compute_loss_with_prediction(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = x0.shape[0]
        timesteps = self._sample_training_timesteps(batch_size=batch_size, device=x0.device)
        noise = torch.randn_like(x0, device=x0.device)
        x_t = self.q_sample(x0=x0, t=timesteps, noise=noise)
        pred_noise, pred_x0 = self.model_predict(x_t, timesteps, context)
        loss_noise = self._noise_loss(pred_noise=pred_noise, noise=noise)
        return {
            "loss_noise": loss_noise,
            "loss": loss_noise,
            "pred_noise": pred_noise,
            "pred_x0": pred_x0,
            "timesteps": timesteps,
            "x_t": x_t,
        }

    def model_predict(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_t.shape[0]
        reshape = (batch_size,) + (1,) * (x_t.ndim - 1)
        pred_noise = self.eps_model(x_t, t, context)
        pred_x0 = (
            self.sqrt_recip_alphas_cumprod[t].reshape(reshape) * x_t
            - self.sqrt_recipm1_alphas_cumprod[t].reshape(reshape) * pred_noise
        )
        return pred_noise, pred_x0

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x_t.shape[0]
        reshape = (batch_size,) + (1,) * (x_t.ndim - 1)
        pred_noise, pred_x0 = self.model_predict(x_t, t, context)
        model_mean = (
            self.posterior_mean_coef1[t].reshape(reshape) * pred_x0
            + self.posterior_mean_coef2[t].reshape(reshape) * x_t
        )
        posterior_variance = self.posterior_variance[t].reshape(reshape)
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(reshape)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x_t.shape[0]
        timesteps = torch.full(
            (batch_size,),
            int(t),
            device=x_t.device,
            dtype=torch.long,
        )
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, timesteps, context)
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        num_samples: int,
        sample_shape: tuple[int, ...],
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        batch_size = context.shape[0]
        flat_context = (
            context[:, None, :, :]
            .expand(batch_size, num_samples, context.shape[1], context.shape[2])
            .reshape(batch_size * num_samples, context.shape[1], context.shape[2])
        )
        x_t = torch.randn(
            batch_size * num_samples,
            *sample_shape,
            device=context.device,
            dtype=context.dtype,
        )
        for timestep in reversed(range(self.timesteps)):
            x_t = self.p_sample(x_t=x_t, t=timestep, context=flat_context)
        return x_t.reshape(batch_size, num_samples, *sample_shape)
