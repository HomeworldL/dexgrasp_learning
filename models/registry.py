from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models.base_model import BaseModel
from models.cvae_model import CVAEModel
from models.dp_model import DPModel
from models.dp_rt_model import DPRTModel
from models.dexdiffuser_model import DexDiffuserModel
from models.dexdiffuser_rt_model import DexDiffuserRTModel
from models.udgm_model import UDGMModel
from models.udgm_rt_model import UDGMRTModel

SUPPORTED_INPUT_ENCODERS = frozenset({"pointnet", "bps"})


@dataclass(frozen=True)
class AlgorithmSpec:
    model_cls: type[BaseModel]
    required_keys: tuple[str, ...] = ()
    encoder_required_keys: dict[str, tuple[str, ...]] = field(default_factory=dict)


ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {
    "cvae": AlgorithmSpec(
        model_cls=CVAEModel,
        required_keys=("model.algorithms.cvae.latent_dim",),
    ),
    "dexdiffuser": AlgorithmSpec(
        model_cls=DexDiffuserModel,
        required_keys=(
            "model.algorithms.dexdiffuser.condition.context_dim",
            "model.algorithms.dexdiffuser.unet.d_model",
            "model.algorithms.dexdiffuser.diffusion.steps",
            "model.algorithms.dexdiffuser.diffusion.schedule.beta",
            "model.algorithms.dexdiffuser.diffusion.schedule.beta_schedule",
        ),
        encoder_required_keys={
            "pointnet": (
                "model.algorithms.dexdiffuser.condition.pointnet.num_condition_tokens",
            ),
            "bps": (
                "model.algorithms.dexdiffuser.condition.bps.num_condition_tokens",
            ),
        },
    ),
    "dexdiffuser_rt": AlgorithmSpec(
        model_cls=DexDiffuserRTModel,
        required_keys=(
            "model.algorithms.dexdiffuser_rt.condition.context_dim",
            "model.algorithms.dexdiffuser_rt.unet.d_model",
            "model.algorithms.dexdiffuser_rt.diffusion.steps",
            "model.algorithms.dexdiffuser_rt.diffusion.schedule.beta",
            "model.algorithms.dexdiffuser_rt.diffusion.schedule.beta_schedule",
        ),
        encoder_required_keys={
            "pointnet": (
                "model.algorithms.dexdiffuser_rt.condition.pointnet.num_condition_tokens",
            ),
            "bps": (
                "model.algorithms.dexdiffuser_rt.condition.bps.num_condition_tokens",
            ),
        },
    ),
    "udgm": AlgorithmSpec(
        model_cls=UDGMModel,
        required_keys=(
            "model.algorithms.udgm.condition_dim",
            "model.algorithms.udgm.flow.hidden_dim",
            "model.algorithms.udgm.flow.num_layers",
            "model.algorithms.udgm.flow.num_blocks_per_layer",
        ),
    ),
    "udgm_rt": AlgorithmSpec(
        model_cls=UDGMRTModel,
        required_keys=(
            "model.algorithms.udgm_rt.condition_dim",
            "model.algorithms.udgm_rt.flow.hidden_dim",
            "model.algorithms.udgm_rt.flow.num_layers",
            "model.algorithms.udgm_rt.flow.num_blocks_per_layer",
        ),
    ),
    "dp": AlgorithmSpec(
        model_cls=DPModel,
        required_keys=(
            "model.algorithms.dp.diffusion.scheduler_type",
            "model.algorithms.dp.diffusion.scheduler.num_train_timesteps",
            "model.algorithms.dp.diffusion.num_inference_timesteps",
            "model.algorithms.dp.diffusion.loss_type",
        ),
    ),
    "dp_rt": AlgorithmSpec(
        model_cls=DPRTModel,
        required_keys=(
            "model.algorithms.dp_rt.diffusion.scheduler_type",
            "model.algorithms.dp_rt.diffusion.scheduler.num_train_timesteps",
            "model.algorithms.dp_rt.diffusion.num_inference_timesteps",
            "model.algorithms.dp_rt.diffusion.loss_type",
            "model.algorithms.dp_rt.regression.hidden_features",
        ),
    ),
}


def normalize_algorithm_name(raw_value: Any) -> str:
    return str(raw_value).strip().lower()


def normalize_input_encoder_name(raw_value: Any) -> str:
    return str(raw_value).strip().lower()
