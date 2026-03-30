from __future__ import annotations

from typing import Any

import torch

from models.base_sc import BaseSCModel
from models.dexdiffuser import (
    BPSConditionTokenizer,
    DDPM,
    DexDiffuserConditionAdapter,
    DexDiffuserUNet,
    DiffusionTargetCodec,
)


class DexDiffuserSCModel(BaseSCModel):
    """适配当前单条件主线的 DexDiffuser 生成器。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        if self.algorithm != "dexdiffuser":
            raise NotImplementedError(
                "DexDiffuserSCModel only supports model.algorithm=dexdiffuser."
            )
        algorithm_config = dict(self.model_config)
        condition_config = dict(algorithm_config.get("condition", {}))
        self.codec = DiffusionTargetCodec(
            init_pose_dim=self.init_pose_dim,
            squeeze_pose_dim=self.squeeze_pose_dim,
            joint_dim=self.joint_dim,
            normalization_config=dict(algorithm_config.get("target_normalization", {})),
        )
        self.condition_adapter = DexDiffuserConditionAdapter(
            point_feat_dim=self.point_feat_dim,
            input_encoder_name=self.input_encoder_name,
            input_encoder_config=dict(self.model_config.get("input_encoder", {})),
            condition_config=condition_config,
        )
        self.bps_condition_tokenizer: BPSConditionTokenizer | None = None
        if self.input_encoder_name == "bps":
            bps_tokenizer_config = dict(self.model_config.get("input_encoder", {}))
            bps_tokenizer_config.update(dict(condition_config.get("bps", {})))
            self.bps_condition_tokenizer = BPSConditionTokenizer(
                basis_path=bps_tokenizer_config.get("basis_path"),
                feature_types=list(bps_tokenizer_config.get("feature_types", ["dists"])),
                bps_type=str(bps_tokenizer_config.get("bps_type", "random_uniform")),
                n_bps_points=int(bps_tokenizer_config.get("n_bps_points", 4096)),
                radius=float(bps_tokenizer_config.get("radius", 1.0)),
                n_dims=int(bps_tokenizer_config.get("n_dims", 3)),
            )

        unet_config = dict(algorithm_config.get("unet", {}))
        context_dim = int(condition_config.get("context_dim", 512))
        self.unet = DexDiffuserUNet(
            d_x=self.target_dim,
            d_model=int(unet_config.get("d_model", 512)),
            context_dim=context_dim,
            time_embed_mult=int(unet_config.get("time_embed_mult", 2)),
            nblocks=int(unet_config.get("nblocks", 4)),
            resblock_dropout=float(unet_config.get("resblock_dropout", 0.0)),
            transformer_num_heads=int(unet_config.get("transformer_num_heads", 8)),
            transformer_dim_head=int(unet_config.get("transformer_dim_head", 64)),
            transformer_dropout=float(unet_config.get("transformer_dropout", 0.1)),
            transformer_depth=int(unet_config.get("transformer_depth", 1)),
            transformer_mult_ff=int(unet_config.get("transformer_mult_ff", 2)),
            use_position_embedding=bool(unet_config.get("use_position_embedding", False)),
        )
        self.ddpm = DDPM(
            config=dict(algorithm_config.get("diffusion", {})),
            eps_model=self.unet,
        )

    def _build_context_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        global_feature, backbone_feature = self.encode_condition_features(batch)
        if self.input_encoder_name == "pointnet":
            return self.condition_adapter.from_pointnet(global_feature, backbone_feature)
        if self.input_encoder_name == "bps":
            if self.bps_condition_tokenizer is None:
                raise RuntimeError("BPSConditionTokenizer was not initialized.")
            bps_tokens = self.bps_condition_tokenizer(batch["point_cloud"])
            return self.condition_adapter.from_bps(global_feature, bps_tokens)
        raise NotImplementedError(
            f"DexDiffuser currently supports pointnet and bps, got {self.input_encoder_name}."
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x0 = self.codec.build_from_batch(batch)
        context_tokens = self._build_context_tokens(batch)
        return self.ddpm.compute_loss(x0=x0, context=context_tokens)

    def infer(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sampled = self.sample(batch=batch, num_samples=1)
        return {
            "pred_init_pose": sampled["pred_init_pose"][:, 0, :],
            "pred_squeeze_pose": sampled["pred_squeeze_pose"][:, 0, :],
            "pred_squeeze_joint": sampled["pred_squeeze_joint"][:, 0, :],
        }

    def sample(
        self,
        batch: dict[str, torch.Tensor],
        num_samples: int,
    ) -> dict[str, torch.Tensor]:
        context_tokens = self._build_context_tokens(batch)
        sampled_x = self.ddpm.sample(
            context=context_tokens,
            num_samples=num_samples,
            sample_shape=(self.target_dim,),
        )
        return self.codec.split(sampled_x)
