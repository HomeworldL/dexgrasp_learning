from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base_model import BaseModel
from models.dexdiffuser import (
    BPSConditionTokenizer,
    DDPM,
    DexDiffuserConditionAdapter,
    DexDiffuserRTHead,
    DexDiffuserUNet,
    InitJointCodec,
    SqueezePoseCodec,
)


class DexDiffuserRTModel(BaseModel):
    """先扩散 squeeze_pose，再回归 init_pose 与 squeeze_joint。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        if self.algorithm != "dexdiffuser_rt":
            raise NotImplementedError(
                "DexDiffuserRTModel only supports model.algorithm=dexdiffuser_rt."
            )
        algorithm_config = dict(self.model_config)
        condition_config = dict(algorithm_config.get("condition", {}))
        normalization_config = dict(algorithm_config.get("target_normalization", {}))
        self.squeeze_pose_codec = SqueezePoseCodec(
            squeeze_pose_dim=self.squeeze_pose_dim,
            normalization_config=normalization_config,
        )
        self.init_joint_codec = InitJointCodec(
            init_pose_dim=self.init_pose_dim,
            joint_dim=self.joint_dim,
            normalization_config=normalization_config,
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
            d_x=self.squeeze_pose_dim,
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

        regression_config = dict(algorithm_config.get("regression", {}))
        self.regression_head = DexDiffuserRTHead(
            point_feat_dim=self.point_feat_dim,
            squeeze_pose_dim=self.squeeze_pose_dim,
            init_pose_dim=self.init_pose_dim,
            joint_dim=self.joint_dim,
            hidden_dims=list(regression_config.get("hidden_dims", [256, 256])),
            activation=str(regression_config.get("activation", "leaky_relu")),
        )
        self.pose_loss = nn.SmoothL1Loss()
        self.joint_loss = nn.SmoothL1Loss()

        algorithm_loss_weights = dict(algorithm_config.get("loss_weights", {}))
        self.loss_weights = {
            "diffusion": float(
                algorithm_loss_weights.get("diffusion", 1.0)
            ),
            "init_pose": float(
                algorithm_loss_weights.get("init_pose", 1.0)
            ),
            "joint": float(algorithm_loss_weights.get("joint", 1.0)),
        }

    def _build_condition_features(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_feature, backbone_feature = self.encode_condition_features(batch)
        if self.input_encoder_name == "pointnet":
            context_tokens = self.condition_adapter.from_pointnet(global_feature, backbone_feature)
        elif self.input_encoder_name == "bps":
            if self.bps_condition_tokenizer is None:
                raise RuntimeError("BPSConditionTokenizer was not initialized.")
            bps_tokens = self.bps_condition_tokenizer(batch["point_cloud"])
            context_tokens = self.condition_adapter.from_bps(global_feature, bps_tokens)
        else:
            raise NotImplementedError(
                "DexDiffuserRT currently supports pointnet and bps, "
                f"got {self.input_encoder_name}."
            )
        return global_feature, context_tokens

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        global_feature, context_tokens = self._build_condition_features(batch)
        squeeze_pose_target = self.squeeze_pose_codec.build_from_batch(batch)
        diffusion_outputs = self.ddpm.compute_loss_with_prediction(
            x0=squeeze_pose_target,
            context=context_tokens,
        )
        predicted_squeeze_pose = self.squeeze_pose_codec.denormalize(diffusion_outputs["pred_x0"])
        init_joint_prediction = self.regression_head(global_feature, predicted_squeeze_pose)
        split_prediction = self.init_joint_codec.split(init_joint_prediction)

        loss_init_pose = self.pose_loss(split_prediction["pred_init_pose"], batch["init_pose"])
        loss_joint = self.joint_loss(
            split_prediction["pred_squeeze_joint"],
            batch["squeeze_joint"],
        )
        loss = (
            self.loss_weights["diffusion"] * diffusion_outputs["loss_diffusion"]
            + self.loss_weights["init_pose"] * loss_init_pose
            + self.loss_weights["joint"] * loss_joint
        )
        return {
            "loss_diffusion": diffusion_outputs["loss_diffusion"],
            "loss_init_pose": loss_init_pose,
            "loss_joint": loss_joint,
            "loss": loss,
        }

    def sample(
        self,
        batch: dict[str, torch.Tensor],
        num_samples: int,
    ) -> dict[str, torch.Tensor]:
        self._validate_num_samples(num_samples)
        global_feature, context_tokens = self._build_condition_features(batch)
        sampled_squeeze_pose = self.ddpm.sample(
            context=context_tokens,
            num_samples=num_samples,
            sample_shape=(self.squeeze_pose_dim,),
        )
        decoded_squeeze_pose = self.squeeze_pose_codec.split(sampled_squeeze_pose)["pred_squeeze_pose"]
        init_joint_prediction = self.regression_head(global_feature, decoded_squeeze_pose)
        return self.init_joint_codec.merge(
            squeeze_pose=decoded_squeeze_pose,
            init_pose_and_joint=init_joint_prediction,
        )
