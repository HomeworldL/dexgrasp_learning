# 点云条件抓取实验表格

本文件用于维护当前主线实验的 Markdown 表格模板。

当前主线目标是：

- 先把 `global + world + cvae + pointnet` 训练稳定
- 优先提升训练集上的采样成功率
- 之后再比较 `camera`、其他生成算法、其他点云编码器与网络超参数

## 指标定义

- `OSR`：物体成功率。对一个 `object-scale`，只要本次评测的多个候选抓取里至少有一次成功，就记该物体成功。
- `GSR`：抓取成功率。成功抓取数除以总评测抓取数。
- `summary_path`：对应 `sim.py` 输出的 summary 路径。

## 填写规则

- 一张表只比较一类变量。
- 每张表未列出的变量都必须保持一致。
- 当前阶段建议先做 `seed=0` 的初筛，再对最优配置补 `seed=1,2`。
- 当前所有结果默认来自 `sim.py`，除非表格备注里明确写成 `sim_dataset.py` oracle。

## 1. 完整点云基线确认

这个表只用于确认当前第一阶段最重要的主线是否已经跑通：`global + world + cvae + pointnet`。
它不是消融表，而是主基线记录表。后面的所有对比都应以这里选出的基线配置为参照。

固定条件：

- `model.algorithm=cvae`
- `model.input_encoder.name=pointnet`
- `data.cloud_type=global`
- `data.frame=world`
- `data.n_points=2048`
- `train.batch_size=64`
- `train.max_steps=5000`
- `sim.num_grasp_samples=16`

| exp_name | seed | model.algorithms.cvae.latent_dim | model.common.point_feat_dim | pointnet.local_conv_hidden_dims | pointnet.global_mlp_hidden_dims | train.lr | OSR | GSR | ckpt_path | summary_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sc_base_global_world_s0 | 0 | 64 | 128 | [64,128,256] | [256] | 1e-3 |  |  |  |  |  |
| sc_base_global_world_s1 | 1 | 64 | 128 | [64,128,256] | [256] | 1e-3 |  |  |  |  |  |
| sc_base_global_world_s2 | 2 | 64 | 128 | [64,128,256] | [256] | 1e-3 |  |  |  |  |  |

## 2. 完整点云坐标系对比

这个表只比较 `world` 和 `camera` 两种坐标系输入对成功率的影响。
除了 `data.frame` 以外，其余配置必须与基线完全一致。

固定条件：

- `model.algorithm=cvae`
- `model.input_encoder.name=pointnet`
- `data.cloud_type=global`
- 其余训练和仿真超参数与“完整点云基线确认”一致

| exp_name | seed | data.frame | OSR | GSR | ckpt_path | summary_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sc_frame_global_world_s0 | 0 | world |  |  |  |  |  |
| sc_frame_global_camera_s0 | 0 | camera |  |  |  |  |  |

## 3. 点云完整性对比

这个表只比较完整点云和局部点云的差异。
建议先在 `world` 系下比较，等 `world` 结论稳定后，再单独做 `camera` 系的同类比较。

固定条件：

- `model.algorithm=cvae`
- `model.input_encoder.name=pointnet`
- 先固定 `data.frame=world`
- 其余训练和仿真超参数与基线一致

| exp_name | seed | data.cloud_type | data.frame | OSR | GSR | ckpt_path | summary_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sc_cloud_world_global_s0 | 0 | global | world |  |  |  |  |  |
| sc_cloud_world_partial_s0 | 0 | partial | world |  |  |  |  |  |

如果后续要做 `camera` 系下的完整/局部对比，建议另开一张同结构表，不和这里混填。

## 4. 生成算法对比

这个表只比较生成算法家族。
为了保证控制变量成立，点云输入固定为最容易先跑稳的 `global + world`，点云编码器固定为 `pointnet`。

固定条件：

- `data.cloud_type=global`
- `data.frame=world`
- `model.input_encoder.name=pointnet`
- 其余训练和仿真超参数与基线一致

| exp_name | seed | model.algorithm | OSR | GSR | ckpt_path | summary_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sc_algo_cvae_s0 | 0 | cvae |  |  |  |  | 当前已实现 |
| sc_algo_diffusion_s0 | 0 | diffusion |  |  |  |  | 预留 |
| sc_algo_flow_s0 | 0 | flow |  |  |  |  | 预留 |

## 5. 点云编码器对比

这个表只比较点云 backbone。
生成算法固定为 `cvae`，输入固定为 `global + world`。

固定条件：

- `model.algorithm=cvae`
- `data.cloud_type=global`
- `data.frame=world`
- 其余训练和仿真超参数与基线一致

| exp_name | seed | model.input_encoder.name | OSR | GSR | ckpt_path | summary_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sc_encoder_pointnet_s0 | 0 | pointnet |  |  |  |  | 当前已实现 |
| sc_encoder_pointnet2_s0 | 0 | pointnet2 |  |  |  |  | 预留 |
| sc_encoder_pct_s0 | 0 | pct |  |  |  |  | 预留 |

## 6. CVAE 潜变量维度对比

这个表只比较 `latent_dim`。
它用于观察生成多样性和可优化性之间的平衡，不改其他结构宽度。

固定条件：

- `model.algorithm=cvae`
- `model.input_encoder.name=pointnet`
- `data.cloud_type=global`
- `data.frame=world`
- `model.common.point_feat_dim=128`
- `pointnet` 与 `train/sim` 其他配置与基线一致

| exp_name | seed | model.algorithms.cvae.latent_dim | OSR | GSR | ckpt_path | summary_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sc_latent_32_s0 | 0 | 32 |  |  |  |  |  |
| sc_latent_64_s0 | 0 | 64 |  |  |  |  |  |
| sc_latent_128_s0 | 0 | 128 |  |  |  |  |  |

## 7. PointNet 特征宽度对比

这个表只比较点云特征宽度。
它用于判断条件编码容量是否已经成为瓶颈。

固定条件：

- `model.algorithm=cvae`
- `model.input_encoder.name=pointnet`
- `data.cloud_type=global`
- `data.frame=world`
- `model.algorithms.cvae.latent_dim=64`
- 其余训练和仿真超参数与基线一致

| exp_name | seed | model.common.point_feat_dim | pointnet.local_conv_hidden_dims | pointnet.global_mlp_hidden_dims | OSR | GSR | ckpt_path | summary_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sc_pointfeat_128_s0 | 0 | 128 | [64,128,256] | [256] |  |  |  |  |  |
| sc_pointfeat_256_s0 | 0 | 256 | [64,128,256] | [256,256] |  |  |  |  |  |
| sc_pointfeat_512_s0 | 0 | 512 | [64,128,256,512] | [512] |  |  |  |  |  |

## 8. CVAE MLP 宽度对比

这个表只比较 CVAE 编码器和解码器 MLP 宽度。
它用于判断当前主线的生成头容量是否足够。

固定条件：

- `model.algorithm=cvae`
- `model.input_encoder.name=pointnet`
- `data.cloud_type=global`
- `data.frame=world`
- `model.algorithms.cvae.latent_dim=64`
- `model.common.point_feat_dim=128`
- 其余训练和仿真超参数与基线一致

| exp_name | seed | model.algorithms.cvae.encoder_hidden_dims | model.algorithms.cvae.decoder_hidden_dims | OSR | GSR | ckpt_path | summary_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sc_cvae_mlp_small_s0 | 0 | [256,256] | [256,256] |  |  |  |  |  |
| sc_cvae_mlp_base_s0 | 0 | [512,256] | [256,256] |  |  |  |  |  |
| sc_cvae_mlp_large_s0 | 0 | [512,512,256] | [512,256] |  |  |  |  |  |

## 9. 训练超参数补充表

这个表只比较训练超参数。
只有在模型结构基本稳定之后才建议使用，否则会把结构变化和优化变化混在一起。

固定条件：

- `model.algorithm=cvae`
- `model.input_encoder.name=pointnet`
- `data.cloud_type=global`
- `data.frame=world`
- 模型结构固定为当前最优基线

| exp_name | seed | train.lr | train.batch_size | train.beta_kld | train.weight_decay | OSR | GSR | ckpt_path | summary_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sc_trainhp_base_s0 | 0 | 1e-3 | 64 | 1e-3 | 1e-4 |  |  |  |  |  |
| sc_trainhp_low_lr_s0 | 0 | 5e-4 | 64 | 1e-3 | 1e-4 |  |  |  |  |  |
| sc_trainhp_high_kld_s0 | 0 | 1e-3 | 64 | 5e-3 | 1e-4 |  |  |  |  |  |

## 推荐执行顺序

建议按下面顺序填表，不要一开始并行铺太多变量：

1. 先填“完整点云基线确认”，拿到一个稳定的 `global + world` 基线。
2. 再填“完整点云坐标系对比”，判断 `camera` 是否明显掉点。
3. 再做 `latent_dim`、`point_feat_dim`、`CVAE MLP` 这三类主结构超参数表。
4. 当 `cvae + pointnet + global` 已经很稳时，再做算法和编码器预留表。
5. 最后再做局部点云迁移。
