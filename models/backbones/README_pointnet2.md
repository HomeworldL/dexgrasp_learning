# PointNet2 Backbone Notes

## 对应脚本
- `models/backbones/pointnet2_semseg.py`
- `models/backbones/pointnet2/pointnet2_modules.py`
- `models/backbones/pointnet2/pointnet2_utils.py`
- `models/backbones/pointnet2/pytorch_utils.py`

## 主要用途
- PointNet++ 层级点云编码路线。
- 既可做逐点分割网络，也可做分层特征提取器。
- 当前仓库里更适合未来承接:
  - DexDiffuser 风格的 scene token 编码
  - DGTR 风格的层级局部几何编码

## `pointnet2_semseg.py` 中的类与函数

### `get_model`
- 位置: `pointnet2_semseg.py:7`
- 作用:
  - 根据 `is_msg` 构造 `Pointnet2MSG` 或 `Pointnet2SSG`。

### `Pointnet2MSG`
- 位置: `pointnet2_semseg.py:25`
- 作用:
  - 多尺度分组的 PointNet2 分割网络。
- 输入:
  - `pointcloud`: `(B, N, 3 + C_in)`
- 输出:
  - `pred_cls`: `(B, N, num_classes)`

### `Pointnet2SSG`
- 位置: `pointnet2_semseg.py:105`
- 作用:
  - 单尺度分组的 PointNet2 分割网络。
- 输入:
  - `pointcloud`: `(B, N, 3 + C_in)`
- 输出:
  - `pred_cls`: `(B, N, num_classes)`

### `get_feature_extractor`
- 位置: `pointnet2_semseg.py:186`
- 作用:
  - 根据 `is_msg` 构造 `Pointnet2MSG_Feature` 或 `Pointnet2SSG_Feature`。

### `Pointnet2MSG_Feature`
- 位置: `pointnet2_semseg.py:202`
- 作用:
  - 仅保留 Set Abstraction 编码部分，返回层级特征。
- 输入:
  - `pointcloud`: `(B, N, 3 + C_in)`
- 输出:
  - `l_xyz`: 列表，典型形状依次为
    - `(B, N, 3)`
    - `(B, 1024, 3)`
    - `(B, 256, 3)`
    - `(B, 64, 3)`
    - `(B, 16, 3)`
  - `l_features`: 列表，典型形状依次为
    - `None` 或 `(B, C_in, N)`
    - `(B, 96, 1024)`
    - `(B, 256, 256)`
    - `(B, 512, 64)`
    - `(B, 1024, 16)`

### `Pointnet2SSG_Feature`
- 位置: `pointnet2_semseg.py:255`
- 作用:
  - 单尺度版本的层级特征提取器。
  - 当前更接近 DexDiffuser 那类“只取高层 scene feature”用法。
- 输入:
  - `pointcloud`: `(B, N, 3 + C_in)`
- 输出:
  - `l_xyz`: 列表，典型形状依次为
    - `(B, N, 3)`
    - `(B, 2048, 3)`
    - `(B, 512, 3)`
    - `(B, 128, 3)`
    - `(B, 16, 3)`
  - `l_features`: 列表，典型形状依次为
    - `None` 或 `(B, C_in, N)`
    - `(B, 64, 2048)`
    - `(B, 128, 512)`
    - `(B, 256, 128)`
    - `(B, 512, 16)`

### `pointnet2_enc_repro`
- 位置: `pointnet2_semseg.py:308`
- 作用:
  - 直接返回 `Pointnet2SSG_Feature`
  - 目前硬编码要求 `num_points == 2048`

## `pointnet2/pointnet2_modules.py`

### `_PointnetSAModuleBase`
- 作用:
  - Set Abstraction 模块基类
- 输入:
  - `xyz`: `(B, N, 3)`
  - `features`: `(B, C, N)` 或 `None`
- 输出:
  - `new_xyz`: `(B, npoint, 3)`
  - `new_features`: `(B, C_out, npoint)`

### `PointnetSAModuleMSG`
- 作用:
  - 多尺度分组 SA 模块

### `PointnetSAModule`
- 作用:
  - 单尺度分组 SA 模块

### `PointnetSAModuleVotes`
- 作用:
  - 带采样索引返回的 SA 模块
  - 更接近 VoteNet / DGTR 一类检测用法

### `PointnetSAModuleMSGVotes`
- 作用:
  - 多尺度版本的 votes SA 模块

### `PointnetFPModule`
- 作用:
  - Feature Propagation 上采样模块
- 输入:
  - `unknown`: `(B, n, 3)`
  - `known`: `(B, m, 3)`
  - `unknow_feats`: `(B, C1, n)`
  - `known_feats`: `(B, C2, m)`
- 输出:
  - `(B, C_out, n)`

### `PointnetLFPModuleMSG`
- 作用:
  - 局部特征传播模块

## `pointnet2/pointnet2_utils.py`
- 作用:
  - PointNet2 CUDA / grouping 相关底层算子封装
- 关键内容:
  - `FurthestPointSampling`
  - `GatherOperation`
  - `ThreeNN`
  - `ThreeInterpolate`
  - `GroupingOperation`
  - `BallQuery`
  - `QueryAndGroup`
  - `GroupAll`

## `pointnet2/pytorch_utils.py`
- 作用:
  - PointNet2 中常用的小模块封装
- 关键内容:
  - `SharedMLP`
  - `Conv1d / Conv2d / Conv3d`
  - `FC`
  - `BNMomentumScheduler`

## 对应算法
- 直接相关:
  - DexDiffuser 风格点云编码
  - DGTR 风格 PointNet2 编码
- 当前仓库阶段:
  - 先保留结构
  - 暂不作为第一阶段训练主干

## 当前状态判断
- `pointnet2_semseg.py` 里仍保留旧路径导入:
  - `from models.model.pointnet2...`
- 说明它还没有完全适配你当前的扁平目录结构。
- 因此现在更适合先作为“保留原始结构的候选 backbone”，而不是当前马上训练的主干。
