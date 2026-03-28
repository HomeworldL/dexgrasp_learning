# MinkUNet Backbone Notes

## 对应脚本
- `models/backbones/mink_unet.py`

## 主要用途
- 稀疏体素卷积路线的点云编码器。
- 更接近 `MinkowskiEngine` / `AnyDexGrasp` 这一类 3D sparse conv 方案。
- 适合处理稀疏体素化后的点云，而不是直接处理原始 `(B, N, 3)` 点集。

## 脚本中的类

### `WrappedMinkUNet`
- 位置: `mink_unet.py:6`
- 作用:
  - 对外包装统一入口。
  - 内部默认实例化 `MinkUNet14(in_channels=3, out_channels=cfg.out_feat_dim)`。
  - 把稀疏卷积输出重新映射回原始点顺序。
- 输入:
  - `data["point_clouds"]`: `(B, N, 3)`
  - `data["coors"]`: `(M, 4)`，稀疏坐标，通常是 `(batch_idx, x, y, z)`
  - `data["feats"]`: `(M, 3)`，体素特征
  - `data["quantize2original"]`: `(B*N,)` 或可 reshape 回 `(B,N)` 的索引
- 输出:
  - `global_feature`: `(B, out_feat_dim)`
  - `local_feature`: `(B, N, out_feat_dim)`

### `ResNetBase` 及其派生类
- 位置: `mink_unet.py:25`
- 作用:
  - 稀疏卷积 `ResNet` 主干基类。
  - `ResNet14/18/34/50/101` 是标准残差网络深度版本。
- 输入:
  - `ME.SparseTensor`
- 输出:
  - 稀疏全局池化后的特征，最终维度一般为 `(num_sparse_entries, out_channels)` 或全局 pooled 结果

### `MinkUNetBase` 及其派生类
- 位置: `mink_unet.py:156`
- 作用:
  - U-Net 结构的 sparse conv 主干。
  - 包含下采样、上采样、skip connection。
- 主要派生类:
  - `MinkUNet14`
  - `MinkUNet18`
  - `MinkUNet34`
  - `MinkUNet50`
  - `MinkUNet101`
  - 多个 `A/B/C/D/...` 变体

## 对应算法
- 更接近:
  - AnyDexGrasp 这类基于稀疏体素卷积的抓取检测/生成前端
- 不属于当前 `cvae / udg / 3dp` 第一阶段的优先路线

## 输入输出维度总结
- 对外最重要的是 `WrappedMinkUNet`
- 它的输出已经是比较友好的两种视图:
  - `global_feature`: `(B, C)`
  - `local_feature`: `(B, N, C)`
- 与 `PointNet` 相比:
  - `PointNet` 的局部特征格式是 `(B, C, N)`
  - `WrappedMinkUNet` 的局部特征格式是 `(B, N, C)`
- 后续统一接口时需要固定一种通用张量格式

## 当前状态判断
- 依赖 `MinkowskiEngine`，环境门槛高。
- 还需要完整的体素化预处理才能进入训练。
- 适合作为未来稀疏卷积路线候选，不适合当前先打通最简训练闭环。
