# PointNet Backbone Notes

## 对应脚本
- `models/backbones/pointnet.py`

## 主要用途
- 当前仓库里最适合先作为 `DexLearn-style` 生成式模型默认点云编码器。
- 适配 `cvae / udg / 3dp` 这类“点云编码 -> 全局条件 -> 低维生成头”的路线。

## 脚本中的类

### `PointNet`
- 位置: `pointnet.py:11`
- 作用:
  - 对输入点云做逐点 `Conv1d(1x1)` 编码。
  - 通过 `max pooling` 生成全局物体特征。
  - 同时保留逐点局部特征图。
- 输入:
  - `x`: `(B, N, K)`
  - 只使用前 `point_feature_dim` 维，通常为 `(B, N, 3)`。
- 输出:
  - `global_feature`: `(B, pc_feature_dim)`
  - `local_feature`: `(B, C_local, N)`，其中 `C_local = local_conv_hidden_layers_dim[-1]`
- 典型维度:
  - 默认 `local_conv_hidden_layers_dim=[64,128,256]`
  - 默认 `global_mlp_hidden_layers_dim=[256]`
  - 默认 `pc_feature_dim=128`
  - 则输出一般为:
    - `global_feature`: `(B, 128)`
    - `local_feature`: `(B, 256, N)`

### `RelPointnet`
- 位置: `pointnet.py:67`
- 作用:
  - 先基于给定的旋转和平移，将输入点云变换到某个相对坐标系下。
  - 再调用 `PointNet` 编码。
- 输入:
  - `x`: `(B, N, K)`，前 3 维必须是 XYZ
  - `rot`: `(B, 3, 3)`
  - `trans`: `(B, 3)`
- 输出:
  - 与 `PointNet` 相同:
    - `global_feature`: `(B, pc_feature_dim)`
    - `local_feature`: `(B, C_local, N)`
- 适用场景:
  - 后续如果要把 `hand_now` 或相机系/手系相对表示引入条件，可以优先考虑这一路。

## 对应算法
- 直接对应:
  - DexLearn 风格的 `CVAE`
  - DexLearn 风格的 `UDG / normalizing flow`
  - DexLearn 风格的 `3DP / diffusion head`
- 可作为统一骨干:
  - 当前仓库最适合先拿它做第一版统一训练接口

## 当前状态判断
- 该脚本本身比较独立，依赖简单，适合先打通训练。
- 它输出的是 `global + local` 两种特征视图，后续统一到 `local token` 路线时也方便保留兼容性。
