# PCT Backbone Notes

## 对应脚本
- `models/backbones/pct.py`

## 主要用途
- 作为候选的局部点云编码器。
- 更偏向输出逐点特征，而不是只输出单个全局向量。
- 如果后续仓库统一到 `local token / local feature` 编码范式，`pct.py` 是重点候选之一。

## 脚本中的类

### `PointCloudTransformerEncoder`
- 位置: `pct.py:6`
- 作用:
  - 先用两层 `Conv1d` 把点特征提升到 128 维。
  - 再串联 4 个自注意力层 `SA_Layer`。
  - 最后融合局部特征和全局池化特征，输出逐点特征。
- 输入:
  - `x`: `(B, N, in_channels)`
- 输出:
  - `(B, N, feature_dim)`
- 中间结构:
  - 前端卷积后特征: `(B, 128, N)`
  - 注意力块输出拼接后: `(B, 512, N)`
  - 融合后逐点输出: `(B, N, feature_dim)`

### `SA_Layer`
- 位置: `pct.py:82`
- 作用:
  - 对逐点特征做自注意力更新。
  - 输入输出通道数保持不变。
- 输入:
  - `x`: `(B, C, N)`
- 输出:
  - `x`: `(B, C, N)`

## 关于脚本下半部分
- 从 `pct.py` 中段开始有一大段三引号包裹的备用实现。
- 其中又定义了一套带旋转位置编码的 `PointCloudTransformerEncoder` / `SA_Layer`。
- 当前这部分不是生效代码，只能视为历史方案或备选方案。
- 现阶段若要接入训练，默认应只看文件顶部那一套生效实现。

## 对应算法
- 当前仓库中尚未直接接入某个生成式算法训练链。
- 更适合作为:
  - 统一 `local feature` 编码器候选
  - token 级条件生成的前端

## 输入输出维度总结
- 输入:
  - `(B, N, in_channels)`
- 输出:
  - `(B, N, feature_dim)`
- 若要导出统一接口，最自然的组织方式是:
  - `token_xyz`: `(B, N, 3)`
  - `token_feat`: `(B, N, feature_dim)`

## 当前状态判断
- 依赖 `flash_attn`，并且是顶层导入，所以即使 `use_flash_attn=False`，环境里通常也要有该库。
- 适合后续作为统一局部编码器做对比实验，但不适合作为当前第一优先打通链路的骨干。
