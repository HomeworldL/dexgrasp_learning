# AGENTS_zh.md

## 仓库使命

本仓库聚焦于基于点云的灵巧手抓取生成。

当前主线目标为：

1. 先做单条件抓取生成，输入使用全局点云。
2. 先在 world 系验证抓取成功率。
3. 再切换到 camera 系，对比成功率。
4. 最后推进到局部点云。

当前基线算法是 CVAE。点云 backbone 必须可替换。算法栈需要保持可扩展，使后续升级到 diffusion、flow 或其他生成式方法时，不需要重写训练与仿真主流水线。

## 核心问题定义

- 训练/推理输入：
  - 物体完整点云或局部点云
  - 可选的当前手位姿条件：`hand_now`
- 输出：
  - 目标抓取位姿：`qpos_init`、`qpos_squeeze`
  - 目标抓取关节向量：`squeeze_joint`
- 坐标系规则：
  - 对单条件局部点云输入，选择的输入坐标系（`world` 或 `camera`）也决定标签和输出坐标系。

## 当前优先级

- 先聚焦单条件流水线。
- 在单条件 train/sim 路径稳定前，不处理双条件训练和评测。
- 评估改动时，优先看抓取成功率，不只看 loss。

## 仓库结构

- `assets/`
  - 数据集软连接、抓取标签、点云标签
  - 仿真资产与场景资源
- `docs/`
  - 设计说明、实验记录、设计决策、持续维护文档
- `models/`
  - 本仓库实际使用的模型实现
- `models_ref/`
  - 外部仓库参考代码或复现代码
  - 默认只读，不直接耦合到主训练流水线
- `src/`
  - 训练/评估模块、仿真辅助模块、共享运行时逻辑
- 仓库根目录
  - 当前活跃的 train/sim 入口脚本

## 归属与边界规则

- 新功能优先落在 `src/` 和 `models/`。
- `models_ref/` 默认只作参考，除非明确要求修复复现代码。
- 模型代码中禁止硬编码数据集路径。
- 仿真逻辑与纯学习逻辑必须解耦。
- `src/mj_ho.py` 在主线开发中视为不可改动文件。它必须与数据集采集程序保持一致；除非用户明确要求做同步上游修改，否则禁止改动。任何训练/仿真适配都必须放在 `src/mj_ho.py` 之外的包装层实现。

## 数据集契约

`datasets/graspdata_YCB_liberhand/<object>/scaleXXX/` 下的典型输出为：

```text
coacd.obj
object.xml
convex_parts/*.obj
grasp.h5
grasp.npy
grasp_fail.h5
grasp_fail.npy
pc_warp/
  global_pc.npy
  cam_in.npy
  cam_ex_XX.npy
  partial_pc_XX.npy
  partial_pc_cam_XX.npy
```

当前 `grasp.h5` / `grasp.npy` 中的抓取数组主要包括：

- `qpos_init`
- `qpos_approach`
- `qpos_prepared`
- `qpos_grasp`
- `qpos_squeeze`

这些数组当前统一按 `float32` 存储。

## 统一数据集格式

内部样本表示必须满足：

- 每个 object-scale 条目必须有一个源文件：`grasp.h5`
- 每个 object-scale 条目必须有一个派生文件：`grasp.npy`
- 每个 object-scale 条目必须有一个失败样本源文件：`grasp_fail.h5`
- 每个 object-scale 条目必须有一个失败样本派生文件：`grasp_fail.npy`
- `grasp.npy` 必须由 `grasp.h5` 转换而来，且抓取数值完全一致
- 点云必须独立存储，不能打包进 `grasp.npy`
- 局部点云渲染输出属于后处理结果，需与抓取数组分离

`grasp.h5` 必需 schema：

- `object_name: str`
- `scale: float`
- `hand_name: str`
- `rot_repr: "wxyz+qpos"`
- `qpos_init: [tx, ty, tz, qw, qx, qy, qz, q1...qN]`
- `qpos_approach: [tx, ty, tz, qw, qx, qy, qz, q1...qN]`
- `qpos_prepared: [tx, ty, tz, qw, qx, qy, qz, q1...qN]`
- `qpos_grasp: [tx, ty, tz, qw, qx, qy, qz, q1...qN]`
- `qpos_squeeze: [tx, ty, tz, qw, qx, qy, qz, q1...qN]`
- `meta: {}`

`grasp_fail.h5` 必需摘要字段：

- `qpos_fail: [tx, ty, tz, qw, qx, qy, qz, q1...qN]`
- `failure_stage: str`

局部点云渲染输出必须包括：

- `global_pc.npy`
- `cam_in.npy`
- `cam_ex_XX.npy`
- `partial_pc_XX.npy`
- `partial_pc_cam_XX.npy`

## 数据集切分规则

`build_dataset_splits.py` 输出：

- `datasets/<dataset_tag>/train.json`
- `datasets/<dataset_tag>/test.json`

切分规则：

- 按 `object_name` 切分，而不是按 object-scale 条目切分
- 同一物体的所有 scale 必须留在同一个 split 中
- 默认对唯一物体数做约 `80/20` 切分，并使用配置中的 `seed` 打乱
- 只有正样本抓取输出、失败样本输出和所需渲染输出都存在，条目才可进入清单

split 记录应包含：

- `grasp_h5_path`
- `grasp_npy_path`
- `grasp_h5_fail_path`
- `grasp_fail_npy_path`
- `global_pc_path`
- `partial_pc_path[]`
- `partial_pc_cam_path[]`
- `cam_ex_path[]`

## 数据与坐标系约定

- 点云与 wrist/base 抓取位姿标签必须处于同一坐标系
- 手指关节保持机器人本体定义
- 坐标变换必须在数据流水线中显式实现，并记录转换来源
- 禁止在模型 `forward` 中隐式做坐标转换

当前数据集约定：

- `mode="train"` 读取 `assets/datasets/graspdata_YCB_liberhand/train.json`
- `mode="eval"` 读取 `assets/datasets/graspdata_YCB_liberhand/test.json`
- 每个 JSON item 对应一个 object-scale 粒度条目，路径相对 `datasets/<dataset_tag>/`
- 点数大于 `n_points` 时，使用 PyTorch3D FPS 下采样
- 点数小于 `n_points` 时，使用随机重复补齐

当前监督目标：

- `qpos_init`
- `qpos_squeeze`
- `squeeze_joint`

## 当前活跃入口

以下入口定义了当前主工作面，文档必须与之保持一致：

- `train_sc.py`：统一的单条件训练入口
- `src/print_dataset.py`：打印单条数据样本用于检查
- `sim_dataset.py`：直接评测数据集内保存抓取的 oracle 入口
- `sim_sc.py`：统一的单条件仿真评测入口

实现优先级：

- 先稳定单条件 train/sim 路径
- 上述路径稳定后，再处理双条件流程

推荐命令风格：

```bash
python train_sc.py --config configs/ycb_liberhand_sc.yaml --set model.algorithm=cvae --set data.frame=world --set data.cloud_type=global
python sim_sc.py --config configs/ycb_liberhand_sc.yaml --set sim.split=train --set sim.ckpt_path=...
python sim_dataset.py --config configs/ycb_liberhand_sc.yaml --split train
python -m src.print_dataset --config configs/ycb_liberhand_sc.yaml --split train --index 0
```

## 配置优先策略

训练、评估、仿真都必须由配置驱动。

当前活跃的单条件根配置为：

- `configs/ycb_liberhand_sc.yaml`

组合实验统一通过 CLI 覆盖，例如：

- `--set model.algorithm=...`
- `--set data.frame=...`
- `--set model.input_encoder.name=...`

单条件 base config 中，模型配置应按以下三层组织：

- `model.common`
- `model.algorithms.<algorithm>`
- `model.input_encoders.<backbone>`

硬性要求：

- 禁止在代码里硬编码数据根路径、机器人路径、checkpoint 路径
- 必填配置项缺失时必须快速失败，并给出明确报错

## 模型演进策略

- 早期验证阶段以 CVAE 为主线基线
- backbone 替换必须低成本且显式
- 保持清晰的算法注册/解析接口，让 CVAE、diffusion、flow 共享同一套数据与运行时接口
- 避免做出把仓库锁死在单一生成范式上的架构决策

## Python 开发规范

- Python `>= 3.10`，除非后续有明确项目级约束说明
- 遵循 PEP 8
- 公共函数与关键数据结构应添加类型注解
- 脚本保持轻量，可复用逻辑下沉到模块
- 长时任务统一使用 `logging`，避免零散 `print`
- 数据变换与坐标变换函数应小而可测
- 避免使用冗余的回退逻辑和不必要的 `try/except`

## 测试规范

仓库应维护以下测试：

- 数据加载与坐标转换正确性
- 模型输入输出形状与关键不变量
- 仿真接口契约；若全仿真代价过高，可先 mock

回归规则：

- 任何“抓取可行性回归”类 bug，都必须补回归测试

当前高优先级检查项：

- `train/eval` manifest 切换是否正确
- PyTorch3D `se3_log_map` 的矩阵约定是否正确
- FPS 下采样与小点云补齐行为
- `candidate_indices` 的固定长度拼 batch 行为

## 实验管理

- 每次实验都必须保存配置、commit id、seed、指标
- 尽量固定随机种子；若存在不可避免的非确定性，需要记录来源
- 关键设计决策和消融记录写入 `docs/`
- `experiment_matrix.csv` 是当前实验协议下的活动结果表，不是长期结果堆积表
- 当活动实验协议发生实质变化时，应删除旧协议结果，避免与新协议混写

当根据用户提供的结果填写实验表时：

- 必须严格遵守现有 `experiment_matrix.csv` 的列结构
- 如果多个实验名称实际对应同一组有效配置，应自动将同一份结果同步到所有匹配行
- 除非用户另有说明，默认以下字段构成主匹配键：
  - `model.algorithm`
  - `data.frame`
  - `model.input_encoder.name`
  - `model.prediction_structure.name`
  - `seed`
  - `sim.num_grasp_samples`
- 填写时保留各自行自己的实验名称，但重复配置行复用相同指标与 `summary_path`

## 提交与 PR 规范

使用 Conventional Commits，例如：

- `feat(src): add dual-condition training entry`
- `fix(data): correct camera-frame transform in loader`

每个 PR 应包含：

- 改了什么，以及为什么改
- 可复现实验/运行命令
- 影响了哪些数据集和配置
- 是否修改了 `models_ref/`

## 文档与计划文件生命周期

当前活动计划文件：

- `TODO.md`

生命周期规则：

- 仓库根目录只保留一个活动计划文件：`TODO.md`
- 替换当前 `TODO.md` 前，必须先归档到 `docs/`，命名为 `docs/<YYYYMMDD>_TODO_<summary>.md`
- 当前 `TODO.md` 全部完成后，也必须先归档，再创建新的根目录 `TODO.md`
- 执行任务时，始终优先读取根目录当前活动的 `TODO.md`；历史计划文件仅供参考

`docs/` 下文档命名为强制规则：

- 格式：`<YYYYMMDD>_<category>_<summary>.md`
- 日期必须在最前
- 类别必须短且明确

推荐类别：

- `TODO`
- `方案`
- `调研`
- `过程`
- `实验`
- `指南`

示例：

- `20260315_方案_双条件训练接口设计.md`
- `20260315_调研_DexLearn与DexDiffuser扩散对比.md`
- `20260315_TODO_模型迁移历史.md`

## Do / Don't

Do：

- 保持架构模块化与配置优先
- 保持坐标转换显式透明
- 保持文档与真实运行行为一致
- 将 `models_ref/` 视为参考实现

Don't：

- 在模型内部隐藏路径假设或坐标系假设
- 让主流水线与 `models_ref/` 强耦合
- 在缺少配置和文档更新的情况下合入大段实验代码

## AGENTS 同步规则

每次更新 `AGENTS.md` 后，必须在仓库根目录创建或更新中文翻译文件 `AGENTS_zh.md`。
