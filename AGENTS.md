# AGENTS.md

## Repository Mission

This repository focuses on dexterous grasp generation from point clouds.

The current mainline goal is:

1. Start with single-condition grasp generation using global point clouds.
2. Validate grasp success rate in the world frame first.
3. Then switch to the camera frame and compare success rate.
4. Finally move to partial point clouds.

The current baseline algorithm is CVAE. The point-cloud backbone must remain swappable. The algorithm stack should stay extensible so later upgrades to diffusion, flow, or other generative methods do not require rewriting the training and simulation pipeline.

## Core Problem Definition

- Inputs for training and inference:
  - Full or partial object point cloud
  - Optional current hand pose condition: `hand_now`
- Outputs:
  - Target grasp poses: `qpos_init`, `qpos_squeeze`
  - Target grasp joint vector: `squeeze_joint`
- Coordinate-setting rule:
  - For single-condition local-point-cloud input, the chosen input frame (`world` or `camera`) also determines the label/output frame.

## Current Priority

- Focus on the single-condition pipeline first.
- Do not spend time on dual-condition training or evaluation until the single-condition train/sim loop is stable.
- Evaluate changes primarily by grasp success metrics, not only by loss reduction.

## Repository Layout

- `assets/`
  - Dataset symlinks, grasp labels, point-cloud labels
  - Simulation assets and scene resources
- `docs/`
  - Design notes, experiment logs, decisions, and running documentation
- `models/`
  - Model implementations actually used by this repository
- `models_ref/`
  - External reference or reproduction code
  - Read-only by default; do not couple it to the main training pipeline
- `src/`
  - Training/evaluation modules, simulation helpers, shared runtime logic
- Repository root
  - Active train/sim entry scripts

## Ownership Rules

- New functionality should go into `src/` and `models/` first.
- `models_ref/` is reference-only unless a change is explicitly requested for reproduction repair.
- Do not hardcode dataset paths in model code.
- Keep simulation logic decoupled from pure learning logic.
- `src/mj_ho.py` is immutable for mainline work. Keep it aligned with the dataset-collection pipeline and do not modify it unless the user explicitly requests a synchronized upstream change. Any training/simulation adaptation must happen in wrapper code outside `src/mj_ho.py`.

## Dataset Contract

Example dataset output under `datasets/graspdata_YCB_liberhand/<object>/scaleXXX/`:

```text
coacd.obj
object.xml
convex_parts/*.obj
grasp.h5
grasp.npy
partial_pc_warp/
  cam_in.npy
  cam_ex_XX.npy
  partial_pc_XX.npy
  partial_pc_cam_XX.npy
```

Current grasp arrays stored in `grasp.h5` / `grasp.npy`:

- `qpos_init`
- `qpos_approach`
- `qpos_prepared`
- `qpos_grasp`
- `qpos_squeeze`

All grasp arrays are currently stored as `float32`.

## Unified Dataset Format

Internal sample representation must follow these rules:

- Each object-scale item must have one required source file: `grasp.h5`
- Each object-scale item must have one required derived file: `grasp.npy`
- `grasp.npy` must be converted from `grasp.h5` with numerically identical grasp values
- Point clouds must stay outside `grasp.npy`
- Partial-point-cloud render outputs are post-processing artifacts and remain separate from grasp arrays

Required `grasp.h5` schema:

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

Required partial-point-cloud outputs:

- `cam_in.npy`
- `cam_ex_XX.npy`
- `partial_pc_XX.npy`
- `partial_pc_cam_XX.npy`

## Dataset Split Rules

`build_dataset_splits.py` writes:

- `datasets/<dataset_tag>/train.json`
- `datasets/<dataset_tag>/test.json`

Split rules:

- Split by `object_name`, not by object-scale entry
- All scales of the same object must stay in the same split
- Default split is approximately `80/20` over unique objects, shuffled by config `seed`
- An entry is valid only if `grasp.h5`, `grasp.npy`, and all required render outputs exist

## Data And Coordinate Conventions

- Point clouds and wrist/base grasp labels must be in the same frame
- Finger joints stay in the robot-native definition
- Coordinate transforms must be explicit in the data pipeline and their source must be recorded
- Do not perform implicit frame conversion inside model `forward`

Current dataset convention:

- `mode="train"` reads `assets/datasets/graspdata_YCB_liberhand/train.json`
- `mode="eval"` reads `assets/datasets/graspdata_YCB_liberhand/test.json`
- Each JSON item corresponds to one object-scale entry relative to `datasets/<dataset_tag>/`
- If the point count is larger than `n_points`, downsample with PyTorch3D FPS
- If the point count is smaller than `n_points`, pad by random repetition

Current supervised targets:

- `qpos_init`
- `qpos_squeeze`
- `squeeze_joint`

## Active Entry Points

These entry points define the current working surface and must stay consistent with the docs:

- `train_sc.py`: unified single-condition training entry
- `src/print_dataset.py`: print one dataset sample for inspection
- `sim_dataset.py`: oracle evaluation using grasps stored in the dataset
- `sim_sc.py`: unified single-condition simulation evaluation entry

Implementation priority:

- Stabilize the single-condition train/sim path first
- Only work on dual-condition flows after the above is stable

Recommended command style:

```bash
python train_sc.py --config configs/ycb_liberhand_sc.yaml --set model.algorithm=cvae --set data.frame=world --set data.cloud_type=global
python sim_sc.py --config configs/ycb_liberhand_sc.yaml --set sim.split=train --set sim.ckpt_path=...
python sim_dataset.py --config configs/ycb_liberhand_sc.yaml --split train
python -m src.print_dataset --config configs/ycb_liberhand_sc.yaml --split train --index 0
```

## Configuration Policy

Training, evaluation, and simulation must all be config-driven.

The active single-condition root config is:

- `configs/ycb_liberhand_sc.yaml`

Use CLI overrides for experiment combinations, for example:

- `--set model.algorithm=...`
- `--set data.frame=...`
- `--set model.input_encoder.name=...`

Single-condition base configs should keep model configuration in these layers:

- `model.common`
- `model.algorithms.<algorithm>`
- `model.input_encoders.<backbone>`

Hard requirements:

- Do not hardcode dataset roots, robot asset paths, or checkpoint paths inside code
- Missing required config values must fail fast with clear errors

## Model Evolution Policy

- Keep the current baseline centered on CVAE for early validation
- Keep backbone replacement cheap and explicit
- Preserve a clean algorithm registry so CVAE, diffusion, and flow can share the same data/runtime interface
- Avoid architecture decisions that lock the repository to a single generative family

## Python Development Rules

- Python `>= 3.10`, unless a different project-level constraint is explicitly documented later
- Follow PEP 8
- Add type annotations to public functions and important data structures
- Keep scripts thin; move reusable logic into modules
- Use `logging` for long-running jobs instead of scattered `print`
- Keep data transforms and coordinate-conversion helpers small and testable
- Avoid redundant fallback branches and unnecessary `try/except` structures

## Testing Requirements

The repository should maintain tests for:

- Data loading and coordinate-conversion correctness
- Model input/output shapes and key invariants
- Simulation interface contracts; use mocks first if full simulation is too expensive

Regression rule:

- Any bug related to grasp feasibility regression must add a regression test

Current high-priority checks:

- Correct `train/eval` manifest switching
- PyTorch3D `se3_log_map` matrix convention usage
- FPS downsampling and small-point-cloud padding behavior
- Fixed-length batch behavior for `candidate_indices`

## Experiment Management

- Every experiment must save config, commit id, seed, and metrics
- Fix random seeds whenever possible and record non-deterministic sources when unavoidable
- Write key decisions and ablations into `docs/`
- `experiment_matrix.csv` is an active protocol table, not a long-term result dump
- When the active protocol changes materially, remove stale rows from `experiment_matrix.csv` instead of mixing old and new protocols

When filling experiment results from user-provided metrics:

- Follow the existing `experiment_matrix.csv` column schema strictly
- If multiple experiment names map to the same effective configuration, sync the same result to all matching rows
- Unless the user says otherwise, use these fields as the primary match key:
  - `model.algorithm`
  - `data.frame`
  - `model.input_encoder.name`
  - `model.prediction_structure.name`
  - `seed`
  - `samples_per_object_scale`
- Keep experiment names unchanged per row, but reuse the same metrics and `summary_path` for duplicate effective configurations

## Commit And PR Rules

Use Conventional Commits, for example:

- `feat(src): add dual-condition training entry`
- `fix(data): correct camera-frame transform in loader`

Each PR should include:

- What changed and why
- Reproducible commands
- Which datasets and configs are affected
- Whether `models_ref/` was modified

## Documentation And Plan Lifecycle

Current active plan file:

- `TODO.md`

Lifecycle rules:

- Keep exactly one active plan file at the repository root: `TODO.md`
- Before replacing the current `TODO.md`, archive it into `docs/` as `docs/<YYYYMMDD>_TODO_<summary>.md`
- When the current `TODO.md` is fully completed, archive it first, then create the next root `TODO.md`
- Always read the current root `TODO.md` first when executing tasks; historical plan files are reference-only

Documentation naming under `docs/` is mandatory:

- Format: `<YYYYMMDD>_<category>_<summary>.md`
- Date must come first
- Category must be short and explicit

Recommended categories:

- `TODO`
- `方案`
- `调研`
- `过程`
- `实验`
- `指南`

Examples:

- `20260315_方案_双条件训练接口设计.md`
- `20260315_调研_DexLearn与DexDiffuser扩散对比.md`
- `20260315_TODO_模型迁移历史.md`

## Do / Don't

Do:

- Keep the architecture modular and config-first
- Keep coordinate conversion explicit and transparent
- Keep documentation aligned with actual runnable behavior
- Treat `models_ref/` as reference-only

Do not:

- Hide path assumptions or frame assumptions inside model internals
- Couple the mainline pipeline tightly to `models_ref/`
- Merge large experimental changes without matching config and documentation updates

## AGENTS Sync Rule

Every time `AGENTS.md` is updated, create or update the Chinese translation at the repository root as `AGENTS_zh.md`.
