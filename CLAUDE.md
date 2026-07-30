# openpi_franka — repo guide

Fork of [Physical Intelligence `openpi`](https://github.com/Physical-Intelligence/openpi)
(π₀ / π₀.₅ vision-language-action models), adapted to fine-tune on **real Franka arm**
LeRobot datasets.

Upstream history is preserved; local work begins around `41178bc`
("updated scripts for joint actions").

> Machine-specific setup (cluster accounts, storage paths, job submission) is
> deliberately **not** in this file — it differs per checkout location.

## Layout

| Path | What it is |
|---|---|
| `src/openpi/training/config.py` | **The main file we edit.** All `TrainConfig`s live in `_CONFIGS` at the bottom. |
| `src/openpi/policies/` | Per-robot input/output transforms. The Franka configs reuse `aloha_policy.py`; `lehome_policy.py` is a local addition for the LeHome dual-arm sim. |
| `src/openpi/models/` | JAX/Flax π₀, π₀.₅, π₀-FAST, Gemma/SigLIP backbones. |
| `src/openpi/models_pytorch/` | PyTorch port of the same models (`train_pytorch.py` path). |
| `scripts/train.py` | JAX training entrypoint. |
| `scripts/compute_norm_stats.py` | Precompute normalization stats — **must run before training a new config**. |
| `scripts/serve_policy.py` | Websocket policy server for robot-side inference. |
| `slurm/*.sbatch` | Job submission scripts (paths inside are machine-specific). |
| `assets/<config>/<repo_id>/norm_stats.json` | Output of `compute_norm_stats.py`. Small; belongs in git. |
| `checkpoints/<config>/<exp_name>/<step>/` | `params/`, `train_state/`, `assets/`. ~1.8 GB per checkpoint. |
| `logs/` | Job stdout/stderr. Untracked. |

## Environment

Dependency management is **uv** (`uv run <cmd>`), Python ≥3.11, JAX on CUDA 12.

```bash
uv run python -c "import jax; print(jax.devices())"   # sanity check GPUs
uv run wandb login                                     # W&B is enabled by default
uv run huggingface-cli login                           # only needed for private datasets
```

Useful env vars: `HF_LEROBOT_HOME` (where LeRobot datasets are downloaded),
`HF_HOME` (HF cache), `JAX_PLATFORMS=cpu` (force CPU, e.g. for norm stats on a
GPU-less node).

## Typical workflow

```bash
# 1. add/modify a TrainConfig in src/openpi/training/config.py
# 2. compute norm stats (note: --config-name is a *flag*, not positional)
uv run scripts/compute_norm_stats.py --config-name <config_name>

# 3. train
uv run scripts/train.py <config_name> --exp-name=<run_name> [--overwrite | --resume]
```

`--checkpoint-base-dir` overrides where checkpoints land (default `./checkpoints`),
which is how runs are pointed at large scratch/project storage.

`compute_norm_stats.py` is **CPU-only and needs no GPU**: it calls
`disable_video_decoding()` and injects zero placeholder images, so it reads only the
`state`/`action` columns out of the dataset's parquet files. Image content cannot
affect state/action statistics. Practical consequence: norm stats can be computed for
a dataset whose *videos* are still uploading, as long as the parquet files and
`meta/info.json` are complete.

## Franka configs

All named `pi05_Franka_*` in `_CONFIGS`. They share a shape:

- `Pi0Config(pi05=True, action_dim=32, action_horizon=16|50, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")` — LoRA on both towers, with a matching `freeze_filter`.
- `data=LeRobotAlohaDataConfig(...)` with `adapt_to_pi=False` (our data is already in
  the right space) and a `RepackTransform` mapping the dataset's camera keys onto
  `cam_high` / `cam_left_wrist` / `cam_right_wrist`.
- `use_delta_joint_actions=True` with `delta_action_mask=make_bool_mask(7, -1)` —
  7 joint dims predicted as deltas from current state, gripper stays absolute.
- `weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`
  — fine-tune from π₀.₅ base.
- `ema_decay=None`, cosine LR schedule, `save_interval=2500`.

### Franka dataset schema

The `saifahmad123/Franka_*` LeRobot datasets (v2.1, `robot_type=franka`, 30 fps) all share:

```
observation.state          float32 (9,)  7 joints + gripper_left + gripper_right
action                     float32 (8,)  7 joints + gripper
observation.images.wrist    video (3,480,640)
observation.images.front_1  video (3,480,640)
observation.images.front_2  video (3,480,640)
```

Because state is 9-dim but action is 8-dim, `make_bool_mask(7, -1)` matches the
*action* layout. The camera remap in every config is deliberately non-obvious —
the wrist camera is fed as the "high" view:

```
cam_high        <- observation.images.wrist
cam_left_wrist  <- observation.images.front_1
cam_right_wrist <- observation.images.front_2
```

Multi-task datasets (several language prompts) set
`base_config=DataConfig(prompt_from_task=True)` so the policy is language-conditioned
on each episode's task string.

Active configs: `pi05_Franka_3_objects_2` (1965 eps / 1.36M frames, 3 tasks) and
`pi05_Franka_GraspNet_Test` (2130 eps / 1.52M frames, 3 tasks: cracker box, mustard
bottle, nivea face wash). Both use `action_horizon=50`, 30k steps, batch 32.

## Local patches to upstream (why they exist)

Don't "clean these up".

1. **`scripts/train.py` — HF pre-download loop.** Before `create_data_loader`, the
   dataset is `snapshot_download`ed in a retry loop that sleeps 320s on failure, then
   sets `HF_HUB_OFFLINE=1`. HuggingFace rate-limits at 1000 req / 5 min and the
   many-worker LeRobot dataset construction trips it. `compute_norm_stats.py` has the
   same block.
2. **`data_loader.py` — `video_backend="pyav"`.** torchcodec imports fine on some HPC
   nodes but its FFmpeg shared libs aren't on the library path, so it dies at *decode*
   time rather than import time. pyav statically bundles FFmpeg.
3. **`pyproject.toml`** — pins `av==13.1.0`, bumps wandb, and adds a
   `[[tool.uv.dependency-metadata]]` block for the git-sourced `lerobot` so uv can
   resolve it without building.
4. **`compute_norm_stats.py` — `disable_video_decoding` + `InjectDummyImages`.**
   Turns an hours-long pass into minutes on large datasets.

## Gotchas

- **`--overwrite` deletes the existing checkpoint dir.** `--overwrite` and `--resume`
  are mutually exclusive and the config raises if both are set. Check which one a
  submission script passes before re-running it.
- **Wall-clock kills are the main failure mode.** 30k steps at ~1.3 s/it on a single
  H100 is ~11h; on 4 GPUs proportionally less. Size the time limit accordingly or
  plan to `--resume`.
- `pyproject.toml` has an `override-dependencies` key under
  `[tool.pytest.ini_options]` where `testpaths` used to be. Pytest ignores it, so
  `uv run pytest` no longer has test paths configured — likely an accidental paste.
- `echo` and `EXIT=0` in the repo root are ~1 MB stray files from a mistyped shell
  redirect (captured training logs, not code). Safe to delete.
- `scratchpad_*.py/json` in the root are ad-hoc norm-stat comparison helpers.
