#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/EXP0 (single-task "pick up the orange cylinder",
# prompt_from_task=True so it stays language-conditioned). Short 2k-step run:
# warmup 200, peak_lr 2.5e-5 -> 2.5e-6 cosine, batch 32, action_horizon=50.
#
# Norm stats already exist at
# assets/pi05_Franka_EXP0/saifahmad123/EXP0/norm_stats.json
# so compute_norm_stats.sbatch does NOT need to be re-run. (A miss there is silent:
# _load_norm_stats() returns None and Normalize passes data through untouched, i.e.
# the run would train unnormalized.)
#
# HF_HUB_OFFLINE is deliberately NOT set: EXP0 is not yet cached under
# HF_LEROBOT_HOME, so train.py's pre-download loop (scripts/train.py:220-247) must be
# able to reach the Hub. It retries with a 320s sleep to wait out the 1000 req / 5 min
# rate limit, then flips itself offline before create_data_loader. Do not run a second
# job populating the same local_dir concurrently — that is what got 8686483-85 killed
# with HTTP 429. Once the download has completed once, add
#     export HF_HUB_OFFLINE=1
# below for subsequent runs (must be exported from the shell, not set from Python:
# huggingface_hub freezes it into a module constant at import time).
#
# Walltime: 2k steps is only ~1-2 h of compute on one A100, but the first run also
# pays for the dataset download, which can take hours and stall on rate limits.
# 24 h is headroom, not an estimate; gengpu allows up to 48 h.
#
# Retention: the point of this run is to find the minimum step count that works, so
# the config sets keep_period=200 to match save_interval=200. Every save then hits
# step % keep_period == 0 and escapes the max_to_keep=1 pruning: all 10 checkpoints
# (200..2000) are kept, ~8.9 GB each, ~89 GB total.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_EXP0_a100
#SBATCH --account=p53063
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_EXP0 \
    --exp-name=EXP0_a100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
