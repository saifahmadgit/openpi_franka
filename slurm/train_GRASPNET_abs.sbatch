#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/Franka_GRASPNET with *absolute* joint actions
# (use_delta_joint_actions=False) instead of the baseline's deltas.
#
# Norm stats are already committed at
# assets/pi05_Franka_GRASPNET_abs/saifahmad123/Franka_GRASPNET/
# These are a genuinely different set from the baseline's, not a copy: the delta
# transform runs before statistics are gathered, so the action stats here describe
# absolute joint angles (mean ~[0.01, -0.29, ..., 2.05, 0.83]) rather than per-step
# deltas (mean ~0). Do not swap the two.
#
# Checkpoints go to project storage (1 TB) instead of home. save_interval=2500 with
# the default keep_period=5000 means 6 dirs survive (5k,10k,15k,20k,25k,30k) at
# ~8.9 GB each, ~53 GB total.
#
# Compute settings match train_GRASPNET.sbatch — dropping the delta transform changes
# no model shapes, so step time should be unchanged (~11 h for 30k steps).
#
#
# A100 (not H100): the h100 queue had ~95 GPUs of higher-priority demand against 60
# cards, so nothing was schedulable inside slurm's 48h backfill window. The a100 pool
# had ~11 free SXM cards and only one higher-priority job pending, so these start
# immediately. --constraint=quest12 pins to the 4-GPU SXM nodes (503G RAM, 80G cards);
# the quest10 PCIe a100 nodes only have 188G of node RAM and so can never satisfy
# --mem=200G anyway.
#
# 50k steps at the ~2.5-3.5 s/it expected on an A100 is ~35-48h. gengpu's MaxTime is
# 2-00:00:00, so 48h is the ceiling and covers the run. Do not drop this back to 24h:
# a running job's TimeLimit can only be shrunk by a user, never grown, so an
# under-sized limit costs a second queue cycle plus a --resume submission.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_GRASPNET_abs
#SBATCH --account=p53063
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=quest12
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache
# The dataset is already fully cached under HF_LEROBOT_HOME (2572 parquet + 7716 mp4,
# downloaded by the pi05_Franka_GRASPNET run). Going offline makes LeRobotDatasetMetadata
# take its local load_metadata() path instead of falling through to the Hub, which is what
# got jobs 8686483-85 killed with HTTP 429 while the first run was still populating the
# same local_dir. Must be exported here, not set from Python: huggingface_hub freezes
# HF_HUB_OFFLINE into a module constant at import time, so train.py setting it later is
# a no-op. Unset this if the dataset ever needs re-downloading.
export HF_HUB_OFFLINE=1

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_GRASPNET_abs \
    --exp-name=GRASPNET_abs \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
