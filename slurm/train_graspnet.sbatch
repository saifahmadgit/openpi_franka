#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/Franka_GraspNet_Test.
#
# Run slurm/compute_norm_stats.sbatch first — training asserts that
# assets/pi05_Franka_GraspNet_Test/.../norm_stats.json exists.
#
# Checkpoints go to project storage (1 TB) instead of home. A single run of
# 10k steps writes 5 checkpoints at ~8.9 GB each.
#
# Resource settings match slurm/train_3_objects_2_1gpu.sbatch, which ran cleanly.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_graspnet
#SBATCH --account=p53063
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_GraspNet_Test \
    --exp-name=graspnet_test \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
