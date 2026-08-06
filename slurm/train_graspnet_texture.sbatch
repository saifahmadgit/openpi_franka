#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/Franka_GraspNet_Test_texture (texture/roughness DR).
#
# Norm stats are already committed at
# assets/pi05_Franka_GraspNet_Test_texture/saifahmad123/Franka_GraspNet_Test_texture/
# so compute_norm_stats.sbatch does NOT need to be re-run for this config.
#
# Checkpoints go to project storage (1 TB) instead of home. save_interval=2500 with
# the default keep_period=5000 means 6 dirs survive (5k,10k,15k,20k,25k,30k) at
# ~8.9 GB each, ~53 GB total.
#
# Compute settings match slurm/train_3_objects_2_1gpu.sbatch. Walltime does not:
# 30k steps at the measured 1.3 s/it on one H100 is ~11 h, and the 4 h in that
# script is what killed the earlier run mid-training. gengpu allows up to 48 h.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_graspnet_texture
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

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_GraspNet_Test_texture \
    --exp-name=graspnet_texture \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
