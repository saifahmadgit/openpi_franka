#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/Franka_GRASPNET (2572 eps / 1.70M frames, 3 tasks).
# Baseline of the GRASPNET family: delta joint actions, SigLIP encoder trained.
#
# Norm stats are already committed at
# assets/pi05_Franka_GRASPNET/saifahmad123/Franka_GRASPNET/
# so compute_norm_stats.sbatch does NOT need to be re-run for this config.
#
# Checkpoints go to project storage (1 TB) instead of home. save_interval=2500 with
# the default keep_period=5000 means 6 dirs survive (5k,10k,15k,20k,25k,30k) at
# ~8.9 GB each, ~53 GB total.
#
# Walltime: 30k steps at the ~1.3 s/it measured on the comparable GraspNet_Test_texture
# run is ~11 h on one H100. 24 h leaves headroom; gengpu allows up to 48 h.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_GRASPNET
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

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_GRASPNET \
    --exp-name=GRASPNET \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
