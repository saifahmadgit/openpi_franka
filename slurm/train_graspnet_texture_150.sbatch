#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/Franka_GraspNet_Test_texture (texture/roughness DR)
# with action_horizon=150 instead of 50.
#
# Norm stats are already committed at
# assets/pi05_Franka_GraspNet_Test_texture_150/saifahmad123/Franka_GraspNet_Test_texture/
# (copied from the horizon-50 config; norm stats cover only the state/action columns
# and do not depend on the action horizon), so compute_norm_stats.sbatch does NOT
# need to be re-run for this config.
#
# Checkpoints go to project storage (1 TB) instead of home. save_interval=2500 with
# the default keep_period=5000 means 6 dirs survive (5k,10k,15k,20k,25k,30k) at
# ~8.9 GB each, ~53 GB total.
#
# Walltime: the horizon-50 run measured ~1.3 s/it on one H100 (~11 h for 30k steps).
# The action expert now attends over 150 action tokens instead of 50, so expect a
# meaningfully slower step time. 48 h is gengpu's max; if it still times out, resubmit
# with --resume instead of --overwrite.
#
# Memory: XLA_PYTHON_CLIENT_MEM_FRACTION is 0.85 here rather than 0.9 — the longer
# horizon grows the action-expert activations, and leaving a little more headroom is
# cheaper than an OOM 10 h in. Drop batch_size in the config if it OOMs anyway.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_graspnet_texture_150
#SBATCH --account=p53063
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run scripts/train.py pi05_Franka_GraspNet_Test_texture_150 \
    --exp-name=graspnet_texture_150 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
