#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/Franka_GraspNet_2 (2990 eps / 1.30M frames, 3 tasks).
# Baseline of the GraspNet_2 pair: the original Franka camera remap (wrist -> cam_high).
#
# Norm stats are committed at
# assets/pi05_Franka_GraspNet_2/saifahmad123/Franka_GraspNet_2/norm_stats.json, so
# compute_norm_stats.sbatch does NOT need to be re-run. The camfix config carries a
# byte-identical copy under its own name (stats are resolved by config name, not repo
# id); if you ever recompute these, update both directories.
#
# Checkpoints go to project storage (1 TB) instead of home. save_interval=2500 with the
# default keep_period=5000 means 6 dirs survive (5k,10k,15k,20k,25k,30k) at ~8.9 GB
# each, ~53 GB total.
#
# Walltime: 30k steps at the ~1.3 s/it measured across the GRASPNET family is ~11 h on
# one H100. 24 h leaves headroom for the cold first epoch and the initial dataset
# download; gengpu allows up to 48 h.
#
# DATASET DOWNLOAD: Franka_GraspNet_2 is NOT yet cached under
# /projects/p53063/data/lerobot/saifahmad123/. This job downloads it via the retry loop
# in train.py. Do NOT submit train_GraspNet_2_camfix.sbatch until that finishes — two
# jobs populating the same local_dir is what got 8686483-85 killed with HTTP 429.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_GraspNet_2
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
# Franka_GraspNet_2 is fully cached (2990 parquet / 8970 mp4, verified). Offline so
# LeRobotDatasetMetadata takes its local load_metadata() path instead of falling through
# to the Hub: jobs 8916322/8916324/8917203/8917204 all died on HTTP 429 doing exactly
# that. Must be exported from the shell, not set from Python: huggingface_hub freezes
# HF_HUB_OFFLINE into a module constant at import time, so train.py setting it is a no-op.
export HF_HUB_OFFLINE=1

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_GraspNet_2 \
    --exp-name=GraspNet_2 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
