#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/Franka_GraspNet_2 (2990 eps / 1.30M frames, 3 tasks)
# with a 100-step action horizon. Identical to train_GraspNet_2.sbatch in every other
# respect -- pi05_Franka_GraspNet_2_100 differs from pi05_Franka_GraspNet_2 only in
# Pi0Config.action_horizon (50 -> 100), verified by dataclass diff.
#
# Norm stats at
# assets/pi05_Franka_GraspNet_2_100/saifahmad123/Franka_GraspNet_2/norm_stats.json
# are a byte-identical copy of pi05_Franka_GraspNet_2's; the horizon does not change
# state/action statistics. The copy is required because train.py resolves stats as
# assets_base_dir / config.name / repo_id (config.py:545), and a miss is silent --
# _load_norm_stats() returns None and Normalize passes the data through untouched.
# compute_norm_stats.sbatch does NOT need to be re-run.
#
# Checkpoints go to project storage (1 TB) instead of home. save_interval=2500 with the
# default keep_period=5000 means 6 dirs survive (5k,10k,15k,20k,25k,30k) at ~8.9 GB
# each, ~53 GB total.
#
# Walltime: the horizon-50 runs did 30k steps in ~11 h at 1.3 s/it (jobs 8934087/8934089).
# Doubling the horizon doubles the action-expert token count, so budget roughly 2x that.
# 48 h is gengpu's MaxTime and covers even a 3x slowdown; a running job's TimeLimit can be
# shrunk by a user but never grown, and wall-clock kills are this repo's usual failure mode.
#
# Account: e32706 (fairshare 0.48 vs p53063's 0.23). The horizon-50 pair sat 8 h behind
# 61 higher-priority jobs on p53063 and started within minutes once moved. Switch back to
# p53063 if that allocation should not carry this run.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_GraspNet_2_100
#SBATCH --account=e32706
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
# Franka_GraspNet_2 is fully cached (2990 parquet / 8970 mp4, verified). Offline so
# LeRobotDatasetMetadata takes its local load_metadata() path instead of falling through
# to the Hub: jobs 8916322/8916324/8917203/8917204 all died on HTTP 429 doing exactly
# that. Must be exported from the shell, not set from Python: huggingface_hub freezes
# HF_HUB_OFFLINE into a module constant at import time, so train.py setting it is a no-op.
export HF_HUB_OFFLINE=1

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_GraspNet_2_100 \
    --exp-name=GraspNet_2_100_steps \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
