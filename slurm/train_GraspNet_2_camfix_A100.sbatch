#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/Franka_GraspNet_2 with the corrected camera
# assignment: front_1 -> cam_high, wrist -> cam_left_wrist, front_2 -> cam_right_wrist,
# with geometric_aug_keys=("base_0_rgb", "right_wrist_0_rgb") so the two static cameras
# (not the wrist) get the random crop/resize/rotate. Data and hyperparameters are
# otherwise identical to pi05_Franka_GraspNet_2, so the pair is a clean A/B on the remap.
#
# Norm stats at
# assets/pi05_Franka_GraspNet_2_camfix/saifahmad123/Franka_GraspNet_2/norm_stats.json
# are a byte-identical copy of pi05_Franka_GraspNet_2's, and are valid for this config:
# the remap only changes which video stream feeds which view, and compute_norm_stats.py
# injects zero placeholder images, so state/action statistics cannot differ. The copy is
# needed because train.py resolves stats as assets_base_dir / config.name / repo_id
# (config.py:545) — sharing a dataset is not enough, the directory has to exist under
# this config's name. A miss there is silent: _load_norm_stats() returns None
# (config.py:200) and Normalize passes data through untouched (transforms.py:127), i.e.
# the run would train unnormalized. Verified loading before this script was finalized.
# compute_norm_stats.sbatch does NOT need to be re-run.
#
# Checkpoints go to project storage (1 TB) instead of home. save_interval=2500 with the
# default keep_period=5000 means 6 dirs survive (5k,10k,15k,20k,25k,30k) at ~8.9 GB
# each, ~53 GB total.
#
# Walltime: same compute profile as the baseline — 30k steps at ~1.3 s/it is ~11 h on
# one H100. 24 h leaves headroom; gengpu allows up to 48 h.
#
# ORDERING: submit this only after train_GraspNet_2.sbatch has finished downloading
# Franka_GraspNet_2 into /projects/p53063/data/lerobot, then uncomment HF_HUB_OFFLINE=1
# below. Two jobs populating the same local_dir concurrently is what got 8686483-85
# killed with HTTP 429.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_GraspNet_2_camfix_a100
#SBATCH --account=p53063
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache
# Uncomment once Franka_GraspNet_2 is fully cached under HF_LEROBOT_HOME. Must be
# exported here, not set from Python: huggingface_hub freezes HF_HUB_OFFLINE into a
# module constant at import time, so train.py setting it later is a no-op.
export HF_HUB_OFFLINE=1

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_GraspNet_2_camfix \
    --exp-name=GraspNet_2_camfix_a100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
