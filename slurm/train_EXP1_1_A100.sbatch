#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/EXP1_1 (single-task "pick up the orange cylinder",
# prompt_from_task=True so it stays language-conditioned). 13 episodes / 3,920 frames
# -- the 1% rung of the EXP1 data-scaling ladder below pi05_Franka_EXP1_10
# (130 eps / 39,625 frames).
#
# Hyperparameters are IDENTICAL to train_EXP1_10_A100.sbatch on purpose: 2k steps,
# warmup 200, peak_lr 2.5e-5 -> 2.5e-6 cosine, batch 32, action_horizon=50. Compute is
# held constant so dataset size is the only variable, which means this run sees
# ~16.3 epochs versus 1.6 on EXP1_10. Epoch-matching instead would need
# num_train_steps scaled down; that was deliberately not done.
#
# Unlike pi05_Franka_EXP3_20, EXP1_1 is a REAL uploaded dataset, not an episodes=
# subset of a bigger repo, so it has its own norm stats at
# assets/pi05_Franka_EXP1_1/saifahmad123/EXP1_1/norm_stats.json
# (computed 2026-08-19). compute_norm_stats.py does NOT need to be re-run. A miss there
# is silent: _load_norm_stats() returns None and Normalize passes data through
# untouched, i.e. the run would train unnormalized. Confirm with the
# "Loaded norm stats from ..." line in the .err log before trusting a run.
#
# HF_HUB_OFFLINE=1 is safe: the dataset (~0.01 GB, tiny) was fully cached under
# HF_LEROBOT_HOME by the norm-stats pass, so train.py's pre-download loop has nothing
# to fetch and no rate limit to wait out. It must be exported from the shell, not set
# from Python: huggingface_hub freezes it into a module constant at import time.
#
# Walltime: 2k steps at this shape ran 2:12-2:18 on an A100 for EXP1_10/EXP3_10/EXP5_10
# INCLUDING their first-time dataset download. With the data already cached this should
# be well under 2 h. 24 h is headroom, not an estimate -- Slurm charges actual elapsed,
# not TimeLimit, and CLAUDE.md flags wall-clock kills as the main failure mode.
#
# Retention: exactly two checkpoints. save_interval=1000 means the only saves written
# are 1000/ and the final one at num_train_steps - 1 = 1999 (not 2000), and both
# survive pruning -- keep_period=1000 spares 1000/, max_to_keep=1 spares the newest.
# ~8.9 GB each, ~18 GB total.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_EXP1_1_a100
# e32706, not p53063. Priority on gengpu is ~99.8% fairshare
# (PriorityWeightFairShare=100000 vs Age=10000) and e32706 holds 7x the RawShares
# (500 vs 70) while under-consuming them. Checkpoints still land in /projects/p53063.
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
# 72G, matching the other EXP A100 scripts. gengpu bills
# TRESBillingWeights=CPU=4.5,Mem=1.0G,gres/gpu=4.5 under PriorityFlags=MAX_TRES, so the
# charge is max(16*4.5, mem_GB, 4.5) -- flat at 72 until mem exceeds 72G, then
# mem-bound. Observed MaxRSS on this job shape is ~57 GiB, so 72G keeps ~26% headroom.
#SBATCH --mem=72G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache
export HF_HUB_OFFLINE=1

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_EXP1_1 \
    --exp-name=EXP1_1_a100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
