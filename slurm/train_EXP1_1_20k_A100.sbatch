#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/EXP1_1 for 20k steps -- the long-schedule arm of the
# EXP1 data-scaling ladder. 13 episodes / 3,920 frames, single task
# ("pick up the orange cylinder"), prompt_from_task=True so it stays language-conditioned.
#
# Sibling of train_EXP1_1_A100.sbatch, which ran the same dataset for 2k steps. The
# only change is num_train_steps 2k -> 20k (with warmup and cosine decay_steps scaled to
# match), so the pair isolates "not enough training" from "not enough data".
# Schedule is identical across the 1/5/10% rungs, so epochs differ by design:
# 163 / 32 / 16 respectively. This one runs 163.3 epochs.
#
# Norm stats already exist at
# assets/pi05_Franka_EXP1_1_20k/saifahmad123/EXP1_1/norm_stats.json
# -- a verified md5-identical copy of the 2k config's file, since the dataset is
# byte-identical. No compute_norm_stats.py pass is needed and this script has NO
# --dependency. A miss would be SILENT: _load_norm_stats() returns None and Normalize
# passes data through untouched, i.e. the run would train unnormalized. Confirm the
# "Loaded norm stats from ..." line in the .err log before trusting the run.
#
# HF_HUB_OFFLINE=1 is safe here because EXP1_1 is already in HF_LEROBOT_HOME from the
# 2k run. If you ever point this at a cold cache, DROP the export or train.py's
# pre-download loop has nothing to fetch with. It must be exported from the shell, not set
# from Python: huggingface_hub freezes it into a module constant at import time.
#
# Walltime: measured ~3.45 s/step on this job shape (EXP1_5 did 2k steps in 1:54:57
# including startup), so 20k is ~19-20 h. 36 h is headroom, not an estimate -- Slurm
# charges actual elapsed, not TimeLimit, and CLAUDE.md flags wall-clock kills as the main
# failure mode. gengpu caps TimeLimit at 2-00:00:00.
#
# Retention: save_interval=5000 with keep_period=10_000. max_to_keep=1 spares the newest
# save plus any step where step % keep_period == 0, so 5000/ is pruned when 10000/ lands
# and 15000/ is pruned when the final 19999/ lands (the last save is at
# num_train_steps - 1, not 20000). End state is exactly two dirs, 10000/ + 19999/,
# ~18 GB; at most three (~27 GB) coexist mid-run. The 5k interval exists only to give
# --resume a recent restart point on a ~20 h run.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_EXP1_1_20k_a100
# e32706, not p53063. Priority on gengpu is ~99.8% fairshare
# (PriorityWeightFairShare=100000 vs Age=10000) and e32706 holds 7x the RawShares
# (500 vs 70) while under-consuming them. Checkpoints still land in /projects/p53063.
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
# 72G, matching the other EXP A100 scripts. gengpu bills
# TRESBillingWeights=CPU=4.5,Mem=1.0G,gres/gpu=4.5 under PriorityFlags=MAX_TRES, so the
# charge is max(16*4.5, mem_GB, 4.5) -- flat at 72 until mem exceeds 72G, then mem-bound.
# Observed MaxRSS on this job shape is ~57 GiB, so 72G keeps ~26% headroom.
#SBATCH --mem=72G
#SBATCH --time=36:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache
export HF_HUB_OFFLINE=1

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_EXP1_1_20k \
    --exp-name=EXP1_1_20k_a100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
