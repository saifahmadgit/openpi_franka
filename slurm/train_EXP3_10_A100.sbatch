#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/EXP3_10 (single-task "pick up the orange cylinder",
# prompt_from_task=True so it stays language-conditioned). Companion to
# train_EXP1_10_A100.sbatch -- pi05_Franka_EXP3_10 is identical to pi05_Franka_EXP1_10
# in every field except repo_id, so the pair is a clean A/B on the dataset alone.
# Short 2k-step run: warmup 200, peak_lr 2.5e-5 -> 2.5e-6 cosine, batch 32,
# action_horizon=50.
#
# Norm stats already exist at
# assets/pi05_Franka_EXP3_10/saifahmad123/EXP3_10/norm_stats.json
# so compute_norm_stats.sbatch does NOT need to be re-run. (A miss there is silent:
# _load_norm_stats() returns None and Normalize passes data through untouched, i.e.
# the run would train unnormalized. Confirm with the "Loaded norm stats from ..." line
# in the .err log before trusting a run.)
#
# HF_HUB_OFFLINE is deliberately NOT set: EXP3_10 is not yet cached under
# HF_LEROBOT_HOME, so train.py's pre-download loop (scripts/train.py:220-247) must be
# able to reach the Hub. It retries with a 320s sleep to wait out the 1000 req / 5 min
# rate limit, then flips itself offline before create_data_loader. Once the download
# has completed once, add
#     export HF_HUB_OFFLINE=1
# below for subsequent runs (must be exported from the shell, not set from Python:
# huggingface_hub freezes it into a module constant at import time).
#
# CONCURRENCY: the HF rate limit is per account, so kicking this off alongside another
# first-time dataset download can trip 429s. The retry loop recovers, it just costs
# walltime. EXP1_10 is ~97 MB / 525 files and took ~19 min to fetch; EXP3_10 should be
# comparable.
#
# Walltime: 2k steps is ~2 h on one A100 (EXP0 runs came in at 1:53 and 1:59), plus the
# dataset download on this first run. 24 h is headroom, not an estimate. Over-requesting
# time is free as far as fairshare goes -- Slurm charges actual elapsed, not TimeLimit
# (verified against sreport) -- and CLAUDE.md flags wall-clock kills as the main failure
# mode, so the headroom is deliberate.
#
# Retention: save_interval=200 with keep_period=1000, and max_to_keep=1 spares the newest
# save plus any step where step % keep_period == 0. So only 1000/ and the final 1999/
# survive (the last save is at num_train_steps - 1, not 2000). ~8.9 GB each, ~18 GB total.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_EXP3_10_a100
# e32706, not p53063. Priority on gengpu is ~99.8% fairshare
# (PriorityWeightFairShare=100000 vs Age=10000), and as of 2026-08-16 the two accounts
# sit at ~0.47 vs ~0.23 -- e32706 holds 7x the RawShares (500 vs 70) and is
# under-consuming them. Same job estimated at ~13 days under p53063 vs minutes under
# e32706. Checkpoints still land in /projects/p53063.
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
# 72G, not the 150G the EXP0 scripts use. gengpu bills
# TRESBillingWeights=CPU=4.5,Mem=1.0G,gres/gpu=4.5 under PriorityFlags=MAX_TRES, so the
# charge is max(16*4.5, mem_GB, 4.5) -- i.e. flat at 72 until mem exceeds 72G, then
# mem-bound. Observed MaxRSS on this job shape is ~57 GiB (9336241: 56.1, 9357372:
# 57.3), so 72G keeps ~26% headroom while halving the fairshare burn vs 150G.
# Raise it if a run OOMs, but past 72G every extra GB is charged 1:1.
#SBATCH --mem=72G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_EXP3_10 \
    --exp-name=EXP3_10_a100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
