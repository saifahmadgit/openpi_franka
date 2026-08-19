#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/EXP3_50 (single-task "pick up the orange cylinder",
# prompt_from_task=True so the policy stays language-conditioned). 5x the data of
# pi05_Franka_EXP3_10: 650 eps / 196,856 frames vs 130 eps / 39,387.
# pi05_Franka_EXP3_50 keeps the EXP-family LR (peak 2.5e-5 -> decay 2.5e-6, cosine)
# with the schedule stretched to the longer run: 10k steps, warmup 1000 (10% of
# training), decay_steps 10000 so the cosine bottoms out at the last step.
# batch 32, action_horizon=50 -> ~1.6 epochs.
#
# Norm stats already exist at
# assets/pi05_Franka_EXP3_50/saifahmad123/EXP3_50/norm_stats.json
# so compute_norm_stats.sbatch does NOT need to be re-run. (A miss there is silent:
# _load_norm_stats() returns None and Normalize passes data through untouched, i.e.
# the run would train unnormalized. Confirm with the "Loaded norm stats from ..." line
# in the .err log before trusting a run.)
#
# HF_HUB_OFFLINE is deliberately NOT set: EXP3_50 is not yet cached under
# HF_LEROBOT_HOME (EXP0, EXP0_20, EXP1_10, EXP3_10, EXP4_10, EXP5_10, EXP6_10 are), so
# train.py's pre-download loop (scripts/train.py:220-247) must be able to reach the Hub.
# It retries with a 320s sleep to wait out the 1000 req / 5 min rate limit, then flips
# itself offline before create_data_loader. Once the download has completed once, add
#     export HF_HUB_OFFLINE=1
# below for subsequent runs (must be exported from the shell, not set from Python:
# huggingface_hub freezes it into a module constant at import time).
#
# Download cost: EXP3_10 is 1,576 files / 98 MB for 130 eps, so EXP3_50 at 650 eps is
# ~7,800 files / ~500 MB. Files, not bytes, are the long pole -- ~8 rate-limit windows
# is a ~40 min floor, call it 1-1.5 h with 429 retries. The limit is per account, so do
# not start another first-time dataset download alongside this one.
#
# H100, 1 GPU: scripts/train.py shards over all visible devices, and every single-GPU
# H100 run of this shape has landed at ~1.34 s/it (pi05_GraspNet_2, job 8934087:
# 30k steps in 11:10:14; GraspNet_2_100, job 9035162: 11:11:14). 10k steps is therefore
# ~3.7 h of compute, ~5-5.5 h including the download. 24 h is headroom, not an estimate.
# Over-requesting time is free as far as fairshare goes -- Slurm charges actual elapsed,
# not TimeLimit (verified against sreport) -- and CLAUDE.md flags wall-clock kills as
# the main failure mode, so the headroom is deliberate.
#
# Retention/disk: save_interval=2000 with keep_period=2000, so every save is kept --
# 2000/, 4000/, 6000/, 8000/ plus the final 9999/ (no 10000/; the last save is at
# num_train_steps - 1) = 5 dirs x ~8.9 GB = ~45 GB. /projects/p53063 had 164 GB free at
# submit time (1.0T, 85% used), so this fits but the margin is thinner than usual --
# check `df -h /projects/p53063` before resubmitting.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_EXP3_50_h100
# e32706, not p53063. Priority on gengpu is ~99.8% fairshare
# (PriorityWeightFairShare=100000 vs Age=10000), and as of 2026-08-16 the two accounts
# sit at ~0.47 vs ~0.23 -- e32706 holds 7x the RawShares (500 vs 70) and is
# under-consuming them. Checkpoints still land in /projects/p53063.
#SBATCH --account=e32706
# gengpu is the only H100 partition open to us: b33210-gpu / b53080 / b32868 / a9009 all
# carry AllowAccounts=<their own buy-in account> and we only hold e32706 and p53063.
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
# 120G, matching train_GRASPNET_FINAL_H100.sbatch. gengpu bills
# TRESBillingWeights=CPU=4.5,Mem=1.0G,gres/gpu=4.5 under PriorityFlags=MAX_TRES, so the
# charge is max(16*4.5, mem_GB, 4.5) -- flat at 72 until mem exceeds 72G, then
# mem-bound. Observed MaxRSS on the A100 EXP runs is ~57 GiB and ~53 GiB on the H100
# GraspNet runs; EXP3_50 has 5x the episodes of EXP3_10, so 120G is the safer side of
# that. Drop to 72G if a completed run's MaxRSS stays well under it.
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_EXP3_50 \
    --exp-name=EXP3_50_h100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
