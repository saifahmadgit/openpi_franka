#!/bin/bash
# Fine-tune pi0.5 on a 260-episode random subset of saifahmad123/EXP3_50 (single-task
# "pick up the orange cylinder", prompt_from_task=True so it stays language-conditioned).
# Data-scaling point between pi05_Franka_EXP3_10 (130 eps) and pi05_Franka_EXP3_50
# (650 eps): 260 eps / 78,096 frames.
#
# THERE IS NO saifahmad123/EXP3_20 DATASET. pi05_Franka_EXP3_20 points at the full
# EXP3_50 repo and sets DataConfig.episodes -- 260 indices drawn with
# np.random.default_rng(20) -- so the subset is selected at load time. Nothing was
# re-uploaded and the on-disk cache is shared with the EXP3_50 run. Changing that seed or
# size silently changes the training set.
#
# The subset is applied by SubsetDataset (data_loader.py), NOT by
# LeRobotDataset(episodes=...): that lerobot path filters the rows but builds
# episode_data_index positionally over the subset while __getitem__ still indexes it with
# the original episode_index, so it dies with "IndexError: index 288 is out of bounds for
# dimension 0 with size 260" mid-epoch (job 9842717). Don't switch it back.
#
# Norm stats are the FULL EXP3_50 stats, copied verbatim to
# assets/pi05_Franka_EXP3_20/saifahmad123/EXP3_50/norm_stats.json
# (asset_id follows repo_id, which is still saifahmad123/EXP3_50). They therefore
# describe all 650 episodes, not the 260 being trained on -- deliberate, per the request.
# compute_norm_stats.py does NOT need to be re-run. (A miss there is silent:
# _load_norm_stats() returns None and Normalize passes data through untouched, i.e. the
# run would train unnormalized. Confirm with the "Loaded norm stats from ..." line in the
# .err log before trusting a run.)
#
# HF_HUB_OFFLINE=1 is safe here: EXP3_50 is already fully cached under HF_LEROBOT_HOME
# (/projects/p53063/data/lerobot/saifahmad123/EXP3_50, 489 MB) from the EXP3_50_h100 run,
# so train.py's pre-download loop has nothing to fetch and no rate limit to wait out.
# It must be exported from the shell, not set from Python: huggingface_hub freezes it
# into a module constant at import time.
#
# Walltime: 6k steps. The 2k-step A100 runs of this shape (EXP1/EXP3/EXP5_10) came in at
# 2:12-2:18 including their first-time dataset download; with the download already done,
# 6k steps should land well under 8 h. 24 h is headroom, not an estimate. Over-requesting
# time is free as far as fairshare goes -- Slurm charges actual elapsed, not TimeLimit --
# and CLAUDE.md flags wall-clock kills as the main failure mode.
#
# Retention: exactly two checkpoints. save_interval=3000 means the only saves written are
# 3000/ and the final one at num_train_steps - 1 = 5999 (not 6000), and both survive
# pruning -- keep_period=3000 spares 3000/, max_to_keep=1 spares the newest.
# ~8.9 GB each, ~18 GB total. /projects/p53063 had 165 GB free at submit time.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_EXP3_20_a100
# e32706, not p53063. Priority on gengpu is ~99.8% fairshare
# (PriorityWeightFairShare=100000 vs Age=10000), and as of 2026-08-16 the two accounts
# sit at ~0.47 vs ~0.23 -- e32706 holds 7x the RawShares (500 vs 70) and is
# under-consuming them. Checkpoints still land in /projects/p53063.
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
# 72G, matching the other EXP A100 scripts. gengpu bills
# TRESBillingWeights=CPU=4.5,Mem=1.0G,gres/gpu=4.5 under PriorityFlags=MAX_TRES, so the
# charge is max(16*4.5, mem_GB, 4.5) -- flat at 72 until mem exceeds 72G, then mem-bound.
# Observed MaxRSS on this job shape is ~57 GiB, so 72G keeps ~26% headroom.
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

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_EXP3_20 \
    --exp-name=EXP3_20_a100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
