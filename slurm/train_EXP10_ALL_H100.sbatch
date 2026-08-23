#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/EXP10_ALL -- the full 8-object pool (cracker box,
# orange cylinder, lemon, mustard bottle, peach, purple cube, red cube, tomato soup can),
# 700 eps / 207,539 frames, prompt_from_task=True so the policy is language-conditioned
# across all 8 objects. This is NOT a rung of the EXP1/EXP3 size ladders; its pair is
# pi05_Franka_EXP9_ALL (686 eps / 205,955 frames), which is identical in every field
# except repo_id, so the comparison is on the data itself.
#
# 20k steps, warmup 2000 (10% of training), peak_lr 2.5e-5 -> 2.5e-6 cosine with
# decay_steps=20000 so it bottoms out at the last step. batch 32, action_horizon=50,
# num_workers=14 -> ~3.1 epochs.
#
# Norm stats are ALREADY committed at
# assets/pi05_Franka_EXP10_ALL/saifahmad123/EXP10_ALL/norm_stats.json, so no
# compute_norm_stats dependency is needed. A miss there would be SILENT:
# _load_norm_stats() returns None and Normalize passes data through untouched, i.e. the
# run would train unnormalized. Confirm the "Loaded norm stats from ..." line in the
# .err log before trusting the run.
#
# HF_HUB_OFFLINE is deliberately NOT set. EXP10_ALL is not in $HF_LEROBOT_HOME (EXP0,
# EXP0_20, EXP1_1/5/10, EXP3_10/30/50, EXP4_10, EXP5_10, EXP6_10, EXP7_10, EXP8_10 and
# the GraspNet sets are), so train.py's pre-download loop must actually reach the Hub;
# it retries with a 320 s sleep to wait out the 1000 req / 5 min rate limit, then sets
# HF_HUB_OFFLINE=1 itself before create_data_loader. Exporting it up front would
# hard-fail on the cold cache. Once this job has downloaded the set, a resubmission can
# export it -- from the shell, not from Python: huggingface_hub freezes it into a module
# constant at import time.
#
# Download cost: EXP3_50 measures 7,816 files / 489 MB for 650 eps, so EXP10_ALL at 700
# eps is ~8,400 files / ~525 MB. Files, not bytes, are the long pole -- ~8 rate-limit
# windows is a ~40 min floor, call it 1-1.5 h with 429 retries.
#
# SUBMIT THIS SECOND. The Hub rate limit is per ACCOUNT, and EXP9_ALL is a cold-cache
# download too, so the two pre-download loops would fight each other (not fatal -- the
# loop retries forever -- but it doubles the stall). Submit as:
#     JID=$(sbatch --parsable slurm/train_EXP9_ALL_H100.sbatch)
#     sbatch --dependency=after:${JID}+90 slurm/train_EXP10_ALL_H100.sbatch
# "after:<id>+90" starts this job 90 minutes after EXP9 BEGINS RUNNING (not after it
# finishes), by which point EXP9's download is done and its GPU work is underway -- the
# two runs then overlap on separate H100s, which is the point. Plain
# `sbatch slurm/train_EXP10_ALL_H100.sbatch` also works if EXP9_ALL is already cached.
#
# H100, 1 GPU: every single-H100 run of this model shape has landed at ~1.34 s/it
# (pi05_GraspNet_2 job 8934087: 30k steps in 11:10:14; GraspNet_2_100 job 9035162:
# 11:11:14), so 20k steps is ~7.5 h of compute, ~9 h including the download. 24 h is
# headroom, not an estimate -- Slurm charges actual elapsed, not TimeLimit, and
# CLAUDE.md flags wall-clock kills as the main failure mode.
#
# Retention/disk: save_interval=10000 with keep_period=10000. train.py saves when
# (step % save_interval == 0 and step > start_step) or step == num_train_steps - 1, so
# over steps 0..19999 only 10000/ and the final 19999/ are written -- 2 dirs x ~8.9 GB
# = ~18 GB (~36 GB with the EXP9 run). /projects/p53063 had 122 GB free (1.0T, 89%
# used) when this was written; check `df -h /projects/p53063` before resubmitting. The
# 10000/ save doubles as the only --resume point if the job hits its walltime.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_EXP10_ALL_h100
# e32706, not p53063. Priority on gengpu is ~99.8% fairshare
# (PriorityWeightFairShare=100000 vs Age=10000) and e32706 holds 7x the RawShares
# (500 vs 70) while under-consuming them. Checkpoints still land in /projects/p53063.
#SBATCH --account=e32706
# gengpu is the only H100 partition open to us: b33210-gpu / b53080 / b32868 / a9009 all
# carry AllowAccounts=<their own buy-in account> and we only hold e32706 and p53063.
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
# 120G, matching train_EXP3_50_H100.sbatch (650 eps, same num_workers=14). gengpu bills
# TRESBillingWeights=CPU=4.5,Mem=1.0G,gres/gpu=4.5 under PriorityFlags=MAX_TRES, so the
# charge is max(16*4.5, mem_GB, 4.5) -- flat at 72 until mem exceeds 72G, then
# mem-bound. Observed MaxRSS is ~57 GiB on the A100 EXP runs and ~53 GiB on the H100
# GraspNet runs; loader RAM tracks workers x batch x action_horizon, not dataset size,
# so 120G is comfortable headroom rather than a fitted number.
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
df -h /projects/p53063

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_EXP10_ALL \
    --exp-name=EXP10_ALL_h100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
