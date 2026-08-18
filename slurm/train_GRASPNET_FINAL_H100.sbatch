#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/GRASPNET_FINAL -- the final, largest GraspNet
# collection (5500 eps / 1.69M frames, 7 tasks: cracker box, lemon, mustard bottle,
# nivea men face wash tube, peach, pear, tomato soup can). prompt_from_task=True, so
# the policy stays language-conditioned across all 7 prompts.
# 50k steps, warmup 1500, peak_lr 2.5e-5 -> 2.5e-6 cosine, batch 32, action_horizon=50.
#
# Norm stats already exist at
# assets/pi05_Franka_GRASPNET_FINAL/saifahmad123/GRASPNET_FINAL/norm_stats.json
# so compute_norm_stats.sbatch does NOT need to be re-run. (A miss there is silent:
# _load_norm_stats() returns None and Normalize passes data through untouched, i.e.
# the run would train unnormalized. Confirm with the "Loaded norm stats from ..." line
# in the .err log before trusting a run.)
#
# HF_HUB_OFFLINE is deliberately NOT set: GRASPNET_FINAL is not yet cached under
# HF_LEROBOT_HOME, so train.py's pre-download loop (scripts/train.py:220-247) must be
# able to reach the Hub. It retries with a 320s sleep to wait out the 1000 req / 5 min
# rate limit, then flips itself offline before create_data_loader. Once the download
# has completed once, add
#     export HF_HUB_OFFLINE=1
# below for subsequent runs (must be exported from the shell, not set from Python:
# huggingface_hub freezes it into a module constant at import time).
#
# DOWNLOAD IS THE LONG POLE, NOT THE BYTES: the repo is only 4.76 GB but it is 22,005
# files (one video per camera per episode). At the 1000 req / 5 min account rate limit
# that is >=22 windows, i.e. ~2 h floor, and the retry loop's 320 s sleeps push it
# higher whenever it trips a 429. Budget 2-4 h before step 0. Do not start another
# first-time dataset download alongside this one -- the limit is per account.
#
# H100, not A100, and only 1 GPU: scripts/train.py shards over all visible devices, but
# every prior single-GPU H100 run of this shape landed at ~1.34 s/it (pi05_GraspNet_2,
# job 8934087: 30k steps in 11:10:14; GraspNet_2_100, job 9035162: 11:11:14). 50k steps
# is therefore ~18.6 h of compute. Plus the download, that is ~21-23 h, which fits the
# 48 h gengpu cap (PartitionName=gengpu MaxTime=2-00:00:00 -- this partition CANNOT be
# given more, and gengpu-long has A100s only, no H100s). The margin is real but not
# huge: if this times out, resubmit with --resume instead of --overwrite and it picks up
# from the last save.
#
# Retention/disk: save_interval=5000 with keep_period=5000, so every save is kept --
# 5000/ ... 45000/ plus the final 49999/ (no 50000/; the last save is num_train_steps-1)
# = 10 dirs x ~8.9 GB = ~89 GB. /projects/p53063 had 240 GB free at submit time
# (1.0T, 77% used), so this fits but does not leave much. Check `df -h /projects/p53063`
# before resubmitting.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_GRASPNET_FINAL_h100
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
# 120G, between the 200G the older H100 scripts requested and the 72G the A100 EXP
# scripts use. gengpu bills TRESBillingWeights=CPU=4.5,Mem=1.0G,gres/gpu=4.5 under
# PriorityFlags=MAX_TRES, so the charge is max(16*4.5, mem_GB, 4.5) -- flat at 72 until
# mem exceeds 72G, then mem-bound. Observed MaxRSS on this job shape is ~53 GiB
# (8934087: 53.35, 9035162: 51.10) and ~57 GiB on the A100 EXP runs; GRASPNET_FINAL has
# 2.6x the episodes but only 11% more frames, so expect ~60 GiB. 120G is ~2x headroom
# at 40% less billing than 200G. Raise it if a run OOMs.
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_GRASPNET_FINAL \
    --exp-name=GRASPNET_FINAL_h100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
