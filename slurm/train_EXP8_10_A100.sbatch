#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/EXP8_10 (single-task "pick up the red cube",
# prompt_from_task=True so it stays language-conditioned). 130 episodes / 39,800 frames.
#
# Third arm of the EXP6/EXP7/EXP8 A/B: pi05_Franka_EXP8_10 matches pi05_Franka_EXP7_10 and
# pi05_Franka_EXP6_10 in every field except repo_id -- 3k steps, warmup 300 (10% of
# training), peak_lr 2.5e-5 -> 2.5e-6 cosine bottoming out at the last step, batch 32,
# action_horizon=50. All three are 130 eps with the SAME task string (EXP6_10 40,979
# frames, EXP7_10 40,573, EXP8_10 39,800), so the comparison is on the data itself, not on
# size or schedule.
#
# Norm stats are ALREADY committed at
# assets/pi05_Franka_EXP8_10/saifahmad123/EXP8_10/norm_stats.json (commit 5aeabde), so
# unlike the EXP6/EXP7 scripts this one needs no --dependency on a norm-stats job. A miss
# there would be SILENT: _load_norm_stats() returns None and Normalize passes data through
# untouched, i.e. the run would train unnormalized. Confirm the "Loaded norm stats from
# ..." line in the .err log before trusting the run.
#
# NO HF_HUB_OFFLINE=1 here -- and that is the one real difference from
# train_EXP7_10_A100.sbatch. EXP8_10 is NOT in $HF_LEROBOT_HOME (only EXP0..EXP7 and the
# GraspNet sets are), so train.py's pre-download loop has to actually fetch it: ~525 files
# from the hub, a few minutes, after which train.py sets HF_HUB_OFFLINE=1 itself for the
# many-worker LeRobotDataset construction. Exporting it up front would hard-fail on the
# cold cache. Once this job has run, the dataset is cached and a resubmission could export
# it -- from the shell, not from Python: huggingface_hub freezes it into a module constant
# at import time.
#
# Walltime: 3k steps is ~3-3.5 h on one A100 (2k came in at 2:12-2:18 on EXP1/EXP3/EXP5,
# and EXP3_20 measured ~3.6 s/step), plus the one-time download. 24 h is headroom, not an
# estimate -- Slurm charges actual elapsed, not TimeLimit, and CLAUDE.md flags wall-clock
# kills as the main failure mode.
#
# Retention: save_interval=1000 with keep_period=1000, and max_to_keep=1 spares the newest
# save plus any step where step % keep_period == 0. So 1000/, 2000/ and the final 2999/
# all survive (the last save is at num_train_steps - 1, not 3000). ~8.9 GB each, ~27 GB
# total -- same as the EXP6_10 and EXP7_10 runs.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_EXP8_10_a100
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
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_EXP8_10 \
    --exp-name=EXP8_10_a100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
