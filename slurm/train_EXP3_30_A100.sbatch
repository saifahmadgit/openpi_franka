#!/bin/bash
# Fine-tune pi0.5 on saifahmad123/EXP3_30 (single-task "pick up the orange cylinder",
# prompt_from_task=True so it stays language-conditioned). 390 episodes / 118,104 frames
# -- the rung between pi05_Franka_EXP3_20 (260 eps) and pi05_Franka_EXP3_50 (650 eps).
#
# Hyperparameters are copied verbatim from pi05_Franka_EXP3_50 so dataset size is the only
# variable: 10k steps, warmup 1000, peak_lr 2.5e-5 -> 2.5e-6 cosine, batch 32,
# action_horizon=50. That is ~2.7 epochs here vs ~1.6 on EXP3_50.
#
# Unlike pi05_Franka_EXP3_20, EXP3_30 is a REAL uploaded dataset, not an episodes= subset
# of a bigger repo, so it has its own norm stats at
# assets/pi05_Franka_EXP3_30/saifahmad123/EXP3_30/norm_stats.json
# (job 9919860, 2026-08-19). A miss there is silent: _load_norm_stats() returns None and
# Normalize passes data through untouched, i.e. the run would train unnormalized. Confirm
# the "Loaded norm stats from ..." line in the .err log before trusting the run.
#
# HF_HUB_OFFLINE=1 is safe ONLY because this job is submitted with
# --dependency=afterok:<normstats job>, and that job snapshot_downloads EXP3_30 into
# HF_LEROBOT_HOME before computing stats. If you resubmit this script standalone against a
# cold cache, DROP the export or train.py's pre-download loop has nothing to fetch with.
# It must be exported from the shell, not set from Python: huggingface_hub freezes it into
# a module constant at import time.
#
# Walltime: measured A100 throughput on this exact job shape is ~3.6 s/step
# (pi05_EXP3_20_a100 job 9843660: 6k steps in ~6 h), so 10k steps is ~10 h. 24 h is ~2.4x
# headroom, and it needs to be generous -- see the retention note below. Slurm charges
# actual elapsed, not TimeLimit, so over-requesting costs no fairshare. gengpu caps at 48 h.
#
# Retention: EXACTLY ONE checkpoint, 9999/, ~8.9 GB. The config sets save_interval=10_000
# with num_train_steps=10_000, and train.py:301 only saves when
# (step % save_interval == 0 and step > start_step) or step == num_train_steps - 1 -- step 0
# is excluded and step 10000 is never reached, so nothing is written until the final step.
# CONSEQUENCE: there is no mid-run checkpoint to --resume from. A wall-clock kill, a node
# failure or an OOM at hour 9 loses the entire run. This is deliberate (requested), but it
# is why the time limit is 24 h rather than 12.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive) --
# though with a single final-step save there will normally be nothing to resume from.
#SBATCH --job-name=pi05_EXP3_30_a100
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
export HF_HUB_OFFLINE=1

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_EXP3_30 \
    --exp-name=EXP3_30_a100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
