#!/bin/bash
# Step 3 of pi05_Franka_GRASPNET_FINAL_moreData: fine-tune pi0.5 on the merged
# collection -- 24,061 eps / 7,971,809 frames / 13 tasks (michaelyeah7/
# Franka_GraspNet_20260817 + saifahmad123/GRASPNET_FINAL + saifahmad123/Franka_3_objects_2).
#
# Run only after slurm/prep_GRASPNET_FINAL_moreData.sbatch AND
# slurm/norm_stats_GRASPNET_FINAL_moreData.sbatch have both finished.
#
# 100k steps x batch 32 = 3.2M samples, ~0.40 epochs. Cosine 2.5e-5 -> 2.5e-6 with
# warmup 3000 (~3% of training, matching pi05_Franka_GRASPNET_FINAL's 1500/50k) and
# decay_steps=100000 so the schedule actually bottoms out at the last step.
#
# Prompt augmentation is ON for this config (transforms.PromptVariants +
# openpi/policies/graspnet_prompts.py): up to 20 phrasings per object over the 13
# objects -- 11 paraphrases, 3 superordinate generics ("pick up the can", "pick up the
# fruit") and 6 misspellings, 239 unique strings -- sampled ~65% specific / ~14% generic
# / ~21% noisy, with the untouched canonical string the single heaviest variant (~18%).
# Generics are only attached where a scene holds at most one object of that category;
# "the cube" and "the cylinder" are deliberately excluded (two cubes share the red-cube
# scenes, and mugs are sometimes handleless cylinders). It is training-only; serve_policy.py passes no
# repack_transforms, so a prompt sent to the served policy is never rewritten. It also
# has zero effect on norm stats (verified: RemoveStrings drops the prompt before the
# stats pass, and stats read only state/action).
#
# 1 H100. Every single-H100 run of this exact model shape has landed at ~1.34 s/it
# (pi05_GraspNet_2 job 8934087: 30k steps in 11:10:14; GraspNet_2_100 job 9035162:
# 11:11:14), so 100k steps is ~37 h of compute, plus the up-front LeRobotDataset
# construction over 24,061 episodes -- call it ~38 h against gengpu's hard 2-day
# MaxTime. That leaves ~10 h of margin, which is real but not generous: if the job is
# killed at 48 h, the newest completed 10k checkpoint is what survives, and the run can
# be continued by swapping --overwrite for --resume below (they are mutually exclusive).
#
# HF_HUB_OFFLINE=1 is REQUIRED: saifahmad123/GRASPNET_FINAL_moreData is a local-only
# merge with no Hub repo, and train.py's pre-download loop (scripts/train.py:220-247)
# sleeps 320 s on *any* exception -- a 404 there hangs the job forever. Offline,
# snapshot_download short-circuits to "returning existing local_dir". Must be exported
# from the shell: huggingface_hub freezes it into a module constant at import time.
#
# Retention/disk: save_interval=10000 with keep_period=10000, so every save is kept --
# 10000/ ... 90000/ plus the final 99999/ (there is no 100000/; the last save is at
# num_train_steps - 1) = 10 dirs x ~8.9 GB = ~89 GB. Budget at submit time: 182 GB free
# on /projects/p53063, minus ~14 GB for the michaelyeah7 download, ~0.3 GB for the
# rewritten parquets and ~2.8 GB for Franka_3_objects_2's videos (which live on a
# different GPFS fileset and so must be copied, not hardlinked) = ~165 GB, leaving ~76 GB after
# the run. Re-check `df -h /projects/p53063` before submitting -- the filesystem is
# shared and already 83% used. If it gets tight mid-run, raising keep_period to 20000
# in the config prunes this to ~45 GB without changing save frequency.
#
# NOTE: --overwrite deletes any existing checkpoint dir for this exp-name.
# To continue an interrupted run, swap it for --resume (they are mutually exclusive).
#SBATCH --job-name=pi05_GN_moreData_h100
# e32706 holds 7x the RawShares of p53063 (500 vs 70) and under-consumes them, and
# gengpu priority is ~99.8% fairshare. Checkpoints still land in /projects/p53063.
#SBATCH --account=e32706
# gengpu is the only H100 partition open to us: b33210-gpu / b53080 / b32868 / a9009 all
# carry AllowAccounts=<their own buy-in account> and we only hold e32706 and p53063.
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
# gengpu bills TRESBillingWeights=CPU=4.5,Mem=1.0G,gres/gpu=4.5 under MAX_TRES, so the
# charge is max(16*4.5, mem_GB, 4.5) = max(72, mem_GB) -- flat until mem exceeds 72G,
# then mem-bound. Observed MaxRSS is ~53 GiB on the 1.69M-frame H100 GraspNet runs at
# the same num_workers=14; loader RAM is driven by workers x batch x action_horizon, not
# by dataset size, so 120G would probably hold. 160G anyway: the extra headroom covers
# the one-off arrow conversion of 24,061 parquet files, and an OOM 30 h into a 38 h run
# costs far more than the billing difference.
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache
export HF_HUB_OFFLINE=1

echo "Running on: $(hostname)"
nvidia-smi
df -h /projects/p53063

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_GRASPNET_FINAL_moreData \
    --exp-name=GRASPNET_FINAL_moreData_h100 \
    --checkpoint-base-dir=/projects/p53063/checkpoints \
    --overwrite
