#!/bin/bash
# Step 2 of pi05_Franka_GRASPNET_FINAL_moreData: normalization statistics.
# Run only after slurm/prep_GRASPNET_FINAL_moreData.sbatch has printed "verify OK".
#
# Writes assets/pi05_Franka_GRASPNET_FINAL_moreData/saifahmad123/GRASPNET_FINAL_moreData/
# norm_stats.json. A miss there is SILENT: _load_norm_stats() returns None and Normalize
# passes data straight through, i.e. training would run unnormalized. Confirm the
# "Loaded norm stats from ..." line in the training .err before trusting the run.
#
# HF_HUB_OFFLINE=1 is REQUIRED, not an optimization. saifahmad123/GRASPNET_FINAL_moreData
# is a local-only merge with no Hub repo, and compute_norm_stats.py unconditionally calls
# snapshot_download(repo_id) inside a loop that sleeps 320 s on *any* exception -- a 404
# there would hang the job forever. With HF_HUB_OFFLINE=1 snapshot_download short-circuits
# to "returning existing local_dir" and the loop exits on the first pass. It must be
# exported from the shell: huggingface_hub freezes it into a module constant at import.
#
# CPU-only: compute_norm_stats.py calls disable_video_decoding() and injects zero
# placeholder images, so it reads only the state/action columns out of the parquet files.
# (LeRobotDataset still asserts every *video* file exists before that kicks in -- hence
# the "run only after prep finishes" above.)
#
# 7,971,809 frames / batch 32 = ~249k batches. Sized generously; this is the ~4.7x-larger
# sibling of pi05_Franka_GRASPNET_FINAL.
#SBATCH --job-name=normstats_GN_moreData
#SBATCH --account=p53063
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Sized to schedule, not to be safe-by-oversizing: the normal partition runs ~300 nodes
# allocated with ~320 jobs queued, and a 32-CPU / 240G / 48h ask sat at an estimated
# 32 h start because nothing that large backfills. 16 CPUs matches config.num_workers=14
# plus the main process, and 120G is what every previous normstats run on this repo used.
# Runtime: past runs of this same model shape (action_horizon=50) logged 11-58 batches/s;
# at 249,119 batches that is ~1.5-6.5 h, so 12 h is ~2x the pessimistic end. The pass is
# not resumable, but re-running it costs only those few hours.
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache
export HF_HUB_OFFLINE=1
# No GPU on this partition; stop JAX from trying to initialize CUDA.
export JAX_PLATFORMS=cpu
# Without this, tqdm output is block-buffered and the log lags minutes behind reality.
export PYTHONUNBUFFERED=1

echo "Running on: $(hostname)"
# --max-frames 1000000, i.e. stats from a random 1M of the 7,971,809 frames (~12.5%).
#
# Why: the merged dataset was built from 24,061 separate parquet files, so the arrow
# table carries ~24k record batches and every LeRobotDataset.__getitem__ pays a
# per-chunk lookup inside hf_dataset.select() for its 50-step action window. Measured
# on the full pass: 1.2-2.0 it/s over 249,119 batches = 34-57 h, against a 2-day
# partition cap. The older pi05_Franka_GraspNet_Test runs hit 11-58 it/s because that
# dataset had 2,130 files -- ~11x fewer chunks, ~11x faster. It is chunk count, not
# frame count, that dominates. (Video decoding is NOT the culprit: disable_video_decoding
# strips the video features and meta.video_keys is a live property off info["features"],
# so decoding really is skipped.)
#
# 1M frames -> 31,250 batches -> ~4-7 h, inside the 12 h limit. Statistically this costs
# nothing that matters: these are per-dimension mean/std/q01/q99 over 9-dim state and
# 8-dim action, and 1M random samples pins the mean to ~sigma/1000 and still leaves
# ~10,000 order statistics in each 1% tail. Sampling is shuffled across the whole
# dataset, so all 13 tasks are represented in proportion.
uv run scripts/compute_norm_stats.py --config-name pi05_Franka_GRASPNET_FINAL_moreData     --max-frames 1000000

echo "=== norm stats written ==="
cat assets/pi05_Franka_GRASPNET_FINAL_moreData/saifahmad123/GRASPNET_FINAL_moreData/norm_stats.json
