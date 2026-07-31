#!/bin/bash
# Download a LeRobot dataset and compute its normalization statistics.
#
# Usage:  sbatch slurm/compute_norm_stats.sbatch [CONFIG_NAME]
#         (defaults to pi05_Franka_GraspNet_Test)
#
# CPU-only: the script disables video decoding, so norm stats are computed from the
# parquet state/action columns alone. No GPU is requested.
#
# The dataset is downloaded to project storage; the resulting norm_stats.json is
# written into the repo's assets/ dir (it is small and belongs in git).
#SBATCH --job-name=normstats
#SBATCH --account=p53063
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

CONFIG_NAME="${1:-pi05_Franka_GraspNet_Test}"

# Keep the dataset off the home filesystem — project storage has 1 TB.
export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache
# No GPU on this partition; stop JAX from trying to initialize CUDA.
export JAX_PLATFORMS=cpu
# Without this, python/tqdm output is block-buffered into the log file and the
# progress you see can lag many minutes behind what is actually happening.
export PYTHONUNBUFFERED=1

echo "Running on: $(hostname)"
echo "Config:     ${CONFIG_NAME}"
echo "Dataset ->  ${HF_LEROBOT_HOME}"

# Heartbeat: the HF download's own progress bar is unreliable in a log file, so
# report real on-disk progress to stdout every 2 minutes. Killed when the job ends.
DATA_DIR="${HF_LEROBOT_HOME}"
(
    while true; do
        sleep 120
        printf '[%s] downloaded: %s in %s files\n' \
            "$(date +%H:%M:%S)" \
            "$(du -sh "${DATA_DIR}" 2>/dev/null | cut -f1)" \
            "$(find "${DATA_DIR}" -type f 2>/dev/null | wc -l)"
    done
) &
HEARTBEAT_PID=$!
trap 'kill ${HEARTBEAT_PID} 2>/dev/null || true' EXIT

uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
