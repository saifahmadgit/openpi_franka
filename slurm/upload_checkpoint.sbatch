#!/bin/bash
# Push one finished checkpoint step dir to the HuggingFace Hub.
#
# Usage:  sbatch [--dependency=afterok:<train job>] slurm/upload_checkpoint.sbatch \
#             <CONFIG_NAME> <EXP_NAME> <STEP> <REPO_ID>
#
# Layout is one HF repo per training RUN with the checkpoint in a step_<N>/ subfolder, so
# more steps of the same run can be added later without creating a second repo:
#
#     saifahmad123/<RunName>/
#     |-- step_9999/
#     |   |-- params/  assets/
#     |   `-- _CHECKPOINT_METADATA
#     `-- step_5000/      <- can be added later
#
# Without --step-subfolder the script instead defaults to a repo PER CHECKPOINT named
# <exp_name>_<step> with files at the repo root -- not what we want.
#
# train_state/ is excluded (the script's default): 5.9 GB instead of 8.9 GB, and serving
# only needs params/ + assets/ (see policy_config.create_trained_policy). Pass
# --include-train-state if a run must stay resumable from the Hub copy.
#
# HF_HOME IS DELIBERATELY NOT SET HERE. The other sbatch scripts export
# HF_HOME=/projects/p53063/hf_cache to keep dataset downloads off the home filesystem, but
# that cache holds NO auth token and the upload dies with "Not logged in". The token lives
# in the default HF_HOME at ~/.cache/huggingface/token (user saifahmad123). Do not "fix"
# this by adding the export.
#
# --no-private publishes the repo. Drop it for a private repo (the script's default).
#
# Sized small: this is a network-bound 5.9 GB PUT, not compute. ~10-20 min in practice.
#SBATCH --job-name=upload_ckpt
#SBATCH --account=p53063
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

CONFIG_NAME="$1"
EXP_NAME="$2"
STEP="$3"
REPO_ID="$4"

CKPT_DIR="/projects/p53063/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${STEP}"

echo "Running on: $(hostname)"
echo "checkpoint: ${CKPT_DIR}"
echo "repo:       ${REPO_ID} (public)"

# Fail loudly rather than uploading a half-written or missing checkpoint. The training job
# writes atomically (orbax renames a .orbax-checkpoint-tmp-* dir into place), so the
# presence of params/ means the save finalized.
if [ ! -d "${CKPT_DIR}/params" ]; then
    echo "ERROR: ${CKPT_DIR}/params does not exist -- did the run reach step ${STEP}?" >&2
    ls -la "$(dirname "${CKPT_DIR}")" >&2 || true
    exit 1
fi

uv run scripts/upload_checkpoint.py \
    --checkpoint-dir "${CKPT_DIR}" \
    --repo-id "${REPO_ID}" \
    --step-subfolder \
    --no-private

echo "=== upload done: https://huggingface.co/${REPO_ID} ==="
