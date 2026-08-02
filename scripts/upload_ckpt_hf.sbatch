#!/bin/bash
# Upload one openpi checkpoint step to the Hugging Face Hub.
#
#   sbatch scripts/upload_ckpt_hf.sbatch <STEP> [EXP_DIR] [REPO]
#
# e.g.  sbatch scripts/upload_ckpt_hf.sbatch 6000
#
# The step lands in the repo under "step_<STEP>/", so multiple checkpoints can
# live in one repo without overwriting each other.
#SBATCH --account=p53063
#SBATCH --partition=data-transfer
#SBATCH --job-name=hf_upload
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/hf_upload_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/hf_upload_%j.out

set -euo pipefail

STEP=${1:?usage: sbatch upload_ckpt_hf.sbatch <STEP> [EXP_DIR] [REPO]}
EXP_DIR=${2:-/projects/p53063/checkpoints/pi05_Franka_GraspNet_Test/graspnet_test}
REPO=${3:-saifahmad123/pi05-franka-graspnet-test}

CKPT="$EXP_DIR/$STEP"
HF=/gpfs/home/wgk1727/openpi_franka/.venv/bin/huggingface-cli

[ -d "$CKPT/params" ] || { echo "ERROR: no params/ under $CKPT"; exit 1; }

# HF_HOME points at project space elsewhere in this repo, which has no token in
# it. Read the token from the home-dir cache and pass it explicitly instead.
export HF_TOKEN=$(cat /gpfs/home/wgk1727/.cache/huggingface/token)
unset HF_HOME || true

echo "Uploading $CKPT -> $REPO:step_$STEP"
du -sh "$CKPT"

"$HF" upload "$REPO" "$CKPT" "step_$STEP" \
    --repo-type=model \
    --commit-message="Add checkpoint step $STEP"

echo "DONE rc=$?"
