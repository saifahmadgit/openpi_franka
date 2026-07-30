#!/bin/bash
#SBATCH --job-name=pi05_3obj2
#SBATCH --account=p53063
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

echo "Running on: $(hostname)"
nvidia-smi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_Franka_3_objects_2 --exp-name=3_objects_2_quest --overwrite
