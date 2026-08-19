#!/bin/bash
# Step 1 of pi05_Franka_GRASPNET_FINAL_moreData: get both source datasets on disk and
# merge them into a single LeRobot v2.1 dataset.
#
#   michaelyeah7/Franka_GraspNet_20260817   16,596 eps / 4,923,914 frames / 10 tasks
#   saifahmad123/GRASPNET_FINAL              5,500 eps / 1,685,732 frames /  7 tasks
#   -> saifahmad123/GRASPNET_FINAL_moreData 22,096 eps / 6,609,646 frames / 10 tasks
#
# The two share an identical schema (v2.1, robot_type=franka, 30 fps, state (9,),
# action (8,), the same three 480x640 video keys), and michaelyeah7's task list is a
# strict superset of GRASPNET_FINAL's -- it adds mug, orange, plum. That is what makes
# the merge safe; scripts/merge_graspnet_moredata.py asserts all of it before writing.
#
# Merge layout: michaelyeah7 goes FIRST so its 16,596 parquets need no rewrite at all
# (episode_index / index / task_index are already correct against the merged task
# table). Only GRASPNET_FINAL's 5,500 parquets are rewritten -- episode_index += 16596,
# index += 4923914, task_index remapped by task *name*. All 66,288 videos are hardlinked,
# so the merged copy costs ~260 MB, not another 19 GB.
#
# GRASPNET_FINAL is already cached under HF_LEROBOT_HOME; only michaelyeah7 is fetched.
#
# HF_HUB_DISABLE_XET=1 is the single most important line in this script. This repo is
# Xet-backed, and hf_xet requests a fresh .../xet-read-token/<hash> per download; that
# endpoint is what HuggingFace throttles. Measured on these exact 66,389 files:
#   xet on,  8 workers  -> ~430 files, then 429 "we had to rate limit your IP", repeat
#                          on a 320 s cycle = ~1.3 files/s = ~14 h
#   xet on, 32 workers  -> 429 within minutes, worse
#   xet off, 8 workers  -> 23 files/s sustained, no 429 at all = ~50 min
# Disabling xet falls back to the plain HTTP CDN path, which is not throttled the same
# way. Keep max_workers at 8; the win here is the transport, not the concurrency.
#
# Partition: data-transfer, not normal. Even at 50 min the 31-day limit (vs 2 days on
# normal/gengpu) means this job can never be wall-clock killed mid-merge, and the node
# has the fattest outbound pipe on the cluster.
#
# Fully resumable: snapshot_download skips what it already has and the merge skips any
# output file that already exists, so re-running this script after any interruption is
# safe and cheap.
#SBATCH --job-name=prep_GN_moreData
#SBATCH --account=p53063
#SBATCH --partition=data-transfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=31-00:00:00
#SBATCH --output=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.out
#SBATCH --error=/gpfs/home/wgk1727/openpi_franka/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/wgk1727/openpi_franka

export HF_LEROBOT_HOME=/projects/p53063/data/lerobot
export HF_HOME=/projects/p53063/hf_cache
# See the header: the xet transport's token endpoint is rate-limited, the plain HTTP
# path is not. This turns a ~14 h download into ~50 min.
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

echo "Running on: $(hostname)"
df -h /projects/p53063

SRC=/projects/p53063/data/lerobot/michaelyeah7/Franka_GraspNet_20260817

# Heartbeat: snapshot_download's progress bar is useless in a log file. Report real
# on-disk counts every 5 min instead. Killed when the job ends.
(
    while true; do
        sleep 300
        printf '[%s] parquet %s/16596  mp4 %s/49788  %s\n' \
            "$(date +%H:%M:%S)" \
            "$(find "${SRC}/data" -name '*.parquet' 2>/dev/null | wc -l)" \
            "$(find "${SRC}/videos" -name '*.mp4' 2>/dev/null | wc -l)" \
            "$(du -sh "${SRC}" 2>/dev/null | cut -f1)"
    done
) &
trap 'kill %1 2>/dev/null || true' EXIT

echo "=== downloading michaelyeah7/Franka_GraspNet_20260817 ==="
uv run python - <<'PY'
import time
from huggingface_hub import snapshot_download

root = "/projects/p53063/data/lerobot/michaelyeah7/Franka_GraspNet_20260817"
attempt = 0
while True:
    attempt += 1
    try:
        snapshot_download(
            "michaelyeah7/Franka_GraspNet_20260817",
            repo_type="dataset",
            local_dir=root,
            max_workers=8,
        )
        print(f"download complete after {attempt} attempt(s)", flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print(f"[attempt {attempt}] {str(e)[:200]} -- sleeping 320s", flush=True)
        time.sleep(320)
PY

echo "=== merging ==="
uv run python scripts/merge_graspnet_moredata.py

df -h /projects/p53063
echo "=== prep done ==="
