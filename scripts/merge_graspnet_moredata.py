"""Merge the GraspNet collections + Franka_3_objects_2 into one LeRobot v2.1 dataset:
saifahmad123/GRASPNET_FINAL_moreData.

  michaelyeah7/Franka_GraspNet_20260817  16,596 eps / 4,923,914 frames / 10 tasks
  saifahmad123/GRASPNET_FINAL             5,500 eps / 1,685,732 frames /  7 tasks (subset of the 10)
  saifahmad123/Franka_3_objects_2         1,965 eps / 1,362,163 frames /  3 tasks (all new)
  -> 24,061 eps / 7,971,809 frames / 13 tasks

All three share an identical schema (v2.1, robot_type=franka, 30 fps, state (9,),
action (8,), the same three 480x640 video keys) and the same simulator, robot and table.
The guards below assert every bit of that before a single file is written.

Sources are concatenated in order. The FIRST source is deliberately the largest one
whose task indices already match the merged table, because that makes its episode_index,
index and task_index columns correct as-is -- its 16,596 parquets are hardlinked
untouched. Every later source gets its parquets rewritten: episode_index and index
shifted by the running offsets, task_index remapped by task *name*.

Videos are hardlinked where the filesystem allows and copied otherwise. michaelyeah7 and
GRASPNET_FINAL live under /projects and hardlink for free; Franka_3_objects_2 sits in
~/.cache, which is a separate GPFS fileset, so its 5,895 videos are a real 2.8 GB copy
(os.link raises EXDEV there and link() falls back to shutil.copy2).

Idempotent: every output file is skipped if it already exists, so re-running after an
interruption is safe and cheap.
"""

import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path("/projects/p53063/data/lerobot")
DEST = ROOT / "saifahmad123/GRASPNET_FINAL_moreData"
DEST_REPO_ID = "saifahmad123/GRASPNET_FINAL_moreData"
CHUNK = 1000
VIDEO_KEYS = ["observation.images.front_1", "observation.images.front_2", "observation.images.wrist"]

# Order matters: the first entry is the one whose parquets are reused untouched.
SOURCES = [
    ROOT / "michaelyeah7/Franka_GraspNet_20260817",
    ROOT / "saifahmad123/GRASPNET_FINAL",
    Path("/home/wgk1727/.cache/huggingface/lerobot/saifahmad123/Franka_3_objects_2"),
]


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def link(src: Path, dst: Path):
    """Hardlink if possible, copy if the source is on another filesystem."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    infos = [json.load(open(s / "meta/info.json")) for s in SOURCES]
    tasks = [jl(s / "meta/tasks.jsonl") for s in SOURCES]
    epss = [jl(s / "meta/episodes.jsonl") for s in SOURCES]
    stats = [jl(s / "meta/episodes_stats.jsonl") for s in SOURCES]

    # --- schema guards: a silent mismatch here would poison training ---
    ref = infos[0]
    for s, info in zip(SOURCES, infos, strict=True):
        assert info["fps"] == ref["fps"] == 30, s
        assert info["codebase_version"] == "v2.1", s
        assert info["robot_type"] == ref["robot_type"], s
        assert info["chunks_size"] == CHUNK, s
        assert set(info["features"]) == set(ref["features"]), s
        for k, v in ref["features"].items():
            assert info["features"][k]["dtype"] == v["dtype"], (s, k)
            assert info["features"][k].get("shape") == v.get("shape"), (s, k)
    for s, info, eps in zip(SOURCES, infos, epss, strict=True):
        assert len(eps) == info["total_episodes"], s
        assert sum(e["length"] for e in eps) == info["total_frames"], s
        assert all(e["episode_index"] == i for i, e in enumerate(eps)), s

    # --- merged task table: source 0's indices are preserved, later tasks appended ---
    assert [t["task_index"] for t in tasks[0]] == list(range(len(tasks[0])))
    name_to_idx = {t["task"]: t["task_index"] for t in tasks[0]}
    for tl in tasks[1:]:
        for t in tl:
            if t["task"] not in name_to_idx:
                name_to_idx[t["task"]] = len(name_to_idx)
    remaps = [{t["task_index"]: name_to_idx[t["task"]] for t in tl} for tl in tasks]
    assert remaps[0] == {i: i for i in range(len(tasks[0]))}, "source 0 must be identity"
    for s, r in zip(SOURCES, remaps, strict=True):
        print(f"{s.name}: task remap {r}")

    total_eps = sum(len(e) for e in epss)
    total_frames = sum(i["total_frames"] for i in infos)
    print(f"merged: {total_eps} eps / {total_frames} frames / {len(name_to_idx)} tasks")

    (DEST / "meta").mkdir(parents=True, exist_ok=True)

    # ---------------- meta ----------------
    with open(DEST / "meta/tasks.jsonl", "w") as f:
        for task, idx in sorted(name_to_idx.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    ep_off = 0
    with open(DEST / "meta/episodes.jsonl", "w") as fe, open(DEST / "meta/episodes_stats.jsonl", "w") as fs:
        for eps, st in zip(epss, stats, strict=True):
            assert all(e["episode_index"] == i for i, e in enumerate(st))
            for e in eps:
                fe.write(json.dumps({**e, "episode_index": e["episode_index"] + ep_off}) + "\n")
            for e in st:
                fs.write(json.dumps({**e, "episode_index": e["episode_index"] + ep_off}) + "\n")
            ep_off += len(eps)

    info = dict(ref)
    info["total_episodes"] = total_eps
    info["total_frames"] = total_frames
    info["total_tasks"] = len(name_to_idx)
    info["total_chunks"] = (total_eps + CHUNK - 1) // CHUNK
    info["total_videos"] = total_eps * len(VIDEO_KEYS)
    info["splits"] = {"train": f"0:{total_eps}"}
    info["repo_id"] = DEST_REPO_ID
    json.dump(info, open(DEST / "meta/info.json", "w"), indent=4)
    print("meta written")

    # ---------------- data + videos ----------------
    ep_off = frame_off = 0
    for src, eps, remap in zip(SOURCES, epss, remaps, strict=True):
        identity = ep_off == 0 and frame_off == 0 and all(k == v for k, v in remap.items())
        n = len(eps)
        for old in tqdm(range(n), desc=f"{src.name[:24]} parquet{'' if identity else ' (rewrite)'}"):
            new = old + ep_off
            oc, nc = old // CHUNK, new // CHUNK
            s_path = src / f"data/chunk-{oc:03d}/episode_{old:06d}.parquet"
            d_path = DEST / f"data/chunk-{nc:03d}/episode_{new:06d}.parquet"
            if identity:
                link(s_path, d_path)
                continue
            if d_path.exists():
                continue
            t = pq.read_table(s_path)
            rows = t.num_rows
            t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index",
                             pa.array([new] * rows, type=pa.int64()))
            t = t.set_column(t.schema.get_field_index("index"), "index",
                             pa.array([i + frame_off for i in t.column("index").to_pylist()], type=pa.int64()))
            t = t.set_column(t.schema.get_field_index("task_index"), "task_index",
                             pa.array([remap[i] for i in t.column("task_index").to_pylist()], type=pa.int64()))
            d_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, d_path)
        for vk in VIDEO_KEYS:
            for old in tqdm(range(n), desc=f"{src.name[:24]} {vk.split('.')[-1]}"):
                new = old + ep_off
                link(src / f"videos/chunk-{old // CHUNK:03d}/{vk}/episode_{old:06d}.mp4",
                     DEST / f"videos/chunk-{new // CHUNK:03d}/{vk}/episode_{new:06d}.mp4")
        ep_off += n
        frame_off += sum(e["length"] for e in eps)

    verify(total_eps, total_frames, len(name_to_idx))
    print("merge complete ->", DEST)


def verify(total_eps: int, total_frames: int, n_tasks: int):
    """Fail loudly on anything the training pipeline would otherwise hit at step 0.

    LeRobotDataset asserts that *every* parquet AND video path listed by
    get_episodes_file_paths() exists locally; a single missing file sends it down the
    download branch, which needs the Hub (and this repo_id has no Hub repo).
    """
    info = json.load(open(DEST / "meta/info.json"))
    assert info["total_episodes"] == total_eps
    assert info["total_frames"] == total_frames
    assert info["total_tasks"] == n_tasks
    assert info["repo_id"] == DEST_REPO_ID

    eps = jl(DEST / "meta/episodes.jsonl")
    assert len(eps) == total_eps
    assert all(e["episode_index"] == i for i, e in enumerate(eps)), "episode_index not contiguous"
    assert sum(e["length"] for e in eps) == total_frames
    st = jl(DEST / "meta/episodes_stats.jsonl")
    assert len(st) == total_eps
    assert all(e["episode_index"] == i for i, e in enumerate(st))

    task_rows = jl(DEST / "meta/tasks.jsonl")
    assert [t["task_index"] for t in task_rows] == list(range(n_tasks))
    name_to_idx = {t["task"]: t["task_index"] for t in task_rows}
    assert {t for e in eps for t in e["tasks"]} <= set(name_to_idx), "episode references an unknown task"

    missing = []
    for ep in tqdm(range(total_eps), desc="verify files"):
        c = ep // CHUNK
        if not (DEST / f"data/chunk-{c:03d}/episode_{ep:06d}.parquet").is_file():
            missing.append(f"data/{ep}")
        for vk in VIDEO_KEYS:
            if not (DEST / f"videos/chunk-{c:03d}/{vk}/episode_{ep:06d}.mp4").is_file():
                missing.append(f"{vk}/{ep}")
    assert not missing, f"{len(missing)} missing files, first 10: {missing[:10]}"

    # Spot-check each source's first/last episode: task_index must agree with episodes.jsonl.
    bounds = [0]
    off = 0
    for s in SOURCES:
        off += json.load(open(s / "meta/info.json"))["total_episodes"]
        bounds += [off - 1, off]
    for ep in sorted({b for b in bounds if 0 <= b < total_eps}):
        c = ep // CHUNK
        t = pq.read_table(DEST / f"data/chunk-{c:03d}/episode_{ep:06d}.parquet")
        assert set(t.column("episode_index").to_pylist()) == {ep}, ep
        want = name_to_idx[eps[ep]["tasks"][0]]
        assert set(t.column("task_index").to_pylist()) == {want}, (ep, want)
        assert t.num_rows == eps[ep]["length"], ep
        print(f"  ep {ep:6d}  task_index {want:2d}  {eps[ep]['tasks'][0]!r}  rows {t.num_rows}")
    print("verify OK")


if __name__ == "__main__":
    main()
