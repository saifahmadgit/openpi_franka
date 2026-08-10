"""Upload a training checkpoint to the HuggingFace Hub.

Point it at a single step directory produced by `scripts/train.py`:

    checkpoints/<config_name>/<exp_name>/<step>/
        params/                 ~6 GB   model weights (needed to serve)
        assets/                 ~2 KB   norm stats  (needed to serve)
        train_state/            ~3 GB   optimizer state (only needed to resume)
        _CHECKPOINT_METADATA

By default the repo id is derived from the folder name and the step number, e.g.

    checkpoints/pi05_Franka_3_objects_2/3_objects_2_quest/10000
        ->  <your-hf-username>/3_objects_2_quest_10000

Examples:
    # dry run first — prints the resolved repo id and what would be sent
    uv run scripts/upload_checkpoint.py --checkpoint-dir checkpoints/pi05_Franka_3_objects_2/3_objects_2_quest/10000 --dry-run

    # real upload (private by default)
    uv run scripts/upload_checkpoint.py --checkpoint-dir checkpoints/pi05_Franka_3_objects_2/3_objects_2_quest/10000

    # public, and include the optimizer state so the run can be resumed from the Hub
    uv run scripts/upload_checkpoint.py --checkpoint-dir <path> --no-private --include-train-state

    # override the derived name entirely
    uv run scripts/upload_checkpoint.py --checkpoint-dir <path> --repo-id myorg/my-policy

    # one repo per experiment, one step_<N>/ subfolder per checkpoint
    uv run scripts/upload_checkpoint.py --checkpoint-dir <path> \
        --repo-id myorg/my-policy --step-subfolder

Requires `uv run huggingface-cli login` (or HF_TOKEN in the environment) beforehand.
Uploads are resumable: re-running after an interruption skips what already made it.
"""

import dataclasses
import pathlib
import shutil
import sys
import tempfile

from huggingface_hub import HfApi
import tyro


@dataclasses.dataclass
class Args:
    # Path to a single checkpoint step directory, e.g.
    # checkpoints/pi05_Franka_3_objects_2/3_objects_2_quest/10000
    checkpoint_dir: pathlib.Path

    # Full "owner/name" to upload to. If omitted, derived as <exp_name>_<step>
    # under your own HuggingFace account.
    repo_id: str | None = None

    # Upload into a subfolder of the repo instead of its root, so one repo can
    # hold several steps side by side. Pass --step-subfolder for "step_<step>/",
    # or give an explicit path here.
    path_in_repo: str | None = None

    # Shorthand for --path-in-repo=step_<step>.
    step_subfolder: bool = False

    # Create the repo as private. Pass --no-private to publish it.
    private: bool = True

    # Also upload train_state/ (~3 GB). Only needed to resume training from the
    # Hub; serving a policy does not use it.
    include_train_state: bool = False

    # Skip writing the generated README.md model card.
    skip_model_card: bool = False

    # Print what would happen and exit without uploading anything.
    dry_run: bool = False

    # Parallel upload workers. Leave as None to let huggingface_hub decide.
    num_workers: int | None = None


def _dir_size(path: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _find_dataset_repo_id(assets_dir: pathlib.Path) -> str | None:
    """Recover the training dataset's repo id from assets/<owner>/<name>/norm_stats.json."""
    for stats in assets_dir.rglob("norm_stats.json"):
        rel = stats.relative_to(assets_dir).parent
        if len(rel.parts) == 2:
            return "/".join(rel.parts)
    return None


def _existing_step_folders(api: HfApi, repo_id: str) -> list[str]:
    """Top-level `step_*` folders already in the repo, if it exists."""
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
    except Exception:
        return []
    return sorted({f.split("/")[0] for f in files if f.startswith("step_") and "/" in f})


def _model_card(
    repo_id: str,
    config_name: str,
    exp_name: str,
    step: str,
    dataset_repo_id: str | None,
    path_in_repo: str | None,
    step_folders: list[str],
) -> str:
    dataset_line = f"- **Training dataset:** [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id})\n" if dataset_repo_id else ""
    tags = "- robotics\n- openpi\n- pi0\n- franka\n"

    if path_in_repo:
        listing = "\n".join(f"- `{s}/`" for s in step_folders) or f"- `{path_in_repo}/`"
        step_section = f"""- **Checkpoints in this repo:**

{listing}

Each folder is one training step and is self-contained.
"""
        prefix = f"{path_in_repo}/"
        download = (
            f'ckpt = snapshot_download("{repo_id}", allow_patterns="{prefix}*") + "/{path_in_repo}"'
        )
    else:
        step_section = f"- **Step:** `{step}`\n"
        prefix = ""
        download = f'ckpt = snapshot_download("{repo_id}")'

    return f"""---
library_name: openpi
tags:
{tags}---

# {repo_id.split("/")[-1]}

openpi π₀.₅ policy checkpoint.

- **Train config:** `{config_name}`
- **Experiment:** `{exp_name}`
{step_section}{dataset_line}
## Contents

| Path | Purpose |
|---|---|
| `{prefix}params/` | Model weights. Required to serve the policy. |
| `{prefix}assets/` | Normalization statistics. Required to serve the policy. |
| `{prefix}train_state/` | Optimizer state. Only present if uploaded with `--include-train-state`; needed to resume training. |

## Usage

```python
from huggingface_hub import snapshot_download
from openpi.policies import policy_config
from openpi.training import config as _config

{download}
train_config = _config.get_config("{config_name}")
policy = policy_config.create_trained_policy(train_config, ckpt)

action_chunk = policy.infer(observation)["actions"]
```
"""


def main(args: Args) -> None:
    ckpt = args.checkpoint_dir.expanduser().resolve()

    if not ckpt.is_dir():
        sys.exit(f"error: not a directory: {ckpt}")
    if not (ckpt / "params").is_dir():
        sys.exit(
            f"error: {ckpt} has no params/ subdirectory, so it is not a checkpoint step directory.\n"
            f"       Point --checkpoint-dir at a step, e.g. .../<config>/<exp_name>/10000"
        )

    step = ckpt.name
    exp_name = ckpt.parent.name
    config_name = ckpt.parent.parent.name

    if not step.isdigit():
        print(f"warning: '{step}' is not a step number; using it verbatim in the repo name.")

    api = HfApi()

    if args.repo_id:
        repo_id = args.repo_id
    else:
        try:
            owner = api.whoami()["name"]
        except Exception as e:
            sys.exit(f"error: could not determine your HuggingFace account ({e}).\n       Run `uv run huggingface-cli login`, or pass --repo-id explicitly.")
        repo_id = f"{owner}/{exp_name}_{step}"

    path_in_repo = args.path_in_repo
    if args.step_subfolder:
        if path_in_repo:
            sys.exit("error: pass either --path-in-repo or --step-subfolder, not both.")
        path_in_repo = f"step_{step}"
    path_in_repo = path_in_repo.strip("/") if path_in_repo else None

    ignore_patterns = None if args.include_train_state else ["train_state/**"]

    upload_bytes = _dir_size(ckpt)
    if not args.include_train_state and (ckpt / "train_state").is_dir():
        upload_bytes -= _dir_size(ckpt / "train_state")

    dataset_repo_id = _find_dataset_repo_id(ckpt / "assets") if (ckpt / "assets").is_dir() else None

    print(f"checkpoint   : {ckpt}")
    print(f"config       : {config_name}")
    print(f"experiment   : {exp_name}")
    print(f"step         : {step}")
    print(f"dataset      : {dataset_repo_id or '(not found in assets/)'}")
    print(f"repo         : {repo_id}  ({'private' if args.private else 'PUBLIC'})")
    print(f"destination  : {path_in_repo + '/' if path_in_repo else '(repo root)'}")
    print(f"train_state  : {'included' if args.include_train_state else 'skipped (--include-train-state to add)'}")
    print(f"upload size  : {_fmt(upload_bytes)}")

    if path_in_repo:
        already = _existing_step_folders(api, repo_id)
        if already:
            print(f"already there : {', '.join(already)}")

    if args.dry_run:
        print("\n--dry-run set; nothing was uploaded.")
        return

    api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(f"\nuploading {_fmt(upload_bytes)} to {repo_id}/{path_in_repo or ''} ...")
    if path_in_repo:
        # upload_large_folder() has no path_in_repo, so subfolder uploads go
        # through upload_folder(). LFS dedupes on re-run, so it is still safe
        # to retry after an interruption.
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(ckpt),
            path_in_repo=path_in_repo,
            repo_type="model",
            ignore_patterns=ignore_patterns,
        )
    else:
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=str(ckpt),
            repo_type="model",
            ignore_patterns=ignore_patterns,
            num_workers=args.num_workers,
        )

    if not args.skip_model_card:
        # Written after the upload so the card can list every step now present.
        step_folders = _existing_step_folders(api, repo_id) if path_in_repo else []
        tmp = pathlib.Path(tempfile.mkdtemp()) / "README.md"
        tmp.write_text(
            _model_card(repo_id, config_name, exp_name, step, dataset_repo_id, path_in_repo, step_folders)
        )
        api.upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        shutil.rmtree(tmp.parent, ignore_errors=True)

    print(f"\ndone: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))
