"""Compute normalization statistics for a config, from a dataset already on this disk.

Identical to scripts/compute_norm_stats.py except for where the dataset comes from:
that one downloads repo_id from the Hub into ~/.cache/huggingface/lerobot, this one
reads the LeRobot directory the converter already wrote (e.g. ~/MagicSim/data/<name>).
Same stats, same output location, no download and no second copy on disk.

Usage:
    uv run scripts/compute_norm_stats_local.py --config-name <config> --data-dir ~/MagicSim/data/<name>
"""

import pathlib

from lerobot.common.datasets import lerobot_dataset
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class InjectDummyImages(transforms.DataTransformFn):
    """Adds tiny zero placeholders for the given (missing) image keys.

    Used together with `disable_video_decoding`: norm stats only need `state`/`actions`,
    so we skip the expensive per-frame video decode but still need the downstream
    repack/policy-input transforms to find the image keys they expect. Image *content*
    does not affect the state/action statistics, so zeros are fine.
    """

    def __init__(self, image_keys: list[str]):
        self._image_keys = image_keys

    def __call__(self, x: dict) -> dict:
        for key in self._image_keys:
            x.setdefault(key, np.zeros((3, 2, 2), dtype=np.uint8))
        return x


def disable_video_decoding(dataset: _data_loader.Dataset) -> list[str]:
    """Stops a LeRobotDataset from decoding videos in __getitem__, returning the dropped video keys.

    Computing norm stats only touches `state`/`actions` (read from the parquet files), so decoding
    the camera videos for every frame is pure wasted work that can turn a minutes-long pass into hours.
    Video decoding is gated on `len(meta.video_keys) > 0`, which derives from `meta.info["features"]`;
    dropping the video features there disables decoding without affecting the parquet reads.
    """
    base = dataset
    while not hasattr(base, "meta") and hasattr(base, "_dataset"):
        base = base._dataset
    if not hasattr(base, "meta"):
        return []
    video_keys = list(base.meta.video_keys)
    base.meta.info["features"] = {
        k: v for k, v in base.meta.info["features"].items() if v["dtype"] != "video"
    }
    return video_keys


def create_local_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    root: pathlib.Path,
) -> _data_loader.Dataset:
    """_data_loader.create_torch_dataset() with an explicit local root.

    This is the whole difference from compute_norm_stats.py. LeRobotDataset only skips
    the Hub when given `root`, and create_torch_dataset() does not take one.
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        video_backend="pyav",
    )

    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def main(config_name: str, data_dir: pathlib.Path, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    root = data_dir.expanduser().resolve()
    print(f"Reading local dataset: {root}  (repo_id: {data_config.repo_id})")

    dataset = create_local_torch_dataset(data_config, config.model.action_horizon, config.model, root)
    video_keys = disable_video_decoding(dataset)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            InjectDummyImages(video_keys),
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // config.batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // config.batch_size
        shuffle = False

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
