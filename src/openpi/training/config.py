"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import numpy as np
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.graspnet_prompts as graspnet_prompts
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.lehome_config as lehome_config
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If set, only these episode indices are loaded from the LeRobot dataset (passed straight
    # through to LeRobotDataset(episodes=...)). Lets a config train on a subset of a larger
    # dataset without materializing a second copy of it on disk or on the Hub.
    episodes: Sequence[int] | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # Optional override for the delta action mask. If None, defaults to make_bool_mask(6, -1, 6, -1) for bimanual Aloha (14-dim).
    # Override this for robots with different action dimensions, e.g. make_bool_mask(7, -2) for a 9-dim Franka.
    delta_action_mask: tyro.conf.Suppress[tuple[bool, ...] | None] = None
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = self.delta_action_mask if self.delta_action_mask is not None else _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# NNX filter matching the SigLIP image encoder (`PaliGemma/img/...`, 23 param arrays).
# Unlike the LLM towers, SigLIP is instantiated without LoRA adapters, so combining this
# with a config's `get_freeze_filter()` leaves the vision encoder completely static and
# trains only the LoRA weights in the Gemma / action-expert towers.
_FreezeSigLip = nnx_utils.PathRegex("^PaliGemma/img/.*")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_magicsim_apple_red",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=False,
            adapt_to_pi=False,
            repo_id="michaelyeah7/panda_grasp_red_apple_409_v21_tag",
            default_prompt="grasp the red apple",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front",
                                "cam_left_wrist": "observation.images.wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=100_000,
        batch_size=2,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=5000,
    ),
    TrainConfig(
        name="pi05_magicsim_apple_red_joint",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_Apple_joint_Cam_Calibrated",
            default_prompt="grasp the red apple",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front_1",
                                "cam_left_wrist": "observation.images.front_2",
                                "cam_right_wrist": "observation.images.wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=100_000,
        batch_size=2,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=25000,
    ),
    TrainConfig(
        name="pi05_magicsim_base",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            default_prompt="pick up the apple",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front_1",
                                "cam_left_wrist": "observation.images.front_2",
                                "cam_right_wrist": "observation.images.wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
    ),
    TrainConfig(
        name="pi05_magicsim_apple",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=False,
            adapt_to_pi=False,
            repo_id="saifahmad123/panda_grasp_saif",
            default_prompt="grasp the red apple",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front",
                                "cam_left_wrist": "observation.images.wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=50000,
        batch_size=2,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=5000,
    ),
    TrainConfig(
        name="pi05_Franka_Real_Random_Data",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions; Franka 8 actions are padded to 32
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Real_Random",
            default_prompt="pick up the orange cylinder",  # or use base_config=DataConfig(prompt_from_task=True)
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front_1",
                                "cam_left_wrist": "observation.images.front_2",
                                "cam_right_wrist": "observation.images.wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=100_000,
        batch_size=2,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=25_000,
    ),
    TrainConfig(
        name="pi05_Franka_Real_Random_New_Data",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions; Franka 8 actions are padded to 32
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Real_Random_New_2p1",
            default_prompt="pick up the orange cylinder",  # or use base_config=DataConfig(prompt_from_task=True)
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front_1",
                                "cam_left_wrist": "observation.images.front_2",
                                "cam_right_wrist": "observation.images.wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=100_000,
        batch_size=8,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=25_000,
    ),
    TrainConfig(
        # Fine-tune pi05-DROID on the same 130-episode Franka dataset.
        # Parallel experiment to pi05_Franka_Real_Random_New_Data (which starts from pi05_base).
        # Key differences vs the base config:
        #   - Starts from pi05_droid checkpoint (pre-trained on full DROID dataset, joint velocity actions)
        #   - Reuses DROID norm stats so the normalization matches the pre-trained action head
        #   - Action format stays delta joint position (same as the base experiment) for a fair comparison;
        #     there is a mild action-space mismatch (velocity vs delta position) but LoRA adapts quickly
        #   - 3rd camera slot (right_wrist_0_rgb) was masked to zeros during DROID pre-training; here it
        #     receives a real wrist image — the model will learn to use it via fine-tuning
        name="pi05_Franka_Real_Random_New_Data_droid",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Real_Random_New_2p1",
            default_prompt="pick up the orange cylinder",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front_1",
                                "cam_left_wrist": "observation.images.front_2",
                                "cam_right_wrist": "observation.images.wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=100_000,
        batch_size=8,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=25_000,
    ),
    TrainConfig(
        name="pi05_Franka_Real_Random_New_Data_2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Real_Random_New_2",
            default_prompt="pick up the orange cylinder",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front_1",
                                "cam_left_wrist": "observation.images.front_2",
                                "cam_right_wrist": "observation.images.wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=60_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=5_000,
    ),
    TrainConfig(
        name="pi05_Franka_Teleop",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Teleop",
            default_prompt="pick up the red object",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps= 20_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    TrainConfig(
        name="pi05_Franka_cube_with_DR",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_cube_with_DR",
            default_prompt="pick up the orange cylinder",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=20_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    TrainConfig(
        name="pi05_Franka_cube_with_DR_slow",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_cube_with_DR_slow",
            default_prompt="pick up the orange cylinder",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    TrainConfig(
        name="pi05_Franka_cube_with_DR_slow_purple",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_cube_with_DR_slow_purple",
            default_prompt="pick up the purple object",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    TrainConfig(
        name="pi05_Franka_3_objects",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_3_objects",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    TrainConfig(
        name="pi05_Franka_3_objects_2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_3_objects_2",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    TrainConfig(
        name="pi05_Franka_EXP0",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/EXP0",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Short 2k-step run: warmup scaled down from 1k so it is only 10% of training.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2.5e-5,
            decay_steps=2_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=2_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=200,
        # This run exists to find the minimum step count that works, so every saved
        # checkpoint has to survive. max_to_keep=1 only spares steps where
        # step % keep_period == 0; the default 5000 never matches below 2k, so all but
        # the last would be pruned. 200 matches every save: 10 dirs, ~89 GB.
        keep_period=200,
    ),
    TrainConfig(
        name="pi05_Franka_EXP0_20",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/EXP0_20",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Short 2k-step run: warmup scaled down from 1k so it is only 10% of training.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2.5e-5,
            decay_steps=2_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=2_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=200,
        # Same as pi05_Franka_EXP0: every saved checkpoint must survive pruning, so
        # keep_period matches save_interval. 10 dirs, ~89 GB.
        keep_period=1000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP1_10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/EXP1_10",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Short 2k-step run: warmup scaled down from 1k so it is only 10% of training.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2.5e-5,
            decay_steps=2_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=2_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Keep only the 1k and the final checkpoint, same as pi05_Franka_EXP0_20.
        # max_to_keep=1 spares the newest save plus any step where
        # step % keep_period == 0, so 1000 survives and every other 200-step save
        # is pruned. The last save is at step num_train_steps - 1 = 1999, so the
        # two dirs on disk are 1000/ and 1999/ (not 2000/).
        save_interval=200,
        keep_period=1000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP1_5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 65 episodes / 19,841 frames -- the 5% rung of the EXP1 data-scaling
            # ladder, a real uploaded dataset (not an episodes= subset of EXP1_10), so it
            # needs its own norm stats.
            repo_id="saifahmad123/EXP1_5",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Identical schedule to pi05_Franka_EXP1_10 so the only variable across the rung
        # is dataset size: same 2k steps, same LR, same batch. That means 3.2 epochs
        # here versus 1.6 on EXP1_10 -- equal compute, not equal epochs.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2.5e-5,
            decay_steps=2_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=2_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Exactly two checkpoints, ~18 GB total. Saves land on multiples of
        # save_interval plus the final step (num_train_steps - 1 = 1999), so only 1000/
        # and 1999/ are ever written: keep_period=1000 spares 1000/ from pruning and
        # max_to_keep=1 (checkpoints.py) spares the newest. Unlike EXP1_10 this writes
        # two dirs rather than writing ten and pruning eight.
        save_interval=1_000,
        keep_period=1_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP1_1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 13 episodes / 3,920 frames -- the 1% rung of the EXP1 data-scaling
            # ladder, a real uploaded dataset (not an episodes= subset of EXP1_10), so it
            # needs its own norm stats.
            repo_id="saifahmad123/EXP1_1",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Identical schedule to pi05_Franka_EXP1_10 so the only variable across the rung
        # is dataset size: same 2k steps, same LR, same batch. That means 16.3 epochs
        # here versus 1.6 on EXP1_10 -- equal compute, not equal epochs.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2.5e-5,
            decay_steps=2_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=2_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Exactly two checkpoints, ~18 GB total. Saves land on multiples of
        # save_interval plus the final step (num_train_steps - 1 = 1999), so only 1000/
        # and 1999/ are ever written: keep_period=1000 spares 1000/ from pruning and
        # max_to_keep=1 (checkpoints.py) spares the newest. Unlike EXP1_10 this writes
        # two dirs rather than writing ten and pruning eight.
        save_interval=1_000,
        keep_period=1_000,
    ),
    # ------------------------------------------------------------------
    # Long-schedule (20k-step) arms of the EXP1 data-scaling ladder.
    #
    # pi05_Franka_EXP1_{1,5,10} above are the 2k-step "equal compute" arms. These three
    # rerun the same three datasets for 10x longer, to separate "the small dataset is
    # not enough data" from "2k steps is not enough training". Schedule, batch size and
    # horizon are identical across all three, so dataset size is the only variable --
    # which also means epochs differ wildly: 163 / 32 / 16 for the 1%, 5% and 10% rungs.
    # The 1% rung at 163 epochs is expected to overfit; that is the measurement.
    #
    # Norm stats are shared with the 2k configs: the datasets are byte-identical, so
    # assets/<this config>/saifahmad123/<repo>/norm_stats.json is a verified copy of the
    # sibling's file rather than a fresh compute_norm_stats.py pass.
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi05_Franka_EXP1_1_20k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 13 episodes / 3,920 frames -> 163.3 epochs at 20k steps, batch 32.
            repo_id="saifahmad123/EXP1_1",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Warmup is 10% of num_train_steps, following the rationale in the 2k configs
        # above rather than the flat warmup_steps=1000 used elsewhere in this file.
        # Identical across all three rungs so dataset size stays the only variable.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=20_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Exactly two checkpoints survive, ~18 GB. Saves fire at step % 5000 == 0
        # (step > start_step) plus the final step num_train_steps - 1 = 19999; step
        # 20000 is never reached. keep_period=10_000 spares only multiples of 10k, so
        # 5000/ is pruned when 10000/ lands and 15000/ is pruned when 19999/ lands,
        # leaving 10000/ + 19999/. max_to_keep=1 (checkpoints.py:48) spares the newest.
        # The 5k interval exists purely for --resume granularity on a ~20 h run; it
        # costs one extra dir (~27 GB) transiently, never a third at the end.
        save_interval=5_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP1_5_20k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 65 episodes / 19,841 frames -> 32.3 epochs at 20k steps, batch 32.
            repo_id="saifahmad123/EXP1_5",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Warmup is 10% of num_train_steps, following the rationale in the 2k configs
        # above rather than the flat warmup_steps=1000 used elsewhere in this file.
        # Identical across all three rungs so dataset size stays the only variable.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=20_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Exactly two checkpoints survive, ~18 GB. Saves fire at step % 5000 == 0
        # (step > start_step) plus the final step num_train_steps - 1 = 19999; step
        # 20000 is never reached. keep_period=10_000 spares only multiples of 10k, so
        # 5000/ is pruned when 10000/ lands and 15000/ is pruned when 19999/ lands,
        # leaving 10000/ + 19999/. max_to_keep=1 (checkpoints.py:48) spares the newest.
        # The 5k interval exists purely for --resume granularity on a ~20 h run; it
        # costs one extra dir (~27 GB) transiently, never a third at the end.
        save_interval=5_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP1_10_20k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 130 episodes / 39,625 frames -> 16.2 epochs at 20k steps, batch 32.
            repo_id="saifahmad123/EXP1_10",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Warmup is 10% of num_train_steps, following the rationale in the 2k configs
        # above rather than the flat warmup_steps=1000 used elsewhere in this file.
        # Identical across all three rungs so dataset size stays the only variable.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=20_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Exactly two checkpoints survive, ~18 GB. Saves fire at step % 5000 == 0
        # (step > start_step) plus the final step num_train_steps - 1 = 19999; step
        # 20000 is never reached. keep_period=10_000 spares only multiples of 10k, so
        # 5000/ is pruned when 10000/ lands and 15000/ is pruned when 19999/ lands,
        # leaving 10000/ + 19999/. max_to_keep=1 (checkpoints.py:48) spares the newest.
        # The 5k interval exists purely for --resume granularity on a ~20 h run; it
        # costs one extra dir (~27 GB) transiently, never a third at the end.
        save_interval=5_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP1_50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 650 episodes / 195,289 frames (13 sites x 50) -> ~1.6 epochs at 10k steps,
            # batch 32. The top rung of the EXP1 ladder above EXP1_1 / EXP1_5 / EXP1_10,
            # and a matched partner for pi05_Franka_EXP3_50 (650 eps / 196,856 frames of
            # the same orange cylinder): same size, same schedule, so EXP1 vs EXP3 stays
            # an A/B on the data alone.
            repo_id="saifahmad123/EXP1_50",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Schedule copied verbatim from pi05_Franka_EXP3_50: same peak/floor LR as the
        # rest of the EXP family, warmup at 10% of training and decay_steps spanning the
        # whole run so the cosine bottoms out at the last step.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=10_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=10_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Same retention as pi05_Franka_EXP3_50: save every 2k and keep them all, since
        # keep_period matches save_interval so 2000/, 4000/, 6000/ and 8000/ survive
        # pruning, and max_to_keep=1 spares the final save at num_train_steps - 1 = 9999.
        # Five dirs on disk, ~45 GB.
        save_interval=2_000,
        keep_period=2_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP3_10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/EXP3_10",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Short 2k-step run: warmup scaled down from 1k so it is only 10% of training.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2.5e-5,
            decay_steps=2_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=2_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Keep only the 1k and the final checkpoint, same as pi05_Franka_EXP1_10:
        # the two dirs on disk end up being 1000/ and 1999/.
        save_interval=200,
        keep_period=1000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP4_10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/EXP4_10",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Half the length of EXP1/EXP3 (1k instead of 2k steps); warmup scaled with it
        # so it stays 10% of training.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=2.5e-5,
            decay_steps=1_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=1_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Same mid + final pattern as EXP1/EXP3, halved: keep_period=500 spares step
        # 500 and max_to_keep=1 spares the newest save. The last save is at
        # num_train_steps - 1 = 999, so the two dirs on disk are 500/ and 999/
        # (there is no 1000/ — nothing is saved at step 1000).
        save_interval=200,
        keep_period=500,
    ),
    TrainConfig(
        name="pi05_Franka_EXP5_10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/EXP5_10",
            # Single-task dataset ("pick up the red cube"), but still take the prompt
            # from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Short 2k-step run: warmup scaled down from 1k so it is only 10% of training.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2.5e-5,
            decay_steps=2_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=2_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Keep only the 1k and the final checkpoint, same as pi05_Franka_EXP3_10:
        # the two dirs on disk end up being 1000/ and 1999/.
        save_interval=200,
        keep_period=1000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP6_10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/EXP6_10",
            # Single-task dataset ("pick up the red cube"), but still take the prompt
            # from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # 3k-step run: same peak/floor LR as pi05_Franka_EXP5_10, with warmup and
        # decay_steps rescaled to the longer schedule (warmup stays 10% of training,
        # decay_steps spans the whole run so the cosine bottoms out at the last step).
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=2.5e-5,
            decay_steps=3_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=3_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Save every 1k. keep_period matches save_interval so 1000/ and 2000/ both
        # survive pruning, and max_to_keep=1 spares the final save at
        # num_train_steps - 1 = 2999. Three dirs on disk, ~18 GB.
        save_interval=1_000,
        keep_period=1_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP7_10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 130 episodes / 40,573 frames -- the same shape and the same task string as
            # pi05_Franka_EXP6_10 (130 eps / 40,979 frames, "pick up the red cube"), so the
            # pair is an A/B on the data itself rather than on size or schedule.
            repo_id="saifahmad123/EXP7_10",
            # Single-task dataset ("pick up the red cube"), but still take the prompt
            # from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Schedule copied verbatim from pi05_Franka_EXP6_10: 3k steps, warmup 300 (10% of
        # training), decay_steps spanning the whole run so the cosine bottoms out at the
        # last step, same peak/floor LR as the rest of the EXP family.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=2.5e-5,
            decay_steps=3_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=3_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Same retention as pi05_Franka_EXP6_10: save every 1k, keep_period matches
        # save_interval so 1000/ and 2000/ survive pruning, and max_to_keep=1 spares the
        # final save at num_train_steps - 1 = 2999. Three dirs on disk, ~27 GB.
        save_interval=1_000,
        keep_period=1_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP8_10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 130 episodes / 39,800 frames -- same shape and same task string as
            # pi05_Franka_EXP6_10 (130 eps / 40,979 frames) and pi05_Franka_EXP7_10
            # (130 eps / 40,573 frames), so the three stay an A/B on the data itself
            # rather than on size or schedule.
            repo_id="saifahmad123/EXP8_10",
            # Single-task dataset ("pick up the red cube"), but still take the prompt
            # from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Schedule copied verbatim from pi05_Franka_EXP7_10: 3k steps, warmup 300 (10% of
        # training), decay_steps spanning the whole run so the cosine bottoms out at the
        # last step, same peak/floor LR as the rest of the EXP family.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=2.5e-5,
            decay_steps=3_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=3_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Same retention as pi05_Franka_EXP7_10: save every 1k, keep_period matches
        # save_interval so 1000/ and 2000/ survive pruning, and max_to_keep=1 spares the
        # final save at num_train_steps - 1 = 2999. Three dirs on disk, ~27 GB.
        save_interval=1_000,
        keep_period=1_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP8_10_10k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # Same dataset as pi05_Franka_EXP8_10 (130 eps / 39,800 frames), trained on
            # the longer pi05_Franka_EXP3_50 schedule instead of 3k steps -> ~8.0 epochs
            # at batch 32. The 3k config is kept as-is so the EXP6/EXP7/EXP8 A/B stays
            # intact; this is the long-schedule sibling, same idea as the *_20k configs.
            repo_id="saifahmad123/EXP8_10",
            # Single-task dataset ("pick up the red cube"), but still take the prompt
            # from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Schedule copied verbatim from pi05_Franka_EXP3_50: same peak/floor LR as the
        # rest of the EXP family, warmup at 10% of training and decay_steps spanning the
        # whole run so the cosine bottoms out at the last step.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=10_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=10_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Same retention as pi05_Franka_EXP3_50: save every 2k and keep them all, since
        # keep_period matches save_interval so 2000/, 4000/, 6000/ and 8000/ survive
        # pruning, and max_to_keep=1 spares the final save at num_train_steps - 1 = 9999.
        # Five dirs on disk, ~45 GB.
        save_interval=2_000,
        keep_period=2_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP9_ALL",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 686 episodes / 205,955 frames across 8 tasks (cracker box, orange
            # cylinder, lemon, mustard bottle, peach, purple cube, red cube, tomato soup
            # can) -- the full-object pool, not a rung of the EXP1/EXP3 size ladders.
            repo_id="saifahmad123/EXP9_ALL",
            # Multi-task: take the prompt from each episode's task string so the policy
            # is language-conditioned across all 8 objects.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # 20k-step run, same peak/floor LR as the rest of the EXP family with warmup at
        # 10% of training and decay_steps spanning the whole run so the cosine bottoms
        # out at the last step. 686 eps / 205,955 frames at batch 32 -> ~3.1 epochs.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=20_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # TWO checkpoints, ~18 GB. train.py saves when
        # (step % save_interval == 0 and step > start_step) or step == num_train_steps - 1,
        # so over steps 0..19999 only step 10000 and the final step 19999 are written.
        # keep_period=10_000 spares 10000/ from pruning and max_to_keep=1
        # (checkpoints.py) spares the newest, 19999/. The midpoint save also doubles as
        # the only --resume point if the job hits its walltime.
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP10_ALL",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 700 episodes / 207,539 frames across 8 tasks (cracker box, orange
            # cylinder, lemon, mustard bottle, peach, purple cube, red cube, tomato soup
            # can) -- the full-object pool, not a rung of the EXP1/EXP3 size ladders.
            repo_id="saifahmad123/EXP10_ALL",
            # Multi-task: take the prompt from each episode's task string so the policy
            # is language-conditioned across all 8 objects.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # 20k-step run, same peak/floor LR as the rest of the EXP family with warmup at
        # 10% of training and decay_steps spanning the whole run so the cosine bottoms
        # out at the last step. 700 eps / 207,539 frames at batch 32 -> ~3.1 epochs.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=20_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # TWO checkpoints, ~18 GB. train.py saves when
        # (step % save_interval == 0 and step > start_step) or step == num_train_steps - 1,
        # so over steps 0..19999 only step 10000 and the final step 19999 are written.
        # keep_period=10_000 spares 10000/ from pruning and max_to_keep=1
        # (checkpoints.py) spares the newest, 19999/. The midpoint save also doubles as
        # the only --resume point if the job hits its walltime.
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP11_10_10k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 130 episodes / 39,573 frames -> ~8.1 epochs at 10k steps, batch 32. Size-
            # matched to pi05_Franka_EXP8_10 (130 eps / 39,800 frames) by design: EXP11
            # is EXP8 with the visual DR switched off, so episode count must not vary.
            # The Hub copy's meta/ matches this exactly (info.json splits train 0:130),
            # but data/ and videos/ still hold 42 orphan files (episodes 130-171) from
            # the pre-trim push. LeRobot ignores them -- info.json caps the split at 130
            # -- so training is unaffected; they only cost download time, since train.py's
            # snapshot_download has no allow_patterns and pulls the whole repo.
            repo_id="saifahmad123/EXP11_10",
            # Single-task dataset ("pick up the red cube"), but still take the prompt
            # from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Schedule copied verbatim from pi05_Franka_EXP3_50: same peak/floor LR as the
        # rest of the EXP family, warmup at 10% of training and decay_steps spanning the
        # whole run so the cosine bottoms out at the last step.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=10_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=10_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Same retention as pi05_Franka_EXP3_50: save every 2k and keep them all, since
        # keep_period matches save_interval so 2000/, 4000/, 6000/ and 8000/ survive
        # pruning, and max_to_keep=1 spares the final save at num_train_steps - 1 = 9999.
        # Five dirs on disk, ~45 GB.
        save_interval=2_000,
        keep_period=2_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP3_50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/EXP3_50",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # 10k-step run (5x pi05_Franka_EXP3_10): same peak/floor LR as the rest of the
        # EXP family, with warmup and decay_steps rescaled to the longer schedule so
        # warmup stays 10% of training and the cosine bottoms out at the last step.
        # 650 eps / 196,856 frames at batch 32 -> ~1.6 epochs.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=10_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=10_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Save every 2k and keep them all: keep_period matches save_interval so
        # 2000/, 4000/, 6000/ and 8000/ all survive pruning, and max_to_keep=1 spares
        # the final save at num_train_steps - 1 = 9999. Five dirs on disk, ~45 GB.
        save_interval=2_000,
        keep_period=2_000,
    ),
    TrainConfig(
        name="pi05_Franka_EXP3_30",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # 390 episodes / 118,104 frames -- the rung between pi05_Franka_EXP3_20
            # (260 eps) and pi05_Franka_EXP3_50 (650 eps). Unlike EXP3_20 this is a real
            # uploaded dataset rather than an episodes= subset of EXP3_50, so it gets its
            # own norm stats under assets/pi05_Franka_EXP3_30/saifahmad123/EXP3_30/.
            repo_id="saifahmad123/EXP3_30",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Schedule copied verbatim from pi05_Franka_EXP3_50 so dataset size stays the only
        # variable across the rung: 10k steps, warmup 10% of training, cosine bottoming
        # out at the last step. 390 eps / 118,104 frames at batch 32 -> ~2.7 epochs.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=10_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=10_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # ONE checkpoint: 9999 only, ~8.9 GB. train.py:301 saves when
        # (step % save_interval == 0 and step > start_step) or step == num_train_steps - 1,
        # so save_interval=10_000 never fires inside a 0..9999 run -- step 0 is excluded by
        # step > start_step and step 10000 is never reached. Only the final save is
        # written, and keep_period=None leaves max_to_keep=1 (checkpoints.py:48) to spare
        # it. Consequence: there is no mid-run checkpoint to --resume from, so an
        # interrupted run restarts from scratch. Sized the sbatch walltime accordingly.
        save_interval=10_000,
        keep_period=None,
    ),
    TrainConfig(
        name="pi05_Franka_EXP3_20",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            # Same dataset as pi05_Franka_EXP3_50, cut down to 260 of its 650 episodes
            # (78,096 of 196,856 frames) instead of being re-uploaded as its own repo.
            repo_id="saifahmad123/EXP3_50",
            # Single-task dataset ("pick up the orange cylinder"), but still take the
            # prompt from the episode task so the policy stays language-conditioned.
            base_config=DataConfig(
                prompt_from_task=True,
                # The 260-episode subset, drawn once without replacement. numpy's Generator
                # stream is stability-guaranteed across versions, so seed 20 reproduces this
                # exact set of episode indices -- do not change the seed or the size without
                # accepting that any checkpoint trained under the old value is no longer
                # reproducible. Norm stats are reused from the full EXP3_50 (see below), so
                # they describe all 650 episodes, not just these 260.
                episodes=tuple(sorted(np.random.default_rng(20).choice(650, size=260, replace=False).tolist())),
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # 6k-step run: same peak/floor LR as the rest of the EXP family, warmup held at
        # 10% of training and decay_steps set to the last step so the cosine bottoms out
        # there. 260 eps / 78,096 frames at batch 32 -> ~2.5 epochs.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=2.5e-5,
            decay_steps=6_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=6_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Exactly two checkpoints: saves land on multiples of save_interval plus the final
        # step (num_train_steps - 1 = 5999), so 3000/ and 5999/ are the only ones written.
        # keep_period=3000 spares 3000/ from pruning and max_to_keep=1 spares 5999/.
        # Two dirs on disk, ~18 GB.
        save_interval=3_000,
        keep_period=3_000,
    ),
    TrainConfig(
        name="pi05_Franka_GraspNet_Test",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_GraspNet_Test",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # batch_size 128 is 4x the usual 32; at 10_000 steps the run sees ~1.28M
        # samples (~0.84 epochs of this dataset).
        # NOTE: decay_steps is NOT derived from num_train_steps — it must be changed
        # alongside it. Leaving it higher would end training partway up the cosine
        # curve with the anneal never applied.
        # peak_lr is sqrt-scaled (2.5e-5 -> 5e-5) for the 4x batch; warmup is ~3% of
        # the run.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=5e-5,
            decay_steps=10_000,
            decay_lr=5e-6,
        ),
        num_train_steps=10_000,
        batch_size=128,
        # Matches the 16 cores the job requests (same as pi05_Franka_3_objects_2).
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2_000,
    ),
    TrainConfig(
        name="pi05_Franka_GraspNet_Test_texture",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_GraspNet_Test_texture",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Same batch size / LR schedule as pi05_Franka_3_objects_2, which trained well
        # on a dataset of comparable size — batch 32 at peak_lr 2.5e-5 for 30k steps,
        # rather than the batch 128 / 5e-5 / 10k of pi05_Franka_GraspNet_Test.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    # Same data / hyperparameters as pi05_Franka_GraspNet_Test_texture, but with a
    # 150-step action horizon instead of 50.
    TrainConfig(
        name="pi05_Franka_GraspNet_Test_texture_150",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=150,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_GraspNet_Test_texture",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    # Full GraspNet dataset (2572 eps / 1.70M frames, same 3 tasks and schema as
    # Franka_GraspNet_Test_texture). Hyperparameters match the texture configs.
    TrainConfig(
        name="pi05_Franka_GRASPNET",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_GRASPNET",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    # Same data / hyperparameters / 50-step horizon as pi05_Franka_GRASPNET, but the
    # policy predicts *absolute* joint targets instead of deltas from the current state.
    # With use_delta_joint_actions=False the DeltaActions/AbsoluteActions pair is dropped
    # from the pipeline entirely, so delta_action_mask no longer applies. Note this needs
    # its own norm stats: the delta transform runs before statistics are gathered, so the
    # action stats here are not interchangeable with pi05_Franka_GRASPNET's.
    TrainConfig(
        name="pi05_Franka_GRASPNET_abs",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=False,
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_GRASPNET",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=50_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=50_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    # Same data / hyperparameters / 50-step horizon as pi05_Franka_GRASPNET, but the
    # SigLIP vision encoder is frozen on top of the usual LoRA freeze filter.
    TrainConfig(
        name="pi05_Franka_GRASPNET_frozen_siglip",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_GRASPNET",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=50_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=50_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=nnx.Any(
            pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            _FreezeSigLip,
        ),
        ema_decay=None,
        save_interval=2500,
    ),
    # pi05_Franka_GRASPNET with both variations combined: absolute joint actions and a
    # frozen SigLIP encoder. Shares its norm stats requirement with
    # pi05_Franka_GRASPNET_abs (same data pipeline); freezing weights does not affect them.
    TrainConfig(
        name="pi05_Franka_GRASPNET_abs_frozen_siglip",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=False,
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_GRASPNET",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=50_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=50_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=nnx.Any(
            pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            _FreezeSigLip,
        ),
        ema_decay=None,
        save_interval=2500,
    ),
    # Same data / hyperparameters as pi05_Franka_GRASPNET, but with a 150-step
    # action horizon instead of 50.
    TrainConfig(
        name="pi05_Franka_GRASPNET_150",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=150,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_GRASPNET",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    # Second GraspNet collection (2990 eps / 1.30M frames). Same schema and same 3 tasks
    # as Franka_GRASPNET; hyperparameters match pi05_Franka_GRASPNET.
    TrainConfig(
        name="pi05_Franka_GraspNet_2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_GraspNet_2",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    # Final, largest GraspNet collection (5500 eps / 1.69M frames, 7 tasks: cracker box,
    # lemon, mustard bottle, nivea men face wash tube, peach, pear, tomato soup can).
    # Same schema and hyperparameters as pi05_Franka_GraspNet_2, scaled to a 50k-step run.
    TrainConfig(
        name="pi05_Franka_GRASPNET_FINAL",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/GRASPNET_FINAL",
            # Multi-task dataset (7 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # decay_steps tracks num_train_steps so the cosine actually bottoms out at
        # decay_lr; warmup scaled up from GraspNet_2's 1k/30k to stay ~3% of training.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_500,
            peak_lr=2.5e-5,
            decay_steps=50_000,
            decay_lr=2.5e-6,
        ),
        # 50k steps x batch 32 = 1.6M samples, ~0.95 epochs of this dataset.
        num_train_steps=50_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Every 5k-step save is kept: keep_period matches save_interval, so pruning
        # (max_to_keep=1) only ever drops non-multiples. Dirs on disk end up being
        # 5000/ ... 45000/ plus the last save at num_train_steps - 1 = 49999/
        # (there is no 50000/) — 10 dirs, ~89 GB.
        save_interval=5_000,
        keep_period=5_000,
    ),
    # Three datasets merged on disk (scripts/merge_graspnet_moredata.py):
    #   michaelyeah7/Franka_GraspNet_20260817  16,596 eps / 4.92M frames / 10 tasks
    #   saifahmad123/GRASPNET_FINAL             5,500 eps / 1.69M frames /  7 (subset)
    #   saifahmad123/Franka_3_objects_2         1,965 eps / 1.36M frames /  3 (all new)
    #   -> 24,061 eps / 7,971,809 frames / 13 tasks
    # Same simulator, robot and table throughout; identical v2.1 schema.
    #
    # Merged on disk ONLY -- there is no Hub repo for
    # "saifahmad123/GRASPNET_FINAL_moreData", so every job touching this config must
    # export HF_HUB_OFFLINE=1 (that turns the pre-download loops in scripts/train.py and
    # scripts/compute_norm_stats.py into no-ops that just return the local dir; without
    # it snapshot_download 404s and the retry loop sleeps forever).
    #
    # Merge layout: michaelyeah7 keeps indices 0..16595 untouched (its task indices
    # already match the merged table, so its parquets are reused as-is); GRASPNET_FINAL
    # lands at 16596..22095 and Franka_3_objects_2 at 22096..24060, both with
    # episode_index/index shifted and task_index remapped by task name. Task table is
    # michaelyeah7's 10 plus orange cylinder / red cube / purple cube at 10..12.
    TrainConfig(
        name="pi05_Franka_GRASPNET_FINAL_moreData",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/GRASPNET_FINAL_moreData",
            # Multi-task dataset (10 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    # Prompt augmentation: up to 20 phrasings per object (11
                    # paraphrases + 3 category generics like "pick up the can" + 6
                    # misspellings), 239 unique strings over the 13 objects, sampled
                    # ~65% specific / ~14% generic / ~21% noisy with the canonical
                    # string weighted heaviest so eval prompts using the original
                    # wording stay in distribution. Must come BEFORE RepackTransform,
                    # which drops the
                    # episode_index/frame_index this keys its deterministic choice on.
                    # Training-only: serve_policy.py passes no repack_transforms, so a
                    # real prompt is never rewritten at inference.
                    _transforms.PromptVariants(graspnet_prompts.PROMPT_VARIANTS),
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # decay_steps tracks num_train_steps so the cosine actually bottoms out at
        # decay_lr; warmup stays ~3% of training, as in pi05_Franka_GRASPNET_FINAL.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        # 100k steps x batch 32 = 3.2M samples, ~0.40 epochs of this 7.97M-frame dataset.
        num_train_steps=100_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # keep_period matches save_interval so every 10k save is kept (Orbax runs with
        # max_to_keep=1, so pruning only ever drops non-multiples of keep_period).
        # On disk: 10000/ ... 90000/ plus the last save at num_train_steps - 1 = 99999/
        # (there is no 100000/) -- 10 dirs x 8.9 GB = ~89 GB. Measured against
        # /projects/p53063 at 168 GB free after the merged dataset lands, leaving ~79 GB.
        # 10k rather than 20k because this runs on a single H100: ~37 h of compute
        # against gengpu's hard 2-day MaxTime, so a wall-clock kill is a live risk and
        # 10k bounds the lost work. If disk gets tight, raising keep_period to 20_000
        # while leaving save_interval at 10_000 keeps 10k-granularity crash recovery but
        # prunes down to ~5 dirs (~45 GB).
        save_interval=10_000,
        keep_period=10_000,
    ),
    # Same data / hyperparameters as pi05_Franka_GraspNet_2, but with a 100-step action
    # horizon instead of 50. Only Pi0Config.action_horizon differs.
    TrainConfig(
        name="pi05_Franka_GraspNet_2_100",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=100,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_GraspNet_2",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.wrist",
                                "cam_left_wrist": "observation.images.front_1",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    # pi05_Franka_GraspNet_2 with the camera slots assigned by what the cameras actually are,
    # instead of the usual Franka remap. Two things were wrong with feeding the wrist camera
    # into cam_high and the static front cameras into the wrist slots:
    #
    #  1. preprocess_observation gates geometric augmentation (random crop/resize/rotate) on
    #     "wrist" not being in the model key, so the remap inverted it -- the wrist camera got
    #     cropped and rotated (destroying the hand-eye relationship it exists to provide) while
    #     both static cameras got none of the robustness to camera placement they need. That is
    #     the augmentation most likely to matter for sim-to-real, since real camera extrinsics
    #     never match the sim exactly.
    #  2. pi05_base was pretrained with cam_high holding a scene-level third-person view and the
    #     wrist slots holding gripper close-ups. The remap hands it the opposite, so adaptation
    #     capacity gets spent relearning slot semantics.
    #
    # front_1 now takes cam_high; the wrist camera takes cam_left_wrist. front_2 has to sit in
    # cam_right_wrist (only one non-wrist slot exists), so geometric_aug_keys names the two
    # static cameras explicitly rather than letting the key-name heuristic decide.
    TrainConfig(
        name="pi05_Franka_GraspNet_2_camfix",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            geometric_aug_keys=("base_0_rgb", "right_wrist_0_rgb"),
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Franka_GraspNet_2",
            # Multi-task dataset (3 prompts): take the prompt from each episode's task
            # so the policy is language-conditioned at inference time.
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front_1",
                                "cam_left_wrist": "observation.images.wrist",
                                "cam_right_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=14,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=2500,
    ),
    TrainConfig(
        name="pi05_Franka_Teleop_Random",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Teleop_Random",
            default_prompt="pick up the red object",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front_1",
                                "cam_left_wrist": "observation.images.front_2",
                                "cam_right_wrist": "observation.images.wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=10_000,
    ),
    TrainConfig(
        name="pi05_Franka_Teleop_Random_2Cam",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Teleop_Random_2Cam",
            default_prompt="pick up the red object",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front_1",
                                "cam_left_wrist": "observation.images.front_2",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=10_000,
    ),
    TrainConfig(
        name="pi05_Franka_Real_Random_Data_droid",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Real_Random",
            default_prompt="pick up the orange cylinder",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front_1",
                                "cam_left_wrist": "observation.images.front_2",
                                "cam_right_wrist": "observation.images.wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=100_000,
        batch_size=2,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        save_interval=25_000,
    ),
    TrainConfig(
        name="pi05_Franka_base",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
        ),
        data=LeRobotAlohaDataConfig(
            use_delta_joint_actions=True,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            adapt_to_pi=False,
            repo_id="saifahmad123/Real_Random",
            default_prompt="pick up the orange cylinder",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.front_1",
                                "cam_left_wrist": "observation.images.front_2",
                                "cam_right_wrist": "observation.images.wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        num_train_steps=50_000,
        batch_size=2,
        ema_decay=None,
        save_interval=25_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
    *lehome_config.get_lehome_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]