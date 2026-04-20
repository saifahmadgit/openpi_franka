"""Training configs for the LeHome Challenge.

Follows the same pattern as roboarena_config.py and polaris_config.py:
imported by config.py via *lehome_config.get_lehome_configs().

Dataset: lehome/dataset_challenge_merged (LeRobot v3.0 format on HuggingFace)
  Features: observation.state (12,), action (12,), three RGB cameras
  No task/prompt field — a fixed default prompt is injected.

Models:
  pi05_lehome_lora — pi0.5 with LoRA on both VLM and action expert.
                     Fine-tuned from gs://openpi-assets/checkpoints/pi05_base.
"""

import dataclasses

import openpi.models.pi0_config as pi0_config
import openpi.policies.lehome_policy as lehome_policy
import openpi.transforms as _transforms

# HuggingFace dataset repo ID for the merged LeHome challenge dataset.
LEHOME_DATASET_REPO_ID = "lehome/dataset_challenge_merged"

# Language prompt injected at training and inference (dataset has no task field).
DEFAULT_PROMPT = lehome_policy.DEFAULT_PROMPT


def _make_repack_transform() -> _transforms.Group:
    """Maps LeRobot dot-key dataset features to the nested dict structure
    that LeHomeInputs expects.

    Source keys (as stored in the LeRobot dataset):
        observation.images.top_rgb   → images.cam_high
        observation.images.left_rgb  → images.cam_left_wrist
        observation.images.right_rgb → images.cam_right_wrist
        observation.state            → state
        action                       → actions  (plural, expected by training loop)
    """
    return _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.top_rgb",
                        "cam_left_wrist": "observation.images.left_rgb",
                        "cam_right_wrist": "observation.images.right_rgb",
                    },
                    "state": "observation.state",
                    "actions": "action",
                }
            )
        ]
    )


def get_lehome_configs():
    # Local imports to avoid circular dependency with config.py.
    from openpi.training.config import (
        AssetsConfig,
        DataConfig,
        DataConfigFactory,
        ModelTransformFactory,
        TrainConfig,
    )
    import openpi.training.weight_loaders as weight_loaders
    from typing_extensions import override
    import pathlib
    import openpi.models.model as _model

    @dataclasses.dataclass(frozen=True)
    class LeRobotLeHomeDataConfig(DataConfigFactory):
        """DataConfig for the LeHome challenge dataset (LeRobot format).

        The dataset uses absolute joint-angle actions (not deltas), so no
        DeltaActions transform is applied.
        """

        @override
        def create(
            self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
        ) -> DataConfig:
            data_transforms = _transforms.Group(
                inputs=[lehome_policy.LeHomeInputs(model_type=model_config.model_type)],
                outputs=[lehome_policy.LeHomeOutputs()],
            )

            model_transforms = ModelTransformFactory(default_prompt=DEFAULT_PROMPT)(
                model_config
            )

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=_make_repack_transform(),
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                # Dataset stores actions under the key "action" (singular).
                # The training data loader uses this to build the action sequence.
                action_sequence_keys=("action",),
            )

    return [
        # ── pi0.5 + LoRA fine-tune on all LeHome garments ─────────────────────
        TrainConfig(
            name="pi05_lehome_lora",
            model=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",       # LoRA on VLM backbone
                action_expert_variant="gemma_300m_lora",  # LoRA on action expert
            ),
            data=LeRobotLeHomeDataConfig(
                repo_id=LEHOME_DATASET_REPO_ID,
                base_config=DataConfig(prompt_from_task=False),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            # Freeze base weights; train only LoRA parameters.
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            num_train_steps=50_000,
            batch_size=8,         # safe for 48 GB GPU with LoRA; increase if memory allows
            save_interval=25000,
            ema_decay=None,       # skip EMA for LoRA fine-tuning
            wandb_enabled=True,
        ),
    ]
