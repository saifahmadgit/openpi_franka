"""Serve the pi0.5-DROID fine-tuned model for MagicSim evaluation.

Uses weights from the pi0.5 model fine-tuned on the full DROID dataset (pi05_droid),
so you get the benefit of large-scale diverse robot data rather than the vanilla base.

The sim sends observations in MagicSim/Aloha format:
    images: {"cam_high": ..., "cam_left_wrist": ..., "cam_right_wrist": ...}
    state:  [j0..j6, gripper]  (8-dim flat)

This script remaps those to DROID input format before running inference:
    observation/exterior_image_1_left  <- cam_high
    observation/wrist_image_left       <- cam_left_wrist (falls back to cam_right_wrist)
    observation/joint_position         <- state[:7]
    observation/gripper_position       <- state[7:8]

Weights are fetched from GCS on first run (~4 GB) and cached locally at:
    ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid/

NOTE: DROID was trained with joint-velocity actions; the sim interprets whatever
actions come back, so treat this as a capability probe rather than a calibrated
deployment.

Usage:
    cd ~/openpi && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python scripts/serve_policy_magicsim_droid.py \\
        --default_prompt "pick up the apple" \\
        --port 8000
"""

import dataclasses
import logging
import socket

import numpy as np
import tyro

from openpi import transforms as _transforms
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


DROID_WEIGHTS_GCS = "gs://openpi-assets/checkpoints/pi05_droid"


@dataclasses.dataclass(frozen=True)
class _MagicSimToDroid(_transforms.DataTransformFn):
    """Bridges MagicSim observation keys to the format DroidInputs expects."""

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        state = np.asarray(data["state"])
        out = dict(data)
        # cam_high      = front_1 (exterior fixed cam)  → DROID exterior slot
        # cam_right_wrist = wrist cam                   → DROID wrist slot
        # cam_left_wrist  = front_2 (second fixed cam)  → unused by DROID
        out["observation/exterior_image_1_left"] = images["cam_high"]
        out["observation/wrist_image_left"] = images.get("cam_right_wrist", images["cam_high"])
        out["observation/joint_position"] = state[:7]
        out["observation/gripper_position"] = state[7:8]
        return out


@dataclasses.dataclass
class Args:
    default_prompt: str = "pick up the apple"
    port: int = 8000


def main(args: Args) -> None:
    train_config = _config.get_config("pi05_droid")
    repack = _transforms.Group(inputs=[_MagicSimToDroid()])

    logging.info("Loading pi0.5-DROID weights from %s (auto-cached on first run)", DROID_WEIGHTS_GCS)
    policy = _policy_config.create_trained_policy(
        train_config,
        DROID_WEIGHTS_GCS,
        repack_transforms=repack,
        default_prompt=args.default_prompt,
    )

    hostname = socket.gethostname()
    logging.info("Starting server on %s:%d", hostname, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
