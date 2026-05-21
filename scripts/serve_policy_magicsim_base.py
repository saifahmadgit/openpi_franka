"""Serve the off-the-shelf pi0.5 base model with MagicSim norm stats.

This is a baseline evaluation script: same observation/action pipeline as the
fine-tuned model (pi05_magicsim_apple_red_joint) but using the original pi0.5
weights with no fine-tuning.  The only thing that differs from a fine-tuned run
is the model weights — normalization, camera mapping, and transforms are
identical — so the result is a clean measure of how much fine-tuning helped.

Weights come from the cached GCS download at:
    ~/.cache/openpi/openpi-assets/checkpoints/pi05_base/

Norm stats come from the fine-tuned checkpoint's assets:
    checkpoints/pi05_magicsim_apple_red_joint/.../assets/.../norm_stats.json

Usage:
    cd ~/openpi && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python scripts/serve_policy_magicsim_base.py \\
        --default_prompt "pick up the apple" \\
        --port 8000
"""

import dataclasses
import logging
import pathlib
import socket

import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


BASE_WEIGHTS_DIR = (
    pathlib.Path.home()
    / ".cache/openpi/openpi-assets/checkpoints/pi05_base"
)

NORM_STATS_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "checkpoints/pi05_magicsim_apple_red_joint"
    / "magicsim_apple_lora_red_joint_cam_cal/50000"
    / "assets/saifahmad123/Franka_Apple_joint_Cam_Calibrated"
)


@dataclasses.dataclass
class Args:
    default_prompt: str = "pick up the apple"
    port: int = 8000
    record: bool = False


def main(args: Args) -> None:
    logging.info("Loading norm stats from %s", NORM_STATS_DIR)
    norm_stats = _normalize.load(NORM_STATS_DIR)

    logging.info("Loading base pi0.5 weights from %s", BASE_WEIGHTS_DIR)
    train_config = _config.get_config("pi05_magicsim_base")
    policy = _policy_config.create_trained_policy(
        train_config,
        BASE_WEIGHTS_DIR,
        norm_stats=norm_stats,
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
