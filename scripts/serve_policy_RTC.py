"""Websocket policy server with real-time chunking (RTC) enabled.

Separate entrypoint from `serve_policy.py`, which is left untouched. Use this one only
when the client is sending `prev_actions` / `prev_actions_start` / `prev_actions_d`; the
server falls back to unconstrained sampling whenever those keys are absent, so it is also
safe to run against a client that has RTC switched off.

The CLI deliberately mirrors `serve_policy.py` exactly, so the same command works with
only the script name changed:

    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python scripts/serve_policy_RTC.py --default_prompt "" --port 8000 \
        policy:checkpoint \
        --policy.config pi05_Franka_GraspNet_2 \
        --policy.dir checkpoints/pi05_Franka_GraspNet_2/GraspNet_2/29999
"""

import dataclasses
import logging
import os
import pathlib
import socket
import time

import jax
import jax.numpy as jnp
import tyro

from openpi.models import model as _model
from openpi.models import rtc as _rtc
from openpi.policies import policy_rtc as _policy_rtc
from openpi.serving import websocket_policy_server
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi import transforms


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi05_Franka_GraspNet_2").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi05_Franka_GraspNet_2/GraspNet_2/29999").
    dir: str


@dataclasses.dataclass
class Default:
    """Placeholder so the CLI matches serve_policy.py; RTC needs an explicit checkpoint."""


@dataclasses.dataclass
class Args:
    """Arguments for the RTC serve_policy script."""

    # Specifies how to load the policy. Mirrors serve_policy.py so the same command line
    # works: `policy:checkpoint --policy.config <name> --policy.dir <path>`.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Port to serve the policy on.
    port: int = 8000
    # Used when the "prompt" key is not present in the data.
    default_prompt: str | None = None

    # --- RTC knobs ---
    # How far into the chunk the soft mask extends, in steps. The reference implementation
    # uses `action_horizon - execute_horizon`. None means "all of the previous chunk that
    # has not been played yet", which is the most constrained setting.
    prefix_attention_horizon: int | None = None
    # Weight decay shape across the soft-mask region.
    prefix_attention_schedule: _rtc.PrefixAttentionSchedule = "exp"
    # Upper bound on the guidance weight (paper default: 5.0).
    max_guidance_weight: float = 5.0
    # Flow-matching denoising steps. RTC adds a backward pass through the action expert per
    # step, so this is the main lever on inference latency.
    num_steps: int = 10

    # Compile both sampler traces before accepting connections. Leave this on: the guided
    # and unguided paths are *separate* JIT traces, so without warmup the guided one
    # compiles on the first request that carries `prev_actions` -- i.e. mid-episode, on a
    # moving arm. Compilation blocks the server's asyncio loop, and the client hardcodes a
    # 20s websocket keepalive (websocket_client_policy.py:37, not overridable), so the
    # connection is dropped rather than merely delayed.
    warmup: bool = True

    # Persistent XLA compilation cache. The two traces cost ~1-3 min to compile; with this
    # set, that is paid once and every later start reads them off disk. Defaults outside
    # the repo so it never shows up in git status. Set to None to disable.
    # The cache is keyed on the computation and the jax/jaxlib version, so it invalidates
    # itself correctly when the sampler code or JAX changes -- a stale hit is not a risk.
    compilation_cache_dir: str | None = "~/.cache/openpi/jax_compilation_cache"


def create_rtc_policy(args: Args) -> tuple[_policy_rtc.RTCPolicy, _config.TrainConfig, int, int]:
    """Mirror of `policy_config.create_trained_policy`, but building an `RTCPolicy`.

    Kept here rather than as an edit to `policy_config.py` so that the existing serving
    path is bit-identical to what it was. If `create_trained_policy` changes upstream, the
    transform lists below need the same change.
    """
    if not isinstance(args.policy, Checkpoint):
        raise ValueError(
            "RTC serving needs an explicit checkpoint. Use:\n"
            "  policy:checkpoint --policy.config <config_name> --policy.dir <checkpoint_dir>"
        )

    train_config = _config.get_config(args.policy.config)
    checkpoint_dir = download.maybe_download(args.policy.dir)

    if os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
        raise ValueError("RTC serving supports the JAX checkpoint path only, not PyTorch.")

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(pathlib.Path(checkpoint_dir) / "params", dtype=jnp.bfloat16))

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id)

    action_dim = int(norm_stats["actions"].mean.shape[-1])
    state_dim = int(norm_stats["state"].mean.shape[-1])

    policy = _policy_rtc.RTCPolicy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(args.default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": args.num_steps},
        metadata=train_config.policy_metadata,
        prefix_attention_horizon=args.prefix_attention_horizon,
        prefix_attention_schedule=args.prefix_attention_schedule,
        max_guidance_weight=args.max_guidance_weight,
        prev_action_dim=action_dim,
    )
    return policy, train_config, state_dim, action_dim


def warmup_policy(policy, train_config, state_dim: int, action_dim: int) -> None:
    """Force both sampler traces to compile while nothing is connected."""
    import numpy as np

    horizon = train_config.model.action_horizon

    def fake_obs(with_prev: bool) -> dict:
        obs = {
            "state": np.zeros((state_dim,), np.float32),
            "images": {
                name: np.zeros((3, 224, 224), np.uint8)
                for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            },
            "prompt": "warmup",
        }
        if with_prev:
            obs["prev_actions"] = np.zeros((horizon, action_dim), np.float32)
            obs["prev_actions_start"] = horizon // 2
            obs["prev_actions_d"] = 3
        return obs

    for label, with_prev in (("unguided", False), ("guided (RTC)", True)):
        t0 = time.monotonic()
        try:
            # First call pays for XLA compilation, which `block_until_ready` correctly
            # charges to the timer. Run a second one to get the number that matters.
            policy.infer(fake_obs(with_prev))
            compile_s = time.monotonic() - t0
            out = policy.infer(fake_obs(with_prev))
        except Exception:
            logging.exception("Warmup failed for the %s path; it will compile on first use.", label)
            continue
        logging.info(
            "Warmup %-12s compile+first call %6.1fs | steady-state infer %6.1f ms | rtc_active=%s",
            label,
            compile_s,
            out["policy_timing"]["infer_ms"],
            out.get("rtc_active"),
        )


def setup_compilation_cache(cache_dir: str) -> None:
    """Enable JAX's persistent compilation cache. Must run before the first compile."""
    path = pathlib.Path(cache_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    # Defaults skip small/fast compiles; ours are neither, but set them explicitly so the
    # behaviour does not depend on the JAX version's defaults.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    existing = len(list(path.glob("*")))
    logging.info("JAX compilation cache: %s (%d existing entries)", path, existing)


def main(args: Args) -> None:
    if args.compilation_cache_dir:
        setup_compilation_cache(args.compilation_cache_dir)

    policy, train_config, state_dim, action_dim = create_rtc_policy(args)

    if args.warmup:
        warmup_policy(policy, train_config, state_dim, action_dim)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating RTC server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
