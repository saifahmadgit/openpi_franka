"""Policy wrapper implementing the server half of real-time chunking (RTC).

Additive: `policy.py` is untouched, so the non-RTC serving path is unchanged. This
subclass only overrides `infer`.

The wire contract with the client (all three keys sent together, or all omitted):

    prev_actions        float32 (action_horizon, D)  the previous infer() reply, verbatim:
                                                     absolute space, unsliced, unrebased
    prev_actions_start  int                          index of the first action the robot
                                                     has not yet *played*
    prev_actions_d      int                          size of the frozen region, in steps

The client deliberately sends absolute actions and does no rebasing of its own, because
the delta transform and the normalization stats live here. Getting that rebase wrong is
silent: `use_delta_joint_actions=True` means the model predicts offsets from the state of
*this* call, so reusing the previous chunk's numbers without rebasing pins the arm to
targets stale by exactly the motion RTC exists to bridge.

Rather than reimplement the rebase, we push `prev_actions` through the policy's own input
transform stack under the `actions` key. That stack is the same one training used --
AlohaInputs -> DeltaActions (rebases against this call's state) -> Normalize ->
PadStatesAndActions -- so the guidance target lands in exactly the model's space and
cannot drift from the training config.
"""

from collections.abc import Sequence
import logging
import time
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import rtc as _rtc
from openpi.models.pi0 import Pi0
from openpi.models.pi0_rtc import sample_actions_rtc
from openpi.policies import policy as _policy
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

logger = logging.getLogger(__name__)


class RTCPolicy(_policy.Policy):
    """A `Policy` that can guide sampling toward the previous action chunk."""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        # How far into the chunk the soft mask extends, in steps. The reference uses
        # `action_horizon - execute_horizon`. None means "the whole available overlap",
        # i.e. attend to all of the previous chunk that has not been played yet.
        prefix_attention_horizon: int | None = None,
        prefix_attention_schedule: _rtc.PrefixAttentionSchedule = "exp",
        max_guidance_weight: float = 5.0,
        # Width, in dims, at which `prev_actions` must enter the input transform stack.
        # This is the action norm-stat width (8 for Franka: 7 joints + gripper), NOT the
        # width the client receives. The reply is sliced to 14 by AlohaOutputs, and the
        # input-side Normalize is not tolerant of a mismatch the way Unnormalize is: it
        # slices the stats to the data width and then fails to broadcast. So the extra
        # dims -- which are meaningless padding the model is trained to emit as zero --
        # get dropped here, on the server, keeping the client free of action-space logic.
        prev_action_dim: int | None = None,
    ):
        if not isinstance(model, Pi0):
            raise ValueError(f"RTCPolicy requires a Pi0/Pi0.5 JAX model, got {type(model).__name__}.")

        super().__init__(
            model,
            rng=rng,
            transforms=transforms,
            output_transforms=output_transforms,
            sample_kwargs=sample_kwargs,
            metadata=metadata,
        )

        self._prefix_attention_horizon = prefix_attention_horizon
        self._prefix_attention_schedule = prefix_attention_schedule
        self._max_guidance_weight = max_guidance_weight
        self._action_horizon = model.action_horizon
        self._prev_action_dim = prev_action_dim

        # `module_jit` wants a bound method of an nnx.Module; bind the free function to the
        # model so the module state is frozen the same way the base class does it.
        bound = types.MethodType(sample_actions_rtc, model)
        self._sample_actions_rtc = nnx_utils.module_jit(bound)

    def _build_guidance(self, obs: dict) -> tuple[np.ndarray | None, int, int, int]:
        """Pop the RTC keys and align the previous chunk. Mutates `obs`.

        Returns (aligned_prev_absolute, start, d_eff, overlap). The first element is None
        when RTC is inactive, in which case sampling is unconstrained.
        """
        prev_actions = obs.pop("prev_actions", None)
        prev_start = obs.pop("prev_actions_start", None)
        prev_d = obs.pop("prev_actions_d", None)

        if prev_actions is None or prev_start is None or prev_d is None:
            if prev_actions is not None or prev_start is not None or prev_d is not None:
                logger.warning("Partial RTC keys received; ignoring. Send all three or none.")
            return None, 0, 0, 0

        prev = np.asarray(prev_actions, dtype=np.float32)
        horizon = self._action_horizon
        if prev.ndim != 2 or prev.shape[0] != horizon:
            logger.warning(
                "prev_actions has shape %s, expected (%d, D); ignoring RTC for this call.", prev.shape, horizon
            )
            return None, 0, 0, 0

        start = int(np.clip(int(prev_start), 0, horizon))
        overlap = horizon - start
        if overlap <= 0:
            # The previous chunk is fully played; there is nothing to pin against, so this
            # is the same situation as the first inference of an episode.
            return None, start, 0, 0

        # Clamp the frozen region to what actually exists. d_eff < d means the reply landed
        # after the old chunk ran dry -- a dropped deadline, reported back to the client.
        d_eff = int(np.clip(int(prev_d), 0, overlap))

        if self._prev_action_dim is not None:
            if prev.shape[1] < self._prev_action_dim:
                logger.warning(
                    "prev_actions has %d dims, expected at least %d; ignoring RTC for this call.",
                    prev.shape[1],
                    self._prev_action_dim,
                )
                return None, start, 0, 0
            prev = prev[:, : self._prev_action_dim]

        # Align so index 0 is the first unplayed action: out[k] <-> prev[start + k].
        aligned = np.empty_like(prev)
        aligned[:overlap] = prev[start:]
        # Tail is never attended to (weights are zero past `overlap`); hold the last action
        # rather than zero-fill so the delta rebase below stays in a sane numeric range.
        aligned[overlap:] = prev[-1]

        return aligned, start, d_eff, overlap

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        aligned_prev, start, d_eff, overlap = self._build_guidance(inputs)

        if aligned_prev is not None:
            # Ride the normal input transform stack so the rebase + normalization + padding
            # match training exactly.
            inputs["actions"] = aligned_prev

        inputs = self._input_transform(inputs)

        prefix_actions = inputs.pop("actions", None)
        prefix_weights = None
        attention_horizon = 0
        if prefix_actions is not None:
            attention_horizon = overlap if self._prefix_attention_horizon is None else self._prefix_attention_horizon
            attention_horizon = int(np.clip(attention_horizon, 0, overlap))
            prefix_weights = _rtc.get_prefix_weights(
                d_eff, attention_horizon, self._action_horizon, self._prefix_attention_schedule
            )

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise
        if prefix_actions is not None:
            sample_kwargs["prefix_actions"] = jnp.asarray(prefix_actions)[np.newaxis, ...]
            sample_kwargs["prefix_weights"] = jnp.asarray(prefix_weights)
            sample_kwargs["max_guidance_weight"] = self._max_guidance_weight

        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()
        actions = self._sample_actions_rtc(sample_rng, observation, **sample_kwargs)
        # NOTE: JAX dispatch is asynchronous. The base Policy stops its timer before the
        # result is materialized, so its `policy_timing.infer_ms` reports dispatch cost
        # (~10ms) rather than inference (~170ms). Block first so this number is real.
        actions = jax.block_until_ready(actions)
        model_time = time.monotonic() - start_time

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)

        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        outputs["rtc_active"] = prefix_actions is not None
        outputs["rtc_d_eff"] = d_eff
        outputs["rtc_start"] = start
        outputs["rtc_overlap"] = overlap
        outputs["rtc_prefix_attention_horizon"] = attention_horizon
        return outputs
