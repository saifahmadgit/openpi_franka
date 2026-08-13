"""Real-time chunking (RTC) sampler for Pi0 / Pi0.5.

Implements "Real-Time Execution of Action Chunking Flow Policies"
(Black, Galliker & Levine, 2025), ported from the reference implementation at
https://github.com/Physical-Intelligence/real-time-chunking-kinetix (`src/model.py`,
`realtime_action`).

This is a *free function* that takes an existing `Pi0` instance rather than a subclass or
a patch, so that `pi0.py` stays untouched and the non-RTC serving path is bit-identical to
what it was before. The cost is that the denoising loop below duplicates the structure of
`Pi0.sample_actions`; if that method changes upstream, this needs the same change.

Two conventions differ from the reference and are the easiest thing to get wrong:

1. Time. openpi uses the diffusion-literature convention where ``time=1`` is noise and
   ``time=0`` is the target. The paper uses the opposite. Below, ``t_rtc = 1 - time``.
2. Velocity sign. openpi's velocity points target -> noise; the paper's points
   noise -> target. So the guidance correction enters with a flipped sign.
"""

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at


@at.typecheck
def sample_actions_rtc(
    model: Pi0,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    # Unions with a scalar array: under `jax.jit` these arrive as tracers, not Python
    # scalars, and `at.typecheck` rejects a bare `int`/`float`. Matches Pi0.sample_actions.
    num_steps: int | at.Int[at.Array, ""] = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
    # The previous action chunk, already rebased into this call's delta space, normalized,
    # and padded to the model action dim -- i.e. pushed through the same input transform
    # stack as a training target. Aligned so index 0 is the first action the robot has not
    # yet played, so that `out[k]` corresponds to `prev[prev_actions_start + k]`.
    prefix_actions: at.Float[at.Array, "b ah ad"] | None = None,
    # Per-timestep prefix attention weight; see `rtc.get_prefix_weights`.
    prefix_weights: at.Float[at.Array, " ah"] | None = None,
    max_guidance_weight: float | at.Float[at.Array, ""] = 5.0,
) -> _model.Actions:
    """Sample an action chunk, optionally guided toward a previous chunk.

    With `prefix_actions`/`prefix_weights` set to None this reduces exactly to
    `Pi0.sample_actions`, which makes it safe to use as a drop-in for the first inference
    of an episode.
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    # Fill the KV cache with a forward pass of the prefix. This is the expensive pass (the
    # 2B vision-language tower); it happens once and is *not* differentiated through, so
    # the RTC guidance below only pays for backward passes through the action expert.
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def velocity(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return model.action_out_proj(suffix_out[:, -model.action_horizon :])

    use_rtc = prefix_actions is not None and prefix_weights is not None

    def step(carry):
        x_t, time = carry

        if not use_rtc:
            return x_t + dt * velocity(x_t, time), time + dt

        # Rather than overwriting the frozen region of x_t, nudge the velocity so the
        # *predicted clean sample* moves toward the previous chunk, weighted per timestep.
        # This is what lets RTC work with a policy that was never trained for it.
        def denoiser(x):
            v = velocity(x, time)
            # time=1 is noise, time=0 is the target, so the clean prediction is x - time*v.
            return x - time * v, v

        x_clean, vjp_fn, v_t = jax.vjp(denoiser, x_t, has_aux=True)
        error = (prefix_actions - x_clean) * prefix_weights[None, :, None]
        correction = vjp_fn(error)[0]

        # Guidance weight, constants from the paper, mapped into our time convention.
        t_rtc = 1.0 - time
        inv_r2 = (t_rtc**2 + time**2) / (time**2)
        c = jnp.nan_to_num(time / t_rtc, posinf=max_guidance_weight)
        guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)

        # Sign flip: our velocity runs target -> noise, the paper's runs noise -> target.
        v_t = v_t - guidance_weight * correction

        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        # robust to floating-point error
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0
