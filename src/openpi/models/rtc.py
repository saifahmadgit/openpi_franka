"""Real-time chunking (RTC) helpers.

Implements the prefix-attention weighting from "Real-Time Execution of Action Chunking
Flow Policies" (Black, Galliker & Levine, 2025). The guidance itself lives in
`Pi0.sample_actions`; this module only builds the per-timestep weight vector, which is
computed on the host so that changing `inference_delay` does not retrigger a JIT
recompile of the sampler.

Ported from the reference implementation at
https://github.com/Physical-Intelligence/real-time-chunking-kinetix (`src/model.py`).
"""

from typing import Literal, TypeAlias

import numpy as np

PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]


def get_prefix_weights(
    start: int, end: int, total: int, schedule: PrefixAttentionSchedule = "exp"
) -> np.ndarray:
    """With start=2, end=6, total=10, the (linear) output is:

        1  1  4/5 3/5 2/5 1/5 0  0  0  0
               ^              ^
             start           end

    `start` (inclusive) is where the chunk starts being allowed to change; below it the
    weight is 1, which is the frozen region. `end` (exclusive) is where the chunk stops
    paying attention to the prefix at all. If start == 0 the whole chunk may change; if
    end == total the entire prefix is attended to.

    `end` takes precedence over `start`: if end < start, start is pushed down to end, so
    end == 0 ignores the prefix entirely.
    """
    start = min(start, end)
    idx = np.arange(total)
    if schedule == "ones":
        w = np.ones(total, dtype=np.float32)
    elif schedule == "zeros":
        w = (idx < start).astype(np.float32)
    elif schedule in ("linear", "exp"):
        w = np.clip((start - 1 - idx) / (end - start + 1) + 1, 0, 1).astype(np.float32)
        if schedule == "exp":
            w = w * np.expm1(w) / (np.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return np.where(idx >= end, 0.0, w).astype(np.float32)
