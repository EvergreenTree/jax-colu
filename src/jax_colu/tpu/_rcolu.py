"""TPU entry point for rCoLU.

The kernel body is shared with the generic Pallas implementation. The public
function is split out so dispatch has a stable TPU import path.
"""

from __future__ import annotations

import jax

from jax_colu.gpu._rcolu import rcolu_gpu


def rcolu_tpu(x: jax.Array, dim: int = 4, eps: float = 1e-7) -> jax.Array:
    return rcolu_gpu(x, dim=dim, eps=eps)
