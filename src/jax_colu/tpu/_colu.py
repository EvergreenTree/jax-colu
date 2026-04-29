"""TPU entry point for explicit-axis CoLU."""

from __future__ import annotations

import jax

from jax_colu.gpu._colu import colu_gpu


def colu_tpu(
    x: jax.Array,
    dim: int = 4,
    eps: float = 1e-7,
    share_axis: bool = False,
) -> jax.Array:
    return colu_gpu(x, dim=dim, eps=eps, share_axis=share_axis)
