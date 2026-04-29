"""TPU entry point for rCoLU."""

from __future__ import annotations

import jax

from jax_colu._reference import rcolu_reference


def rcolu_tpu(x: jax.Array, dim: int = 4, eps: float = 1e-7) -> jax.Array:
    return rcolu_reference(x, dim=dim, eps=eps)
