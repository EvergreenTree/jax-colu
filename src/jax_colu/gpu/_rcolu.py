"""Pallas rCoLU kernel for NVIDIA GPU backends (Triton lowering)."""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from jax_colu._kernel_math import rcolu_bwd_kernel, rcolu_fwd_kernel

_BLOCK_CANDIDATES = (256, 128, 64, 32, 16, 8, 4, 2, 1)


def _pick_block_m(NG: int, S: int) -> int:
    env = os.environ.get("JAX_COLU_BLOCK", "").strip()
    if env:
        try:
            request = int(env)
        except ValueError:
            request = 0
        if request > 0 and NG % request == 0:
            return request
    target = max(_BLOCK_CANDIDATES[-1], min(_BLOCK_CANDIDATES[0], 1024 // max(S, 1)))
    for candidate in _BLOCK_CANDIDATES:
        if candidate <= target and NG % candidate == 0:
            return candidate
    return 1


def _tile(BM: int, S: int) -> pl.BlockSpec:
    return pl.BlockSpec((BM, S), lambda i: (i, 0))


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def rcolu_gpu(x: jax.Array, dim: int = 4, eps: float = 1e-7) -> jax.Array:
    """Fused hard-scaling rCoLU for NVIDIA GPU backends."""
    return _fwd(x, dim, eps)[0]


def _fwd(x: jax.Array, dim: int, eps: float):
    if dim <= 0:
        raise ValueError("`dim` must be positive.")
    if x.size % dim != 0:
        raise ValueError(f"input size {x.size} is not divisible by dim={dim}.")

    S = int(dim)
    NG = x.size // S
    BM = _pick_block_m(NG, S)
    x_flat = x.reshape(NG, S)
    group = _tile(BM, S)

    o = pl.pallas_call(
        functools.partial(rcolu_fwd_kernel, S=S, eps=eps),
        out_shape=jax.ShapeDtypeStruct((NG, S), x.dtype),
        in_specs=[group],
        out_specs=group,
        grid=(NG // BM,),
    )(x_flat)
    return o.reshape(x.shape), (x_flat,)


def _bwd(dim: int, eps: float, residuals, g: jax.Array):
    (x_flat,) = residuals
    S = int(dim)
    NG = x_flat.shape[0]
    BM = _pick_block_m(NG, S)
    g_flat = g.reshape(NG, S)
    group = _tile(BM, S)

    gx = pl.pallas_call(
        functools.partial(rcolu_bwd_kernel, S=S, eps=eps),
        out_shape=jax.ShapeDtypeStruct((NG, S), g.dtype),
        in_specs=[group, group],
        out_specs=group,
        grid=(NG // BM,),
    )(x_flat, g_flat)
    return (gx.reshape(g.shape),)


rcolu_gpu.defvjp(_fwd, _bwd)
