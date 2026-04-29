"""Pallas rCoLU kernel for TPU backends (Mosaic lowering).

The TPU MXU/VPU prefers tiles whose leading dim is a multiple of the 8-row
sublane group and whose lane dim aligns to 128. We choose ``BM`` (rows of
cone groups per program) as a multiple of 8 and pad the count of cone
groups ``NG`` up to that multiple with zeros (zero input maps to zero
output under hard scaling). The cone width ``S`` is taken as-is — small
``S`` simply leaves trailing lanes unused, which is acceptable when the
multi-row block keeps each program busy.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from jax_colu._kernel_math import rcolu_bwd_kernel, rcolu_fwd_kernel

_SUBLANE = 8
_BLOCK_CANDIDATES = (512, 256, 128, 64, 32, 16, 8)


def _pick_block_m(NG: int, S: int) -> int:
    env = os.environ.get("JAX_COLU_BLOCK", "").strip()
    if env:
        try:
            request = int(env)
        except ValueError:
            request = 0
        if request >= _SUBLANE and request % _SUBLANE == 0:
            return request
    target = max(_SUBLANE, min(_BLOCK_CANDIDATES[0], 4096 // max(S, 1)))
    for candidate in _BLOCK_CANDIDATES:
        if candidate <= target:
            return candidate
    return _SUBLANE


def _pad_to_block(x_flat: jax.Array, BM: int) -> tuple[jax.Array, int]:
    NG = x_flat.shape[0]
    rem = NG % BM
    if rem == 0:
        return x_flat, NG
    pad = BM - rem
    return jnp.pad(x_flat, ((0, pad), (0, 0))), NG + pad


def _tile(BM: int, S: int) -> pl.BlockSpec:
    return pl.BlockSpec((BM, S), lambda i: (i, 0))


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def rcolu_tpu(x: jax.Array, dim: int = 4, eps: float = 1e-7) -> jax.Array:
    """Fused hard-scaling rCoLU for TPU backends."""
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
    x_padded, NG_padded = _pad_to_block(x_flat, BM)
    group = _tile(BM, S)

    o_padded = pl.pallas_call(
        functools.partial(rcolu_fwd_kernel, S=S, eps=eps),
        out_shape=jax.ShapeDtypeStruct((NG_padded, S), x.dtype),
        in_specs=[group],
        out_specs=group,
        grid=(NG_padded // BM,),
    )(x_padded)
    o = o_padded[:NG] if NG_padded != NG else o_padded
    return o.reshape(x.shape), (x_flat,)


def _bwd(dim: int, eps: float, residuals, g: jax.Array):
    (x_flat,) = residuals
    S = int(dim)
    NG = x_flat.shape[0]
    BM = _pick_block_m(NG, S)
    g_flat = g.reshape(NG, S)
    x_padded, NG_padded = _pad_to_block(x_flat, BM)
    g_padded, _ = _pad_to_block(g_flat, BM)
    group = _tile(BM, S)

    gx_padded = pl.pallas_call(
        functools.partial(rcolu_bwd_kernel, S=S, eps=eps),
        out_shape=jax.ShapeDtypeStruct((NG_padded, S), g.dtype),
        in_specs=[group, group],
        out_specs=group,
        grid=(NG_padded // BM,),
    )(x_padded, g_padded)
    gx = gx_padded[:NG] if NG_padded != NG else gx_padded
    return (gx.reshape(g.shape),)


rcolu_tpu.defvjp(_fwd, _bwd)
