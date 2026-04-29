"""Pallas implementation of rCoLU for GPU-like backends."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _fwd_kernel(x_ref, o_ref, t_ref, r_ref, sc_ref, *, S: int, eps: float):
    x = x_ref[0, :].astype(jnp.float32)
    inv_s = 1.0 / math.sqrt(float(S))
    e = jnp.full((S,), inv_s, dtype=jnp.float32)

    t = jnp.sum(x) * inv_s
    w = x - t * e
    r = jnp.sqrt(jnp.sum(w * w))
    t_pos = jnp.maximum(t, 0.0)
    sc = jnp.minimum(t_pos / (r + eps), 1.0)

    o_ref[0, :] = (t_pos * e + sc * w).astype(o_ref.dtype)
    t_ref[0, 0] = t
    r_ref[0, 0] = r
    sc_ref[0, 0] = sc


def _bwd_kernel(x_ref, t_ref, r_ref, sc_ref, go_ref, gx_ref, *, S: int, eps: float):
    x = x_ref[0, :].astype(jnp.float32)
    t = t_ref[0, 0]
    r = r_ref[0, 0]
    sc = sc_ref[0, 0]
    go = go_ref[0, :].astype(jnp.float32)

    inv_s = 1.0 / math.sqrt(float(S))
    e = jnp.full((S,), inv_s, dtype=jnp.float32)
    w = x - t * e
    go_t = jnp.sum(go * e)
    go_w = go - go_t * e
    dot_go_w = jnp.sum(go_w * w)
    den = r + eps
    r_safe = jnp.maximum(r, eps)

    active = (t > 0.0) & (sc < 1.0)
    saturated = (t > 0.0) & (sc >= 1.0)

    gt_active = go_t + dot_go_w / den
    gw_active = sc * go_w - (t * dot_go_w / (den * den * r_safe)) * w
    gx_active = gt_active * e + gw_active
    gx = jnp.where(saturated, go, jnp.where(active, gx_active, jnp.zeros_like(go)))
    gx_ref[0, :] = gx.astype(gx_ref.dtype)


def _tile(S: int):
    group = pl.BlockSpec((1, S), lambda i: (i, 0))
    scalar = pl.BlockSpec((1, 1), lambda i: (i, 0))
    return group, scalar


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def rcolu_gpu(x: jax.Array, dim: int = 4, eps: float = 1e-7) -> jax.Array:
    """Fused hard-scaling rCoLU for Pallas GPU backends."""
    return _fwd(x, dim, eps)[0]


def _fwd(x: jax.Array, dim: int, eps: float):
    if dim <= 0:
        raise ValueError("`dim` must be positive.")
    if x.size % dim != 0:
        raise ValueError(f"input size {x.size} is not divisible by dim={dim}.")

    shape = x.shape
    S = int(dim)
    NG = x.size // S
    x_flat = x.reshape(NG, S)
    group, scalar = _tile(S)

    o, t, r, sc = pl.pallas_call(
        functools.partial(_fwd_kernel, S=S, eps=eps),
        out_shape=[
            jax.ShapeDtypeStruct((NG, S), x.dtype),
            jax.ShapeDtypeStruct((NG, 1), jnp.float32),
            jax.ShapeDtypeStruct((NG, 1), jnp.float32),
            jax.ShapeDtypeStruct((NG, 1), jnp.float32),
        ],
        in_specs=[group],
        out_specs=[group, scalar, scalar, scalar],
        grid=(NG,),
    )(x_flat)
    return o.reshape(shape), (x_flat, t, r, sc)


def _bwd(dim: int, eps: float, residuals, g: jax.Array):
    x_flat, t, r, sc = residuals
    S = int(dim)
    NG = x_flat.shape[0]
    g_flat = g.reshape(NG, S)
    group, scalar = _tile(S)

    gx = pl.pallas_call(
        functools.partial(_bwd_kernel, S=S, eps=eps),
        out_shape=jax.ShapeDtypeStruct((NG, S), g.dtype),
        in_specs=[group, scalar, scalar, scalar, group],
        out_specs=group,
        grid=(NG,),
    )(x_flat, t, r, sc, g_flat)
    return (gx.reshape(g.shape),)


rcolu_gpu.defvjp(_fwd, _bwd)
