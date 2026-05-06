"""Pallas:Mosaic GPU implementation of rCoLU for Hopper+ GPUs."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


_MGPU_BLOCK = 128


def _sqrt_newton(x):
    y = jnp.maximum(x, 1.0)
    for _ in range(20):
        y = 0.5 * (y + x / y)
    return jnp.where(x > 0.0, y, 0.0)


def _fwd_kernel(x_ref, e_ref, o_ref, t_ref, r_ref, sc_ref, *, P: int, eps: float):
    x = x_ref[0, :].astype(jnp.float32)
    e = e_ref[0, :].astype(jnp.float32)

    t = jnp.sum(x * e)
    w = x - t * e
    r = _sqrt_newton(jnp.sum(w * w))
    t_pos = jnp.maximum(t, 0.0)
    sc = jnp.minimum(t_pos / (r + eps), 1.0)

    o_ref[0, :] = (t_pos * e + sc * w).astype(o_ref.dtype)
    t_ref[0, :] = jnp.full((P,), t, dtype=jnp.float32)
    r_ref[0, :] = jnp.full((P,), r, dtype=jnp.float32)
    sc_ref[0, :] = jnp.full((P,), sc, dtype=jnp.float32)


def _bwd_kernel(
    x_ref, e_ref, t_ref, r_ref, sc_ref, go_ref, gx_ref, *, P: int, eps: float
):
    x = x_ref[0, :].astype(jnp.float32)
    e = e_ref[0, :].astype(jnp.float32)
    t = t_ref[0, 0]
    r = r_ref[0, 0]
    sc = sc_ref[0, 0]
    go = go_ref[0, :].astype(jnp.float32)

    w = x - t * e
    go_t = jnp.sum(go * e)
    go_w = go - go_t * e
    dot_go_w = jnp.sum(go_w * w)
    den = r + eps
    r_safe = jnp.maximum(r, eps)

    active = (t_ref[0, :] > 0.0) & (sc_ref[0, :] < 1.0)
    saturated = (t_ref[0, :] > 0.0) & (sc_ref[0, :] >= 1.0)

    gt_active = go_t + dot_go_w / den
    gw_active = sc * go_w - (t * dot_go_w / (den * den * r_safe)) * w
    gx_active = gt_active * e + gw_active
    gx = jnp.where(saturated, go, jnp.where(active, gx_active, jnp.zeros_like(go)))
    gx_ref[0, :] = gx.astype(gx_ref.dtype)


def _tile(S: int):
    group = pl.BlockSpec((1, S), lambda i: (i, 0))
    scalar = pl.BlockSpec((1, S), lambda i: (i, 0))
    const = pl.BlockSpec((1, S), lambda i: (0, 0))
    return group, scalar, const


def _compiler_params() -> plgpu.CompilerParams:
    return plgpu.CompilerParams(
        reduction_scratch_bytes=6144,
        lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def rcolu_mgpu(x: jax.Array, dim: int = 4, eps: float = 1e-7) -> jax.Array:
    """Fused hard-scaling rCoLU for the Pallas:Mosaic GPU backend."""
    return _fwd(x, dim, eps)[0]


def _fwd(x: jax.Array, dim: int, eps: float):
    if dim <= 0:
        raise ValueError("`dim` must be positive.")
    if x.size % dim != 0:
        raise ValueError(f"input size {x.size} is not divisible by dim={dim}.")

    shape = x.shape
    S = int(dim)
    P = max(_MGPU_BLOCK, S)
    NG = x.size // S
    x_flat = jnp.pad(x.reshape(NG, S), ((0, 0), (0, P - S)))
    inv_s = 1.0 / math.sqrt(float(S))
    e = jnp.pad(jnp.full((1, S), inv_s, dtype=jnp.float32), ((0, 0), (0, P - S)))
    group, scalar, const = _tile(P)

    o, t, r, sc = pl.pallas_call(
        functools.partial(_fwd_kernel, P=P, eps=eps),
        out_shape=[
            jax.ShapeDtypeStruct((NG, P), x.dtype),
            jax.ShapeDtypeStruct((NG, P), jnp.float32),
            jax.ShapeDtypeStruct((NG, P), jnp.float32),
            jax.ShapeDtypeStruct((NG, P), jnp.float32),
        ],
        in_specs=[group, const],
        out_specs=[group, scalar, scalar, scalar],
        grid=(NG,),
        compiler_params=_compiler_params(),
    )(x_flat, e)
    return o[:, :S].reshape(shape), (x_flat, e, t, r, sc)


def _bwd(dim: int, eps: float, residuals, g: jax.Array):
    x_flat, e, t, r, sc = residuals
    S = int(dim)
    P = x_flat.shape[1]
    NG = x_flat.shape[0]
    g_flat = jnp.pad(g.reshape(NG, S), ((0, 0), (0, P - S)))
    group, scalar, const = _tile(P)

    gx = pl.pallas_call(
        functools.partial(_bwd_kernel, P=P, eps=eps),
        out_shape=jax.ShapeDtypeStruct((NG, P), g.dtype),
        in_specs=[group, const, scalar, scalar, scalar, group],
        out_specs=group,
        grid=(NG,),
        compiler_params=_compiler_params(),
    )(x_flat, e, t, r, sc, g_flat)
    return (gx[:, :S].reshape(g.shape),)


rcolu_mgpu.defvjp(_fwd, _bwd)
