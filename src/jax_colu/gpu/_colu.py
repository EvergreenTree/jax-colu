"""Pallas implementation of explicit-axis CoLU for GPU-like backends."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _fwd_kernel(
    x_ref,
    o_ref,
    t_ref,
    r_ref,
    sc_ref,
    *,
    C: int,
    G: int,
    S: int,
    share_axis: bool,
    eps: float,
):
    x = x_ref[0, :].astype(jnp.float32)
    t_size = 1 if share_axis else G
    body = x[t_size:].reshape((G, S - 1))
    t_in = x[0] + jnp.zeros((G,), dtype=jnp.float32) if share_axis else x[:G]

    r = jnp.sqrt(jnp.sum(body * body, axis=1))
    t_pos = jnp.maximum(t_in, 0.0)
    sc = jnp.clip(t_pos / (r + eps), 0.0, 1.0)
    body_out = sc[:, None] * body
    t_out = jnp.maximum(x[:t_size], 0.0)
    y = jnp.concatenate([t_out, body_out.reshape((G * (S - 1),))], axis=0)

    o_ref[0, :] = y.astype(o_ref.dtype)
    t_ref[0, :] = t_in
    r_ref[0, :] = r
    sc_ref[0, :] = sc


def _bwd_kernel(
    x_ref,
    t_ref,
    r_ref,
    sc_ref,
    go_ref,
    gx_ref,
    *,
    C: int,
    G: int,
    S: int,
    share_axis: bool,
    eps: float,
):
    x = x_ref[0, :].astype(jnp.float32)
    go = go_ref[0, :].astype(jnp.float32)
    t_size = 1 if share_axis else G

    t_in = t_ref[0, :]
    r = r_ref[0, :]
    sc = sc_ref[0, :]
    body = x[t_size:].reshape((G, S - 1))
    go_t = go[:t_size]
    go_body = go[t_size:].reshape((G, S - 1))

    dot_go_w = jnp.sum(go_body * body, axis=1)
    den = r + eps
    r_safe = jnp.maximum(r, eps)
    active = (t_in > 0.0) & (sc < 1.0)
    saturated = (t_in > 0.0) & (sc >= 1.0)

    body_active = sc[:, None] * go_body - (
        t_in * dot_go_w / (den * den * r_safe)
    )[:, None] * body
    body_grad = jnp.where(
        saturated[:, None],
        go_body,
        jnp.where(active[:, None], body_active, jnp.zeros_like(go_body)),
    )

    t_body_grad = jnp.where(active, dot_go_w / den, jnp.zeros_like(dot_go_w))
    if share_axis:
        t_direct = jnp.where(x[0] > 0.0, go_t[0], 0.0)
        t_grad = jnp.array([t_direct + jnp.sum(t_body_grad)], dtype=jnp.float32)
    else:
        t_direct = jnp.where(x[:G] > 0.0, go_t, jnp.zeros_like(go_t))
        t_grad = t_direct + t_body_grad

    gx = jnp.concatenate([t_grad, body_grad.reshape((G * (S - 1),))], axis=0)
    gx_ref[0, :] = gx.astype(gx_ref.dtype)


def _tile(C: int, G: int):
    row = pl.BlockSpec((1, C), lambda i: (i, 0))
    scalars = pl.BlockSpec((1, G), lambda i: (i, 0))
    return row, scalars


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def colu_gpu(
    x: jax.Array,
    dim: int = 4,
    eps: float = 1e-7,
    share_axis: bool = False,
) -> jax.Array:
    """Fused hard-scaling CoLU for Pallas GPU backends."""
    return _fwd(x, dim, eps, share_axis)[0]


def _shape_params(x: jax.Array, dim: int, share_axis: bool) -> tuple[int, int, int]:
    if dim <= 1:
        raise ValueError("`dim` must be greater than 1 for `colu_gpu`.")
    C = x.shape[-1]
    stride = C - 1 if share_axis else C
    inner = dim - (1 if share_axis else 0)
    if inner <= 0 or stride % inner != 0:
        raise ValueError(
            f"last dimension {C} is incompatible with dim={dim}, "
            f"share_axis={share_axis}."
        )
    G = stride // inner
    return C, G, int(dim)


def _fwd(x: jax.Array, dim: int, eps: float, share_axis: bool):
    C, G, S = _shape_params(x, int(dim), bool(share_axis))
    outer = x.size // C
    x_flat = x.reshape(outer, C)
    row, scalars = _tile(C, G)

    o, t, r, sc = pl.pallas_call(
        functools.partial(
            _fwd_kernel, C=C, G=G, S=S, share_axis=bool(share_axis), eps=eps
        ),
        out_shape=[
            jax.ShapeDtypeStruct((outer, C), x.dtype),
            jax.ShapeDtypeStruct((outer, G), jnp.float32),
            jax.ShapeDtypeStruct((outer, G), jnp.float32),
            jax.ShapeDtypeStruct((outer, G), jnp.float32),
        ],
        in_specs=[row],
        out_specs=[row, scalars, scalars, scalars],
        grid=(outer,),
    )(x_flat)
    return o.reshape(x.shape), (x_flat, t, r, sc, x.shape)


def _bwd(dim: int, eps: float, share_axis: bool, residuals, g: jax.Array):
    x_flat, t, r, sc, shape = residuals
    C, G, S = _shape_params(g, int(dim), bool(share_axis))
    outer = x_flat.shape[0]
    g_flat = g.reshape(outer, C)
    row, scalars = _tile(C, G)

    gx = pl.pallas_call(
        functools.partial(
            _bwd_kernel, C=C, G=G, S=S, share_axis=bool(share_axis), eps=eps
        ),
        out_shape=jax.ShapeDtypeStruct((outer, C), g.dtype),
        in_specs=[row, scalars, scalars, scalars, row],
        out_specs=row,
        grid=(outer,),
    )(x_flat, t, r, sc, g_flat)
    return (gx.reshape(shape),)


colu_gpu.defvjp(_fwd, _bwd)
