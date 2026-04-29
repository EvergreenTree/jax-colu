"""Shared Pallas kernel bodies for rCoLU.

Both the GPU (Triton) and TPU (Mosaic) backends lower the same closed-form
math; only the tile geometry chosen by each backend wrapper differs.

Closed form (hard scaling, last axis = cone width S):

    inv_s = 1 / sqrt(S)
    t   = sum(x)        * inv_s
    r^2 = max(sum(x*x) - t*t, 0)
    w   = x - t * inv_s
    sc  = clip(relu(t) / (r + eps), 0, 1)
    y   = relu(t) * inv_s + sc * w

Backward closed form (mirrors `_rcolu_hard_bwd` in `_reference.py`):

    g_t = sum(go) * inv_s
    g_w = go - g_t * inv_s
    a   = <g_w, w>
    den = r + eps
    active    = (t > 0) & (sc < 1)
    saturated = (t > 0) & (sc >= 1)
    g_t_active = g_t + a / den
    g_w_active = sc * g_w - (t * a / (den^2 * max(r, eps))) * w
    g_x_active = g_t_active * inv_s + g_w_active
    gx = where(saturated, go, where(active, g_x_active, 0))

Both kernels recompute t, r, sc from x in the bwd path so the only residual
is x itself. That halves residual memory and avoids storing scalar-per-row
buffers whose width-1 last dim plays badly with TPU lane geometry.
"""

from __future__ import annotations

import math

import jax.numpy as jnp


def rcolu_fwd_kernel(x_ref, o_ref, *, S: int, eps: float) -> None:
    x = x_ref[...].astype(jnp.float32)
    inv_s = 1.0 / math.sqrt(float(S))
    sum_x = jnp.sum(x, axis=-1, keepdims=True)
    sum_xx = jnp.sum(x * x, axis=-1, keepdims=True)
    t = sum_x * inv_s
    r = jnp.sqrt(jnp.maximum(sum_xx - t * t, 0.0))
    w = x - t * inv_s
    t_pos = jnp.maximum(t, 0.0)
    sc = jnp.minimum(t_pos / (r + eps), 1.0)
    o_ref[...] = (t_pos * inv_s + sc * w).astype(o_ref.dtype)


def rcolu_bwd_kernel(x_ref, go_ref, gx_ref, *, S: int, eps: float) -> None:
    x = x_ref[...].astype(jnp.float32)
    go = go_ref[...].astype(jnp.float32)
    inv_s = 1.0 / math.sqrt(float(S))

    sum_x = jnp.sum(x, axis=-1, keepdims=True)
    sum_xx = jnp.sum(x * x, axis=-1, keepdims=True)
    t = sum_x * inv_s
    r = jnp.sqrt(jnp.maximum(sum_xx - t * t, 0.0))
    w = x - t * inv_s
    t_pos = jnp.maximum(t, 0.0)
    sc = jnp.minimum(t_pos / (r + eps), 1.0)

    go_t = jnp.sum(go, axis=-1, keepdims=True) * inv_s
    go_w = go - go_t * inv_s
    dot_go_w = jnp.sum(go_w * w, axis=-1, keepdims=True)
    den = r + eps
    r_safe = jnp.maximum(r, eps)

    active = (t > 0.0) & (sc < 1.0)
    saturated = (t > 0.0) & (sc >= 1.0)
    gt_active = go_t + dot_go_w / den
    gw_active = sc * go_w - (t * dot_go_w / (den * den * r_safe)) * w
    gx_active = gt_active * inv_s + gw_active
    gx = jnp.where(saturated, go, jnp.where(active, gx_active, jnp.zeros_like(go)))
    gx_ref[...] = gx.astype(gx_ref.dtype)
