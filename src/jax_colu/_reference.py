"""Pure-JAX implementations of CoLU and rCoLU.

These functions are the CPU and Apple Metal/MPS fallback path. They also serve
as the correctness oracle for backend kernels.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

_SCALINGS = frozenset({"hard", "soft", "sqrt", "log"})


def _validate_scaling(scaling: str) -> None:
    if scaling not in _SCALINGS:
        raise ValueError(
            f"Unknown scaling {scaling!r}; expected one of {sorted(_SCALINGS)}."
        )


def _normalise_axis(ndim: int, axis: int) -> int:
    """Convert an axis to its canonical negative form."""
    if ndim == 0:
        return -1
    if axis >= 0:
        axis -= ndim
    if not (-ndim <= axis < 0):
        raise ValueError(f"axis {axis} out of range for ndim {ndim}")
    return axis


def _resolve_groups(
    C: int,
    dim: Optional[int],
    num_groups: Optional[int],
    share_axis: bool,
) -> tuple[int, int]:
    """Return ``(num_groups, dim)`` given exactly one of dim/num_groups."""
    if (dim is None) == (num_groups is None):
        raise ValueError("Specify exactly one of `dim` and `num_groups`.")

    stride = C - 1 if share_axis else C
    if stride <= 0:
        raise ValueError("The channel dimension is too small for the requested cone.")

    if num_groups is not None:
        if num_groups <= 0:
            raise ValueError("`num_groups` must be positive.")
        if stride % num_groups != 0:
            raise ValueError(
                f"channel stride {stride} is not divisible by num_groups={num_groups}."
            )
        inner = stride // num_groups
        return num_groups, inner + (1 if share_axis else 0)

    if dim is None or dim <= 0:
        raise ValueError("`dim` must be positive.")
    inner = dim - (1 if share_axis else 0)
    if inner <= 0:
        raise ValueError("`dim` is too small for the requested cone layout.")
    if stride % inner != 0:
        raise ValueError(f"channel stride {stride} is not divisible by dim={dim}.")
    return stride // inner, dim


def _as_dim(dim: int | Sequence[int] | None) -> int | None:
    if isinstance(dim, Sequence) and not isinstance(dim, (str, bytes)):
        raise ValueError("Mixed cone dimensions are not supported; pass one integer.")
    return dim


def _apply_scale(raw: jax.Array, scaling: str) -> jax.Array:
    """Transform the raw ratio according to the selected scaling mode."""
    _validate_scaling(scaling)
    if scaling == "hard":
        return raw.clip(0.0, 1.0)
    if scaling == "soft":
        return jax.nn.sigmoid(raw - 0.5)
    if scaling == "sqrt":
        return jnp.sqrt(jnp.maximum(raw, 0.0))
    return jnp.log1p(jnp.maximum(raw, 0.0))


def _axis_activation(t: jax.Array, scaling: str) -> jax.Array:
    # Soft mode intentionally uses SiLU, which is negative for negative t.
    # Hard mode is the ReLU/cone-clipping path; sqrt/log clamp only because
    # their scale transforms are defined on nonnegative ratios.
    if scaling == "soft":
        return jax.nn.silu(t)
    if scaling == "hard":
        return jax.nn.relu(t)
    return jnp.maximum(t, 0.0)


def _tuple_sum_and_sum_squares(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    zero = jnp.array(0, dtype=x.dtype)

    def reducer(a, b):
        return (a[0] + b[0], a[1] + b[1])

    return lax.reduce((x, x * x), (zero, zero), reducer, dimensions=(x.ndim - 1,))


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _rcolu_hard_inner(x: jax.Array, eps: float) -> jax.Array:
    return _rcolu_hard_fwd(x, eps)[0]


def _rcolu_hard_fwd(x: jax.Array, eps: float):
    out_dtype = x.dtype
    if x.dtype in (jnp.float16, jnp.bfloat16):
        x = x.astype(jnp.float32)

    S = x.shape[-1]
    inv_s = 1.0 / math.sqrt(float(S))

    sum_x, sum_xx = _tuple_sum_and_sum_squares(x)
    t = sum_x[..., None] * inv_s
    r2 = jnp.maximum(sum_xx[..., None] - t * t, 0.0)
    r = jnp.sqrt(r2)
    w = x - t * inv_s
    t_out = jnp.maximum(t, 0.0)
    sc = jnp.minimum(t_out / (r + eps), 1.0)
    return (t_out * inv_s + sc * w).astype(out_dtype), (x, t, r, sc)


def _rcolu_hard_bwd(eps: float, residuals, go: jax.Array):
    x, t, r, sc = residuals
    go_dtype = go.dtype
    go = go.astype(x.dtype)
    S = x.shape[-1]
    inv_s = 1.0 / math.sqrt(float(S))
    w = x - t * inv_s

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
    return (gx.astype(go_dtype),)


_rcolu_hard_inner.defvjp(_rcolu_hard_fwd, _rcolu_hard_bwd)


def _rcolu_inner(x: jax.Array, eps: float, scaling: str) -> jax.Array:
    """Apply rCoLU to groups exposed on the last dimension."""
    if scaling == "hard":
        return _rcolu_hard_inner(x, eps)

    S = x.shape[-1]
    inv_s = 1.0 / math.sqrt(float(S))
    t = jnp.sum(x, axis=-1, keepdims=True) * inv_s
    w = x - t * inv_s
    r = jnp.linalg.norm(w, axis=-1, keepdims=True)
    t_out = _axis_activation(t, scaling)
    sc = _apply_scale(t_out / (r + eps), scaling)
    return t_out * inv_s + sc * w


@functools.partial(
    jax.jit,
    static_argnames=["dim", "num_groups", "scaling", "axis", "eps"],
)
def rcolu_reference(
    x: jax.Array,
    dim: Optional[int] = 4,
    num_groups: Optional[int] = None,
    scaling: str = "hard",
    axis: int = -1,
    eps: float = 1e-7,
) -> jax.Array:
    """Homogeneous-axis Conic Linear Unit.

    The target axis is partitioned into non-overlapping groups of ``dim``
    elements. Each group is clipped to the cone around the homogeneous axis
    ``ones / sqrt(dim)``.
    """
    _validate_scaling(scaling)
    dim = _as_dim(dim)
    if x.ndim == 0:
        return _axis_activation(x, scaling)

    axis = _normalise_axis(x.ndim, axis)
    G, S = _resolve_groups(x.shape[axis], dim, num_groups, share_axis=False)
    x_ = jnp.moveaxis(x, axis, -1)
    x_ = x_.reshape(x_.shape[:-1] + (G, S))
    y_ = _rcolu_inner(x_, eps, scaling)
    y_ = y_.reshape(x_.shape[:-2] + (G * S,))
    return jnp.moveaxis(y_, -1, axis)


@functools.partial(
    jax.jit,
    static_argnames=["channel_axis", "scaling", "eps", "num_groups", "dim", "share_axis"],
)
def colu_reference(
    x: jax.Array,
    channel_axis: int = -1,
    scaling: str = "hard",
    eps: float = 1e-7,
    num_groups: Optional[int] = None,
    dim: Optional[int] = 4,
    share_axis: bool = False,
) -> jax.Array:
    """Explicit-axis Conic Linear Unit."""
    _validate_scaling(scaling)
    dim = _as_dim(dim)
    if x.ndim == 0:
        return _axis_activation(x, scaling)

    channel_axis = _normalise_axis(x.ndim, channel_axis)
    C = x.shape[channel_axis]
    G, S = _resolve_groups(C, dim, num_groups, share_axis)
    t_size = 1 if share_axis else G

    if S == 1:
        return _axis_activation(x, scaling)

    t, w = jnp.split(x, [t_size], axis=channel_axis)
    axis_pos = channel_axis % x.ndim
    prefix = x.shape[:axis_pos]
    suffix = x.shape[axis_pos + 1 :]
    t_group_count = 1 if share_axis else G

    t = t.reshape(prefix + (t_group_count, 1) + suffix)
    w = w.reshape(prefix + (G, S - 1) + suffix)
    body_axis = len(prefix) + 1
    r = jnp.linalg.norm(w, axis=body_axis, keepdims=True)

    t_out = _axis_activation(t, scaling)
    sc = _apply_scale(t_out / (r + eps), scaling)
    w = sc * w

    t = t_out.reshape(prefix + (t_size,) + suffix)
    w = w.reshape(prefix + (G * (S - 1),) + suffix)
    return jnp.concatenate([t, w], axis=channel_axis)
