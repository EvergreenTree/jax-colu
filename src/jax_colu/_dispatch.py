"""Backend dispatch for public activations."""

from __future__ import annotations

import functools
from collections.abc import Sequence

import jax

from jax_colu._reference import colu_reference, rcolu_reference

_GPU_BACKENDS = frozenset({"gpu", "cuda", "rocm"})
_TPU_BACKENDS = frozenset({"tpu"})
_METAL_BACKENDS = frozenset({"metal", "mps"})


def _backend() -> str:
    return jax.default_backend().lower()


def _is_scalar_dim(dim: object) -> bool:
    return not isinstance(dim, Sequence) or isinstance(dim, (str, bytes))


def _can_use_pallas_rcolu(
    *,
    backend: str,
    dim: object,
    num_groups: object,
    scaling: str,
    axis: int,
) -> bool:
    return (
        backend in (_GPU_BACKENDS | _TPU_BACKENDS)
        and scaling == "hard"
        and axis == -1
        and num_groups is None
        and _is_scalar_dim(dim)
    )


def _can_use_pallas_colu(
    *,
    backend: str,
    dim: object,
    num_groups: object,
    scaling: str,
    channel_axis: int,
) -> bool:
    return (
        backend in (_GPU_BACKENDS | _TPU_BACKENDS)
        and scaling == "hard"
        and channel_axis == -1
        and num_groups is None
        and _is_scalar_dim(dim)
    )


@functools.partial(
    jax.jit,
    static_argnames=["dim", "num_groups", "scaling", "axis", "eps"],
)
def rcolu(
    x,
    dim=4,
    num_groups=None,
    scaling="hard",
    axis=-1,
    eps=1e-7,
):
    """Homogeneous-axis Conic Linear Unit."""
    backend = _backend()
    if _can_use_pallas_rcolu(
        backend=backend,
        dim=dim,
        num_groups=num_groups,
        scaling=scaling,
        axis=axis,
    ):
        if backend in _GPU_BACKENDS:
            from jax_colu.gpu._rcolu import rcolu_gpu

            return rcolu_gpu(x, dim=dim, eps=eps)
        if backend in _TPU_BACKENDS:
            from jax_colu.tpu._rcolu import rcolu_tpu

            return rcolu_tpu(x, dim=dim, eps=eps)

    return rcolu_reference(
        x, dim=dim, num_groups=num_groups, scaling=scaling, axis=axis, eps=eps
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "channel_axis",
        "scaling",
        "eps",
        "num_groups",
        "dim",
        "share_axis",
    ],
)
def colu(
    x,
    channel_axis=-1,
    scaling="hard",
    eps=1e-7,
    num_groups=None,
    dim=4,
    share_axis=False,
):
    """Explicit-axis Conic Linear Unit."""
    backend = _backend()
    if _can_use_pallas_colu(
        backend=backend,
        dim=dim,
        num_groups=num_groups,
        scaling=scaling,
        channel_axis=channel_axis,
    ):
        if backend in _GPU_BACKENDS:
            from jax_colu.gpu._colu import colu_gpu

            return colu_gpu(x, dim=dim, eps=eps, share_axis=share_axis)
        if backend in _TPU_BACKENDS:
            from jax_colu.tpu._colu import colu_tpu

            return colu_tpu(x, dim=dim, eps=eps, share_axis=share_axis)

    return colu_reference(
        x,
        channel_axis=channel_axis,
        scaling=scaling,
        eps=eps,
        num_groups=num_groups,
        dim=dim,
        share_axis=share_axis,
    )
