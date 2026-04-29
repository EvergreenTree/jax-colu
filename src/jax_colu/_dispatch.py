"""Backend dispatch for public activations."""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence

import jax

from jax_colu._reference import colu_reference, rcolu_reference

_GPU_BACKENDS = frozenset({"gpu", "cuda", "rocm"})
_TPU_BACKENDS = frozenset({"tpu"})

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Public dispatch only routes to the GPU Pallas kernel on Ampere-or-newer
# NVIDIA devices; older NVIDIA generations and non-NVIDIA GPUs fall back to
# the JAX reference path. Users can override either way with
# JAX_COLU_FORCE_PALLAS / JAX_COLU_DISABLE_PALLAS.
_AMPERE_PLUS_TOKENS = (
    "a100", "a800", "a10", "a16", "a2", "a30", "a40",
    "l4", "l40",
    "rtx 30", "rtx 40", "rtx 50", "rtx a", "ada",
    "h100", "h200", "hopper",
    "b100", "b200", "gb200", "blackwell",
)


def _backend() -> str:
    return jax.default_backend().lower()


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _is_scalar_dim(dim: object) -> bool:
    return not isinstance(dim, Sequence) or isinstance(dim, (str, bytes))


def _normalise_device_kind(device_kind: str) -> str:
    return f" {device_kind.lower().replace('-', ' ')} "


def _is_ampere_plus(device_kind: str) -> bool:
    haystack = _normalise_device_kind(device_kind)
    return any(token in haystack for token in _AMPERE_PLUS_TOKENS)


def _gpu_device_kinds() -> tuple[str, ...]:
    try:
        devices = jax.devices("gpu")
    except Exception:
        devices = [d for d in jax.devices() if d.platform in _GPU_BACKENDS]
    return tuple(getattr(d, "device_kind", str(d)) for d in devices)


def _gpu_backend_supports_pallas() -> bool:
    if _env_truthy("JAX_COLU_DISABLE_PALLAS"):
        return False
    if _env_truthy("JAX_COLU_FORCE_PALLAS"):
        return True
    kinds = _gpu_device_kinds()
    return bool(kinds) and all(_is_ampere_plus(k) for k in kinds)


def _has_tpu_devices() -> bool:
    try:
        return bool(jax.devices("tpu"))
    except Exception:
        return False


def _tpu_backend_supports_pallas() -> bool:
    if _env_truthy("JAX_COLU_DISABLE_PALLAS"):
        return False
    if _env_truthy("JAX_COLU_FORCE_PALLAS"):
        return True
    return _has_tpu_devices()


def _can_use_pallas_rcolu(
    *,
    backend: str,
    dim: object,
    num_groups: object,
    scaling: str,
    axis: int,
) -> bool:
    if not (
        scaling == "hard"
        and axis == -1
        and num_groups is None
        and _is_scalar_dim(dim)
    ):
        return False
    if backend in _GPU_BACKENDS:
        return _gpu_backend_supports_pallas()
    if backend in _TPU_BACKENDS:
        return _tpu_backend_supports_pallas()
    return False


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
    """Explicit-axis Conic Linear Unit.

    Public dispatch always routes to the JAX reference implementation. The
    Pallas CoLU kernel is being rewritten around padded power-of-two blocks
    and is not part of the public API yet.
    """
    return colu_reference(
        x,
        channel_axis=channel_axis,
        scaling=scaling,
        eps=eps,
        num_groups=num_groups,
        dim=dim,
        share_axis=share_axis,
    )
