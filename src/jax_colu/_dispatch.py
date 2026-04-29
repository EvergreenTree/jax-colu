"""Backend dispatch for public activations."""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence

import jax

from jax_colu._reference import colu_reference, rcolu_reference

_GPU_BACKENDS = frozenset({"gpu", "cuda", "rocm"})
_TPU_BACKENDS = frozenset({"tpu"})
_METAL_BACKENDS = frozenset({"metal", "mps"})

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Pallas/Triton GPU lowering is not universally safe across NVIDIA generations.
# The public dispatch path is conservative: pre-Ampere NVIDIA GPUs use the JAX
# reference path unless the user explicitly opts in with JAX_COLU_FORCE_PALLAS=1.
_PRE_AMPERE_NVIDIA_TOKENS = (
    "tesla t",
    " t4",
    "tesla v",
    " v100",
    "tesla p",
    " p100",
    " p40",
    " p4",
    "tesla k",
    " k80",
    "tesla m",
    " m60",
    "quadro rtx",
    "rtx 20",
    "rtx 2060",
    "rtx 2070",
    "rtx 2080",
    "gtx ",
    "titan",
)
_AMPERE_OR_NEWER_NVIDIA_TOKENS = (
    "a100",
    "a800",
    "a10",
    "a16",
    "a2",
    "a30",
    "a40",
    "l4",
    "l40",
    "rtx 30",
    "rtx 40",
    "rtx 50",
    "rtx a",
    "ada",
    "h100",
    "h200",
    "hopper",
    "b100",
    "b200",
    "gb200",
    "blackwell",
)


def _backend() -> str:
    return jax.default_backend().lower()


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _is_scalar_dim(dim: object) -> bool:
    return not isinstance(dim, Sequence) or isinstance(dim, (str, bytes))


def _normalise_device_kind(device_kind: str) -> str:
    return f" {device_kind.lower().replace('-', ' ')} "


def _looks_like_nvidia(device_kind: str) -> bool:
    kind = _normalise_device_kind(device_kind)
    return any(
        token in kind
        for token in (
            "nvidia",
            "tesla",
            "geforce",
            "quadro",
            "rtx",
            "gtx",
            "titan",
        )
    )


def _is_pre_ampere_nvidia(device_kind: str) -> bool:
    kind = _normalise_device_kind(device_kind)
    return any(token in kind for token in _PRE_AMPERE_NVIDIA_TOKENS)


def _is_ampere_or_newer_nvidia(device_kind: str) -> bool:
    kind = _normalise_device_kind(device_kind)
    return any(token in kind for token in _AMPERE_OR_NEWER_NVIDIA_TOKENS)


def _gpu_device_kinds() -> tuple[str, ...]:
    try:
        devices = jax.devices("gpu")
    except Exception:
        devices = [device for device in jax.devices() if device.platform in _GPU_BACKENDS]
    return tuple(getattr(device, "device_kind", str(device)) for device in devices)


def _gpu_backend_supports_pallas() -> bool:
    if _env_truthy("JAX_COLU_DISABLE_PALLAS"):
        return False
    if _env_truthy("JAX_COLU_FORCE_PALLAS"):
        return True

    kinds = _gpu_device_kinds()
    if not kinds:
        return False

    for kind in kinds:
        if not _looks_like_nvidia(kind):
            return False
        if _is_pre_ampere_nvidia(kind):
            return False
        if not _is_ampere_or_newer_nvidia(kind):
            return False
    return True


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
        and (backend not in _GPU_BACKENDS or _gpu_backend_supports_pallas())
    )


def _can_use_pallas_colu(
    *,
    backend: str,
    dim: object,
    num_groups: object,
    scaling: str,
    channel_axis: int,
) -> bool:
    del backend, dim, num_groups, scaling, channel_axis
    # The experimental Pallas CoLU kernel currently hits GPU lowering limits
    # for slice/concat-like patterns and non-power-of-two blocks. Public
    # dispatch stays on the reference implementation until that kernel is
    # rewritten with padded power-of-two blocks and masks.
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
