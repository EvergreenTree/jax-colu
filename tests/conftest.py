from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _platforms() -> set[str]:
    return {d.platform.lower() for d in jax.devices()}


def has_gpu_backend() -> bool:
    return bool(_platforms() & {"gpu", "cuda", "rocm"})


def has_tpu_backend() -> bool:
    return "tpu" in _platforms()


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="requires CUDA/ROCm Pallas backend")
    skip_tpu = pytest.mark.skip(reason="requires TPU Pallas backend")
    for item in items:
        if "gpu" in item.keywords and not has_gpu_backend():
            item.add_marker(skip_gpu)
        if "tpu" in item.keywords and not has_tpu_backend():
            item.add_marker(skip_tpu)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture(params=[4, 8, 16], ids=lambda d: f"S{d}")
def dim(request):
    return request.param


@pytest.fixture(params=[jnp.float32], ids=["f32"])
def dtype(request):
    return request.param


@pytest.fixture(params=[jnp.float32, jnp.bfloat16], ids=["f32", "bf16"])
def gpu_dtype(request):
    return request.param


def randn(rng, shape, dtype=jnp.float32, scale=1.0):
    values = rng.normal(size=shape).astype(np.float32) * scale
    if jax.default_backend().lower() in {"metal", "mps"}:
        arr = jax.device_put(values, jax.devices("cpu")[0])
    else:
        arr = jnp.asarray(values)
    return arr.astype(dtype)
