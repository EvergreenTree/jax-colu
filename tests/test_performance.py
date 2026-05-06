from __future__ import annotations

import time

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from conftest import (
    has_supported_gpu_pallas_backend,
    has_tpu_backend,
    randn,
    require_supported_gpu_pallas_backend,
)
from jax_colu._reference import rcolu_reference


def _time_ms(fn, x, n=50):
    for _ in range(10):
        jax.block_until_ready(fn(x))
    t0 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn(x))
    return (time.perf_counter() - t0) / n * 1e3


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("dim", [8, 16, 32])
def test_rcolu_pallas_not_much_slower_than_reference(dim):
    require_supported_gpu_pallas_backend()
    from jax_colu.gpu._rcolu import rcolu_gpu

    x = randn(np.random.default_rng(0), (2048, dim), jnp.float32)
    ref = jax.jit(lambda z: rcolu_reference(z, dim=dim))
    gpu = jax.jit(lambda z: rcolu_gpu(z, dim=dim))
    t_ref = _time_ms(ref, x)
    t_gpu = _time_ms(gpu, x)
    assert t_gpu <= t_ref * 1.5


@pytest.mark.tpu
@pytest.mark.slow
@pytest.mark.parametrize("dim", [4, 8, 16, 32])
def test_rcolu_tpu_pallas_not_much_slower_than_reference(dim):
    if not has_tpu_backend():
        pytest.skip("requires TPU backend")
    from jax_colu.tpu._rcolu import rcolu_tpu

    x = randn(np.random.default_rng(0), (4096, 256), jnp.float32)
    ref = jax.jit(lambda z: rcolu_reference(z, dim=dim))
    tpu = jax.jit(lambda z: rcolu_tpu(z, dim=dim))
    t_ref = _time_ms(ref, x)
    t_tpu = _time_ms(tpu, x)
    assert t_tpu <= t_ref * 1.25
