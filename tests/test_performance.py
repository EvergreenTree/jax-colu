from __future__ import annotations

import time

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from conftest import randn
from jax_colu._reference import colu_reference, rcolu_reference


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
    from jax_colu.gpu._rcolu import rcolu_gpu

    x = randn(np.random.default_rng(0), (2048, dim), jnp.float32)
    t_ref = _time_ms(lambda z: rcolu_reference(z, dim=dim), x)
    t_gpu = _time_ms(lambda z: rcolu_gpu(z, dim=dim), x)
    assert t_gpu <= t_ref * 1.25


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("share_axis", [False, True])
def test_colu_pallas_smoke(share_axis):
    from jax_colu.gpu._colu import colu_gpu

    dim = 8
    G = 32
    C = 1 + G * (dim - 1) if share_axis else G * dim
    x = randn(np.random.default_rng(0), (512, C), jnp.float32)
    t_ref = _time_ms(lambda z: colu_reference(z, dim=dim, share_axis=share_axis), x)
    t_gpu = _time_ms(lambda z: colu_gpu(z, dim=dim, share_axis=share_axis), x)
    assert t_gpu <= t_ref * 1.5
