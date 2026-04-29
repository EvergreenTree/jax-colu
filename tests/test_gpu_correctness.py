from __future__ import annotations

import numpy as np
import pytest

from jax_colu._reference import colu_reference, rcolu_reference
from conftest import randn


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(64, 32), (8, 16, 16), (2, 64, 14, 16)])
def test_rcolu_gpu_matches_reference(shape, dim, gpu_dtype, rng):
    if shape[-1] % dim != 0:
        pytest.skip("shape is incompatible with dim")
    from jax_colu.gpu._rcolu import rcolu_gpu

    x = randn(rng, shape, gpu_dtype)
    ref = rcolu_reference(x, dim=dim)
    got = rcolu_gpu(x, dim=dim)
    atol = 3e-3 if str(gpu_dtype) == "<class 'jax.numpy.bfloat16'>" else 2e-5
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=atol, rtol=atol)


@pytest.mark.gpu
@pytest.mark.parametrize("share_axis", [False, True])
def test_colu_gpu_matches_reference(dim, share_axis, gpu_dtype, rng):
    from jax_colu.gpu._colu import colu_gpu

    G = 8
    C = 1 + G * (dim - 1) if share_axis else G * dim
    x = randn(rng, (32, C), gpu_dtype)
    ref = colu_reference(x, dim=dim, share_axis=share_axis)
    got = colu_gpu(x, dim=dim, share_axis=share_axis)
    atol = 3e-3 if str(gpu_dtype) == "<class 'jax.numpy.bfloat16'>" else 2e-5
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=atol, rtol=atol)
