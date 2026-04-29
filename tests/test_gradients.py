from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jax_colu._reference import rcolu_reference
from conftest import randn, require_supported_gpu_pallas_backend


@pytest.mark.gpu
@pytest.mark.parametrize("dim", [4, 8, 16])
def test_rcolu_gpu_vjp_matches_reference(dim, rng):
    require_supported_gpu_pallas_backend()
    from jax_colu.gpu._rcolu import rcolu_gpu

    x = randn(rng, (16, dim), jnp.float32, scale=1.5)
    g = randn(np.random.default_rng(7), (16, dim), jnp.float32)
    _, ref_vjp = jax.vjp(lambda z: rcolu_reference(z, dim=dim), x)
    _, got_vjp = jax.vjp(lambda z: rcolu_gpu(z, dim=dim), x)
    np.testing.assert_allclose(
        np.array(got_vjp(g)[0]), np.array(ref_vjp(g)[0]), atol=2e-4, rtol=2e-4
    )
