from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jax_colu import colu, rcolu
from conftest import randn
from jax_colu._reference import colu_reference, rcolu_reference


@pytest.mark.tpu
def test_public_rcolu_tpu_dispatch_matches_reference(rng):
    x = randn(rng, (16, 8), jnp.float32)
    np.testing.assert_allclose(
        np.array(rcolu(x, dim=4)),
        np.array(rcolu_reference(x, dim=4)),
        atol=2e-5,
    )


@pytest.mark.tpu
def test_public_colu_tpu_dispatch_matches_reference(rng):
    x = randn(rng, (16, 16), jnp.float32)
    np.testing.assert_allclose(
        np.array(colu(x, dim=4)),
        np.array(colu_reference(x, dim=4)),
        atol=2e-5,
    )


@pytest.mark.tpu
@pytest.mark.parametrize("shape", [(16, 8), (8, 16, 16), (2, 64, 14, 16)])
@pytest.mark.parametrize("dim", [4, 8, 16])
def test_rcolu_tpu_pallas_matches_reference(shape, dim, rng):
    if shape[-1] % dim != 0:
        pytest.skip("shape is incompatible with dim")
    from jax_colu.tpu._rcolu import rcolu_tpu

    x = randn(rng, shape, jnp.float32)
    np.testing.assert_allclose(
        np.array(rcolu_tpu(x, dim=dim)),
        np.array(rcolu_reference(x, dim=dim)),
        atol=2e-5,
    )


@pytest.mark.tpu
@pytest.mark.parametrize("dim", [4, 8, 16])
def test_rcolu_tpu_vjp_matches_reference(dim, rng):
    from jax_colu.tpu._rcolu import rcolu_tpu

    x = randn(rng, (16, dim), jnp.float32, scale=1.5)
    g = randn(np.random.default_rng(7), (16, dim), jnp.float32)
    _, ref_vjp = jax.vjp(lambda z: rcolu_reference(z, dim=dim), x)
    _, got_vjp = jax.vjp(lambda z: rcolu_tpu(z, dim=dim), x)
    np.testing.assert_allclose(
        np.array(got_vjp(g)[0]),
        np.array(ref_vjp(g)[0]),
        atol=2e-4,
        rtol=2e-4,
    )
