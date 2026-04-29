from __future__ import annotations

import numpy as np
import pytest
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
def test_rcolu_tpu_matches_reference(rng):
    from jax_colu.tpu._rcolu import rcolu_tpu

    x = randn(rng, (16, 8), jnp.float32)
    np.testing.assert_allclose(
        np.array(rcolu_tpu(x, dim=4)),
        np.array(rcolu_reference(x, dim=4)),
        atol=2e-5,
    )


@pytest.mark.tpu
@pytest.mark.xfail(
    reason="experimental CoLU Pallas kernel needs padded/masked lowering rewrite",
    strict=False,
)
def test_colu_tpu_matches_reference(rng):
    from jax_colu.tpu._colu import colu_tpu

    x = randn(rng, (16, 16), jnp.float32)
    np.testing.assert_allclose(
        np.array(colu_tpu(x, dim=4)),
        np.array(colu_reference(x, dim=4)),
        atol=2e-5,
    )
