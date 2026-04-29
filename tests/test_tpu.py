from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from conftest import randn
from jax_colu._reference import colu_reference, rcolu_reference


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
def test_colu_tpu_matches_reference(rng):
    from jax_colu.tpu._colu import colu_tpu

    x = randn(rng, (16, 16), jnp.float32)
    np.testing.assert_allclose(
        np.array(colu_tpu(x, dim=4)),
        np.array(colu_reference(x, dim=4)),
        atol=2e-5,
    )
