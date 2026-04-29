from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import math

from jax_colu import colu, colu_reference, rcolu, rcolu_reference
from conftest import randn


def _assert_cone_rcolu(y, dim: int):
    y = np.array(y).reshape((-1, dim))
    inv_s = 1.0 / np.sqrt(float(dim))
    t = np.sum(y, axis=-1) * inv_s
    w = y - t[:, None] * inv_s
    r = np.linalg.norm(w, axis=-1)
    assert np.all(t >= -2e-5)
    assert np.all(r <= t + 2e-5)


def _assert_cone_colu(y, dim: int, share_axis: bool):
    y = np.array(y)
    C = y.shape[-1]
    if share_axis:
        G = (C - 1) // (dim - 1)
        t = np.broadcast_to(y[..., :1], y.shape[:-1] + (G,))
        w = y[..., 1:].reshape(y.shape[:-1] + (G, dim - 1))
    else:
        G = C // dim
        t = y[..., :G]
        w = y[..., G:].reshape(y.shape[:-1] + (G, dim - 1))
    r = np.linalg.norm(w, axis=-1)
    assert np.all(t >= -2e-5)
    assert np.all(r <= t + 2e-5)


@pytest.mark.parametrize("shape", [(5, 16), (2, 3, 32), (2, 16, 4)])
def test_rcolu_reference_shape_and_cone(shape, dim, rng, dtype):
    if shape[-1] % dim != 0:
        pytest.skip("shape is incompatible with dim")
    x = randn(rng, shape, dtype)
    y = rcolu_reference(x, dim=dim)
    assert y.shape == x.shape
    _assert_cone_rcolu(y, dim)


@pytest.mark.parametrize("share_axis", [False, True])
def test_colu_reference_shape_and_cone(dim, share_axis, rng, dtype):
    G = 4
    C = 1 + G * (dim - 1) if share_axis else G * dim
    x = randn(rng, (3, C), dtype)
    y = colu_reference(x, dim=dim, share_axis=share_axis)
    assert y.shape == x.shape
    _assert_cone_colu(y, dim, share_axis)


def test_colu_reference_non_last_channel_axis(rng):
    x = randn(rng, (2, 12, 3, 2))
    y = colu_reference(x, channel_axis=1, dim=4)
    assert y.shape == x.shape
    yt = colu_reference(jnp.moveaxis(x, 1, -1), channel_axis=-1, dim=4)
    np.testing.assert_allclose(np.array(jnp.moveaxis(yt, -1, 1)), np.array(y), atol=1e-5)


def test_public_dispatch_matches_reference(rng):
    x = randn(rng, (4, 16))
    np.testing.assert_allclose(
        np.array(rcolu(x, dim=4)), np.array(rcolu_reference(x, dim=4)), atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(colu(x, dim=4)), np.array(colu_reference(x, dim=4)), atol=1e-6
    )


def test_jit_vmap_and_scan(rng):
    x = randn(rng, (6, 8))

    jitted = jax.jit(lambda z: rcolu_reference(z, dim=4))(x)
    np.testing.assert_allclose(np.array(jitted), np.array(rcolu_reference(x, dim=4)))

    vmapped = jax.vmap(lambda z: rcolu_reference(z, dim=4))(x)
    np.testing.assert_allclose(np.array(vmapped), np.array(rcolu_reference(x, dim=4)))

    def step(carry, z):
        y = rcolu_reference(z, dim=4)
        return carry + jnp.sum(y), y

    total, ys = jax.lax.scan(step, x.reshape(-1)[0] * 0.0, x)
    assert ys.shape == x.shape
    assert total.shape == ()


def test_rcolu_reference_custom_vjp_matches_autodiff(rng):
    def autodiff_baseline(z):
        dim = 4
        groups = z.shape[-1] // dim
        q = z.reshape(z.shape[:-1] + (groups, dim))
        inv_s = 1.0 / math.sqrt(float(dim))
        t = jnp.sum(q, axis=-1, keepdims=True) * inv_s
        w = q - t * inv_s
        r = jnp.sqrt(jnp.sum(w * w, axis=-1, keepdims=True))
        t_pos = jnp.maximum(t, 0.0)
        sc = jnp.minimum(t_pos / (r + 1e-7), 1.0)
        return (t_pos * inv_s + sc * w).reshape(z.shape)

    x = randn(rng, (5, 12))
    g = randn(np.random.default_rng(7), (5, 12))
    _, custom_vjp = jax.vjp(lambda z: rcolu_reference(z, dim=4), x)
    _, autodiff_vjp = jax.vjp(autodiff_baseline, x)
    np.testing.assert_allclose(
        np.array(custom_vjp(g)[0]), np.array(autodiff_vjp(g)[0]), atol=1e-5
    )


@pytest.mark.parametrize("scaling", ["soft", "sqrt", "log"])
def test_non_hard_scalings_run(scaling, rng):
    x = randn(rng, (3, 16))
    assert rcolu_reference(x, dim=4, scaling=scaling).shape == x.shape
    assert colu_reference(x, dim=4, scaling=scaling).shape == x.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 4, "num_groups": 4},
        {"dim": None, "num_groups": None},
        {"dim": 5},
        {"axis": 5},
        {"scaling": "bad"},
    ],
)
def test_rcolu_invalid_args(kwargs, rng):
    x = randn(rng, (2, 16))
    with pytest.raises(ValueError):
        rcolu_reference(x, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 4, "num_groups": 4},
        {"dim": None, "num_groups": None},
        {"dim": 5},
        {"channel_axis": 5},
        {"scaling": "bad"},
    ],
)
def test_colu_invalid_args(kwargs, rng):
    x = randn(rng, (2, 16))
    with pytest.raises(ValueError):
        colu_reference(x, **kwargs)
