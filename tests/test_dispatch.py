from __future__ import annotations

import pytest

from jax_colu import _dispatch as dispatch


def _allow_default_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JAX_COLU_DISABLE_PALLAS", raising=False)
    monkeypatch.delenv("JAX_COLU_FORCE_PALLAS", raising=False)


def _set_gpu_kinds(monkeypatch: pytest.MonkeyPatch, *kinds: str) -> None:
    _allow_default_env(monkeypatch)
    monkeypatch.setattr(dispatch, "_gpu_device_kinds", lambda: kinds)


def _set_tpu_present(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    _allow_default_env(monkeypatch)
    monkeypatch.setattr(dispatch, "_has_tpu_devices", lambda: present)


@pytest.mark.parametrize(
    ("device_kind", "is_ampere_or_newer"),
    [
        ("Tesla T4", False),
        ("NVIDIA Tesla V100-SXM2-16GB", False),
        ("NVIDIA A100-SXM4-40GB", True),
        ("NVIDIA L4", True),
        ("NVIDIA H100 80GB HBM3", True),
        ("NVIDIA B200", True),
        ("AMD Radeon Pro", False),
    ],
)
def test_gpu_architecture_detection(device_kind: str, is_ampere_or_newer: bool) -> None:
    assert dispatch._is_ampere_plus(device_kind) is is_ampere_or_newer


def test_gpu_pallas_guard_requires_known_ampere_or_newer_nvidia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gpu_kinds(monkeypatch, "Tesla T4")
    assert not dispatch._gpu_backend_supports_pallas()

    _set_gpu_kinds(monkeypatch, "NVIDIA Unknown GPU")
    assert not dispatch._gpu_backend_supports_pallas()

    _set_gpu_kinds(monkeypatch, "AMD Radeon Pro")
    assert not dispatch._gpu_backend_supports_pallas()

    _set_gpu_kinds(monkeypatch, "NVIDIA A100-SXM4-40GB")
    assert dispatch._gpu_backend_supports_pallas()


def test_gpu_pallas_guard_handles_mixed_gpu_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gpu_kinds(monkeypatch, "NVIDIA A100-SXM4-40GB", "Tesla T4")
    assert not dispatch._gpu_backend_supports_pallas()


def test_gpu_pallas_guard_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_gpu_kinds(monkeypatch, "Tesla T4")
    monkeypatch.setenv("JAX_COLU_FORCE_PALLAS", "1")
    assert dispatch._gpu_backend_supports_pallas()

    _set_gpu_kinds(monkeypatch, "NVIDIA A100-SXM4-40GB")
    monkeypatch.setenv("JAX_COLU_DISABLE_PALLAS", "1")
    assert not dispatch._gpu_backend_supports_pallas()


def test_tpu_pallas_guard_requires_tpu_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_tpu_present(monkeypatch, False)
    assert not dispatch._tpu_backend_supports_pallas()

    _set_tpu_present(monkeypatch, True)
    assert dispatch._tpu_backend_supports_pallas()


def test_tpu_pallas_guard_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_tpu_present(monkeypatch, False)
    monkeypatch.setenv("JAX_COLU_FORCE_PALLAS", "1")
    assert dispatch._tpu_backend_supports_pallas()

    _set_tpu_present(monkeypatch, True)
    monkeypatch.setenv("JAX_COLU_DISABLE_PALLAS", "1")
    assert not dispatch._tpu_backend_supports_pallas()


def test_public_rcolu_dispatch_falls_back_on_pre_ampere_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gpu_kinds(monkeypatch, "Tesla T4")
    assert not dispatch._can_use_pallas_rcolu(
        backend="gpu",
        dim=4,
        num_groups=None,
        scaling="hard",
        axis=-1,
    )

    _set_gpu_kinds(monkeypatch, "NVIDIA A100-SXM4-40GB")
    assert dispatch._can_use_pallas_rcolu(
        backend="gpu",
        dim=4,
        num_groups=None,
        scaling="hard",
        axis=-1,
    )


def test_public_rcolu_dispatch_routes_tpu_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_tpu_present(monkeypatch, False)
    assert not dispatch._can_use_pallas_rcolu(
        backend="tpu",
        dim=4,
        num_groups=None,
        scaling="hard",
        axis=-1,
    )

    _set_tpu_present(monkeypatch, True)
    assert dispatch._can_use_pallas_rcolu(
        backend="tpu",
        dim=4,
        num_groups=None,
        scaling="hard",
        axis=-1,
    )


@pytest.mark.parametrize(
    ("scaling", "axis", "num_groups", "dim"),
    [
        ("soft", -1, None, 4),
        ("hard", 0, None, 4),
        ("hard", -1, 8, None),
        ("hard", -1, None, [4, 4]),
    ],
)
def test_public_rcolu_dispatch_falls_back_for_unsupported_args(
    monkeypatch: pytest.MonkeyPatch,
    scaling: str,
    axis: int,
    num_groups: object,
    dim: object,
) -> None:
    _set_gpu_kinds(monkeypatch, "NVIDIA A100-SXM4-40GB")
    assert not dispatch._can_use_pallas_rcolu(
        backend="gpu",
        dim=dim,
        num_groups=num_groups,
        scaling=scaling,
        axis=axis,
    )
