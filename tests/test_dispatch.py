from __future__ import annotations

import pytest

from jax_colu import _dispatch as dispatch


def _allow_default_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JAX_COLU_DISABLE_PALLAS", raising=False)
    monkeypatch.delenv("JAX_COLU_FORCE_PALLAS", raising=False)


def _set_gpu_kinds(monkeypatch: pytest.MonkeyPatch, *kinds: str) -> None:
    _allow_default_env(monkeypatch)
    monkeypatch.setattr(dispatch, "_gpu_device_kinds", lambda: kinds)


@pytest.mark.parametrize(
    ("device_kind", "is_pre_ampere", "is_ampere_or_newer"),
    [
        ("Tesla T4", True, False),
        ("NVIDIA Tesla V100-SXM2-16GB", True, False),
        ("NVIDIA A100-SXM4-40GB", False, True),
        ("NVIDIA L4", False, True),
        ("NVIDIA H100 80GB HBM3", False, True),
        ("NVIDIA B200", False, True),
        ("AMD Radeon Pro", False, False),
    ],
)
def test_gpu_architecture_detection(
    device_kind: str, is_pre_ampere: bool, is_ampere_or_newer: bool
) -> None:
    assert dispatch._is_pre_ampere_nvidia(device_kind) is is_pre_ampere
    assert dispatch._is_ampere_or_newer_nvidia(device_kind) is is_ampere_or_newer


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


@pytest.mark.parametrize("backend", ["gpu", "tpu"])
def test_public_colu_dispatch_stays_on_reference_path(backend: str) -> None:
    assert not dispatch._can_use_pallas_colu(
        backend=backend,
        dim=4,
        num_groups=None,
        scaling="hard",
        channel_axis=-1,
    )
