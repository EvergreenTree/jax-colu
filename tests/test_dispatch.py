from __future__ import annotations

import pytest

from jax_colu import _dispatch as dispatch


def _allow_default_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JAX_COLU_DISABLE_PALLAS", raising=False)
    monkeypatch.delenv("JAX_COLU_FORCE_PALLAS", raising=False)
    monkeypatch.delenv("JAX_COLU_GPU_BACKEND", raising=False)


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
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", True),
        ("AMD Radeon Pro", False),
    ],
)
def test_gpu_architecture_detection(device_kind: str, is_ampere_or_newer: bool) -> None:
    assert dispatch._is_ampere_plus(device_kind) is is_ampere_or_newer


@pytest.mark.parametrize(
    ("device_kind", "is_hopper_or_newer"),
    [
        ("NVIDIA A100-SXM4-40GB", False),
        ("NVIDIA L4", False),
        ("NVIDIA RTX 4090", False),
        ("NVIDIA H100 80GB HBM3", True),
        ("NVIDIA H200", True),
        ("NVIDIA B200", True),
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", True),
    ],
)
def test_gpu_hopper_or_newer_detection(
    device_kind: str, is_hopper_or_newer: bool
) -> None:
    assert dispatch._is_hopper_or_newer_nvidia(device_kind) is is_hopper_or_newer


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


def test_gpu_backend_env_defaults_to_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_default_env(monkeypatch)
    assert dispatch._gpu_pallas_backend() == "triton"


def test_gpu_backend_env_accepts_mgpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_default_env(monkeypatch)
    monkeypatch.setenv("JAX_COLU_GPU_BACKEND", "mgpu")
    assert dispatch._gpu_pallas_backend() == "mgpu"


def test_gpu_backend_env_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_default_env(monkeypatch)
    monkeypatch.setenv("JAX_COLU_GPU_BACKEND", "cuda")
    with pytest.raises(ValueError, match="invalid JAX_COLU_GPU_BACKEND"):
        dispatch._gpu_pallas_backend()


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


def test_public_rcolu_dispatch_uses_default_triton_on_ampere_gpu(
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
    assert (
        dispatch._rcolu_gpu_backend(
            backend="gpu",
            dim=4,
            num_groups=None,
            scaling="hard",
            axis=-1,
        )
        == "triton"
    )
    assert dispatch._can_use_pallas_rcolu(
        backend="gpu",
        dim=4,
        num_groups=None,
        scaling="hard",
        axis=-1,
    )


def test_public_rcolu_dispatch_uses_mgpu_on_hopper_or_blackwell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gpu_kinds(monkeypatch, "NVIDIA H100 80GB HBM3")
    monkeypatch.setenv("JAX_COLU_GPU_BACKEND", "mgpu")
    assert (
        dispatch._rcolu_gpu_backend(
            backend="gpu",
            dim=4,
            num_groups=None,
            scaling="hard",
            axis=-1,
        )
        == "mgpu"
    )

    _set_gpu_kinds(
        monkeypatch, "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    )
    monkeypatch.setenv("JAX_COLU_GPU_BACKEND", "mgpu")
    assert (
        dispatch._rcolu_gpu_backend(
            backend="gpu",
            dim=4,
            num_groups=None,
            scaling="hard",
            axis=-1,
        )
        == "mgpu"
    )


@pytest.mark.parametrize(
    "device_kind",
    [
        "Tesla T4",
        "NVIDIA A100-SXM4-40GB",
        "NVIDIA L4",
        "NVIDIA RTX 4090",
        "NVIDIA Unknown GPU",
        "AMD Radeon Pro",
    ],
)
def test_public_rcolu_dispatch_rejects_mgpu_on_non_hopper_gpu(
    monkeypatch: pytest.MonkeyPatch, device_kind: str
) -> None:
    _set_gpu_kinds(monkeypatch, device_kind)
    monkeypatch.setenv("JAX_COLU_GPU_BACKEND", "mgpu")
    assert (
        dispatch._rcolu_gpu_backend(
            backend="gpu",
            dim=4,
            num_groups=None,
            scaling="hard",
            axis=-1,
        )
        is None
    )


def test_public_rcolu_dispatch_disable_pallas_wins_for_mgpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gpu_kinds(monkeypatch, "NVIDIA H100 80GB HBM3")
    monkeypatch.setenv("JAX_COLU_GPU_BACKEND", "mgpu")
    monkeypatch.setenv("JAX_COLU_DISABLE_PALLAS", "1")
    assert (
        dispatch._rcolu_gpu_backend(
            backend="gpu",
            dim=4,
            num_groups=None,
            scaling="hard",
            axis=-1,
        )
        is None
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
