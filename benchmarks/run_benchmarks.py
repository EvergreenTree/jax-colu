"""Benchmark raw JAX activations against jax-colu.

The default run benchmarks local CPU and Apple Metal devices when present:

    python benchmarks/run_benchmarks.py --out results/

Rows are written even for unavailable devices, with ``status=error`` and the
backend error message. This is useful for Apple Metal, where the backend may be
installed but unable to accept arrays for a given JAX/jax-metal combination.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jax_colu import colu, rcolu

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def raw_jax_rcolu(x: jax.Array, dim: int, eps: float = 1e-7) -> jax.Array:
    """Naive raw JAX rCoLU baseline using the most direct norm expression."""
    G = x.shape[-1] // dim
    q = x.reshape(x.shape[:-1] + (G, dim))
    inv_s = 1.0 / math.sqrt(float(dim))
    t = jnp.sum(q, axis=-1, keepdims=True) * inv_s
    w = q - t * inv_s
    r = jnp.linalg.norm(w, axis=-1, keepdims=True)
    t_pos = jax.nn.relu(t)
    sc = jnp.clip(t_pos / (r + eps), 0.0, 1.0)
    return (t_pos * inv_s + sc * w).reshape(x.shape)


def rcolu_static_e(x: jax.Array, dim: int, eps: float = 1e-7) -> jax.Array:
    """rCoLU with an explicit homogeneous axis vector."""
    G = x.shape[-1] // dim
    q = x.reshape(x.shape[:-1] + (G, dim))
    inv_s = 1.0 / math.sqrt(float(dim))
    e = jnp.full((dim,), inv_s, dtype=x.dtype)
    t = jnp.sum(q * e, axis=-1, keepdims=True)
    w = q - t * e
    r = jnp.linalg.norm(w, axis=-1, keepdims=True)
    t_pos = jax.nn.relu(t)
    sc = jnp.clip(t_pos / (r + eps), 0.0, 1.0)
    return (t_pos * e + sc * w).reshape(x.shape)


def rcolu_single_kernel(x: jax.Array, dim: int, eps: float = 1e-7) -> jax.Array:
    """Adjacent separate reductions for sum(x) and sum(x*x)."""
    G = x.shape[-1] // dim
    q = x.reshape(x.shape[:-1] + (G, dim))
    inv_s = 1.0 / math.sqrt(float(dim))
    sum_x = jnp.sum(q, axis=-1, keepdims=True)
    sum_xx = jnp.sum(q * q, axis=-1, keepdims=True)
    t = sum_x * inv_s
    r = jnp.sqrt(jnp.maximum(sum_xx - t * t, 0.0))
    w = q - t * inv_s
    t_pos = jnp.maximum(t, 0.0)
    sc = jnp.minimum(t_pos / (r + eps), 1.0)
    return (t_pos * inv_s + sc * w).reshape(x.shape)


def _tuple_sum_and_sum_squares(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    zero = jnp.array(0, dtype=x.dtype)

    def reducer(a, b):
        return (a[0] + b[0], a[1] + b[1])

    return jax.lax.reduce(
        (x, x * x), (zero, zero), reducer, dimensions=(x.ndim - 1,)
    )


def rcolu_two_pass_reduce(x: jax.Array, dim: int, eps: float = 1e-7) -> jax.Array:
    """Tuple-reduce sum(x) and sum(x*x), then use ||w||^2 = ||x||^2 - t^2."""
    G = x.shape[-1] // dim
    q = x.reshape(x.shape[:-1] + (G, dim))
    inv_s = 1.0 / math.sqrt(float(dim))
    sum_x, sum_xx = _tuple_sum_and_sum_squares(q)
    t = sum_x[..., None] * inv_s
    r = jnp.sqrt(jnp.maximum(sum_xx[..., None] - t * t, 0.0))
    w = q - t * inv_s
    t_pos = jnp.maximum(t, 0.0)
    sc = jnp.minimum(t_pos / (r + eps), 1.0)
    return (t_pos * inv_s + sc * w).reshape(x.shape)


def rcolu_mgpu_direct(x: jax.Array, dim: int, eps: float = 1e-7) -> jax.Array:
    """Direct Pallas:Mosaic GPU rCoLU backend."""
    from jax_colu.gpu._rcolu_mgpu import rcolu_mgpu

    return rcolu_mgpu(x, dim=dim, eps=eps)


def raw_jax_colu(x: jax.Array, dim: int, eps: float = 1e-7) -> jax.Array:
    """Naive raw JAX explicit-axis CoLU baseline."""
    G = x.shape[-1] // dim
    t, w = jnp.split(x, [G], axis=-1)
    w = w.reshape(x.shape[:-1] + (G, dim - 1))
    r = jnp.linalg.norm(w, axis=-1, keepdims=True)
    t_pos = jax.nn.relu(t)
    sc = jnp.clip(t_pos.reshape(x.shape[:-1] + (G, 1)) / (r + eps), 0.0, 1.0)
    w_out = (sc * w).reshape(x.shape[:-1] + (G * (dim - 1),))
    return jnp.concatenate([t_pos, w_out], axis=-1)


def _devices(requested: list[str]) -> list[tuple[str, jax.Device | None, str]]:
    devices = []
    for name in requested:
        try:
            dev = jax.devices(name)[0]
            devices.append((name, dev, ""))
        except Exception as exc:  # noqa: BLE001 - surfaced in CSV output.
            devices.append((name, None, f"{type(exc).__name__}: {exc}"))
    return devices


def _device_put(data: np.ndarray, device: jax.Device) -> jax.Array:
    return jax.device_put(data, device)


def _time_ms(
    fn: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    device: jax.Device,
    *,
    warmup: int,
    repeat: int,
    compile_with_jit: bool = True,
) -> tuple[float, float, float]:
    compiled = jax.jit(fn, device=device) if compile_with_jit else fn
    compiled(x).block_until_ready()
    for _ in range(warmup):
        compiled(x).block_until_ready()

    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        compiled(x).block_until_ready()
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples)), float(np.mean(samples)), float(np.std(samples))


def _ok_row(
    *,
    device_name: str,
    device: jax.Device,
    op: str,
    implementation: str,
    dim: int | str,
    shape: tuple[int, int],
    mean_ms: float,
    median_ms: float,
    std_ms: float,
) -> dict[str, str | int | float]:
    return {
        "device": device_name,
        "platform": device.platform,
        "op": op,
        "implementation": implementation,
        "dim": dim,
        "batch": shape[0],
        "channels": shape[1],
        "median_ms": round(median_ms, 5),
        "mean_ms": round(mean_ms, 5),
        "std_ms": round(std_ms, 5),
        "status": "ok",
        "error": "",
    }


def _error_row(
    *,
    device_name: str,
    op: str,
    implementation: str,
    dim: int | str,
    shape: tuple[int, int],
    error: str,
) -> dict[str, str | int | float]:
    return {
        "device": device_name,
        "platform": "",
        "op": op,
        "implementation": implementation,
        "dim": dim,
        "batch": shape[0],
        "channels": shape[1],
        "mean_ms": "",
        "median_ms": "",
        "std_ms": "",
        "status": "error",
        "error": error,
    }


def benchmark_device(
    *,
    device_name: str,
    device: jax.Device | None,
    device_error: str,
    data: np.ndarray,
    dims: list[int],
    warmup: int,
    repeat: int,
) -> list[dict[str, str | int | float]]:
    shape = data.shape
    rows: list[dict[str, str | int | float]] = []
    cases: list[
        tuple[str, str, int | str, Callable[[jax.Array], jax.Array], bool]
    ] = []

    for dim in dims:
        if shape[-1] % dim != 0:
            continue
        cases.extend(
            [
                ("rcolu", "naive", dim, lambda x, dim=dim: raw_jax_rcolu(x, dim), False),
                ("rcolu", "naive_jit", dim, lambda x, dim=dim: raw_jax_rcolu(x, dim), True),
                ("rcolu", "static_e", dim, lambda x, dim=dim: rcolu_static_e(x, dim), True),
                ("rcolu", "two_pass", dim, lambda x, dim=dim: rcolu_two_pass_reduce(x, dim), True),
                ("rcolu", "single", dim, lambda x, dim=dim: rcolu_single_kernel(x, dim), True),
                ("rcolu", "custom_vjp", dim, lambda x, dim=dim: rcolu(x, dim=dim), False),
                ("rcolu", "mgpu", dim, lambda x, dim=dim: rcolu_mgpu_direct(x, dim), True),
                ("colu", "raw_jax", dim, lambda x, dim=dim: raw_jax_colu(x, dim), True),
                ("colu", "jax_colu", dim, lambda x, dim=dim: colu(x, dim=dim), False),
            ]
        )

    cases.extend(
        [
            ("relu", "jax.nn", "", jax.nn.relu, True),
            ("silu", "jax.nn", "", jax.nn.silu, True),
            ("gelu", "jax.nn", "", jax.nn.gelu, True),
        ]
    )

    if device is None:
        return [
            _error_row(
                device_name=device_name,
                op=op,
                implementation=impl,
                dim=dim,
                shape=shape,
                error=device_error,
            )
            for op, impl, dim, _, _ in cases
        ]

    try:
        x = _device_put(data, device)
    except Exception as exc:  # noqa: BLE001 - surfaced in CSV output.
        error = f"{type(exc).__name__}: {exc}"
        return [
            _error_row(
                device_name=device_name,
                op=op,
                implementation=impl,
                dim=dim,
                shape=shape,
                error=error,
            )
            for op, impl, dim, _, _ in cases
        ]

    for op, impl, dim, fn, compile_with_jit in cases:
        try:
            median, mean, std = _time_ms(
                fn,
                x,
                device,
                warmup=warmup,
                repeat=repeat,
                compile_with_jit=compile_with_jit,
            )
            rows.append(
                _ok_row(
                    device_name=device_name,
                    device=device,
                    op=op,
                    implementation=impl,
                    dim=dim,
                    shape=shape,
                    median_ms=median,
                    mean_ms=mean,
                    std_ms=std,
                )
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in CSV output.
            rows.append(
                _error_row(
                    device_name=device_name,
                    op=op,
                    implementation=impl,
                    dim=dim,
                    shape=shape,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return rows


def write_csv(rows: list[dict[str, str | int | float]], out_dir: Path) -> Path:
    path = out_dir / "results.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def fig_latency(rows: list[dict[str, str | int | float]], out_dir: Path) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok" and r["device"] == "cpu"]
    if not ok_rows:
        return

    labels = []
    values = []
    colors = []
    for row in ok_rows:
        dim = f"S{row['dim']}" if row["dim"] != "" else ""
        labels.append(f"{row['op']} {dim}\\n{row['implementation']}")
        values.append(float(row["median_ms"]))
        colors.append("#2E86AB" if row["implementation"] == "jax_colu" else "#A23B72")

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.55), 4.8))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("CPU latency, median over repeated calls")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_cpu_latency.pdf", bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: list[dict[str, str | int | float]]) -> None:
    for row in rows:
        if row["status"] == "ok":
            print(
                f"{row['device']:>5s} {row['op']:<6s} {str(row['dim']):>3s} "
                f"{row['implementation']:<10s} {float(row['median_ms']):8.4f} ms"
            )
        else:
            print(
                f"{row['device']:>5s} {row['op']:<6s} {str(row['dim']):>3s} "
                f"{row['implementation']:<10s} ERROR {row['error']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results", type=Path)
    parser.add_argument("--devices", nargs="+", default=["cpu", "METAL"])
    parser.add_argument("--batch", default=4096, type=int)
    parser.add_argument("--channels", default=256, type=int)
    parser.add_argument("--dims", nargs="+", default=[4, 8, 16, 32], type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--repeat", default=200, type=int)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(0).normal(size=(args.batch, args.channels)).astype(
        np.float32
    )

    rows: list[dict[str, str | int | float]] = []
    for device_name, device, error in _devices(args.devices):
        rows.extend(
            benchmark_device(
                device_name=device_name,
                device=device,
                device_error=error,
                data=data,
                dims=args.dims,
                warmup=args.warmup,
                repeat=args.repeat,
            )
        )

    print_summary(rows)
    print(f"Saved {write_csv(rows, args.out)}")
    fig_latency(rows, args.out)


if __name__ == "__main__":
    main()
