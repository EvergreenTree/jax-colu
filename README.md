# jax-colu

Fused JAX/Pallas kernels for **Conic Linear Units** (CoLU & rCoLU).

- **Paper:** [Conic Activation Functions, PMLR 285](https://proceedings.mlr.press/v285/fu24a.html)
- **Repo:** https://github.com/EvergreenTree/jax-colu

## Quick start

```python
import jax
import jax.numpy as jnp
from jax_colu import rcolu, colu

x = jax.random.normal(jax.random.PRNGKey(0), (8, 32))
y = rcolu(x, dim=4)         # homogeneous-axis variant
z = colu(x, dim=4)          # heterogeneous explicit-apex variant
```

## What's a Conic Linear Unit?

CoLU projects each contiguous group of `dim` channels onto a Lorentz cone.

- `rcolu(x, dim=4)` uses a homogeneous cone axis, `ones / sqrt(dim)`, inside each channel group.
- `colu(x, dim=4)` uses an explicit apex channel per group followed by `dim - 1` body channels.
- Pass exactly one of `dim` or `num_groups`.
- `axis` / `channel_axis` choose the grouped channel axis.
- `scaling` supports `"hard"`, `"soft"`, `"sqrt"`, and `"log"`.
- `share_axis=True` for `colu` shares one apex channel across all cone groups.

## Backend behavior

Public functions are conservative by default:

- CPU and Apple Metal/MPS use the JAX reference path.
- `rcolu` uses Pallas on GPU only for supported hard-scaling calls on known Ampere-or-newer NVIDIA devices.
- Pre-Ampere NVIDIA GPUs such as T4/V100, unknown GPUs, and unsupported argument combinations fall back to the JAX reference path.
- `colu` currently uses the JAX reference path publicly. The direct `jax_colu.gpu._colu.colu_gpu` kernel remains experimental until it is rewritten around power-of-two padded blocks and masks.
- TPU Pallas paths are hardware-gated by tests and should be validated on a real TPU before release.

For local experiments, set `JAX_COLU_FORCE_PALLAS=1` to bypass the GPU guard or `JAX_COLU_DISABLE_PALLAS=1` to force reference dispatch.

Current local Metal support can be installed but still fail basic `device_put`, so benchmarks record those failures explicitly.

Blackwell tuning comes after clean kernel lowering. The expected optimization surface is processing multiple cone groups per program, power-of-two padded group widths with masks, dim/dtype specialization, fp32 accumulation for low-precision inputs, and per-architecture benchmarking of block size, num warps, and program shape.

## Install

```bash
pip install -e ".[dev]"
```

The package metadata targets current upstream JAX:

```toml
jax >= 0.10.0
jaxlib >= 0.10.0
```

Apple Metal users should use the `jax`, `jaxlib`, and `jax-metal` combination supported by Apple.

## Benchmarks

Raw activation latency:

```bash
python benchmarks/run_benchmarks.py --out results/
JAX_PLATFORMS=cpu python benchmarks/run_benchmarks.py --devices cpu --out results/cpu
```

The benchmark reports median latency over repeated calls and includes:

- rCoLU variants: `naive`, `naive_jit`, `static_e`, `two_pass`, `single`, `custom_vjp`
- CoLU: raw JAX and `jax_colu`
- `jax.nn.relu`, `jax.nn.silu`, `jax.nn.gelu`

CPU medians from local runs:

| shape | S | naive | naive_jit | static_e | two_pass | single | custom_vjp | relu | silu | gelu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `(1024, 64)` | 8 | 0.2167 | 0.0291 | 0.0435 | 0.0457 | 0.0442 | 0.0459 | 0.0240 | 0.0408 | 0.0422 |
| `(4096, 128)` | 8 | 0.7567 | 0.1682 | 0.1578 | 0.1593 | 0.1587 | 0.1653 | 0.0815 | 0.1749 | 0.1809 |
| `(4096, 128)` | 16 | 0.8512 | 0.1646 | 0.1587 | 0.1482 | 0.1627 | 0.1485 | 0.0815 | 0.1749 | 0.1809 |
| `(8192, 256)` | 16 | 2.8630 | 0.5633 | 0.5845 | 0.4813 | 0.5308 | 0.4845 | 0.3150 | 0.6145 | 0.6288 |

Times are milliseconds. On this machine, `two_pass` and public `custom_vjp` are the fastest rCoLU paths for the larger shapes. The custom VJP is mainly expected to help backward/training workloads.

## Cap rotation experiment

```bash
python benchmarks/train_cap_rotation.py --out results/cap_rotation
python benchmarks/train_cap_rotation.py --activations relu colu rcolu
```

This trains a one-hidden-layer network to map a 3D spherical cap to a fixed rotated copy of the cap. It writes:

- `cap_rotation_curves.csv`
- `cap_rotation_curves.pdf`
- `cap_rotation_final.pdf`

Local CPU result after 400 steps:

| activation | eval MSE |
|---|---:|
| ReLU | `1.48e-3` |
| CoLU | `7.26e-4` |
| rCoLU | `8.59e-4` |

## Development

```bash
pytest -q
pytest -m gpu
pytest -m tpu
```

GPU and TPU tests are marked and skipped automatically when the hardware backend is unavailable. The GPU suite should be run on at least one supported Ampere-or-newer NVIDIA device before publishing. Direct CoLU Pallas tests are tracked as non-blocking xfails until the padded/masked kernel rewrite is done; public `colu()` fallback tests remain blocking.

Before publishing to PyPI, run the GPU and TPU suites on real hardware and only
then push a `v*.*.*` tag. The `publish.yml` workflow is tag-only, so normal
branch pushes do not publish a package.

## License

Apache 2.0
