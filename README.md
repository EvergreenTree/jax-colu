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
- TPU currently uses the JAX reference path. TPU Pallas kernels should be added only after a TPU-specific padded/masked lowering rewrite and validation on real hardware.

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

GPU validation on 2026-04-29:

- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 580.82.07, CUDA 13.0.
- Software: Python 3.12.13, JAX 0.7.2, jaxlib 0.7.2.
- Test command: `python -m pytest -m gpu`.
- Result: `54 passed, 49 deselected, 16 xfailed`.
- Full publish validation: `python -m pytest` reported `97 passed, 6 skipped, 16 xfailed`.
- Package build: `python -m build --no-isolation` successfully built `jax_colu-0.2.0.tar.gz` and `jax_colu-0.2.0-py3-none-any.whl`; `twine check dist/*` passed for both files.
- Passing coverage: f32 and bf16 public `rcolu()` GPU dispatch, public `colu()` fallback dispatch, direct rCoLU Pallas correctness, rCoLU GPU VJP checks, and the rCoLU Pallas performance smoke test.
- Expected xfails: direct experimental CoLU Pallas tests.

Blackwell GPU raw activation medians from:

```bash
python benchmarks/run_benchmarks.py --devices gpu --out results/gpu_blackwell --batch 4096 --channels 256 --dims 4 8 16 32 --warmup 10 --repeat 200
```

| op | S | naive | naive_jit | static_e | two_pass | single | jax_colu/custom_vjp | raw_jax | relu | silu | gelu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rCoLU | 4 | 0.4529 | 0.0310 | 0.0312 | 0.0630 | 0.0211 | 0.1999 | - | - | - | - |
| rCoLU | 8 | 0.4438 | 0.0306 | 0.0307 | 0.0388 | 0.0273 | 0.1158 | - | - | - | - |
| rCoLU | 16 | 0.4539 | 0.0305 | 0.0308 | 0.0291 | 0.0274 | 0.0802 | - | - | - | - |
| rCoLU | 32 | 0.4588 | 0.0283 | 0.0308 | 0.0286 | 0.0276 | 0.0613 | - | - | - | - |
| CoLU | 4 | - | - | - | - | - | 0.0250 | 0.0247 | - | - | - |
| CoLU | 8 | - | - | - | - | - | 0.0273 | 0.0271 | - | - | - |
| CoLU | 16 | - | - | - | - | - | 0.0273 | 0.0273 | - | - | - |
| CoLU | 32 | - | - | - | - | - | 0.0280 | 0.0270 | - | - | - |
| JAX activations | - | - | - | - | - | - | - | - | 0.0255 | 0.0250 | 0.0253 |

Times are milliseconds. On this Blackwell run, the public rCoLU Pallas path is slower than the best raw JAX rCoLU variants for forward-only latency. Public CoLU matches raw JAX because public dispatch intentionally uses the reference implementation.

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
