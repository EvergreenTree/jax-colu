# jax-colu

Fused JAX/Pallas kernels for **Conic Linear Units** (CoLU & rCoLU).

- **Paper:** [Conic Activation Functions, PMLR 285](https://proceedings.mlr.press/v285/fu24a.html)
- **Repo:** https://github.com/EvergreenTree/jax-colu

## Cap rotation experiment

```bash
python benchmarks/train_cap_rotation.py --out results/cap_rotation
python benchmarks/train_cap_rotation.py --activations relu colu rcolu
```

This trains a one-hidden-layer network to map a 3D spherical cap to a fixed rotated copy of the cap.

Local CPU result after 400 steps:

| activation | eval MSE |
|---|---:|
| ReLU | `1.48e-3` |
| CoLU | `7.26e-4` |
| rCoLU | `8.59e-4` |

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
- `rcolu` uses a Pallas kernel on supported hard-scaling calls with `axis=-1`, scalar `dim`, and no `num_groups`.
- GPU defaults to the Pallas:Triton path on known Ampere-or-newer NVIDIA devices.
- Set `JAX_COLU_GPU_BACKEND=mgpu` to opt in to the Pallas:Mosaic GPU rCoLU path on known Hopper/Blackwell NVIDIA devices.
- TPU uses its TPU Pallas path when TPU devices are available.
- `colu` always uses the JAX reference path publicly.

Both Pallas backends share the fused rCoLU math in `src/jax_colu/_kernel_math.py`. The default GPU path processes multiple cone groups per program; `JAX_COLU_BLOCK=N` overrides the auto-picked block size. Set `JAX_COLU_FORCE_PALLAS=1` to bypass the default Triton/TPU guard, or `JAX_COLU_DISABLE_PALLAS=1` to force reference dispatch. `JAX_COLU_FORCE_PALLAS=1` does not bypass the Hopper/Blackwell guard for `JAX_COLU_GPU_BACKEND=mgpu`.

The Mosaic GPU backend is a correctness and architecture foothold for Hopper/Blackwell-specific work such as async TMA copies, explicit shared-memory layout control, and warp scheduling. Standalone rCoLU is bandwidth-bound, so large speedups are not expected until fused `rCoLU + {LayerNorm, residual}` kernels are added.

## Quick start

```python
import jax
from jax_colu import rcolu, colu

x = jax.random.normal(jax.random.PRNGKey(0), (8, 32))
y = rcolu(x, dim=4)         # homogeneous-axis variant
z = colu(x, dim=4)          # heterogeneous explicit-apex variant
```

## Install

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.11, JAX >= 0.10.0, and jaxlib >= 0.10.0.

## Benchmarks

Raw activation latency:

```bash
python benchmarks/run_benchmarks.py --out results/
JAX_PLATFORMS=cpu python benchmarks/run_benchmarks.py --devices cpu --out results/cpu
```

The benchmark reports median latency over repeated calls and includes:

- rCoLU variants: `naive`, `naive_jit`, `static_e`, `two_pass`, `single`, `custom_vjp`, `mgpu`
- CoLU: raw JAX and `jax_colu`
- `jax.nn.relu`, `jax.nn.silu`, `jax.nn.gelu`

CPU medians from local runs:

| shape | S | naive | naive_jit | static_e | two_pass | single | custom_vjp | relu | silu | gelu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `(1024, 64)` | 8 | 0.2167 | 0.0291 | 0.0435 | 0.0457 | 0.0442 | 0.0459 | 0.0240 | 0.0408 | 0.0422 |
| `(4096, 128)` | 8 | 0.7567 | 0.1682 | 0.1578 | 0.1593 | 0.1587 | 0.1653 | 0.0815 | 0.1749 | 0.1809 |
| `(4096, 128)` | 16 | 0.8512 | 0.1646 | 0.1587 | 0.1482 | 0.1627 | 0.1485 | 0.0815 | 0.1749 | 0.1809 |
| `(8192, 256)` | 16 | 2.8630 | 0.5633 | 0.5845 | 0.4813 | 0.5308 | 0.4845 | 0.3150 | 0.6145 | 0.6288 |

Times are milliseconds. On this machine, `two_pass` and public `custom_vjp` are the fastest rCoLU paths for the larger shapes.

Blackwell GPU validation on 2026-05-06:

- Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 590.48.01, CUDA 13 wheel stack.
- Software: Python 3.12.3, JAX 0.10.0, jaxlib 0.10.0.
- Default GPU test command: `python -m pytest -m gpu -q`.
- Mosaic GPU test command: `JAX_COLU_GPU_BACKEND=mgpu python -m pytest -m gpu -q`.
- Result for both commands: `93 passed, 69 deselected, 16 xfailed`.

Blackwell GPU medians below are for `batch=4096`, `channels=256`.

```bash
python benchmarks/run_benchmarks.py --devices gpu --out results/gpu_blackwell --batch 4096 --channels 256 --dims 4 8 16 32 --warmup 10 --repeat 200
JAX_COLU_GPU_BACKEND=mgpu python benchmarks/run_benchmarks.py --devices gpu --out results/gpu_mgpu_blackwell --batch 4096 --channels 256 --dims 4 8 16 32 --warmup 10 --repeat 200
```

| S | naive_jit | two_pass | single | jax_colu default | mgpu direct | relu | silu | gelu |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.0327 | 0.0630 | 0.0237 | 0.0274 | 0.6990 | 0.0191 | 0.0283 | 0.0162 |
| 8 | 0.0291 | 0.0609 | 0.0255 | 0.0233 | 0.3754 | 0.0191 | 0.0283 | 0.0162 |
| 16 | 0.0275 | 0.0284 | 0.0255 | 0.0235 | 0.1771 | 0.0191 | 0.0283 | 0.0162 |
| 32 | 0.0282 | 0.0257 | 0.0257 | 0.0277 | 0.1102 | 0.0191 | 0.0283 | 0.0162 |

Times are milliseconds. The default multi-group GPU path is now close to the best raw-JAX rCoLU variants for this shape. Mosaic GPU does not show a standalone activation boost; it is intended for fused Hopper/Blackwell follow-up kernels.

## Development

```bash
pytest -q
pytest -m gpu
pytest -m tpu
```

GPU and TPU tests are marked and skipped automatically when the hardware backend is unavailable. Public `colu()` always falls back to the reference implementation; experimental CoLU Pallas modules are not part of public dispatch.

Before publishing to PyPI, run the GPU and TPU suites on real hardware and only
then push a `v*.*.*` tag. The `publish.yml` workflow is tag-only, so normal
branch pushes do not publish a package.

## License

Apache 2.0
