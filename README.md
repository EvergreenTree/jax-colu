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
- `rcolu` uses a Pallas kernel on (a) Ampere-or-newer NVIDIA GPUs and (b) TPUs, for hard-scaling calls with `axis=-1` and a scalar `dim`.
- Pre-Ampere NVIDIA GPUs such as T4/V100, unknown GPUs, and unsupported argument combinations fall back to the JAX reference path.
- `colu` always uses the JAX reference path. A padded/masked Pallas CoLU kernel will land in a later release; the previously experimental `jax_colu.gpu._colu.colu_gpu` and `jax_colu.tpu._colu.colu_tpu` modules have been removed because they were unreachable from public dispatch.

Both Pallas backends share a single fused kernel body (`src/jax_colu/_kernel_math.py`); the GPU and TPU wrappers differ only in tile geometry. The forward kernel processes a contiguous block of cone groups per program (multi-group blocking) — `JAX_COLU_BLOCK=N` overrides the auto-picked block size. The backward kernel recomputes `t`, `r`, `sc` from the saved input rather than carrying scalar-per-row residuals; this halves residual memory and avoids `(BLOCK_M, 1)` lane-padding pitfalls on TPU.

For local experiments, set `JAX_COLU_FORCE_PALLAS=1` to bypass the architecture guard or `JAX_COLU_DISABLE_PALLAS=1` to force reference dispatch (both env vars apply on GPU and TPU).

Current local Metal support can be installed but still fail basic `device_put`, so benchmarks record those failures explicitly.

Blackwell-specific tuning is exposed via `JAX_COLU_BLOCK`; the auto-picker targets ~1024 elements per program on GPU and the largest sublane-aligned block (multiple of 8) on TPU. Future work: power-of-two padded group widths with masks for non-pow2 `dim`, dim/dtype specialization, and per-architecture sweeps of block size and program shape.

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
- Test command: `python -m pytest -m gpu` (`54 passed`).

Blackwell GPU medians below are for `batch=4096`, `channels=256`; times are milliseconds. The public `rcolu` path uses the multi-group Pallas kernel on this backend, while public `colu` remains the JAX reference path.

```bash
python benchmarks/run_benchmarks.py --devices gpu --out results/gpu_blackwell --batch 4096 --channels 256 --dims 4 8 16 32 --warmup 10 --repeat 200
```

<table>
  <thead>
    <tr>
      <th rowspan="2">shape</th>
      <th rowspan="2">S</th>
      <th colspan="6">rCoLU</th>
      <th colspan="2">CoLU</th>
      <th colspan="3">jax.nn</th>
    </tr>
    <tr>
      <th>naive</th>
      <th>naive_jit</th>
      <th>static_e</th>
      <th>two_pass</th>
      <th>single</th>
      <th>jax_colu</th>
      <th>raw_jax</th>
      <th>jax_colu</th>
      <th>relu</th>
      <th>silu</th>
      <th>gelu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>(4096, 256)</code></td>
      <td align="right">4</td>
      <td align="right">0.5137</td>
      <td align="right">0.0327</td>
      <td align="right">0.0275</td>
      <td align="right">0.0630</td>
      <td align="right">0.0237</td>
      <td align="right">0.0274</td>
      <td align="right">0.0268</td>
      <td align="right">0.0272</td>
      <td align="right">0.0191</td>
      <td align="right">0.0283</td>
      <td align="right">0.0162</td>
    </tr>
    <tr>
      <td><code>(4096, 256)</code></td>
      <td align="right">8</td>
      <td align="right">0.4947</td>
      <td align="right">0.0291</td>
      <td align="right">0.0279</td>
      <td align="right">0.0609</td>
      <td align="right">0.0255</td>
      <td align="right">0.0233</td>
      <td align="right">0.0257</td>
      <td align="right">0.0260</td>
      <td align="right">0.0191</td>
      <td align="right">0.0283</td>
      <td align="right">0.0162</td>
    </tr>
    <tr>
      <td><code>(4096, 256)</code></td>
      <td align="right">16</td>
      <td align="right">0.3655</td>
      <td align="right">0.0275</td>
      <td align="right">0.0514</td>
      <td align="right">0.0284</td>
      <td align="right">0.0255</td>
      <td align="right">0.0235</td>
      <td align="right">0.0250</td>
      <td align="right">0.0257</td>
      <td align="right">0.0191</td>
      <td align="right">0.0283</td>
      <td align="right">0.0162</td>
    </tr>
    <tr>
      <td><code>(4096, 256)</code></td>
      <td align="right">32</td>
      <td align="right">0.4038</td>
      <td align="right">0.0282</td>
      <td align="right">0.0282</td>
      <td align="right">0.0257</td>
      <td align="right">0.0257</td>
      <td align="right">0.0277</td>
      <td align="right">0.0297</td>
      <td align="right">0.0298</td>
      <td align="right">0.0191</td>
      <td align="right">0.0283</td>
      <td align="right">0.0162</td>
    </tr>
  </tbody>
</table>

The fused GPU path is now in the same latency range as the best raw-JAX rCoLU variants for this shape.

GPU optimization follow-up should focus on large channel widths with small `S`
(`S=4` and `S=8`). The next useful steps are to sweep larger `JAX_COLU_BLOCK`
values for those dims, benchmark wider shapes such as `(1024, 4096)` and
`(1024, 8192)`, and consider dim-specialized unrolled Pallas kernels if block
tuning does not beat the best raw-JAX baseline. Forward+backward benchmarks
should be tracked separately, since saving compact per-group residuals may be
worth comparing against the current recompute-in-backward path.

TPU v0.2.0 medians (batch=4096, channels=256, repeat=50) — also from before the new TPU Pallas kernel landed:

| op | implementation | dim=4 | dim=8 | dim=16 | dim=32 |
|---|---|---:|---:|---:|---:|
| rcolu | custom_vjp (reference fallback) | 0.1715 | 0.1483 | 0.1520 | 0.1340 |
| rcolu | two_pass | 0.1704 | 0.1507 | 0.1302 | 0.1346 |
| rcolu | static_e | 0.2175 | 0.1508 | 0.1274 | 0.1502 |
| colu | jax_colu (reference) | 0.1802 | 0.1514 | 0.1353 | 0.1335 |
| jax.nn | relu / silu / gelu | 0.1257 | 0.1288 | 0.1265 | — |

Times are milliseconds. The v0.3.0 TPU rCoLU Pallas kernel uses sublane-aligned blocking (BM multiple of 8); refresh this table on a real TPU host once the kernel is benchmarked.

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

GPU and TPU tests are marked and skipped automatically when the hardware backend is unavailable. The GPU suite should be run on at least one supported Ampere-or-newer NVIDIA device, and the TPU suite on a real TPU host, before publishing. Public `colu()` always falls back to the reference implementation; the experimental CoLU Pallas modules and their xfails were removed.

Before publishing to PyPI, run the GPU and TPU suites on real hardware and only
then push a `v*.*.*` tag. The `publish.yml` workflow is tag-only, so normal
branch pushes do not publish a package.

## License

Apache 2.0
