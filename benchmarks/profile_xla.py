"""Dump XLA HLO for reference or Pallas variants."""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import numpy as np

SHAPE = (2048, 32)
DIM = 32


def _x():
    return jnp.asarray(np.random.default_rng(0).normal(size=SHAPE).astype(np.float32))


def _get_fn(variant: str):
    if variant == "rcolu-pallas":
        from jax_colu.gpu._rcolu import rcolu_gpu

        return lambda z: rcolu_gpu(z, dim=DIM)
    if variant == "colu-pallas":
        from jax_colu.gpu._colu import colu_gpu

        return lambda z: colu_gpu(z, dim=DIM)
    if variant == "colu-ref":
        from jax_colu._reference import colu_reference

        return lambda z: colu_reference(z, dim=DIM)
    from jax_colu._reference import rcolu_reference

    return lambda z: rcolu_reference(z, dim=DIM)


def dump_hlo(variant: str, out: str):
    os.makedirs(out, exist_ok=True)
    os.environ["XLA_FLAGS"] = (
        f"--xla_dump_to={out} --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*"
    )
    fn = _get_fn(variant)
    jax.block_until_ready(fn(_x()))

    files = sorted(glob.glob(f"{out}/module_*.optimizations.txt"))
    if not files:
        print("No HLO dumps found.")
        return
    text = Path(files[-1]).read_text()
    print(f"=== HLO summary ({variant}) ===")
    print(f"fusion clusters : {text.count('kFusion')}")
    print(f"custom-call ops : {text.count('custom-call')}")
    print(f"reduce ops      : {text.count('reduce(')}")
    print(f"full dump       : {files[-1]}")


def run_nsys(variant: str):
    script = (
        "import sys, pathlib, numpy as np, jax, jax.numpy as jnp\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        f"from benchmarks.profile_xla import _get_fn, SHAPE, DIM\n"
        f"fn = _get_fn({variant!r})\n"
        "x = jnp.asarray(np.random.default_rng(0).normal(size=SHAPE).astype(np.float32))\n"
        "[jax.block_until_ready(fn(x)) for _ in range(20)]\n"
    )
    report = f"nsys_{variant}"
    subprocess.run(
        ["nsys", "profile", "-o", report, "--trace", "cuda,nvtx", "python", "-c", script],
        check=True,
    )
    print(f"Open: nsys-ui {report}.nsys-rep")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["rcolu-ref", "rcolu-pallas", "colu-ref", "colu-pallas"],
        default="rcolu-ref",
    )
    parser.add_argument("--out", default="/tmp/jax_colu_hlo")
    parser.add_argument("--nsys", action="store_true")
    args = parser.parse_args()
    run_nsys(args.variant) if args.nsys else dump_hlo(args.variant, args.out)


if __name__ == "__main__":
    main()
