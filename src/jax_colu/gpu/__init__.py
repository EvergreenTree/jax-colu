"""GPU Pallas backend for jax-colu."""

from __future__ import annotations

from jax_colu.gpu._rcolu import rcolu_gpu

__all__ = ["rcolu_gpu"]
