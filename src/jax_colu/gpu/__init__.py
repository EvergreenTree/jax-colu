"""GPU Pallas backend for jax-colu."""

from __future__ import annotations

from jax_colu.gpu._rcolu import rcolu_gpu
from jax_colu.gpu._rcolu_mgpu import rcolu_mgpu

__all__ = ["rcolu_gpu", "rcolu_mgpu"]
