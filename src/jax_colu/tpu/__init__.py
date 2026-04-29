"""TPU Pallas backend for jax-colu."""

from __future__ import annotations

from jax_colu.tpu._rcolu import rcolu_tpu

__all__ = ["rcolu_tpu"]
