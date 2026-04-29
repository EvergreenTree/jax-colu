"""TPU backend entry points for jax-colu."""

from __future__ import annotations

from jax_colu.tpu._colu import colu_tpu
from jax_colu.tpu._rcolu import rcolu_tpu

__all__ = ["colu_tpu", "rcolu_tpu"]
