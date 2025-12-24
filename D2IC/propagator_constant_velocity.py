from __future__ import annotations

from typing import Optional
from .types import Array
from .propagator_base import DisplacementPropagatorBase

import jax.numpy as jnp


class ConstantVelocityPropagator(DisplacementPropagatorBase):
    """
    Warm-start strategy assuming constant velocity:
        u_k \u2248 u_{k-1} + (u_{k-1} - u_{k-2}) = 2*u_prev - u_prevprev

    Fallback:
      - if u_prev is None: return None
      - if u_prevprev is None: return u_prev
    """

    def propagate(
        self,
        u_prev: Optional[Array],
        u_prevprev: Optional[Array] = None,
    ) -> Optional[Array]:
        if u_prev is None:
            return None
        if u_prevprev is None:
            return u_prev
        return 2.0 * u_prev - u_prevprev
