from __future__ import annotations

from typing import Optional
from .types import Array
from .propagator_base import DisplacementPropagatorBase


class PreviousDisplacementPropagator(DisplacementPropagatorBase):
    """
    Warm-start strategy: u_warm = u_prev.
    """

    def propagate(
        self,
        u_prev: Optional[Array],
        u_prevprev: Optional[Array] = None,
    ) -> Optional[Array]:
        return u_prev
