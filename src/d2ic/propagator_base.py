from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from .types import Array


class DisplacementPropagatorBase(ABC):
    """
    Strategy interface for warm-start displacement propagation across a sequence.
    """

    @abstractmethod
    def propagate(
        self,
        u_prev: Optional[Array],
        u_prevprev: Optional[Array] = None,
    ) -> Optional[Array]:
        """
        Produce a warm-start displacement from previously computed displacements.

        Parameters
        ----------
        u_prev:
            Displacement from previous frame (k-1), shape (Nn, 2).
        u_prevprev:
            Displacement from frame (k-2), shape (Nn, 2), optional.

        Returns
        -------
        u_warm:
            Proposed warm-start displacement, or None if not available.
        """
        raise NotImplementedError
