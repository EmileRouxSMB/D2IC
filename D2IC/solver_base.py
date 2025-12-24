from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from .types import Array


class SolverBase(ABC):
    """
    Base contract: solver owns compilation (JIT) and solve loops.
    """

    @abstractmethod
    def compile(self, assets: Any) -> None:
        """
        Prepare/jit functions based on asset shapes.
        Stage-1: may be a no-op, but should exist.
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self, state: Any, def_image: Array) -> Any:
        """
        Run solver and return a solver-specific result object.
        """
        raise NotImplementedError
