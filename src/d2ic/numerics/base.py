from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol
from ..core.types import Array
from ..mesh.assets import MeshAssets
from ..core.dataclasses import MeshDICConfig


class SolverState(Protocol):
    """Minimal structural state shared by all DIC solvers."""

    ref_image: Array
    assets: MeshAssets


class MeshSolverState(SolverState, Protocol):
    """State required by nodal mesh solvers."""

    config: MeshDICConfig
    nodes_xy_device: Array
    u0_nodal: Array | None


class SolverBase(ABC):
    """
    Base contract: solver owns compilation (JIT) and solve loops.
    """

    @abstractmethod
    def compile(self, assets: MeshAssets) -> None:
        """
        Prepare/jit functions based on asset shapes.
        May be a no-op for some solvers, but should exist.
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self, state: SolverState, def_image: Array) -> object:
        """
        Run solver and return a solver-specific result object.
        """
        raise NotImplementedError
