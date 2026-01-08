from __future__ import annotations

from abc import ABC, abstractmethod
from .types import Array
from .mesh_assets import MeshAssets
from .dataclasses import DICResult


class DICBase(ABC):
    """
    Base interface for DIC pipelines.

    The typical usage pattern is:
    1) call :meth:`prepare` once for a reference image and mesh assets
    2) call :meth:`run` for each deformed image
    """

    @abstractmethod
    def prepare(self, ref_image: Array, assets: MeshAssets) -> None:
        """
        Prepare internal state for a given reference image.

        Parameters
        ----------
        ref_image:
            Reference image array (H, W).
        assets:
            Precomputed mesh assets (neighbors, pixel caches, ...).
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, def_image: Array) -> DICResult:
        """
        Run DIC for a single deformed image.

        Parameters
        ----------
        def_image:
            Deformed image array (H, W).

        Returns
        -------
        DICResult
            Displacement/strain outputs plus diagnostics.
        """
        raise NotImplementedError
