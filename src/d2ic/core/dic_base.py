from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import Array
from .dataclasses import DICResult


class DICBase(ABC):
    """
    Base interface for DIC pipelines.

    The typical usage pattern is:
    1) call :meth:`prepare` once for a reference image and domain assets
    2) call :meth:`run` for each deformed image

    Subclasses choose their own domain representation: mesh assets for FE-DIC,
    subset grids for local DIC, or another method-specific support.
    """

    @abstractmethod
    def prepare(self, ref_image: Array, domain: Any) -> None:
        """
        Prepare internal state for a given reference image.

        Parameters
        ----------
        ref_image:
            Reference image array (H, W).
        domain:
            Method-specific domain data (mesh assets, subset centers, custom
            support, ...).
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
