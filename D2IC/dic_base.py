from __future__ import annotations

from abc import ABC, abstractmethod
from .types import Array
from .mesh_assets import MeshAssets
from .dataclasses import DICResult


class DICBase(ABC):
    """
    Base DIC pipeline: prepare() then run().
    """

    @abstractmethod
    def prepare(self, ref_image: Array, assets: MeshAssets) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, def_image: Array) -> DICResult:
        raise NotImplementedError
