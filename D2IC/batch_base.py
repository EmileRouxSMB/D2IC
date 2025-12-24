from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
from .types import Array
from .dataclasses import BatchResult


class BatchBase(ABC):
    """
    Abstract batch runner:
    - before(): pre-calculation preparation
    - sequence(): process the image series
    - end(): post-processing / finalization
    - run(): orchestrates these 3 steps
    """

    def __init__(self) -> None:
        super().__init__()

    def run(self, images: Sequence[Array]) -> BatchResult:
        """
        Orchestrate the batch process. Minimal, deterministic control flow.

        Parameters
        ----------
        images:
            Sequence of deformed images. The reference image is assumed to be
            provided in the subclass constructor or via before().
        """
        self.before(images)
        result = self.sequence(images)
        result = self.end(result)
        return result

    @abstractmethod
    def before(self, images: Sequence[Array]) -> None:
        raise NotImplementedError

    @abstractmethod
    def sequence(self, images: Sequence[Array]) -> BatchResult:
        raise NotImplementedError

    @abstractmethod
    def end(self, result: BatchResult) -> BatchResult:
        raise NotImplementedError
