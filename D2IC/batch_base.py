from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
from .types import Array
from .dataclasses import BatchResult


class BatchBase(ABC):
    """
    Abstract batch runner.

    Lifecycle
    ---------
    - :meth:`before`: pre-calculation preparation (compilation, warmup, etc.)
    - :meth:`sequence`: process the image sequence
    - :meth:`end`: optional post-processing/finalization
    - :meth:`run`: orchestrates the three steps above
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
        """Prepare the batch run for a given image sequence."""
        raise NotImplementedError

    @abstractmethod
    def sequence(self, images: Sequence[Array]) -> BatchResult:
        """Run the per-frame processing loop and return the batch result."""
        raise NotImplementedError

    @abstractmethod
    def end(self, result: BatchResult) -> BatchResult:
        """Finalize the batch output (e.g. aggregation, saving) and return it."""
        raise NotImplementedError
