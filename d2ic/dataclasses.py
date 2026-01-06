from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence
from .types import Array


@dataclass(frozen=True)
class InitMotionConfig:
    """
    Configuration for coarse motion initialization (element-center matching).
    """
    win: int = 41                 # patch size (odd)
    search: int = 24              # translation search radius (pixels)
    score_min: Optional[float] = 0.2
    max_centers: Optional[int] = None
    # Later: pyramid levels, chunk sizes, sub-sampling policies, etc.


@dataclass(frozen=True)
class MeshDICConfig:
    """
    Configuration for mesh-based DIC refinement.
    """
    max_iters: int = 50
    tol: float = 1e-6
    reg_strength: float = 0.0
    strain_gauge_length: float = 0.0
    strain_eps: float = 1e-8
    save_history: bool = True


@dataclass(frozen=True)
class DICDiagnostics:
    """
    Free-form diagnostics container.

    Notes
    -----
    This is intentionally unstructured: solvers and pipelines can attach any
    metadata needed for downstream inspection (iteration counts, timings, etc.).
    """
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DICResult:
    """
    Standard DIC output.

    Attributes
    ----------
    u_nodal:
        Nodal displacement field with shape ``(Nn, 2)`` in ``(ux, uy)`` order.
    strain:
        Nodal strain in Voigt form with shape ``(Nn, 3)`` (typically ``[E11, E22, E12]``).
        Some pipelines may return zeros if strain post-processing is disabled.
    diagnostics:
        Dictionary-like diagnostics payload.
    history:
        Optional solver history (implementation-defined).
    """
    u_nodal: Array
    strain: Array
    diagnostics: DICDiagnostics
    history: Array | None = None


# ----------------------------
# Batch processing dataclasses
# ----------------------------


@dataclass(frozen=True)
class BatchConfig:
    """
    Configuration for batch DIC processing.
    """
    # If True, run init motion on each frame to get a u0 guess
    use_init_motion: bool = True

    # If True, warm-start each frame with previous frame result (common in DIC sequences)
    warm_start_from_previous: bool = True

    # Flags for controlling init-motion usage across frames
    init_motion_first_frame_only: bool = False
    init_motion_every_frame: bool = False
    prefer_init_motion_over_propagation: bool = False
    # Progress and logging
    verbose: bool = False
    progress: bool = False
    # Optional per-frame saving (npz with u_nodal/strain)
    save_per_frame: bool = False
    per_frame_dir: Optional[str] = None
    # Optional per-frame PNG export (driven by BatchMeshBased)
    export_png: bool = False
    export_frames: Sequence[int] | None = None
    png_dir: Optional[str] = None
    plot_fields: Sequence[str] = ("u1", "u2", "e11", "e22", "e12")
    plot_include_discrepancy: bool = False
    plot_cmap: str = "jet"
    plot_alpha: float = 0.6
    plot_mesh: bool = True
    plot_dpi: int = 200
    plot_binning: float = 1.0
    plot_projection: str | bool = "fast"


@dataclass(frozen=True)
class BatchDiagnostics:
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResult:
    """
    Batch output: per-frame results plus batch-level diagnostics.
    """
    results: list[DICResult]
    diagnostics: BatchDiagnostics
