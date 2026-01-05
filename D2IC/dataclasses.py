from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence
from .types import Array


@dataclass(frozen=True)
class InitMotionConfig:
    """
    Configuration for coarse motion initialization (element-center matching only).
    Stage-1: placeholders only.
    """
    win: int = 41                 # patch size (odd)
    search: int = 24              # translation search radius (pixels)
    score_min: Optional[float] = 0.2
    max_centers: Optional[int] = None
    # Later: pyramid levels, chunk sizes, sub-sampling policies, etc.


@dataclass(frozen=True)
class MeshDICConfig:
    """
    Configuration for fine mesh-based DIC.
    Stage-1: placeholders only.
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
    """
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DICResult:
    """
    Standard DIC output.
    - u_nodal: nodal displacement field (Nn, 2)
    - strain: placeholder strain output
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
    Stage-1: minimal knobs.
    """
    # If True, run init motion on each frame to get u0 guess (placeholder flow)
    use_init_motion: bool = True

    # If True, warm-start each frame with previous frame result (common in DIC sequences)
    warm_start_from_previous: bool = True

    # Stage-2 flags for controlling init-motion usage across frames
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
    Batch output: a list of per-frame DICResult.
    """
    results: list[DICResult]
    diagnostics: BatchDiagnostics
