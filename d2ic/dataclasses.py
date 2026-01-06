from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence
from .types import Array


@dataclass(frozen=True)
class InitMotionConfig:
    """
    Configuration for coarse motion initialization (element-center matching).

    This stage provides an initial displacement guess (typically a translation)
    before running the full mesh-based refinement.

    Attributes
    ----------
    win:
        Square patch size (in pixels) used to compare image content around an
        element center. Must be odd so that the patch has a well-defined center.
    search:
        Maximum translation magnitude (in pixels) explored in both x and y
        around each element center.
    score_min:
        Minimum acceptable similarity score to accept a match. Set to ``None``
        to disable score-based rejection.
    max_centers:
        Optional cap on the number of element centers evaluated (useful to
        reduce cost on very large meshes). Set to ``None`` for no cap.
    """
    win: int = 41
    search: int = 24
    score_min: Optional[float] = 0.2
    max_centers: Optional[int] = None
    # Later: pyramid levels, chunk sizes, sub-sampling policies, etc.


@dataclass(frozen=True)
class MeshDICConfig:
    """
    Configuration for mesh-based DIC refinement.

    Attributes
    ----------
    max_iters:
        Maximum number of nonlinear/outer iterations allowed.
    tol:
        Convergence tolerance on the solver stopping criterion (implementation-defined,
        e.g., relative update norm or residual norm).
    reg_strength:
        Regularization weight applied to the displacement field (0 disables
        regularization).
    strain_gauge_length:
        Characteristic length scale (in mesh units) used by strain post-processing
        or smoothing. Set to ``0`` to disable length-based smoothing.
    strain_eps:
        Small positive number used to prevent division-by-zero or ill-conditioning
        in strain-related computations.
    save_history:
        If True, retain per-iteration history in the result (e.g., residuals,
        step norms, energies) when the solver provides it.
    compute_discrepancy_map:
        If True, compute a pixel-wise discrepancy map (grey-level residual) after
        solving. The map is stored in `DICResult.pixel_maps["discrepancy_ref"]`.
    """
    max_iters: int = 50
    tol: float = 1e-6
    reg_strength: float = 0.0
    strain_gauge_length: float = 0.0
    strain_eps: float = 1e-8
    save_history: bool = True
    compute_discrepancy_map: bool = False


@dataclass(frozen=True)
class DICDiagnostics:
    """
    Free-form diagnostics container.

    Notes
    -----
    This is intentionally unstructured: solvers and pipelines can attach any
    metadata needed for downstream inspection (iteration counts, timings, etc.).

    Attributes
    ----------
    info:
        Arbitrary key/value payload with solver- and pipeline-specific metadata.
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
        Diagnostics payload for traceability and debugging.
    history:
        Optional solver history (implementation-defined), typically storing
        per-iteration scalars and/or intermediate states.
    fields:
        Optional user-defined scalar nodal fields. Each value must have shape
        ``(Nn,)`` and can be plotted via `DICPlotter.plot(<name>)`.
    pixel_maps:
        Optional user-defined pixel-wise maps. Each value must have shape ``(H, W)``.
    """
    u_nodal: Array
    strain: Array
    diagnostics: DICDiagnostics
    history: Array | None = None
    fields: Dict[str, Array] = field(default_factory=dict)
    pixel_maps: Dict[str, Array] = field(default_factory=dict)


# ----------------------------
# Batch processing dataclasses
# ----------------------------


@dataclass(frozen=True)
class BatchConfig:
    """
    Configuration for batch DIC processing.

    This configuration controls how an image sequence is processed frame-by-frame,
    how initial guesses are propagated, and how results/plots are exported.

    Attributes
    ----------
    use_init_motion:
        If True, run the coarse init-motion stage to compute an initial guess.
    warm_start_from_previous:
        If True, initialize each frame using the previous frame solution (common
        for temporally smooth DIC sequences).
    init_motion_first_frame_only:
        If True, run init-motion only on the first frame, then rely on temporal
        propagation for subsequent frames.
    init_motion_every_frame:
        If True, force init-motion on every frame regardless of propagation.
    prefer_init_motion_over_propagation:
        If True and both init-motion and propagation are available, prefer the
        init-motion guess over the propagated guess.
    verbose:
        If True, enable more detailed logging.
    progress:
        If True, display a progress indicator during batch processing.
    save_per_frame:
        If True, save per-frame numerical outputs (e.g., ``u_nodal`` and ``strain``)
        to disk.
    per_frame_dir:
        Output directory for per-frame numerical files (required when
        ``save_per_frame`` is True).
    export_png:
        If True, export per-frame visualization images (PNG).
    export_frames:
        Optional list/sequence of frame indices to export. If ``None``, export
        all frames when ``export_png`` is enabled.
    png_dir:
        Output directory for PNG exports (required when ``export_png`` is True).
    plot_fields:
        Field identifiers to plot (implementation-defined, commonly displacement
        components and strain components).
    plot_include_discrepancy:
        If True, include an additional discrepancy/error field when supported.
    plot_cmap:
        Matplotlib colormap name used for scalar fields.
    plot_alpha:
        Alpha blending factor for overlay plots (0 fully transparent, 1 opaque).
    plot_mesh:
        If True, draw the mesh overlay on plots when supported.
    plot_dpi:
        Resolution (dots per inch) of exported PNG figures.
    plot_binning:
        Image binning / downsampling factor applied for visualization speed.
    plot_projection:
        Plot projection mode; ``"fast"`` selects a faster approximate projection.
        Some pipelines accept ``False`` to disable projection.
    """
    use_init_motion: bool = True

    warm_start_from_previous: bool = True

    init_motion_first_frame_only: bool = False
    init_motion_every_frame: bool = False
    prefer_init_motion_over_propagation: bool = False
    verbose: bool = False
    progress: bool = False
    save_per_frame: bool = False
    per_frame_dir: Optional[str] = None
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
    """
    Free-form diagnostics container for batch processing.

    Attributes
    ----------
    info:
        Arbitrary key/value payload with batch-level metadata (timings, frame
        counts, failure modes, etc.).
    """
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResult:
    """
    Batch output: per-frame results plus batch-level diagnostics.

    Attributes
    ----------
    results:
        Ordered list of per-frame `DICResult`.
    diagnostics:
        Batch-level diagnostics payload.
    """
    results: list[DICResult]
    diagnostics: BatchDiagnostics
