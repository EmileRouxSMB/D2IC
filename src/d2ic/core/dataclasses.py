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


@dataclass(frozen=True, init=False)
class DICResult:
    """
    Generic DIC output defined on a set of material/image points.

    Attributes
    ----------
    points_xy:
        Coordinates where the displacement is represented, shape ``(Np, 2)``.
        For FE-DIC this is the nodal coordinate array.
    displacement:
        Displacement field with shape ``(Np, 2)`` in ``(ux, uy)`` order.
    strain:
        Strain in Voigt form with shape ``(Np, 3)`` (typically ``[E11, E22, E12]``).
        Some pipelines may return zeros if strain post-processing is disabled.
    geometry:
        Optional representation-specific geometry (mesh, grid, point cloud,
        custom method support, ...). Plotters and projectors can use it when
        available.
    diagnostics:
        Diagnostics payload for traceability and debugging.
    history:
        Optional solver history (implementation-defined), typically storing
        per-iteration scalars and/or intermediate states.
    fields:
        Optional user-defined scalar fields. Each value must have shape ``(Np,)``
        and can be plotted via `DICPlotter.plot(<name>)`.
    pixel_maps:
        Optional user-defined pixel-wise maps. Each value must have shape ``(H, W)``.

    Notes
    -----
    ``u_nodal`` remains available as a compatibility alias for FE-DIC code.
    """
    points_xy: Array | None
    displacement: Array
    strain: Array | None
    diagnostics: DICDiagnostics
    geometry: Any | None
    history: Any | None
    fields: Dict[str, Array]
    pixel_maps: Dict[str, Array]

    def __init__(
        self,
        displacement: Array | None = None,
        strain: Array | None = None,
        diagnostics: DICDiagnostics | None = None,
        *,
        points_xy: Array | None = None,
        geometry: Any | None = None,
        history: Any | None = None,
        fields: Dict[str, Array] | None = None,
        pixel_maps: Dict[str, Array] | None = None,
        u_nodal: Array | None = None,
    ) -> None:
        if displacement is None:
            displacement = u_nodal
        if displacement is None:
            raise ValueError("DICResult requires `displacement` (or compatibility alias `u_nodal`).")
        object.__setattr__(self, "points_xy", points_xy)
        object.__setattr__(self, "displacement", displacement)
        object.__setattr__(self, "strain", strain)
        object.__setattr__(self, "diagnostics", DICDiagnostics() if diagnostics is None else diagnostics)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "fields", {} if fields is None else fields)
        object.__setattr__(self, "pixel_maps", {} if pixel_maps is None else pixel_maps)

    @property
    def u_nodal(self) -> Array:
        """Compatibility alias for mesh-based DIC displacement."""

        return self.displacement


@dataclass(frozen=True, init=False)
class MeshDICResult(DICResult):
    """
    DIC result represented on mesh nodes.

    ``mesh`` and ``geometry`` point to the same mesh object by default. The
    generic fields ``points_xy`` and ``displacement`` remain the canonical API.
    """

    mesh: Any | None

    def __init__(
        self,
        displacement: Array | None = None,
        strain: Array | None = None,
        diagnostics: DICDiagnostics | None = None,
        *,
        mesh: Any | None = None,
        points_xy: Array | None = None,
        geometry: Any | None = None,
        history: Any | None = None,
        fields: Dict[str, Array] | None = None,
        pixel_maps: Dict[str, Array] | None = None,
        u_nodal: Array | None = None,
    ) -> None:
        if mesh is not None and points_xy is None:
            points_xy = mesh.nodes_xy
        if geometry is None:
            geometry = mesh
        super().__init__(
            displacement=displacement,
            strain=strain,
            diagnostics=diagnostics,
            points_xy=points_xy,
            geometry=geometry,
            history=history,
            fields=fields,
            pixel_maps=pixel_maps,
            u_nodal=u_nodal,
        )
        object.__setattr__(self, "mesh", mesh if mesh is not None else geometry)


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
    export_vtu:
        If True, export one ParaView VTU file per frame and a PVD time
        collection. Fields are attached to the mesh nodes.
    vtu_dir:
        Output directory for VTU/PVD exports. Defaults to
        ``_outputs/paraview``.
    pvd_filename:
        Name of the PVD collection file.
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
    # Stream per-frame numeric outputs by default to avoid unbounded memory growth
    # on long sequences (and to allow early inspection of partial results).
    save_per_frame: bool = True
    per_frame_dir: Optional[str] = None
    keep_results: bool = True
    export_png: bool = False
    export_vtu: bool = False
    export_vtk_images: bool = False
    vtu_dir: Optional[str] = None
    pvd_filename: str = "dic_sequence.pvd"
    image_pvd_filename: str = "image_sequence.pvd"
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
