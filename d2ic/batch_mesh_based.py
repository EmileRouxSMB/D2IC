from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional, Iterable

import numpy as np
import jax.numpy as jnp

from .batch_base import BatchBase
from .dic_mesh_based import DICMeshBased
from .mesh_assets import MeshAssets
from .types import Array
from .dataclasses import BatchConfig, BatchResult, BatchDiagnostics, DICResult
from .propagator_base import DisplacementPropagatorBase


@dataclass
class BatchState:
    ref_image: Array
    assets: MeshAssets
    is_prepared: bool = False


class BatchMeshBased(BatchBase):
    """
    Concrete batch runner for mesh-based DIC.

    - Uses injected DICMeshBased instances.
    - before(): prepares both pipelines with ref_image/assets.
    - sequence(): runs per-frame DICMeshBased.run(def_image), optionally using a propagator.
    - end(): optional post-processing hook.

    Notes:
    - This class assumes ref_image and assets are fixed for the whole batch.
    """

    def __init__(
        self,
        ref_image: Array,
        assets: MeshAssets,
        dic_mesh: DICMeshBased,
        batch_config: BatchConfig,
        dic_local: Optional[DICMeshBased] = None,
        propagator: Optional[DisplacementPropagatorBase] = None,
    ) -> None:
        super().__init__()
        self.ref_image = ref_image
        self.assets = assets
        self.dic_mesh = dic_mesh
        self.dic_local = dic_local
        self.config = batch_config
        self.propagator = propagator
        self._state = BatchState(ref_image=ref_image, assets=assets, is_prepared=False)

    def before(self, images: Sequence[Array]) -> None:
        # Prepare mesh-based DIC pipeline
        self.dic_mesh.prepare(self.ref_image, self.assets)

        # Prepare local refinement pipeline if provided
        if self.dic_local is not None:
            self.dic_local.prepare(self.ref_image, self.assets)

        self._state.is_prepared = True

    def sequence(self, images: Sequence[Array] | Iterable[Array]) -> BatchResult:
        if not self._state.is_prepared:
            raise RuntimeError("BatchMeshBased.before() must be called before sequence().")

        save_per_frame = bool(getattr(self.config, "save_per_frame", False))
        keep_results = bool(getattr(self.config, "keep_results", True))
        per_frame_dir = getattr(self.config, "per_frame_dir", None)
        if save_per_frame:
            if per_frame_dir is None:
                per_frame_dir = Path.cwd() / "_outputs" / "per_frame_fields"
            else:
                per_frame_dir = Path(per_frame_dir)
            per_frame_dir.mkdir(parents=True, exist_ok=True)

        progress = bool(getattr(self.config, "progress", False))
        verbose = bool(getattr(self.config, "verbose", False))

        export_png = bool(getattr(self.config, "export_png", False))
        export_frames = getattr(self.config, "export_frames", None)
        png_dir = None
        plot_fields = None
        plot_cmap = None
        plot_alpha = None
        plot_mesh = None
        plot_dpi = None
        plot_binning = None
        plot_projection = None
        pixel_assets = None
        if export_png:
            from .plotter import DICPlotter
            from .pixel_assets import build_pixel_assets
            import matplotlib.pyplot as plt

            png_dir_cfg = getattr(self.config, "png_dir", None)
            if png_dir_cfg is None:
                png_dir = Path.cwd() / "_outputs" / "png"
            else:
                png_dir = Path(png_dir_cfg)
            png_dir.mkdir(parents=True, exist_ok=True)

            plot_fields = list(getattr(self.config, "plot_fields", ("u1", "u2", "e11", "e22", "e12")))
            if bool(getattr(self.config, "plot_include_discrepancy", False)):
                plot_fields.append("discrepancy")
            plot_cmap = getattr(self.config, "plot_cmap", "jet")
            plot_alpha = float(getattr(self.config, "plot_alpha", 0.6))
            plot_mesh = bool(getattr(self.config, "plot_mesh", True))
            plot_dpi = int(getattr(self.config, "plot_dpi", 200))
            plot_binning = float(getattr(self.config, "plot_binning", 1.0))
            plot_projection = getattr(self.config, "plot_projection", "fast")

            if export_frames is not None:
                export_frames = set(int(f) for f in export_frames)

            pixel_assets = self.assets.pixel_data
            if pixel_assets is None:
                roi_mask = getattr(self.assets, "roi_mask", None)
                if roi_mask is not None and roi_mask.shape != self.ref_image.shape:
                    roi_mask = None
                pixel_assets = build_pixel_assets(
                    mesh=self.assets.mesh,
                    ref_image=self.ref_image,
                    binning=plot_binning,
                    roi_mask=roi_mask,
                )

        per_frame: list[DICResult] = []
        n_processed = 0
        u_prev = None
        u_prevprev = None

        n_frames: int | None
        try:
            n_frames = len(images)  # type: ignore[arg-type]
        except TypeError:
            n_frames = None

        for k, Idef in enumerate(images):
            n_processed += 1
            if progress or verbose:
                if n_frames is None:
                    print(f"[Batch] Frame {k + 1}: start")
                else:
                    print(f"[Batch] Frame {k + 1}/{n_frames}: start")
            if self.propagator is not None:
                u_warm = self.propagator.propagate(u_prev=u_prev, u_prevprev=u_prevprev)
                if verbose:
                    print(f"  init: propagator={self.propagator.__class__.__name__}")
            else:
                u_warm = u_prev if self.config.warm_start_from_previous else None
                if verbose:
                    if u_warm is None:
                        print("  init: none")
                    else:
                        print("  init: warm-start from previous frame")

            if u_warm is not None:
                # Copy to avoid donated buffers invalidating warm-start history.
                self.dic_mesh.set_initial_guess(jnp.copy(jnp.asarray(u_warm)))

            cg_res = self.dic_mesh.run(Idef)
            if verbose:
                _print_history(cg_res.history, label="CG")
            res = cg_res
            if self.dic_local is not None:
                # Local refinement directly chained after CG for each frame.
                # Copy to keep CG output valid after donated local solve.
                self.dic_local.set_initial_guess(jnp.copy(jnp.asarray(cg_res.u_nodal)))
                res = self.dic_local.run(Idef)
                if verbose:
                    _print_history(res.history, label="Local")
            if keep_results:
                per_frame.append(res)
            if save_per_frame:
                out_path = per_frame_dir / f"frame_{k:04d}.npz"
                payload = {
                    "u_nodal": np.asarray(res.u_nodal),
                    "strain": np.asarray(res.strain),
                }
                pixel_maps = getattr(res, "pixel_maps", None)
                if isinstance(pixel_maps, dict):
                    disc = pixel_maps.get("discrepancy_ref")
                    if disc is not None:
                        payload["discrepancy_ref"] = np.asarray(disc)
                np.savez_compressed(out_path, **payload)
                if verbose:
                    print(f"  save: {out_path}")

            # Update warm-start history for next frame.
            # Keep JAX arrays on-device; only transfer to host for I/O.
            u_prevprev = u_prev
            u_prev = res.u_nodal
            if progress or verbose:
                if n_frames is None:
                    print(f"[Batch] Frame {k + 1}: done")
                else:
                    print(f"[Batch] Frame {k + 1}/{n_frames}: done")

            if export_png:
                should_export = export_frames is None or k in export_frames
                if should_export:
                    prefix = png_dir / f"frame_{k:04d}"
                    # One plotter + one figure reused across fields.
                    plotter = DICPlotter(
                        result=res,
                        mesh=self.assets.mesh,
                        def_image=np.asarray(Idef),
                        ref_image=np.asarray(self.ref_image),
                        binning=plot_binning,
                        pixel_assets=pixel_assets,
                        project_on_deformed=plot_projection,
                    )
                    fig = None
                    for field in plot_fields:
                        fig, ax = plotter.plot(
                            field=field,
                            image_alpha=plot_alpha,
                            cmap=plot_cmap,
                            plotmesh=plot_mesh,
                        )
                        label = _plot_label(field)
                        ax.set_title(f"{label} (frame {k})")
                        fig.savefig(prefix.with_name(f"{prefix.name}_{label}.png"), dpi=plot_dpi)
                    if fig is not None:
                        plt.close(fig)

        diag = BatchDiagnostics(
            info={
                "stage": "batch_mesh_based",
                "n_frames": int(n_processed),
                "warm_start_from_previous": self.config.warm_start_from_previous,
                "keep_results": keep_results,
                "save_per_frame": save_per_frame,
            }
        )
        return BatchResult(results=per_frame, diagnostics=diag)

    def end(self, result: BatchResult) -> BatchResult:
        # Placeholder for post-processing (e.g. stacking, saving, summary stats).
        # Keep as identity; update diagnostics as needed.
        updated = BatchDiagnostics(info={**result.diagnostics.info, "post": "end() placeholder"})
        return BatchResult(results=result.results, diagnostics=updated)


def _plot_label(field: str) -> str:
    key = field.strip().lower().replace("_", "")
    mapping = {
        "u1": "U1",
        "u2": "U2",
        "e11": "E11",
        "e22": "E22",
        "e12": "E12",
        "discrepancy": "Discrepancy",
    }
    return mapping.get(key, field)


def _print_history(history, *, label: str) -> None:
    if label.strip().upper() == "CG":
        return
    if history is None:
        print(f"  {label}: history disabled")
        return
    arr = np.asarray(history)
    if arr.ndim != 2 or arr.shape[0] == 0:
        print(f"  {label}: empty history")
        return
    if arr.shape[1] == 3:
        # CG: [J, ||grad||, alpha]
        valid = np.isfinite(arr[:, 0])
        arr = arr[valid]
        print(f"  {label}: {arr.shape[0]} iters")
        for i, (J, g, a) in enumerate(arr):
            print(f"    - iter {i:02d}: J={J:.3e}, |grad|={g:.3e}, alpha={a:.3e}")
        return
    if arr.shape[1] == 2:
        # Local: [r_rms, step_rms]
        valid = np.isfinite(arr[:, 0])
        arr = arr[valid]
        print(f"  {label}: {arr.shape[0]} sweeps")
        for i, (r, step) in enumerate(arr):
            print(f"    - iter {i:02d}: r_rms={r:.3e}, step_rms={step:.3e}")
        return
    print(f"  {label}: history shape {arr.shape}")
