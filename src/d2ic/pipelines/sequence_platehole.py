from __future__ import annotations

from pathlib import Path
import time

import matplotlib
import numpy as np

try:  # optional mesh writer
    import meshio
except Exception:  # pragma: no cover
    meshio = None

from ..app_utils import configure_jax_platform, imread_gray, list_deformed_images, prepare_image, make_roi_mask
from ..batch_mesh_based import BatchMeshBased
from ..config import load_sequence_config
from ..dataclasses import BatchConfig, MeshDICConfig
from ..dic_mesh_based import DICMeshBased
from ..mesh_assets import make_mesh_assets
from ..mask2mesh import mask_to_mesh_assets, mask_to_mesh_assets_gmsh
from ..propagator_constant_velocity import ConstantVelocityPropagator
from ..propagator_previous import PreviousDisplacementPropagator
from ..solver_global_cg import GlobalCGSolver
from ..solver_local_gn import LocalGaussNewtonSolver


def run_sequence_from_config(cfg_path: Path) -> None:
    """Run the sequential DIC pipeline from a TOML config file."""
    cfg = load_sequence_config(Path(cfg_path))

    _configure_runtime(cfg)
    _log_verbose(cfg, f"[Config] Using config: {cfg_path}")
    t_total = time.perf_counter()

    # Non-interactive backend so figures can be saved without a display.
    matplotlib.use("Agg")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = cfg.out_dir / "png"
    if cfg.export_png:
        png_dir.mkdir(parents=True, exist_ok=True)

    ref_path = cfg.img_dir / cfg.ref_image_name
    mask_path = cfg.img_dir / cfg.mask_filename
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")

    def_paths = list_deformed_images(cfg.img_dir, cfg.image_pattern, exclude_name=cfg.ref_image_name)
    if not def_paths:
        raise RuntimeError(
            f"No deformed images found in {cfg.img_dir} with pattern '{cfg.image_pattern}'."
        )

    t0 = time.perf_counter()
    ref_image = prepare_image(ref_path, binning=cfg.image_binning)
    _log_elapsed(cfg, "Loaded reference image", t0)
    _log_verbose(cfg, f"[Images] ref shape: {ref_image.shape}")

    t0 = time.perf_counter()
    def_images = [prepare_image(p, binning=cfg.image_binning) for p in def_paths]
    _log_elapsed(cfg, f"Loaded {len(def_images)} deformed images", t0)
    if def_images:
        _log_verbose(cfg, f"[Images] def shape: {def_images[0].shape}")

    t0 = time.perf_counter()
    mask_image = imread_gray(mask_path)
    mask = make_roi_mask(mask_image)
    _log_elapsed(cfg, "Loaded mask image", t0)
    unique_vals = np.unique(mask_image)
    unique_count = unique_vals.size
    unique_min = float(unique_vals.min()) if unique_count else 0.0
    unique_max = float(unique_vals.max()) if unique_count else 0.0
    _log_verbose(
        cfg,
        f"[Mask] shape: {mask.shape}, coverage: {mask.mean():.3f}, "
        f"min/max: {unique_min:.3f}/{unique_max:.3f}, unique: {unique_count}"
    )
    if mask.mean() > 0.95:
        print(
            "[Mask] Warning: ROI covers >95% of the image. "
            "If this is unexpected, invert the mask or check roi.tif."
        )

    t0 = time.perf_counter()
    try:
        mesh, assets = mask_to_mesh_assets_gmsh(
            mask=mask,
            element_size_px=cfg.mesh_element_size_px,
            binning=cfg.image_binning,
            remove_islands=cfg.mesh_remove_islands,
            min_island_area_px=cfg.mesh_min_island_area_px,
            contour_step_px=cfg.mesh_gmsh_contour_step_px,
            optimize=cfg.mesh_gmsh_optimize,
            verbose=cfg.mesh_gmsh_verbose or cfg.verbose,
        )
        print("Generated mesh via Gmsh-based pipeline.")
    except Exception as exc:
        print(f"Gmsh meshing failed ({exc}); falling back to structured grid mesher.")
        mesh, _ = mask_to_mesh_assets(
            mask=mask,
            element_size_px=cfg.mesh_element_size_px,
            binning=cfg.image_binning,
            remove_islands=cfg.mesh_remove_islands,
            min_island_area_px=cfg.mesh_min_island_area_px,
        )
        assets = make_mesh_assets(mesh, with_neighbors=True)
    _log_elapsed(cfg, "Mesh generation", t0)
    _log_verbose(cfg, f"[Mesh] nodes: {len(mesh.nodes_xy)}, elements: {len(mesh.elements)}")

    if cfg.save_mesh:
        _export_mesh(cfg.out_dir, assets.mesh.nodes_xy, assets.mesh.elements)
    overlay_dir = png_dir if cfg.export_png else cfg.out_dir
    overlay_path = overlay_dir / "mesh_overlay.png"
    try:
        _save_mesh_overlay_png(overlay_path, ref_image, mesh, dpi=cfg.plot_dpi)
        print(f"Saved mesh overlay to {overlay_path}")
    except Exception as exc:
        print(f"Mesh overlay PNG failed ({exc}); continuing without overlay export.")

    mesh_cfg = MeshDICConfig(
        max_iters=cfg.max_iters,
        tol=cfg.tol,
        reg_strength=cfg.reg_strength,
        strain_gauge_length=cfg.strain_gauge_length,
        save_history=True,
        compute_discrepancy_map=(cfg.local_sweeps <= 0),
    )

    dic_mesh = DICMeshBased(
        mesh=mesh,
        solver=GlobalCGSolver(interpolation=cfg.interpolation, verbose=cfg.verbose),
        config=mesh_cfg,
    )

    batch_cfg = BatchConfig(
        use_init_motion=False,
        warm_start_from_previous=True,
        verbose=cfg.verbose,
        progress=True,
        keep_results=cfg.keep_results,
        export_png=cfg.export_png,
        export_frames=cfg.export_frames,
        png_dir=str(png_dir),
        plot_fields=cfg.plot_fields,
        plot_include_discrepancy=cfg.plot_include_discrepancy,
        plot_cmap=cfg.plot_cmap,
        plot_alpha=cfg.plot_alpha,
        plot_mesh=cfg.plot_mesh,
        plot_dpi=cfg.plot_dpi,
        plot_binning=cfg.plot_binning,
        plot_projection=cfg.plot_projection,
        save_per_frame=True,
    )

    propagator = ConstantVelocityPropagator() if cfg.use_velocity else PreviousDisplacementPropagator()
    dic_local = None
    if cfg.local_sweeps > 0:
        local_cfg = MeshDICConfig(
            max_iters=cfg.local_sweeps,
            tol=cfg.tol,
            reg_strength=cfg.reg_strength,
            strain_gauge_length=cfg.strain_gauge_length,
            save_history=True,
            compute_discrepancy_map=cfg.plot_include_discrepancy,
        )
        local_solver = LocalGaussNewtonSolver(
            lam=0.1,
            max_step=0.2,
            omega=0.5,
            interpolation=cfg.interpolation,
        )
        dic_local = DICMeshBased(mesh=mesh, solver=local_solver, config=local_cfg)

    batch = BatchMeshBased(
        ref_image=ref_image,
        assets=assets,
        dic_mesh=dic_mesh,
        batch_config=batch_cfg,
        dic_local=dic_local,
        propagator=propagator,
    )

    print("Preparing pipelines (JIT compile may take a while on first run)...")
    print(f"Running batch on {len(def_images)} frame(s).")
    t0 = time.perf_counter()
    batch_result = batch.run(def_images)
    _log_elapsed(cfg, "Batch run", t0)
    print("Batch run completed.")

    per_frame = batch_result.results
    if cfg.save_npz and not cfg.keep_results:
        print("[Output] save_npz disabled because keep_results=false (zero-accumulation mode).")
    if cfg.save_npz and cfg.keep_results:
        nodes_xy = np.asarray(assets.mesh.nodes_xy)
        u_stack = np.stack([np.asarray(r.u_nodal) for r in per_frame], axis=0)
        strain_stack = np.stack([np.asarray(r.strain) for r in per_frame], axis=0)
        discrepancy_stack = _stack_discrepancy(per_frame)

        payload = {
            "nodes_xy": nodes_xy,
            "u_nodal": u_stack,
            "strain": strain_stack,
            "ref_path": str(ref_path),
            "def_paths": [str(p) for p in def_paths],
        }
        if discrepancy_stack is not None:
            payload["discrepancy_ref"] = discrepancy_stack
        out_path = cfg.out_dir / "fields_sequence.npz"
        np.savez_compressed(out_path, **payload)
        print(f"Saved NPZ results to {out_path}")

    _log_elapsed(cfg, "Total pipeline", t_total)


def _configure_runtime(cfg: object) -> None:
    import jax

    if getattr(cfg, "jax_enable_x64", False):
        jax.config.update("jax_enable_x64", True)
    matmul_precision = getattr(cfg, "jax_matmul_precision", None)
    if matmul_precision:
        jax.config.update("jax_default_matmul_precision", matmul_precision)
    configure_jax_platform(
        preferred=getattr(cfg, "jax_preferred", "gpu"),
        fallback=getattr(cfg, "jax_fallback", "cpu"),
    )


def _log_verbose(cfg: object, msg: str) -> None:
    if bool(getattr(cfg, "verbose", False)):
        print(msg)


def _log_elapsed(cfg: object, label: str, t0: float) -> None:
    if bool(getattr(cfg, "verbose", False)):
        elapsed = time.perf_counter() - t0
        print(f"[Timing] {label}: {elapsed:.2f}s")


def _stack_discrepancy(per_frame: list) -> np.ndarray | None:
    discrepancy_frames = []
    for res in per_frame:
        pixel_maps = getattr(res, "pixel_maps", None)
        disc = None
        if isinstance(pixel_maps, dict):
            disc = pixel_maps.get("discrepancy_ref")
        if disc is None:
            discrepancy_frames = []
            break
        discrepancy_frames.append(np.asarray(disc, dtype=np.float32))
    if discrepancy_frames:
        return np.stack(discrepancy_frames, axis=0)
    return None


def _export_mesh(out_dir: Path, nodes_xy: np.ndarray, elements: np.ndarray) -> None:
    mesh_path = out_dir / "roi_mesh.msh"
    out_dir.mkdir(parents=True, exist_ok=True)
    points = np.column_stack([nodes_xy, np.zeros((nodes_xy.shape[0],), dtype=nodes_xy.dtype)])
    if meshio is None:
        _write_gmsh_v2_quad(mesh_path, points, np.asarray(elements, dtype=np.int64))
        print(f"Saved mesh to {mesh_path} (gmsh v2 writer)")
        return
    meshio_mesh = meshio.Mesh(points=points, cells=[("quad", np.asarray(elements, dtype=np.int32))])
    meshio.write(str(mesh_path), meshio_mesh, file_format="gmsh")
    print(f"Saved mesh to {mesh_path}")


def _write_gmsh_v2_quad(path: Path, points: np.ndarray, elements: np.ndarray) -> None:
    n_nodes = points.shape[0]
    n_elems = elements.shape[0]
    with path.open("w", encoding="ascii") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$Nodes\n{n_nodes}\n")
        for idx, (x, y, z) in enumerate(points, start=1):
            handle.write(f"{idx} {x:.6f} {y:.6f} {z:.6f}\n")
        handle.write("$EndNodes\n")
        handle.write(f"$Elements\n{n_elems}\n")
        for eid, conn in enumerate(elements, start=1):
            n1, n2, n3, n4 = (int(v) + 1 for v in conn)
            handle.write(f"{eid} 3 2 0 0 {n1} {n2} {n3} {n4}\n")
        handle.write("$EndElements\n")


def _save_mesh_overlay_png(
    out_path: Path,
    ref_image: np.ndarray,
    mesh,
    *,
    dpi: int = 200,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    nodes = np.asarray(mesh.nodes_xy)
    elements = np.asarray(mesh.elements, dtype=int)
    verts = nodes[elements] if elements.size else np.empty((0, 4, 2), dtype=float)

    fig, ax = plt.subplots()
    ax.imshow(ref_image, cmap="gray", origin="lower", alpha=1.0)
    if verts.size:
        mesh_collection = PolyCollection(
            verts,
            facecolors="none",
            edgecolors="red",
            linewidths=0.5,
        )
        ax.add_collection(mesh_collection)
    ax.set_aspect("equal")
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
