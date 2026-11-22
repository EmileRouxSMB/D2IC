from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import matplotlib
import numpy as np

try:  # optional mesh writer
    import meshio
except Exception:  # pragma: no cover
    meshio = None

from ..io.app_utils import configure_jax_platform, imread_gray, list_deformed_images, prepare_image, read_roi_mask, erode_correlation_roi
from ..batch.mesh_based import BatchMeshBased
from ..io.config import load_sequence_config
from ..io.result_store import ResultStore
from ..core.dataclasses import BatchConfig, DICResult, MeshDICConfig
from ..methods.mesh_based import DICMeshBased
from ..mesh.generation import mask_to_mesh_assets_gmsh
from ..propagation.constant_velocity import ConstantVelocityPropagator
from ..propagation.previous import PreviousDisplacementPropagator
from ..numerics.global_cg import GlobalCGSolver
from ..numerics.local_gauss_newton import LocalGaussNewtonSolver


def run_sequence_from_config(
    cfg_path: Path, *, output_dir: Path | None = None,
    output_prefix: str | None = None, collection_dir: Path | None = None,
) -> None:
    """Run sequential DIC, optionally overriding its generated-output directory."""
    cfg = load_sequence_config(Path(cfg_path))
    if output_dir is not None:
        cfg = replace(cfg, out_dir=Path(output_dir).resolve())

    _configure_runtime(cfg)
    _log_verbose(cfg, f"[Config] Using config: {cfg_path}")
    t_total = time.perf_counter()

    # Non-interactive backend so figures can be saved without a display.
    matplotlib.use("Agg")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    structured = output_prefix is not None
    prefix = output_prefix or "frame"
    collection_dir = cfg.out_dir if collection_dir is None else Path(collection_dir).resolve()
    png_dir = cfg.out_dir / "png"
    npz_dir = cfg.out_dir if structured else cfg.out_dir / "npz"
    paraview_dir = cfg.out_dir if structured else cfg.out_dir / "paraview"
    if cfg.export_png:
        png_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_npz:
        npz_dir.mkdir(parents=True, exist_ok=True)
    if cfg.export_vtu or cfg.save_mesh:
        paraview_dir.mkdir(parents=True, exist_ok=True)

    ref_path = cfg.img_dir / cfg.ref_image_name
    mask_path = cfg.mask_path
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")

    if cfg.image_paths is None:
        sequence_paths = list_deformed_images(cfg.img_dir, cfg.image_pattern)
        sampled_paths = sequence_paths[:: cfg.image_stride]
        def_paths = [p for p in sampled_paths if p.name != cfg.ref_image_name]
    else:
        def_paths = list(cfg.image_paths)
        if any(not p.is_file() or p.resolve().parent != cfg.img_dir.resolve()
               or p.name == cfg.ref_image_name for p in def_paths):
            raise ValueError("paths.image_paths must contain existing deformed images in paths.img_dir.")
    if cfg.max_deformed_images is not None:
        if cfg.max_deformed_images < 1:
            raise ValueError("paths.max_deformed_images must be >= 1.")
        def_paths = def_paths[: cfg.max_deformed_images]
    if not def_paths:
        raise RuntimeError(
            f"No deformed images found in {cfg.img_dir} with pattern '{cfg.image_pattern}'."
        )

    t0 = time.perf_counter()
    ref_image = prepare_image(ref_path, binning=cfg.image_binning)
    _log_elapsed(cfg, "Loaded reference image", t0)
    _log_verbose(cfg, f"[Images] ref shape: {ref_image.shape}")

    t0 = time.perf_counter()
    mask_image = imread_gray(mask_path)
    mask = read_roi_mask(mask_path, mask_image)
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
    except Exception as exc:
        raise RuntimeError(
            "Gmsh mesh generation is required by the sequential DIC pipeline; "
            "no fallback mesh was generated."
        ) from exc
    print("Generated mesh via Gmsh-based pipeline.")
    # Geometry remains the original ROI; only image observations are eroded.
    # FE mask binning retains partial edge blocks, while image binning drops
    # them. Clip only the observation mask so its shape matches ref_image;
    # mesh nodes/elements remain those generated from the original ROI.
    observation_roi = assets.roi_mask[:ref_image.shape[0], :ref_image.shape[1]]
    correlation_mask = erode_correlation_roi(
        observation_roi, cfg.roi_erosion_px, image_binning=cfg.image_binning)
    assets = replace(assets, roi_mask=correlation_mask, pixel_data=None)
    print(f"[ROI] correlation erosion={cfg.roi_erosion_px:g} original px; "
          f"retained pixels={correlation_mask.sum()}")
    _log_elapsed(cfg, "Mesh generation", t0)
    _log_verbose(cfg, f"[Mesh] nodes: {len(mesh.nodes_xy)}, elements: {len(mesh.elements)}")

    if cfg.save_mesh:
        _export_mesh(paraview_dir, assets.mesh.nodes_xy, assets.mesh.elements, filename=f"{prefix}_mesh.msh" if structured else "roi_mesh.msh")
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
        regularization=cfg.regularization,
        strain_gauge_length=cfg.strain_gauge_length,
        save_history=True,
        compute_discrepancy_map=(cfg.local_sweeps <= 0),
    )

    dic_mesh = DICMeshBased(
        solver=GlobalCGSolver(interpolation=cfg.interpolation, verbose=cfg.verbose),
        config=mesh_cfg,
    )

    frame_times = None
    if cfg.abaqus_comparison_enabled:
        from ..io.abaqus_comparison import image_time_from_name
        frame_times = tuple(image_time_from_name(path) for path in def_paths)

    batch_cfg = BatchConfig(
        use_init_motion=False,
        warm_start_from_previous=True,
        verbose=cfg.verbose,
        progress=True,
        keep_results=cfg.keep_results,
        export_png=cfg.export_png,
        export_vtu=cfg.export_vtu,
        export_vtk_images=cfg.export_vtk_images,
        vtu_dir=str(paraview_dir),
        pvd_filename=str(collection_dir / f"{prefix}.pvd") if structured else "sequential_dic_platehole.pvd",
        image_pvd_filename=str(collection_dir / f"{prefix}_images.pvd") if structured else "sequential_dic_platehole_images.pvd",
        frame_times=frame_times,
        file_prefix=prefix,
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
        save_per_frame=False,
        per_frame_dir=str(npz_dir / f"{prefix}_frames" if structured else npz_dir / "frames"),
    )

    propagator = ConstantVelocityPropagator() if cfg.use_velocity else PreviousDisplacementPropagator()
    dic_local = None
    if cfg.local_sweeps > 0:
        local_cfg = MeshDICConfig(
            max_iters=cfg.local_sweeps,
            tol=cfg.tol,
            reg_strength=cfg.reg_strength,
            regularization=cfg.regularization,
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
        dic_local = DICMeshBased(solver=local_solver, config=local_cfg)

    result_store = None
    result_fingerprint = None
    resume_offset = 0
    if cfg.save_npz:
        store_dir = npz_dir / "dic_store"
        input_records = []
        for path in [ref_path, mask_path, *def_paths]:
            stat = path.stat()
            input_records.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns,
                                  _sha256_file(path)))
        fingerprint_payload = {
            "inputs": input_records,
            "config": repr(cfg),
            "mesh": repr((cfg.mesh_element_size_px, cfg.mesh_remove_islands,
                           cfg.mesh_min_island_area_px, cfg.mesh_gmsh_contour_step_px)),
        }
        result_fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        metadata = {
            "fingerprint": result_fingerprint,
            "ref_path": str(ref_path),
            "def_paths": [str(path) for path in def_paths],
            "image_stride": cfg.image_stride,
            "image_binning": cfg.image_binning,
            "roi_erosion_px": cfg.roi_erosion_px,
            "image_shape": list(ref_image.shape),
            "regularization": cfg.regularization,
            "reg_strength": cfg.reg_strength,
            "local_sweeps": cfg.local_sweeps,
            "time_provenance": "frame_times from configured image names" if frame_times else "frame index",
            "frame_times": list(frame_times) if frame_times is not None else None,
        }
        result_store = ResultStore(store_dir, metadata=metadata)
        result_store.write_mesh(np.asarray(assets.mesh.nodes_xy), np.asarray(assets.mesh.elements))
        result_store.write_roi_mask(mask, correlation_mask=correlation_mask)
        checkpoint = result_store.load_checkpoint(result_fingerprint)
        if checkpoint is not None:
            candidate = int(checkpoint["completed_frames"])
            committed_files_exist = all(
                (result_store.frames_path / f"frame_{k:04d}.npz").exists()
                for k in range(candidate)
            )
            if committed_files_exist:
                resume_offset = candidate
        result_store.update_progress(status="running", total_frames=len(def_paths),
                                     completed_frames=resume_offset,
                                     last_frame=resume_offset - 1 if resume_offset else None)

    # Decode only frames that still need solving after a verified checkpoint.
    def_images = (prepare_image(path, binning=cfg.image_binning)
                  for path in def_paths[resume_offset:])

    batch = BatchMeshBased(
        ref_image=ref_image,
        assets=assets,
        dic_mesh=dic_mesh,
        batch_config=batch_cfg,
        dic_local=dic_local,
        propagator=propagator,
        result_store=result_store,
        result_fingerprint=result_fingerprint,
        frame_offset=resume_offset,
    )

    print("Preparing pipelines (JIT compile may take a while on first run)...")
    print(f"Running batch on {len(def_paths)} frame(s).")
    t0 = time.perf_counter()
    batch_result = batch.run(def_images)
    _log_elapsed(cfg, "Batch run", t0)
    print("Batch run completed.")

    per_frame = batch_result.results
    if cfg.abaqus_comparison_enabled:
        if not cfg.keep_results:
            raise ValueError("Abaqus comparison requires output.keep_results = true.")
        from ..io.abaqus_comparison import compare_abaqus_dic_sequence

        comparison_dir = cfg.out_dir / "abaqus_comparison"
        comparison = compare_abaqus_dic_sequence(
            assets.mesh, _restore_comparison_results(result_store, per_frame, resume_offset), def_paths,
            abaqus_npz=cfg.abaqus_npz,
            metadata_json=cfg.abaqus_metadata_json,
            output_dir=comparison_dir,
            vtu_dir=paraview_dir,
            vtu_prefix=prefix,
            pvd_path=(collection_dir / f"{prefix}.pvd") if structured else (
                paraview_dir / "sequential_dic_platehole.pvd"
            ),
            image_binning=cfg.image_binning,
            max_extrapolation_px=cfg.abaqus_max_extrapolation_px,
        )
        print(f"Saved Abaqus comparison JSON to {comparison['json']}")
        print(f"Saved Abaqus comparison CSV to {comparison['csv']}")
    if cfg.save_npz and result_store is not None:
        if cfg.export_legacy_npz:
            out_path = npz_dir / (f"{prefix}_results.npz" if structured else "fields_sequence.npz")
            result_store.export_npz(out_path)
            print(f"Saved NPZ results to {out_path}")
        result_store.update_progress(status="completed", total_frames=len(def_paths),
                                     completed_frames=result_store.frame_count,
                                     last_frame=result_store.frame_count - 1 if result_store.frame_count else None)
    elif cfg.save_npz:
        stack_context = TemporaryDirectory(dir=npz_dir) if not cfg.keep_results else None
        u_stack = strain_stack = discrepancy_stack = None
        try:
            if cfg.keep_results:
                u_stack = np.stack([np.asarray(r.u_nodal) for r in per_frame], axis=0)
                strain_stack = np.stack([np.asarray(r.strain) for r in per_frame], axis=0)
                discrepancy_stack = _stack_discrepancy(per_frame)
            else:
                frame_dir = npz_dir / (f"{prefix}_frames" if structured else "frames")
                frame_paths = [
                    frame_dir / f"{prefix}_{k:04d}.npz" for k in range(len(def_paths))
                ]
                u_stack, strain_stack, discrepancy_stack = _stack_frame_npz(
                    frame_paths, Path(stack_context.name)
                )
            payload = {
                "nodes_xy": np.asarray(assets.mesh.nodes_xy),
                "elements": np.asarray(assets.mesh.elements),
                "u_nodal": u_stack,
                "strain": strain_stack,
                "roi_mask": np.asarray(mask, dtype=bool),
                "correlation_mask": correlation_mask,
                "roi_erosion_px": np.asarray(cfg.roi_erosion_px),
                "image_shape": np.asarray(ref_image.shape, dtype=np.int64),
                "image_binning": np.asarray(cfg.image_binning, dtype=np.int64),
                "strain_shear_convention": np.asarray("tensorial_E12"),
                "ref_path": str(ref_path),
                "def_paths": np.asarray([str(p) for p in def_paths]),
                "image_stride": np.asarray(cfg.image_stride, dtype=np.int64),
                "regularization": np.asarray(cfg.regularization),
                "reg_strength": np.asarray(cfg.reg_strength),
                "local_sweeps": np.asarray(cfg.local_sweeps, dtype=np.int64),
            }
            if discrepancy_stack is not None:
                payload["discrepancy_ref"] = discrepancy_stack
            out_path = npz_dir / (
                f"{prefix}_results.npz" if structured else "fields_sequence.npz"
            )
            np.savez_compressed(out_path, **payload)
            print(f"Saved NPZ results to {out_path}")
        finally:
            if stack_context is not None:
                for array in (u_stack, strain_stack, discrepancy_stack):
                    if isinstance(array, np.memmap):
                        array.flush()
                        array._mmap.close()
                stack_context.cleanup()

    _log_elapsed(cfg, "Total pipeline", t_total)


def _restore_comparison_results(result_store: ResultStore, new_results: list,
                                resume_offset: int) -> list[DICResult]:
    """Rebuild only the stored nodal results needed by Abaqus comparison."""
    if resume_offset == 0:
        return new_results
    previous = []
    for index in range(resume_offset):
        frame = result_store.read_frame(index)
        previous.append(DICResult(displacement=frame["u_nodal"], strain=frame["strain"]))
    return previous + new_results


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Fingerprint source image bytes so same-size replacements invalidate resume."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def _stack_frame_npz(
    frame_paths: list[Path], temporary_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Assemble a sequence on disk from per-frame NPZ files."""
    if not frame_paths:
        raise ValueError("A sequence needs at least one frame.")
    with np.load(frame_paths[0], allow_pickle=False) as first:
        u_shape, u_dtype = first["u_nodal"].shape, first["u_nodal"].dtype
        e_shape, e_dtype = first["strain"].shape, first["strain"].dtype
        has_discrepancy = "discrepancy_ref" in first
        if has_discrepancy:
            d_shape, d_dtype = first["discrepancy_ref"].shape, first["discrepancy_ref"].dtype

    count = len(frame_paths)
    u = np.lib.format.open_memmap(
        temporary_dir / "u_nodal.npy", mode="w+", dtype=u_dtype, shape=(count, *u_shape)
    )
    strain = np.lib.format.open_memmap(
        temporary_dir / "strain.npy", mode="w+", dtype=e_dtype, shape=(count, *e_shape)
    )
    discrepancy = (
        np.lib.format.open_memmap(
            temporary_dir / "discrepancy.npy", mode="w+", dtype=d_dtype,
            shape=(count, *d_shape),
        )
        if has_discrepancy else None
    )
    for k, path in enumerate(frame_paths):
        with np.load(path, allow_pickle=False) as frame:
            u[k] = frame["u_nodal"]
            strain[k] = frame["strain"]
            if discrepancy is not None:
                if "discrepancy_ref" not in frame:
                    raise ValueError(f"Missing discrepancy_ref in {path}")
                discrepancy[k] = frame["discrepancy_ref"]
    return u, strain, discrepancy


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


def _export_mesh(
    out_dir: Path, nodes_xy: np.ndarray, elements: np.ndarray, *, filename: str = "roi_mesh.msh"
) -> None:
    mesh_path = out_dir / filename
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
    ax.imshow(ref_image, cmap="gray", origin="upper", alpha=1.0)
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
