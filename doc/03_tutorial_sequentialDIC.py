"""
Step-by-step (non-notebook) DIC workflow on the "ButterFly" image SEQUENCE.
Designed for users not familiar with Python: simply tweak the parameters below.

How to run:
    python doc/04_tutorial_buterFly_sequence_step_by_step.py

What the script does automatically:
 1) generate the mesh from the ROI binary mask,
 2) estimate the displacements frame by frame,
 3) compute nodal strains,
 4) export PNGs of the Ux, Uy, Exx/Exy/Eyy fields,
 5) save all fields compactly into a .npz file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib
import numpy as np

try:  # optional mesh writer
    import meshio
except Exception:  # pragma: no cover
    meshio = None


from d2ic import (
    MeshDICConfig,
    BatchConfig,
    mask_to_mesh_assets,
    mask_to_mesh_assets_gmsh,
    DICMeshBased,
    GlobalCGSolver,
    LocalGaussNewtonSolver,
    PreviousDisplacementPropagator,
    ConstantVelocityPropagator,
)
from d2ic import BatchMeshBased
from d2ic.mesh_assets import make_mesh_assets
from d2ic.app_utils import configure_jax_platform, imread_gray, list_deformed_images, prepare_image

# Non-interactive backend so figures can be saved without a display.
matplotlib.use("Agg")

configure_jax_platform()

# --------------------------------------------------------------------------- #
#                PARAMETERS TO ADJUST (SECTION FOR NON-EXPERTS)               #
# --------------------------------------------------------------------------- #
# Script folder (this file lives inside `D2IC/doc/`).
PWD = Path(__file__).resolve().parent

IMG_DIR = PWD / "img" / "PlateHole"
REF_IMAGE_NAME = "ohtcfrp_00.tif"  # reference image
MASK_FILENAME = "roi.tif"  # binary ROI mask
IMAGE_PATTERN = "ohtcfrp_*.tif"  # pattern used to list deformed images

# Output folder for the mesh, figures, and .npz file.
OUT_DIR = PWD / "_outputs" / "sequence_platehole"

# ROI mesh generation parameter.
MESH_ELEMENT_SIZE_PX = 40.0

# Global DIC (CG solver) and local refinement parameters.
DIC_MAX_ITER = 400
DIC_TOL = 1e-2
DIC_ALPHA_REG = 1e-2
LOCAL_SWEEPS = 3  # set to 0 to disable nodal refinement

# Initialization options for subsequent frames.
USE_VELOCITY = True
VEL_SMOOTHING = 0.5

# Strain computation parameters.
STRAIN_GAUGE_LENGTH = 80.0

# Frames to export (leave None for all).
FRAMES_TO_PLOT = None #None

# Image export settings: colormap and alpha.
PLOT_CMAP = "jet"
PLOT_ALPHA = 0.6
PLOT_MESH = True
PLOT_FIELDS = ("u1", "u2", "e11", "e22", "e12")
PLOT_INCLUDE_DISCREPANCY = False

# Downsample factor for reference/deformed images (1 keeps native resolution).
IMAGE_BINNING = 1
# Interpolation used inside solvers: "cubic" (im_jax) or "linear" (dm_pix/bilinear fallback).
INTERPOLATION = "cubic"
# Debug prints from inside JAX-compiled loops (CG iterations).
VERBOSE = True


def run_pipeline_sequence() -> Tuple[np.ndarray, np.ndarray]:
    """Run the full `d2ic` batch pipeline using the parameters defined above."""
    if IMAGE_BINNING < 1:
        raise ValueError("IMAGE_BINNING must be >= 1.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_dir = OUT_DIR / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    ref_path = IMG_DIR / REF_IMAGE_NAME
    mask_path = IMG_DIR / MASK_FILENAME
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")

    def_paths = list_deformed_images(IMG_DIR, IMAGE_PATTERN, exclude_name=REF_IMAGE_NAME)
    if not def_paths:
        raise RuntimeError(f"No deformed images found in {IMG_DIR} with pattern '{IMAGE_PATTERN}'.")

    ref_image = prepare_image(ref_path, binning=IMAGE_BINNING)
    def_images = [prepare_image(p, binning=IMAGE_BINNING) for p in def_paths]
    mask = imread_gray(mask_path) > 0.5

    try:
        mesh, assets = mask_to_mesh_assets_gmsh(
            mask=mask,
            element_size_px=MESH_ELEMENT_SIZE_PX,
            binning=IMAGE_BINNING,
            remove_islands=True,
            min_island_area_px=64,
        )
        print("Generated mesh via Gmsh-based pipeline.")
    except Exception as exc:
        print(f"Gmsh meshing failed ({exc}); falling back to structured grid mesher.")
        mesh, _ = mask_to_mesh_assets(
            mask=mask,
            element_size_px=MESH_ELEMENT_SIZE_PX,
            binning=IMAGE_BINNING,
            remove_islands=True,
            min_island_area_px=64,
        )
        assets = make_mesh_assets(mesh, with_neighbors=True)

    mesh_cfg = MeshDICConfig(
        max_iters=DIC_MAX_ITER,
        tol=DIC_TOL,
        reg_strength=DIC_ALPHA_REG,
        strain_gauge_length=STRAIN_GAUGE_LENGTH,
        save_history=True,
    )

    dic_mesh = DICMeshBased(
        mesh=mesh,
        solver=GlobalCGSolver(interpolation=INTERPOLATION, verbose=VERBOSE),
        config=mesh_cfg,
    )

    batch_cfg = BatchConfig(
        use_init_motion=False,
        warm_start_from_previous=True,
        verbose=True,
        progress=True,
        export_png=True,
        export_frames=FRAMES_TO_PLOT,
        png_dir=str(OUT_DIR / "png"),
        plot_fields=PLOT_FIELDS,
        plot_include_discrepancy=PLOT_INCLUDE_DISCREPANCY,
        plot_cmap=PLOT_CMAP,
        plot_alpha=PLOT_ALPHA,
        plot_mesh=PLOT_MESH,
        plot_dpi=200,
        plot_binning=IMAGE_BINNING,
        plot_projection="fast",
    )

    propagator = ConstantVelocityPropagator() if USE_VELOCITY else PreviousDisplacementPropagator()
    dic_local = None
    if LOCAL_SWEEPS > 0:
        local_cfg = MeshDICConfig(
            max_iters=LOCAL_SWEEPS,
            tol=DIC_TOL,
            reg_strength=DIC_ALPHA_REG,
            strain_gauge_length=STRAIN_GAUGE_LENGTH,
            save_history=True,
        )
        local_solver = LocalGaussNewtonSolver(
            lam=0.1,
            max_step=0.2,
            omega=0.5,
            interpolation=INTERPOLATION,
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
    batch_result = batch.run(def_images)
    print("Batch run completed.")
    per_frame = batch_result.results

    # Local sweeps are now chained inside BatchMeshBased if LOCAL_SWEEPS > 0.

    nodes_xy = np.asarray(assets.mesh.nodes_xy)
    u_stack = np.stack([np.asarray(r.u_nodal) for r in per_frame], axis=0)
    strain_stack = np.stack([np.asarray(r.strain) for r in per_frame], axis=0)

    np.savez_compressed(
        OUT_DIR / "fields_sequence.npz",
        nodes_xy=nodes_xy,
        u_nodal=u_stack,
        strain=strain_stack,
        ref_path=str(ref_path),
        def_paths=[str(p) for p in def_paths],
    )
    print(f"Saved NPZ results to {OUT_DIR/'fields_sequence.npz'}")

    if meshio is None:
        print("meshio is not available; skipping mesh export.")
    else:
        mesh_path = OUT_DIR / "roi_mesh.msh"
        points = np.column_stack([nodes_xy, np.zeros((nodes_xy.shape[0],), dtype=nodes_xy.dtype)])
        elements = np.asarray(assets.mesh.elements, dtype=np.int32)
        meshio_mesh = meshio.Mesh(points=points, cells=[("quad", elements)])
        meshio.write(str(mesh_path), meshio_mesh, file_format="gmsh")
        print(f"Saved mesh to {mesh_path}")

    if VEL_SMOOTHING != 0.5:
        print("Note: velocity smoothing is not currently exposed.")

    return u_stack, strain_stack

def main() -> None:
    run_pipeline_sequence()


if __name__ == "__main__":
    main()
