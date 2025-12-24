"""
Rigid-body validation on the Sample3 DIC Challenge sequence.

The dataset increments the rigid translation by 0.1 px per frame in both
directions. This script runs sequential pixelwise DIC without the big motion
detector or nodal refinements, then compares the measured mean nodal
displacements against the expected ground truth.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

from d2ic import (
    MeshDICConfig,
    BatchConfig,
    DICMeshBased,
    BatchMeshBased,
    GlobalCGSolver,
    mask_to_mesh_assets,
    mask_to_mesh_assets_gmsh,
)
from d2ic.mesh_assets import make_mesh_assets

# Non-interactive backend so the plot is saved even on headless systems.
matplotlib.use("Agg")

# Set a clean style suitable for publication-quality plots.
plt.rcParams.update(
    {
        "figure.figsize": (8.5, 6.5),
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    }
)


ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "img" / "Sample3"
REF_IMAGE_NAME = "Sample3 Reference.tif"
MASK_FILENAME = "roi.tif"
IMAGE_PATTERN = "Sample3-*.tif"
OUT_DIR = ROOT / "_outputs" / "validation_sample3"
MESH_ELEMENT_SIZE_PX = 40.0
DIC_MAX_ITER = 400
DIC_TOL = 1e-3
DIC_REG_TYPE = "spring"
DIC_ALPHA_REG = 1e-2
USE_GMSH_MESH = False
INTERPOLATION = "cubic"  # "cubic" (im_jax) or "linear" (dm_pix/bilinear)
GROUND_TRUTH_DEFAULT_STEP = 0.1  # px per frame if parsing fails


def _parse_expected_displacement(path: Path) -> Tuple[float, float]:
    """Extract the analytical shift encoded in the filename."""
    match = re.search(r"X(?P<dx>-?\d+\.\d+)\s+Y(?P<dy>-?\d+\.\d+)", path.stem)
    if not match:
        idx = int(re.search(r"-(\d+)", path.stem).group(1))
        value = idx * GROUND_TRUTH_DEFAULT_STEP
        return value, value
    return float(match.group("dx")), float(match.group("dy"))


def _load_sequence() -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[Path]]:
    """Read the reference image, the deformed frames, and their ground truth."""
    t0 = time.perf_counter()
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")
    im_ref = imread(IMG_DIR / REF_IMAGE_NAME).astype(float)
    frame_paths = sorted(IMG_DIR.glob(IMAGE_PATTERN))
    deformed_paths = [p for p in frame_paths if p.name != REF_IMAGE_NAME]
    if not deformed_paths:
        raise FileNotFoundError(f"No deformed images matching {IMAGE_PATTERN}.")
    images_def = [imread(path).astype(float) for path in deformed_paths]
    expected = np.array([_parse_expected_displacement(p) for p in deformed_paths], dtype=np.float64)
    t1 = time.perf_counter()
    print(f"[timing] load_sequence: {(t1 - t0):.2f}s ({len(images_def)} frames)")
    return im_ref, images_def, expected, deformed_paths


def _load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(imread(path))
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[..., :3]
        mask = mask.mean(axis=2)
    return mask > 0


def _solve_sequence(im_ref: np.ndarray, images_def: List[np.ndarray]) -> np.ndarray:
    """Run DIC sequentially with zero initial guess and no refinements."""
    if DIC_REG_TYPE != "spring":
        raise ValueError("This validation expects spring regularization.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = IMG_DIR / MASK_FILENAME
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    t0 = time.perf_counter()
    mask = _load_mask(mask_path)
    print(f"[timing] load_mask: {(time.perf_counter() - t0):.2f}s")

    t_mesh = time.perf_counter()
    if USE_GMSH_MESH:
        print("[timing] mesh_gmsh: start")
        try:
            mesh, assets = mask_to_mesh_assets_gmsh(
                mask=mask,
                element_size_px=MESH_ELEMENT_SIZE_PX,
                remove_islands=True,
                min_island_area_px=64,
            )
            print(f"[timing] mesh_gmsh: {(time.perf_counter() - t_mesh):.2f}s")
        except Exception as exc:
            print(f"Gmsh meshing failed ({exc}); falling back to structured grid mesher.")
            mesh, _ = mask_to_mesh_assets(
                mask=mask,
                element_size_px=MESH_ELEMENT_SIZE_PX,
                remove_islands=True,
                min_island_area_px=64,
            )
            assets = make_mesh_assets(mesh, with_neighbors=True)
            print(f"[timing] mesh_structured: {(time.perf_counter() - t_mesh):.2f}s")
    else:
        print("[timing] mesh_structured: start (USE_GMSH_MESH=False)")
        mesh, _ = mask_to_mesh_assets(
            mask=mask,
            element_size_px=MESH_ELEMENT_SIZE_PX,
            remove_islands=True,
            min_island_area_px=64,
        )
        assets = make_mesh_assets(mesh, with_neighbors=True)
        print(f"[timing] mesh_structured: {(time.perf_counter() - t_mesh):.2f}s")

    mesh_cfg = MeshDICConfig(
        max_iters=DIC_MAX_ITER,
        tol=DIC_TOL,
        reg_strength=DIC_ALPHA_REG,
    )
    dic_mesh = DICMeshBased(
        mesh=mesh,
        solver=GlobalCGSolver(interpolation=INTERPOLATION),
        config=mesh_cfg,
    )
    batch_cfg = BatchConfig(
        use_init_motion=False,
        warm_start_from_previous=True,
    )
    batch = BatchMeshBased(
        ref_image=im_ref,
        assets=assets,
        dic_mesh=dic_mesh,
        batch_config=batch_cfg,
        dic_init=None,
        propagator=None,
    )
    t_batch = time.perf_counter()
    batch_result = batch.run(images_def)
    print(f"[timing] batch_run: {(time.perf_counter() - t_batch):.2f}s")
    per_frame = batch_result.results
    t_stack = time.perf_counter()
    disp_history = np.stack([np.asarray(res.u_nodal) for res in per_frame], axis=0)
    print(f"[timing] stack_results: {(time.perf_counter() - t_stack):.2f}s")
    return disp_history


def _plot_mean_displacements(
    mean_disp: np.ndarray,
    std_disp: np.ndarray,
    expected: np.ndarray,
    frame_labels: List[str],
) -> None:
    """Create a two-panel publication-ready plot for Ux and Uy statistics."""
    x = np.arange(1, mean_disp.shape[0] + 1, dtype=int)
    fig, axes = plt.subplots(2, 1, sharex=True)
    colors = ("#1b9e77", "#d95f02")
    for ax, comp_idx, title, color in zip(axes, (0, 1), ("Ux", "Uy"), colors):
        mean = mean_disp[:, comp_idx]
        std = std_disp[:, comp_idx]
        ax.plot(x, mean, marker="o", color=color, label=f"Measured {title}")
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
            label=f"{title} ±1σ",
        )
        ax.plot(
            x,
            expected[:, comp_idx],
            linestyle="--",
            color="#4c72b0",
            label="Expected 0.1 px/frame",
        )
        ax.set_ylabel(f"{title} [px]")
        ax.set_title(f"Mean {title} with 1σ uncertainty")
        ax.grid(True)
        ax.legend(loc="upper left", frameon=False)

    axes[-1].set_xlabel("Frame index")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([lbl.split()[0] for lbl in frame_labels], rotation=45, ha="right")
    fig.tight_layout()
    fig_path = OUT_DIR / "mean_displacement_validation.png"
    fig.savefig(fig_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mean displacement plot to {fig_path}")


def main() -> None:
    im_ref, images_def, expected, frame_paths = _load_sequence()
    disp_history = _solve_sequence(im_ref, images_def)
    mean_disp = disp_history.mean(axis=1)
    std_disp = disp_history.std(axis=1)
    _plot_mean_displacements(
        mean_disp,
        std_disp,
        expected,
        [p.stem for p in frame_paths],
    )


if __name__ == "__main__":
    main()
