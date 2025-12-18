"""
Rigid-body validation on the Sample3 DIC Challenge sequence.

The dataset increments the rigid translation by 0.1 px per frame in both
directions. This script runs sequential pixelwise DIC without the big motion
detector or nodal refinements, then compares the measured mean nodal
displacements against the expected ground truth.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

from D2IC import generate_roi_mesh
from D2IC.dic import Dic

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
DIC_ALPHA_REG = 1e-4
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
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")
    im_ref = imread(IMG_DIR / REF_IMAGE_NAME).astype(float)
    frame_paths = sorted(IMG_DIR.glob(IMAGE_PATTERN))
    deformed_paths = [p for p in frame_paths if p.name != REF_IMAGE_NAME]
    if not deformed_paths:
        raise FileNotFoundError(f"No deformed images matching {IMAGE_PATTERN}.")
    images_def = [imread(path).astype(float) for path in deformed_paths]
    expected = np.array([_parse_expected_displacement(p) for p in deformed_paths], dtype=np.float64)
    return im_ref, images_def, expected, deformed_paths


def _solve_sequence(im_ref: np.ndarray, images_def: List[np.ndarray]) -> np.ndarray:
    """Run DIC sequentially with zero initial guess and no refinements."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh_path = OUT_DIR / f"roi_mesh_{int(MESH_ELEMENT_SIZE_PX)}px_validation.msh"
    mesh_generated = generate_roi_mesh(
        IMG_DIR / MASK_FILENAME,
        element_size=MESH_ELEMENT_SIZE_PX,
        msh_path=str(mesh_path),
    )
    mesh_path = Path(mesh_generated)
    dic = Dic(mesh_path=str(mesh_path))
    dic.precompute_pixel_data(im_ref)

    n_nodes = int(dic.node_coordinates.shape[0])
    disp_history = np.zeros((len(images_def), n_nodes, 2))
    disp_guess = np.zeros((n_nodes, 2))

    for i, im_def in enumerate(images_def):
        disp_opt, _ = dic.run_dic(
            im_ref,
            im_def,
            disp_guess=disp_guess,
            max_iter=DIC_MAX_ITER,
            tol=DIC_TOL,
            reg_type=DIC_REG_TYPE,
            alpha_reg=DIC_ALPHA_REG,
        )
        disp_np = np.asarray(disp_opt)
        disp_history[i] = disp_np
        disp_guess = disp_np  # propagate to the next frame

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
