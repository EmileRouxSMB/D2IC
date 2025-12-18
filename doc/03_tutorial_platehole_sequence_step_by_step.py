"""
Step-by-step (non-notebook) DIC workflow on the PlateHole image SEQUENCE.
Designed for users not familiar with Python: simply tweak the parameters below.

How to run:
    python doc/tutorial_buterFly_sequence_step_by_step.py

What the script does automatically:
 1) generate the mesh from the ROI binary mask,
 2) estimate the displacements frame by frame,
 3) compute nodal strains,
 4) export PNGs of the Ux, Uy, Exx/Exy/Eyy fields,
 5) save all fields compactly into a .npz file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import jax
import matplotlib
import numpy as np

from D2IC.app_utils import run_pipeline_sequence as run_pipeline_sequence_app

# Non-interactive backend so figures can be saved without a display.
matplotlib.use("Agg")


def _configure_jax_platform(preferred: str = "gpu", fallback: str = "cpu") -> None:
    """Force a backend when available, otherwise fall back to CPU to avoid crashes."""
    try:
        devices = jax.devices(preferred)
    except RuntimeError:
        devices = []
    if devices:
        jax.config.update("jax_platform_name", preferred)
        print(f"JAX backend: {preferred} ({len(devices)} device(s) detected)")
    else:
        jax.config.update("jax_platform_name", fallback)
        print(f"JAX backend: {preferred} unavailable, falling back to {fallback}.")


_configure_jax_platform()
# eanble 64-bit floats for better accuracy
jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------- #
#                PARAMETERS TO ADJUST (SECTION FOR NON-EXPERTS)               #
# --------------------------------------------------------------------------- #
#                PARAMETERS TO ADJUST (SECTION FOR NON-EXPERTS)               #
# --------------------------------------------------------------------------- #
# Repository root (leave default if the script stays inside doc/).
PWD = Path.cwd().resolve()


# Folder containing the image sequence + ROI mask.
IMG_DIR = PWD / "img" / "PlateHole"
REF_IMAGE_NAME = "ohtcfrp_00.tif"  # reference image
MASK_FILENAME = "roi.tif"  # binary ROI mask
IMAGE_PATTERN = "ohtcfrp_*.tif"  # pattern used to list deformed images

# Output folder for the mesh, figures, and .npz file.
OUT_DIR = PWD / "_outputs" / "sequence_platehole"

# ROI mesh generation parameter.
MESH_ELEMENT_SIZE_PX = 20.

# Global DIC (CG solver) and local refinement parameters.
DIC_MAX_ITER = 4000
DIC_TOL = 1e-3
DIC_REG_TYPE = "spring"
DIC_ALPHA_REG = 0.1
LOCAL_SWEEPS = 10  # set to 0 to disable nodal refinement

# Initialization options for subsequent frames.
USE_VELOCITY = True
VEL_SMOOTHING = 0.5

# Strain computation parameters.
STRAIN_K_RING = 2
STRAIN_GAUGE_LENGTH = 40.0

# Frames to export (leave None for all).
FRAMES_TO_PLOT = None

# Image export settings: colormap and alpha.
PLOT_CMAP = "jet"
PLOT_ALPHA = 0.6


def run_pipeline_sequence() -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper around the full pipeline using the parameters defined above."""
    return run_pipeline_sequence_app(
        img_dir=IMG_DIR,
        ref_image_name=REF_IMAGE_NAME,
        mask_filename=MASK_FILENAME,
        image_pattern=IMAGE_PATTERN,
        out_dir=OUT_DIR,
        mesh_element_size_px=MESH_ELEMENT_SIZE_PX,
        dic_max_iter=DIC_MAX_ITER,
        dic_tol=DIC_TOL,
        dic_reg_type=DIC_REG_TYPE,
        dic_alpha_reg=DIC_ALPHA_REG,
        local_sweeps=LOCAL_SWEEPS,
        use_velocity=USE_VELOCITY,
        vel_smoothing=VEL_SMOOTHING,
        strain_k_ring=STRAIN_K_RING,
        strain_gauge_length=STRAIN_GAUGE_LENGTH,
        frames_to_plot=FRAMES_TO_PLOT,
        plot_cmap=PLOT_CMAP,
        plot_alpha=PLOT_ALPHA,
    )


def main() -> None:
    run_pipeline_sequence()


if __name__ == "__main__":
    main()
