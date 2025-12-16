"""
Tutoriel pas à pas (hors notebook) de corrélation DIC sur une SÉQUENCE d'images "ButterFly".
Pensé pour les personnes peu à l'aise avec Python : il suffit d'ajuster les paramètres ci-dessous.

Comment lancer :
    python doc/tutorial_buterFly_sequence_step_by_step.py

Ce que le script produit automatiquement :
 1) génération du maillage depuis le masque binaire ROI,
 2) estimation des déplacements image par image,
 3) calcul des déformations nodales,
 4) export PNG des champs Ux, Uy, Exx/Exy/Eyy,
 5) sauvegarde compacte de tous les champs dans un fichier .npz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import jax
import matplotlib
import numpy as np

from D2IC.app_utils import run_pipeline_sequence as run_pipeline_sequence_app

# Backend non interactif pour sauvegarder les figures sans écran.
matplotlib.use("Agg")

# Configuration JAX : float64 pour se rapprocher du notebook, CPU par défaut pour la portabilité.
jax.config.update("jax_enable_x64", True)


def _configure_jax_platform(preferred: str = "gpu", fallback: str = "cpu") -> None:
    """Force un backend si disponible, sinon bascule sur CPU pour éviter les plantages."""
    try:
        devices = jax.devices(preferred)
    except RuntimeError:
        devices = []
    if devices:
        jax.config.update("jax_platform_name", preferred)
        print(f"Backend JAX : {preferred} ({len(devices)} device(s) détecté(s))")
    else:
        jax.config.update("jax_platform_name", fallback)
        print(f"Backend JAX : {preferred} indisponible, bascule sur {fallback}.")


_configure_jax_platform()

# --------------------------------------------------------------------------- #
#                PARAMÈTRES À ADAPTER (SECTION POUR NON EXPERTS)              #
# --------------------------------------------------------------------------- #
# Racine du dépôt (laisser par défaut si le script reste dans doc/).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Dossier contenant la séquence d'images + le masque ROI.
IMG_DIR = REPO_ROOT / "doc" / "img" / "PlateHole"
REF_IMAGE_NAME = "ohtcfrp_00.tif"  # image de référence
MASK_FILENAME = "roi.tif"  # masque binaire de la zone d'intérêt
IMAGE_PATTERN = "ohtcfrp_*.tif"  # motif pour lister les images déformées

# Dossier où seront écrits le maillage, les figures et le .npz.
OUT_DIR = Path(__file__).resolve().parent / "_outputs" / "sequence_platehole"

# Paramètres de génération du maillage ROI.
MESH_ELEMENT_SIZE_PX = 40.0

# Paramètres DIC globaux (solver CG) et locaux.
DIC_MAX_ITER = 400
DIC_TOL = 1e-3
DIC_REG_TYPE = "spring"
DIC_ALPHA_REG = 0.1
LOCAL_SWEEPS = 3  # mettre 0 pour désactiver le raffinement nodal

# Options d'initialisation pour les frames suivantes.
USE_VELOCITY = True
VEL_SMOOTHING = 0.5

# Paramètres de calcul de déformations.
STRAIN_K_RING = 2
STRAIN_GAUGE_LENGTH = 200.0

# Liste des frames à exporter (laisser None pour toutes).
FRAMES_TO_PLOT = None

# Export des images : colormap et transparence.
PLOT_CMAP = "jet"
PLOT_ALPHA = 0.6


def run_pipeline_sequence() -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper du pipeline complet avec les paramètres définis en tête de fichier."""
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
