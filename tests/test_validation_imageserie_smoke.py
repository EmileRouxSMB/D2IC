from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "doc" / "00_Validation_ImageSerie.py"
DATA_DIR = PROJECT_ROOT / "doc" / "img" / "Sample3"


def _load_validation_module():
    spec = importlib.util.spec_from_file_location("validation_imageserie", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load 00_Validation_ImageSerie module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_validation_imageserie_smoke() -> None:
    if not DATA_DIR.exists():
        pytest.skip("Sample3 image sequence not available.")

    module = _load_validation_module()
    module.DIC_MAX_ITER = 50
    module.DIC_TOL = 1e-2
    module.USE_GMSH_MESH = False
    module.MESH_ELEMENT_SIZE_PX = 50.0

    im_ref, images_def, _, _ = module._load_sequence()
    images_def = images_def[:2]
    disp_history = module._solve_sequence(im_ref, images_def)

    assert disp_history.shape[0] == len(images_def)
    assert disp_history.shape[2] == 2
    assert np.isfinite(disp_history).all()
