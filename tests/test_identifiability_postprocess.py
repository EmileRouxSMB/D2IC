import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_postprocess_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "img" / "DP600L1" / "PostWorkDir" / "07_identifiability_postprocess.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("identifiability_postprocess", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_fisher_detects_rank_deficit_then_full_excitation() -> None:
    module = _load_postprocess_module()
    sensitivity = np.zeros((3, 2, 3))
    sensitivity[0, :, 0] = 1.0
    sensitivity[1, :, 1] = 1.0
    sensitivity[2, :, 2] = 1.0
    quality = np.ones((3, 2))
    minimum, condition, correlation = module.fisher_indicators(
        sensitivity, quality, window=3
    )
    assert np.all(minimum[0] < minimum[2])
    assert np.all(condition[0] > condition[2])
    np.testing.assert_allclose(correlation[2], np.eye(3), atol=1e-10)
