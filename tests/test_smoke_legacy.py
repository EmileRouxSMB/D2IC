from __future__ import annotations

from pathlib import Path
import sys
import importlib.util

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOTION_INIT_PATH = PROJECT_ROOT / "d2ic" / "_legacy" / "motion_init.py"


def _load_motion_init_module():
    spec = importlib.util.spec_from_file_location("legacy_motion_init", MOTION_INIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load legacy motion_init module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


translation_patch_matches_zncc = _load_motion_init_module().translation_patch_matches_zncc


def _shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate ``image`` by integer pixels without wraparound (zero fill)."""
    h, w = image.shape
    out = np.zeros_like(image)
    if abs(dx) >= w or abs(dy) >= h:
        return out

    if dx >= 0:
        src_x = slice(0, w - dx)
        dst_x = slice(dx, w)
    else:
        src_x = slice(-dx, w)
        dst_x = slice(0, w + dx)

    if dy >= 0:
        src_y = slice(0, h - dy)
        dst_y = slice(dy, h)
    else:
        src_y = slice(-dy, h)
        dst_y = slice(0, h + dy)

    out[dst_y, dst_x] = image[src_y, src_x]
    return out


def test_translation_patch_matches_recovers_integer_shift() -> None:
    rng = np.random.default_rng(42)
    shape = (80, 80)
    base = rng.random(shape, dtype=np.float32)

    true_shift = (3, -2)  # (dx, dy)
    moved = _shift_image(base, *true_shift)

    margin = 20
    xs = np.arange(margin, shape[1] - margin, 12)
    ys = np.arange(margin, shape[0] - margin, 12)
    centers = np.array([[float(x), float(y)] for y in ys for x in xs], dtype=np.float64)

    win = 15
    search = 8
    pts_ref, pts_def, scores = translation_patch_matches_zncc(
        base,
        moved,
        centers,
        win=win,
        search=search,
        pred=(0.0, 0.0),
        score_min=None,
        max_centers=centers.shape[0],
    )

    assert pts_ref.shape[0] > 0, "No matches returned by translation_patch_matches_zncc"
    disp = pts_def - pts_ref
    mean_disp = disp.mean(axis=0)

    assert np.allclose(mean_disp, np.array(true_shift, dtype=np.float64), atol=1.0)
    assert np.all(scores <= 1.0 + 1e-5)
