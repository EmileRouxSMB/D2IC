from __future__ import annotations

import numpy as np

from d2ic.solver_translation_zncc import _match_centers_jax


def test_match_centers_jax_static_win_search() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(64, 64)).astype(np.float32)

    dx, dy = 3, 2
    deformed = np.zeros_like(ref)
    deformed[dy:, dx:] = ref[:-dy, :-dx]

    centers = np.array(
        [
            [32.0, 32.0],
            [40.0, 24.0],
            [24.0, 40.0],
        ],
        dtype=np.float32,
    )

    u, scores = _match_centers_jax(centers, ref, deformed, win=9, search=6)

    u_np = np.asarray(u)
    scores_np = np.asarray(scores)

    assert u_np.shape == (centers.shape[0], 2)
    assert scores_np.shape == (centers.shape[0],)
    assert np.isfinite(scores_np).all()
    assert np.allclose(u_np, np.array([dx, dy], dtype=np.float32), atol=0.5)
