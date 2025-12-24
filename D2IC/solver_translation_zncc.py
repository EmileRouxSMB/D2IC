from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import jax
import jax.numpy as jnp

from .solver_base import SolverBase
from .types import Array
from .mesh_assets import MeshAssets
from .dataclasses import InitMotionConfig


@dataclass(frozen=True)
class TranslationZNCCResult:
    """
    Coarse motion estimate at element centers.
    """
    u_centers: Array  # (Ne, 2)
    scores: Array     # (Ne,)


class TranslationZNCCSolver(SolverBase):
    """
    Translation-only patch matching around each element center using ZNCC.

    Implementation follows the legacy ``motion_init.translation_patch_matches_zncc``
    logic but keeps the heavy lifting in JAX for normalization/scoring while
    controlling memory via per-center evaluation.
    """

    def __init__(self, config: InitMotionConfig, chunk_size: int = 128) -> None:
        if config.win % 2 == 0:
            raise ValueError("InitMotionConfig.win must be odd for ZNCC matching.")
        self.config = config
        self.chunk_size = max(1, int(chunk_size))
        self._compiled = False
        self._zncc_argmax = jax.jit(_zncc_argmax)

    def compile(self, assets: Any) -> None:
        # No-op placeholder: shapes determined at runtime, but keep flag for API parity.
        self._compiled = True

    def solve(self, state: Any, def_image: Array) -> TranslationZNCCResult:
        if not self._compiled:
            raise RuntimeError("TranslationZNCCSolver.compile() must be called before solve().")

        assets: MeshAssets = state.assets
        ref_image = np.asarray(state.ref_image)
        def_image_np = np.asarray(def_image)
        centers = np.asarray(assets.element_centers_xy)

        max_centers = self.config.max_centers
        if max_centers is not None:
            centers = centers[: int(max_centers)]

        n_centers = centers.shape[0]
        u_centers = np.zeros((n_centers, 2), dtype=np.float32)
        scores = np.full((n_centers,), np.nan, dtype=np.float32)

        win = int(self.config.win)
        search = int(self.config.search)
        score_min = self.config.score_min

        for start in range(0, n_centers, self.chunk_size):
            end = min(n_centers, start + self.chunk_size)
            for idx in range(start, end):
                disp_score = _match_single_center(
                    centers[idx],
                    ref_image,
                    def_image_np,
                    win=win,
                    search=search,
                    zncc_fn=self._zncc_argmax,
                )
                if disp_score is None:
                    continue
                disp, score = disp_score
                if score_min is not None and score < float(score_min):
                    continue
                u_centers[idx] = disp
                scores[idx] = score

        return TranslationZNCCResult(
            u_centers=jnp.asarray(u_centers),
            scores=jnp.asarray(scores),
        )


# ---------------------------------------------------------------------
# JAX kernels + helpers
# ---------------------------------------------------------------------

@jax.jit
def _zncc_argmax(template: Array, windows: Array) -> tuple[jnp.ndarray, jnp.ndarray]:
    template = template - jnp.mean(template)
    template = template / (jnp.std(template) + 1e-12)
    windows = windows - jnp.mean(windows, axis=(1, 2), keepdims=True)
    windows = windows / (jnp.std(windows, axis=(1, 2), keepdims=True) + 1e-12)
    scores = jnp.mean(windows * template, axis=(1, 2))
    best_idx = jnp.argmax(scores)
    best_score = scores[best_idx]
    return best_idx, best_score


def _match_single_center(
    center_xy: np.ndarray,
    ref_image: np.ndarray,
    def_image: np.ndarray,
    *,
    win: int,
    search: int,
    zncc_fn,
) -> Optional[tuple[np.ndarray, float]]:
    r = win // 2
    h_ref, w_ref = ref_image.shape[:2]
    h_def, w_def = def_image.shape[:2]

    x = int(round(float(center_xy[0])))
    y = int(round(float(center_xy[1])))

    if (
        y - r < 0
        or y + r >= h_ref
        or x - r < 0
        or x + r >= w_ref
    ):
        return None

    template = ref_image[y - r : y + r + 1, x - r : x + r + 1].astype(np.float32, copy=False)

    xs = x - search
    ys = y - search
    xe = x + search
    ye = y + search

    if (
        ys - r < 0
        or ye + r >= h_def
        or xs - r < 0
        or xe + r >= w_def
    ):
        return None

    patch = def_image[ys - r : ye + r + 1, xs - r : xe + r + 1].astype(np.float32, copy=False)
    windows = _extract_windows(patch, win)
    if windows.size == 0:
        return None

    best_idx, best_score = zncc_fn(jnp.asarray(template), jnp.asarray(windows))
    best_idx_int = int(best_idx)
    best_score_f = float(best_score)

    windows_per_axis = 2 * search + 1
    ky, kx = divmod(best_idx_int, windows_per_axis)
    yy = ys + ky
    xx = xs + kx
    disp = np.array([xx - x, yy - y], dtype=np.float32)
    return disp, best_score_f


def _extract_windows(patch: np.ndarray, win: int) -> np.ndarray:
    """Return all ``win``-sized windows from ``patch`` as ``(N, win, win)``."""
    view = np.lib.stride_tricks.sliding_window_view(patch, (win, win))
    windows = view.reshape(-1, win, win)
    return windows
