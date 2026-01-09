from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
from jax import lax
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

    The solver evaluates a zero-mean normalized cross-correlation (ZNCC) score
    between a reference patch and a grid of candidate patches in the deformed
    image, then returns the argmax translation per element center.

    The heavy lifting runs in JAX for normalization/scoring, while the outer
    loop stays in Python to control memory and allow per-center boundary checks.
    """

    def __init__(self, config: InitMotionConfig, chunk_size: int = 128) -> None:
        if config.win % 2 == 0:
            raise ValueError("InitMotionConfig.win must be odd for ZNCC matching.")
        self.config = config
        self.chunk_size = max(1, int(chunk_size))
        self._compiled = False
        self._zncc_argmax = jax.jit(_zncc_argmax)

    def compile(self, assets: Any) -> None:
        # Shapes are determined at runtime; keep a compile hook for API parity.
        self._compiled = True

    def solve(self, state: Any, def_image: Array) -> TranslationZNCCResult:
        if not self._compiled:
            raise RuntimeError("TranslationZNCCSolver.compile() must be called before solve().")

        assets: MeshAssets = state.assets
        ref_image = jnp.asarray(state.ref_image)
        def_image_jnp = jnp.asarray(def_image)
        centers = jnp.asarray(assets.element_centers_xy)

        max_centers = self.config.max_centers
        if max_centers is not None:
            centers = centers[: int(max_centers)]

        win = int(self.config.win)
        search = int(self.config.search)
        score_min = self.config.score_min

        u_centers, scores = _match_centers_jax(
            centers,
            ref_image,
            def_image_jnp,
            win=win,
            search=search,
        )
        if score_min is not None:
            valid = scores >= float(score_min)
            u_centers = jnp.where(valid[:, None], u_centers, jnp.zeros_like(u_centers))
            scores = jnp.where(valid, scores, jnp.nan)

        return TranslationZNCCResult(
            u_centers=u_centers,
            scores=scores,
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


@partial(jax.jit, static_argnames=("win", "search"))
def _extract_windows_jax(patch: Array, win: int, search: int) -> Array:
    """Return all ``win``-sized windows from ``patch`` as ``(N, win, win)``."""
    windows_per_axis = 2 * search + 1
    ys, xs = jnp.meshgrid(
        jnp.arange(windows_per_axis, dtype=jnp.int32),
        jnp.arange(windows_per_axis, dtype=jnp.int32),
        indexing="ij",
    )
    offsets = jnp.stack([ys.reshape(-1), xs.reshape(-1)], axis=1)

    def slice_at(offset):
        return lax.dynamic_slice(patch, (offset[0], offset[1]), (win, win))

    return jax.vmap(slice_at)(offsets)


@partial(jax.jit, static_argnames=("win", "search"))
def _match_centers_jax(
    centers: Array,
    ref_image: Array,
    def_image: Array,
    *,
    win: int,
    search: int,
) -> tuple[Array, Array]:
    r = win // 2
    h_ref, w_ref = ref_image.shape[:2]
    h_def, w_def = def_image.shape[:2]
    patch_size = win + 2 * search
    windows_per_axis = 2 * search + 1

    def match_one(center_xy):
        x = jnp.rint(center_xy[0]).astype(jnp.int32)
        y = jnp.rint(center_xy[1]).astype(jnp.int32)

        tpl_ok = (y - r >= 0) & (y + r < h_ref) & (x - r >= 0) & (x + r < w_ref)
        xs = x - search
        ys = y - search
        xe = x + search
        ye = y + search
        patch_ok = (ys - r >= 0) & (ye + r < h_def) & (xs - r >= 0) & (xe + r < w_def)
        ok = tpl_ok & patch_ok

        tpl_start = jnp.array([y - r, x - r], dtype=jnp.int32)
        tpl_start = jnp.clip(tpl_start, jnp.array([0, 0]), jnp.array([h_ref - win, w_ref - win]))
        patch_start = jnp.array([ys - r, xs - r], dtype=jnp.int32)
        patch_start = jnp.clip(
            patch_start,
            jnp.array([0, 0]),
            jnp.array([h_def - patch_size, w_def - patch_size]),
        )

        template = lax.dynamic_slice(ref_image, tpl_start, (win, win))
        patch = lax.dynamic_slice(def_image, patch_start, (patch_size, patch_size))
        windows = _extract_windows_jax(patch, win=win, search=search)
        best_idx, best_score = _zncc_argmax(template, windows)

        ky = best_idx // windows_per_axis
        kx = best_idx % windows_per_axis
        yy = ys + ky
        xx = xs + kx
        disp = jnp.array([xx - x, yy - y], dtype=jnp.float32)
        disp = jnp.where(ok, disp, jnp.zeros_like(disp))
        score = jnp.where(ok, best_score, jnp.nan)
        return disp, score

    return lax.map(match_one, centers)
