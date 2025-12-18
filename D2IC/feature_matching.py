"""Feature-based matching utilities shared by DIC pipelines."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from skimage.measure import ransac
from skimage.transform import AffineTransform, SimilarityTransform
from skimage.util import view_as_windows


def _points_in_mesh_area(points, mesh_nodes=None, mesh_elements=None, dilation: float = 0.0):
    """Return a boolean mask marking points inside the FE mesh footprint (with optional dilation)."""
    if mesh_nodes is None or mesh_elements is None:
        return np.ones(points.shape[0], dtype=bool)

    nodes = np.asarray(mesh_nodes)
    elements = np.asarray(mesh_elements, dtype=np.int32)
    if nodes.ndim != 2 or nodes.shape[1] < 2:
        raise ValueError("Mesh nodes must be provided as (n_nodes, >=2) coordinates.")
    if elements.size == 0:
        return np.ones(points.shape[0], dtype=bool)

    nodes_xy = nodes[:, :2]
    mask = np.zeros(points.shape[0], dtype=bool)
    dilation = float(dilation) if dilation else 0.0
    for quad in elements:
        path = Path(nodes_xy[np.asarray(quad, dtype=np.int32)])
        mask |= path.contains_points(points, radius=dilation)
    return mask


def refine_matches_ncc(I0, I1, pts0, pts1_pred, win: int = 31, search: int = 3):
    """Subpixel NCC refinement returning ``(dx, dy)`` corrections for each input match."""
    I0 = np.asarray(I0)
    I1 = np.asarray(I1)
    pts0 = np.asarray(pts0)
    pts1_pred = np.asarray(pts1_pred)

    def ncc_subpixel(template, patch):
        # template: (w, w), patch: larger window; returns subpixel (dx, dy)
        w = template.shape[0]
        windows = view_as_windows(patch, (w, w))
        hs, ws = windows.shape[:2]
        windows = windows.reshape(hs * ws, w, w)
        template_norm = (template - template.mean()) / (template.std() + 1e-12)
        windows_norm = (windows - windows.mean((1, 2), keepdims=True)) / (
            windows.std((1, 2), keepdims=True) + 1e-12
        )
        scores = (windows_norm * template_norm[None, :, :]).sum(axis=(1, 2)) / (w * w)
        best_idx = int(np.argmax(scores))
        ky, kx = divmod(best_idx, ws)
        scores_reshaped = scores.reshape(hs, ws)

        def parabolic_offset(a, b, c):
            denom = 2 * (a - 2 * b + c)
            return 0.0 if denom == 0 else (a - c) / denom

        offx = (
            parabolic_offset(
                scores_reshaped[ky, max(0, kx - 1)],
                scores_reshaped[ky, kx],
                scores_reshaped[ky, min(ws - 1, kx + 1)],
            )
            if 0 < kx < ws - 1
            else 0.0
        )
        offy = (
            parabolic_offset(
                scores_reshaped[max(0, ky - 1), kx],
                scores_reshaped[ky, kx],
                scores_reshaped[min(hs - 1, ky + 1), kx],
            )
            if 0 < ky < hs - 1
            else 0.0
        )
        return (kx + offx, ky + offy)

    height, width = I0.shape
    radius = win // 2
    refined = []
    for (x0, y0), (x1p, y1p) in zip(pts0, pts1_pred):
        x0i, y0i = int(round(x0)), int(round(y0))
        if y0i - radius < 0 or y0i + radius >= height or x0i - radius < 0 or x0i + radius >= width:
            continue
        template = I0[y0i - radius : y0i + radius + 1, x0i - radius : x0i + radius + 1].astype(
            np.float64
        )
        xs, ys = int(round(x1p)) - search, int(round(y1p)) - search
        xe, ye = int(round(x1p)) + search, int(round(y1p)) + search
        if ys - radius < 0 or ye + radius >= height or xs - radius < 0 or xe + radius >= width:
            continue
        patch = I1[ys - radius : ye + radius + 1, xs - radius : xe + radius + 1].astype(
            np.float64
        )
        kx, ky = ncc_subpixel(template, patch)
        x_best = xs + kx
        y_best = ys + ky
        refined.append([x_best - x0, y_best - y0])
    if not refined:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(refined, dtype=np.float64)


def local_ransac_outlier_filter(
    pts_ref: np.ndarray,
    pts_def: np.ndarray,
    model: str = "similarity",
    k_neighbors: int = 30,
    residual_threshold: float = 2.5,
    max_trials: int = 500,
) -> np.ndarray:
    """Mark inliers using localized RANSAC fits around each correspondence."""
    pts_ref = np.asarray(pts_ref, dtype=np.float64)
    pts_def = np.asarray(pts_def, dtype=np.float64)
    n = pts_ref.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    model_class = AffineTransform if model == "affine" else SimilarityTransform
    min_samples = 3 if model == "affine" else 2
    k = int(max(1, min(k_neighbors, n)))

    diff = pts_ref[None, :, :] - pts_ref[:, None, :]
    dist2 = np.sum(diff ** 2, axis=-1)

    inlier_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        neigh_idx = np.argsort(dist2[i])[:k]
        if neigh_idx.size < min_samples:
            inlier_mask[i] = True
            continue
        src = pts_ref[neigh_idx]
        dst = pts_def[neigh_idx]
        try:
            local_model, _ = ransac(
                (src, dst),
                model_class,
                min_samples=min_samples,
                residual_threshold=residual_threshold,
                max_trials=max_trials,
            )
        except Exception:
            inlier_mask[i] = True
            continue
        if local_model is None:
            inlier_mask[i] = True
            continue
        pt_pred = local_model(pts_ref[i][None, :])[0]
        res = np.linalg.norm(pt_pred - pts_def[i])
        inlier_mask[i] = res <= residual_threshold
    return inlier_mask


__all__ = [
    "_points_in_mesh_area",
    "refine_matches_ncc",
    "local_ransac_outlier_filter",
]
