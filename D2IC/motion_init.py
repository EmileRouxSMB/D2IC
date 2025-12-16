"""Coarse initialization tools for large displacements."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import sobel
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp_polar
from skimage.util import view_as_windows


def _ensure_gray_float(image: np.ndarray) -> np.ndarray:
    """Convert an optional RGB image into floating-point grayscale."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = rgb2gray(arr)
    return arr.astype(np.float64, copy=False)


def phase_corr_shift(I0: np.ndarray, I1: np.ndarray, eps: float = 1e-12, subpix: bool = False) -> np.ndarray:
    """Estimate the global translation between two images via phase correlation.

    Parameters
    ----------
    I0, I1 : np.ndarray
        Reference and deformed 2D images already converted to grayscale.
    eps : float, optional
        Small term to avoid division by zero in the frequency domain.
    subpix : bool, optional
        Enables sub-pixel refinement (not implemented in this version).

    Returns
    -------
    np.ndarray, shape (2,)
        Estimated translation ``(dx, dy)``.
    """
    ref = _ensure_gray_float(I0)
    mov = _ensure_gray_float(I1)
    upsample = 10 if subpix else 1
    shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=upsample)
    dy, dx = shift  # phase_cross_correlation retourne (row, col)
    return np.array([-dx, -dy], dtype=np.float64)


def rotation_scale_logpolar(I0: np.ndarray, I1: np.ndarray, center: tuple[float, float] | None = None) -> tuple[float, float]:
    """Recover rotation and scale mismatch via a log-polar transform.

    Parameters
    ----------
    I0, I1 : np.ndarray
        Reference and deformed grayscale images (2D).
    center : tuple of float, optional
        Center to use for the polar transform; defaults to the image center.

    Returns
    -------
    tuple of float
        ``(dtheta, scale)`` representing the rotation (rad) and scale factor.
    """
    ref = _ensure_gray_float(I0)
    mov = _ensure_gray_float(I1)
    if center is None:
        center = (ref.shape[0] / 2.0, ref.shape[1] / 2.0)
    radius = np.hypot(*center)
    P0 = warp_polar(ref, center=center, radius=radius, scaling="log")
    P1 = warp_polar(mov, center=center, radius=radius, scaling="log")
    dx, dy = phase_corr_shift(P0, P1, subpix=False)
    dtheta = (dx / P0.shape[1]) * 2.0 * np.pi
    scale = np.exp(dy / P0.shape[0])
    return dtheta, scale


def select_patch_centers(I: np.ndarray, K: int = 16, min_dist: int = 32) -> np.ndarray:
    """Select textured patch centers spread across the image.

    Parameters
    ----------
    I : np.ndarray
        2D grayscale image.
    K : int, optional
        Number of desired centers.
    min_dist : int, optional
        Minimum distance (pixels) between retained centers.

    Returns
    -------
    np.ndarray, shape (<=K, 2)
        Selected centers as ``(x, y)`` coordinates.
    """
    img = _ensure_gray_float(I)
    response = corner_harris(img, method="k", k=0.05, sigma=1.2)
    candidates = corner_peaks(
        response,
        min_distance=max(1, min_dist // 2),
        threshold_rel=0.1,
        num_peaks=5 * K,
    )
    if candidates.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    texture = sobel(img)
    scores = texture[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(scores)[::-1]
    chosen: list[tuple[int, int]] = []
    for idx in order:
        y, x = map(int, candidates[idx])
        if all((x - cx) ** 2 + (y - cy) ** 2 >= min_dist ** 2 for cy, cx in chosen):
            chosen.append((y, x))
        if len(chosen) >= K:
            break
    if not chosen:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.array([(float(x), float(y)) for y, x in chosen], dtype=np.float64)
    return pts


def match_patch_zncc(
    I0: np.ndarray,
    I1: np.ndarray,
    p: np.ndarray,
    win: int = 41,
    search: int = 16,
    pred: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, float] | None:
    """Compare a patch using ZNCC before/after to retrieve the most likely displacement.

    Parameters
    ----------
    I0, I1 : np.ndarray
        Reference and deformed grayscale images.
    p : np.ndarray
        Patch center in ``I0`` expressed as ``(x, y)``.
    win : int, optional
        Odd patch window size.
    search : int, optional
        Half-width of the search area around the prediction.
    pred : tuple of float, optional
        Initial displacement estimate ``(dx, dy)``.

    Returns
    -------
    tuple (np.ndarray, float) or None
        Displacement ``(dx, dy)`` and ZNCC score. ``None`` if the patch is invalid.
    """
    I0g = _ensure_gray_float(I0)
    I1g = _ensure_gray_float(I1)
    x, y = map(int, np.round(p))
    r = win // 2
    if y - r < 0 or y + r >= I0g.shape[0] or x - r < 0 or x + r >= I0g.shape[1]:
        return None

    template = I0g[y - r : y + r + 1, x - r : x + r + 1].astype(np.float64, copy=False)
    template = (template - template.mean()) / (template.std() + 1e-12)

    xs = x + int(round(pred[0])) - search
    ys = y + int(round(pred[1])) - search
    xe = x + int(round(pred[0])) + search
    ye = y + int(round(pred[1])) + search
    if ys - r < 0 or ye + r >= I1g.shape[0] or xs - r < 0 or xe + r >= I1g.shape[1]:
        return None

    patch = I1g[ys - r : ye + r + 1, xs - r : xe + r + 1].astype(np.float64, copy=False)
    windows = view_as_windows(patch, (win, win))
    hs, ws = windows.shape[:2]
    windows = windows.reshape(hs * ws, win, win)
    windows = (windows - windows.mean((1, 2), keepdims=True)) / (
        windows.std((1, 2), keepdims=True) + 1e-12
    )
    scores = (windows * template).sum(axis=(1, 2)) / (win * win)
    best_idx = int(np.argmax(scores))
    ky, kx = divmod(best_idx, ws)
    score = float(scores[best_idx])
    yy = ys + ky
    xx = xs + kx
    x_best = xx
    y_best = yy
    disp = np.array([x_best - x, y_best - y], dtype=np.float64)
    return disp, score


def _inverse_distance_weighting(
    pts: np.ndarray, values: np.ndarray, power: float = 2.0
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a predictor based on inverse-distance weighting."""

    def predict(xy: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        out = np.zeros((q.shape[0], values.shape[1]), dtype=np.float64)
        for i, pt in enumerate(q):
            dist = np.linalg.norm(pts - pt, axis=1)
            if dist.size == 0:
                continue
            close = dist < 1e-9
            if close.any():
                out[i] = values[close][0]
                continue
            w = 1.0 / np.power(dist + 1e-12, power)
            w /= w.sum()
            out[i] = (w[:, None] * values).sum(axis=0)
        return out

    return predict


def interpolate_displacement_field(
    pts_ref: np.ndarray,
    pts_def: np.ndarray,
    image_shape: tuple[int, int],
    method: str = "rbf",
    smooth: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Interpolate a smooth displacement field from sparse correspondences.

    Parameters
    ----------
    pts_ref, pts_def : np.ndarray
        Matching points ``(x, y)`` in the reference and deformed images.
    image_shape : tuple of int
        Reference image size ``(H, W)``. Only used for scaling since the
        interpolant returns a continuous predictor.
    method : {"rbf", "tps"}
        Requested interpolant type. ``tps`` uses a thin-plate-spline RBF.
    smooth : float, optional
        Smoothing parameter passed to the RBF solver. ``None`` keeps the default.

    Returns
    -------
    callable
        Function ``u(xy)`` returning an array ``(m, 2)`` of displacements for
        the provided ``xy`` positions.
    """
    pts_ref = np.asarray(pts_ref, dtype=np.float64)
    pts_def = np.asarray(pts_def, dtype=np.float64)
    disp = pts_def - pts_ref
    n = pts_ref.shape[0]

    if n == 0:
        return lambda xy: np.zeros((np.atleast_2d(xy).shape[0], 2), dtype=np.float64)

    # Degenerate cases: too few points for a rich model -> use the average displacement.
    if n < 3:
        mean_disp = disp.mean(axis=0)

        def predict(xy: np.ndarray) -> np.ndarray:
            q = np.atleast_2d(np.asarray(xy, dtype=np.float64))
            return np.repeat(mean_disp[None, :], q.shape[0], axis=0)

        return predict

    method = (method or "rbf").lower()
    smooth_value = 0.0 if smooth is None else float(smooth)
    has_scipy = False
    try:
        from scipy.interpolate import Rbf

        has_scipy = True
    except Exception:
        has_scipy = False

    if has_scipy and method in {"rbf", "tps"}:
        func = "thin_plate" if method == "tps" else "multiquadric"
        try:
            rbf_x = Rbf(
                pts_ref[:, 0],
                pts_ref[:, 1],
                disp[:, 0],
                function=func,
                smooth=smooth_value,
            )
            rbf_y = Rbf(
                pts_ref[:, 0],
                pts_ref[:, 1],
                disp[:, 1],
                function=func,
                smooth=smooth_value,
            )

            def predict(xy: np.ndarray) -> np.ndarray:
                q = np.atleast_2d(np.asarray(xy, dtype=np.float64))
                # Rbf extrapolates smoothly outside the convex hull, but limit
                # numerical growth to stay safe.
                dx = rbf_x(q[:, 0], q[:, 1])
                dy = rbf_y(q[:, 0], q[:, 1])
                res = np.vstack([dx, dy]).T
                return res.astype(np.float64, copy=False)

            return predict
        except Exception:
            # Fall back to a local scheme if the RBF fit fails (singular matrix).
            pass

    # Fallback without SciPy or in case of failure: simple IDW interpolation.
    idw = _inverse_distance_weighting(pts_ref, disp)
    return idw


def make_displacement_predictor(
    pts_ref: np.ndarray,
    pts_def: np.ndarray,
    image_shape: tuple[int, int],
    method: str = "rbf",
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a continuous displacement predictor from sparse matches.

    Parameters
    ----------
    pts_ref, pts_def : np.ndarray
        Matching points ``(x, y)`` in the reference and deformed images.
    image_shape : tuple of int
        Size of the reference image ``(H, W)``.
    method : str, optional
        Interpolation method to use (``"rbf"`` or ``"tps"``).

    Returns
    -------
    callable
        Function ``pred(p)`` returning ``(dx, dy)`` for one or many points.
    """
    pts_ref = np.asarray(pts_ref, dtype=np.float64)
    pts_def = np.asarray(pts_def, dtype=np.float64)
    n = pts_ref.shape[0]

    if n == 0:
        mean_disp = np.zeros(2, dtype=np.float64)
    else:
        mean_disp = (pts_def - pts_ref).mean(axis=0)

    if n >= 3:
        core = interpolate_displacement_field(pts_ref, pts_def, image_shape, method=method)
    else:
        # Too few points: fall back to a robust constant prediction.
        core = lambda xy: np.repeat(mean_disp[None, :], np.atleast_2d(xy).shape[0], axis=0)

    def wrapper(p: np.ndarray) -> np.ndarray:
        arr = np.asarray(p, dtype=np.float64)
        pts = arr.reshape(-1, 2)
        disp = core(pts)
        disp = np.asarray(disp, dtype=np.float64)
        return disp[0] if arr.ndim == 1 else disp

    return wrapper


def big_motion_sparse_matches(
    I0: np.ndarray,
    I1: np.ndarray,
    K: int = 16,
    win: int = 41,
    search: int = 24,
    interp_method: str | None = "rbf",
    centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Obtain robust sparse correspondences suited for large displacements.

    Parameters
    ----------
    I0, I1 : np.ndarray
        Reference and deformed images (2D or 3D).
    K : int, optional
        Target number of valid patches.
    win : int, optional
        Odd patch size for ZNCC comparisons.
    search : int, optional
        Half-width of the search zone for each patch.
    interp_method : {"rbf", "tps", None}, optional
        Interpolation strategy used to improve local displacement predictions
        from already-found matches. ``None`` disables that refinement.
    centers : np.ndarray, optional
        Explicit centers ``(x, y)`` for patch matching. If ``None`` (default),
        centers are selected automatically.

    Returns
    -------
    tuple of np.ndarray
        ``(pts_ref, pts_def, scores)`` where each array has shape ``(n, 2)``.
    """
    ref = _ensure_gray_float(I0)
    mov = _ensure_gray_float(I1)

    theta, scale = rotation_scale_logpolar(ref, mov)
    center_xy = np.array([ref.shape[1] / 2.0, ref.shape[0] / 2.0], dtype=np.float64)
    sim = SimilarityTransform(scale=scale, rotation=theta)

    gdx, gdy = phase_corr_shift(ref, mov)
    global_shift = np.array([gdx, gdy], dtype=np.float64)

    if centers is None:
        pts0 = select_patch_centers(ref, K=K, min_dist=max(win, 24))
    else:
        pts0 = np.asarray(centers, dtype=np.float64)
    if pts0.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    matches: list[tuple[np.ndarray, np.ndarray, float]] = []
    pred_func: Callable[[np.ndarray], np.ndarray] | None = None
    for p in pts0:
        if pred_func is not None:
            pred_disp = np.asarray(pred_func(p), dtype=np.float64)
        else:
            rotated = sim(np.atleast_2d(p - center_xy))[0]
            predicted_point = rotated + center_xy + global_shift
            pred_disp = predicted_point - p
        out = match_patch_zncc(ref, mov, p, win=win, search=search, pred=tuple(pred_disp))
        if out is not None:
            disp, score = out
            matches.append((p, p + disp, score))
            # Once enough matches are available, interpolate a smooth field to
            # better guide subsequent patches (non-affine motions).
            if interp_method is not None and len(matches) >= 3:
                pts_ref = np.stack([m[0] for m in matches], axis=0)
                pts_def = np.stack([m[1] for m in matches], axis=0)
                pred_func = make_displacement_predictor(
                    pts_ref,
                    pts_def,
                    ref.shape,
                    method=interp_method,
                )

    if not matches:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    pts_ref = np.stack([m[0] for m in matches], axis=0).astype(np.float64, copy=False)
    pts_def = np.stack([m[1] for m in matches], axis=0).astype(np.float64, copy=False)
    scores = np.array([m[2] for m in matches], dtype=np.float64)
    return pts_ref, pts_def, scores


__all__ = [
    "phase_corr_shift",
    "rotation_scale_logpolar",
    "select_patch_centers",
    "match_patch_zncc",
    "interpolate_displacement_field",
    "make_displacement_predictor",
    "big_motion_sparse_matches",
]
