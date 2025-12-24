"""Helpers that estimate large rigid motions before running DIC."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import sobel
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp_polar
from skimage.util import view_as_windows


def _ensure_gray_float(image: np.ndarray) -> np.ndarray:
    """Return a grayscale float view regardless of the input channels."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = rgb2gray(arr)
    return arr.astype(np.float64, copy=False)


def phase_corr_shift(I0: np.ndarray, I1: np.ndarray, eps: float = 1e-12, subpix: bool = False) -> np.ndarray:
    """Phase-correlation shift estimate returning ``(dx, dy)`` in pixel units.

    ``eps`` guards the FFT ratio; ``subpix`` toggles a denser upsampling factor
    even though we still rely on ``phase_cross_correlation`` for refinement.
    """
    ref = _ensure_gray_float(I0)
    mov = _ensure_gray_float(I1)
    upsample = 10 if subpix else 1
    shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=upsample)
    dy, dx = shift  # phase_cross_correlation reports (row, col) shifts
    return np.array([-dx, -dy], dtype=np.float64)


def rotation_scale_logpolar(
    I0: np.ndarray, I1: np.ndarray, center: Optional[tuple[float, float]] = None
) -> tuple[float, float]:
    """Estimate rotation (rad) and scale using a log-polar remap plus phase correlation.

    The polar center defaults to the image center unless ``center`` is provided.
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
    """Pick up to ``K`` corner-like points, spaced at least ``min_dist`` pixels apart."""
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


@jax.jit
def _zncc_argmax(template: jnp.ndarray, windows: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the (argmax, score) of the normalized cross-correlation over ``windows``."""
    template = template - jnp.mean(template)
    template = template / (jnp.std(template) + 1e-12)
    windows = windows - jnp.mean(windows, axis=(1, 2), keepdims=True)
    windows = windows / (jnp.std(windows, axis=(1, 2), keepdims=True) + 1e-12)
    scores = jnp.mean(windows * template, axis=(1, 2))
    best_idx = jnp.argmax(scores)
    best_score = scores[best_idx]
    return best_idx, best_score


def match_patch_zncc(
    I0: np.ndarray,
    I1: np.ndarray,
    p: np.ndarray,
    win: int = 41,
    search: int = 24,
    pred: tuple[float, float] = (0.0, 0.0),
) -> Optional[tuple[np.ndarray, float]]:
    """ZNCC patch matcher returning ``(dx, dy)`` and the best score, or ``None`` when invalid."""
    I0g = _ensure_gray_float(I0)
    I1g = _ensure_gray_float(I1)
    x, y = map(int, np.round(p))
    r = win // 2
    if y - r < 0 or y + r >= I0g.shape[0] or x - r < 0 or x + r >= I0g.shape[1]:
        return None

    template = I0g[y - r : y + r + 1, x - r : x + r + 1].astype(np.float64, copy=False)

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
    best_idx_jax, best_score_jax = _zncc_argmax(
        jnp.asarray(template),
        jnp.asarray(windows),
    )
    best_idx = int(best_idx_jax)
    ky, kx = divmod(best_idx, ws)
    score = float(best_score_jax)
    yy = ys + ky
    xx = xs + kx
    x_best = xx
    y_best = yy
    disp = np.array([x_best - x, y_best - y], dtype=np.float64)
    return disp, score


def translation_patch_matches_zncc(
    I0: np.ndarray,
    I1: np.ndarray,
    centers: np.ndarray,
    win: int = 41,
    search: int = 24,
    pred: tuple[float, float] = (0.0, 0.0),
    score_min: float | None = None,
    max_centers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match fixed patches around ``centers`` via pure translations scored by ZNCC.

    Parameters
    ----------
    centers: array_like
        Coordinates ``(x, y)`` of candidate patches (typically element centers).
    win: int
        Patch size (pixels) in both directions; must be odd.
    search: int
        Half-width of the square search window explored in ``I1``.
    pred: tuple
        Optional global prediction ``(dx, dy)`` added to each center before searching.
    score_min: float or None
        Keep only matches whose ZNCC score is >= ``score_min`` when set.
    max_centers: int or None
        Evaluate at most this many centers (useful for quick tests).
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must be an array of shape (N, 2).")
    if win % 2 == 0:
        raise ValueError("win must be odd.")
    pred_dx, pred_dy = map(float, pred)
    max_n = centers.shape[0] if max_centers is None else int(max_centers)
    pts_ref: list[np.ndarray] = []
    pts_def: list[np.ndarray] = []
    scores: list[float] = []

    print("Starting translation patch matches ZNCC...")
    print(f"  Number of centers to evaluate: {min(centers.shape[0], max_n)}")
    print(f"  Patch size: {win}x{win}, Search radius: {search}")
    print(f"  Initial prediction: {pred}")
    print(f"  Minimum score threshold: {score_min}")


    for idx in range(min(centers.shape[0], max_n)):
        p = centers[idx]
        disp_score = match_patch_zncc(
            I0,
            I1,
            p,
            win=win,
            search=search,
            pred=(pred_dx, pred_dy),
        )
        if disp_score is None:
            continue
        disp, zncc = disp_score
        if score_min is not None and zncc < score_min:
            continue
        pts_ref.append(p)
        pts_def.append(p + disp)
        scores.append(float(zncc))

    if not pts_ref:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    pts_ref_arr = np.asarray(pts_ref, dtype=np.float64)
    pts_def_arr = np.asarray(pts_def, dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    print(f"translation_patch_matches_zncc: Found {pts_ref_arr.shape[0]} valid matches.")
    return pts_ref_arr, pts_def_arr, scores_arr


def _inverse_distance_weighting(
    pts: np.ndarray, values: np.ndarray, power: float = 2.0
) -> Callable[[np.ndarray], np.ndarray]:
    """Return an inverse-distance predictor with exponent ``power``."""

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
    """Fit a smooth displacement interpolant from sparse matches.

    Uses SciPy RBFs when available (``method`` controls kernel) and falls back
    to inverse-distance weighting otherwise.
    """
    pts_ref = np.asarray(pts_ref, dtype=np.float64)
    pts_def = np.asarray(pts_def, dtype=np.float64)
    disp = pts_def - pts_ref
    n = pts_ref.shape[0]

    if n == 0:
        return lambda xy: np.zeros((np.atleast_2d(xy).shape[0], 2), dtype=np.float64)

    # When there are too few matches, just reuse the mean displacement.
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
                # Rbf extrapolates smoothly outside the convex hull; keep everything in float64.
                dx = rbf_x(q[:, 0], q[:, 1])
                dy = rbf_y(q[:, 0], q[:, 1])
                res = np.vstack([dx, dy]).T
                return res.astype(np.float64, copy=False)

            return predict
        except Exception:
            # If the RBF solve blows up (singular matrix, etc.), fall back to IDW.
            pass

    # Without SciPy or after a failure, use the IDW fallback.
    idw = _inverse_distance_weighting(pts_ref, disp)
    return idw


def make_displacement_predictor(
    pts_ref: np.ndarray,
    pts_def: np.ndarray,
    image_shape: tuple[int, int],
    method: str = "rbf",
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap ``interpolate_displacement_field`` and handle degenerate cases.

    Always returns a callable providing ``(dx, dy)`` for one or many ``(x, y)`` queries.
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
        # Not enough points for interpolation: keep a constant predictor.
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
    mode: str = "robust",
    pred: tuple[float, float] = (0.0, 0.0),
    score_min: float | None = None,
    max_centers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find sparse correspondences either via robust rotation+scale search or pure translation ZNCC.

    Parameters
    ----------
    mode: {"robust", "translation_zncc"}
        ``"robust"`` keeps the default rotation/scale-insensitive pipeline (Harris + RANSAC).
        ``"translation_zncc"`` skips feature detection and performs pure translations scored via ZNCC
        using the provided ``centers`` (typically mesh element centers).
    """
    ref = _ensure_gray_float(I0)
    mov = _ensure_gray_float(I1)

    if mode.lower() in {"translation", "translation_zncc"}:
        if centers is None:
            raise ValueError("translation_zncc mode requires explicit centers.")
        pts_ref_t, pts_def_t, scores_t = translation_patch_matches_zncc(
            ref,
            mov,
            centers,
            win=win,
            search=search,
            pred=pred,
            score_min=score_min,
            max_centers=max_centers,
        )
        return pts_ref_t, pts_def_t, scores_t

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
            # After a few matches, build a smooth predictor to guide the next patches.
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
    "translation_patch_matches_zncc",
    "interpolate_displacement_field",
    "make_displacement_predictor",
    "big_motion_sparse_matches",
]


def _test_translation_patch_matches_zncc() -> None:
    """Quick sanity check for ``translation_patch_matches_zncc``."""
    rng = np.random.default_rng(0)
    H, W = 64, 64
    base = rng.normal(size=(H, W)).astype(np.float64)
    dx, dy = 3, -2

    def shift_image(im: np.ndarray, dx_px: int, dy_px: int) -> np.ndarray:
        out = np.zeros_like(im)
        if dx_px >= 0:
            src_x0, dst_x0 = 0, dx_px
            width = W - dx_px
        else:
            src_x0, dst_x0 = -dx_px, 0
            width = W + dx_px
        if dy_px >= 0:
            src_y0, dst_y0 = 0, dy_px
            height = H - dy_px
        else:
            src_y0, dst_y0 = -dy_px, 0
            height = H + dy_px
        out[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = im[src_y0 : src_y0 + height, src_x0 : src_x0 + width]
        return out

    shifted = shift_image(base, dx, dy)
    centers = np.array([[20.0, 20.0], [30.0, 30.0]], dtype=np.float64)
    pts_ref, pts_def, scores = translation_patch_matches_zncc(
        base,
        shifted,
        centers,
        win=21,
        search=6,
        pred=(0.0, 0.0),
    )
    disp = pts_def - pts_ref
    print("Estimated displacement:", disp)
    print("Scores:", scores)
    assert np.allclose(disp, np.array([[dx, dy]] * disp.shape[0]), atol=1.0), "Translation estimates off."
    assert np.all(scores > 0.8), "ZNCC scores too low."
