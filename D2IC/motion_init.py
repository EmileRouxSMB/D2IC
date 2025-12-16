"""Outils d'initialisation grossière pour les grands déplacements."""

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
    """Convertir une image éventuelle RGB en niveaux de gris flottants."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = rgb2gray(arr)
    return arr.astype(np.float64, copy=False)


def phase_corr_shift(I0: np.ndarray, I1: np.ndarray, eps: float = 1e-12, subpix: bool = False) -> np.ndarray:
    """Estimer la translation globale entre deux images par corrélation de phase.

    Parameters
    ----------
    I0, I1 : np.ndarray
        Images de référence et déformée (2D) déjà converties en niveaux de gris.
    eps : float, optional
        Petit terme pour éviter la division par zéro dans le domaine fréquentiel.
    subpix : bool, optional
        Active un affinage sub-pixel (non implémenté dans cette version).

    Returns
    -------
    np.ndarray, shape (2,)
        Translation estimée ``(dx, dy)``.
    """
    ref = _ensure_gray_float(I0)
    mov = _ensure_gray_float(I1)
    upsample = 10 if subpix else 1
    shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=upsample)
    dy, dx = shift  # phase_cross_correlation retourne (row, col)
    return np.array([-dx, -dy], dtype=np.float64)


def rotation_scale_logpolar(I0: np.ndarray, I1: np.ndarray, center: tuple[float, float] | None = None) -> tuple[float, float]:
    """Récupérer l'écart de rotation et d'échelle via transformation log-polaire.

    Parameters
    ----------
    I0, I1 : np.ndarray
        Images de référence et déformée (2D) en niveaux de gris.
    center : tuple of float, optional
        Centre à utiliser pour la transformation polaire; par défaut centre de l'image.

    Returns
    -------
    tuple of float
        ``(dtheta, scale)`` représentant la rotation (rad) et le facteur d'échelle.
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
    """Sélectionner des centres de patchs texturés et répartis sur l'image.

    Parameters
    ----------
    I : np.ndarray
        Image (2D) en niveaux de gris.
    K : int, optional
        Nombre de centres souhaité.
    min_dist : int, optional
        Distance minimale (pixels) entre deux centres retenus.

    Returns
    -------
    np.ndarray, shape (<=K, 2)
        Centres retenus sous la forme ``(x, y)``.
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
    """Comparer un patch par ZNCC avant/après pour retrouver le déplacement le plus probable.

    Parameters
    ----------
    I0, I1 : np.ndarray
        Images de référence et déformée (2D) en niveaux de gris.
    p : np.ndarray
        Centre du patch dans ``I0`` sous la forme ``(x, y)``.
    win : int, optional
        Taille (impair) de la fenêtre de patch.
    search : int, optional
        Demi-largeur de la zone de recherche autour de la prédiction.
    pred : tuple of float, optional
        Déplacement initial estimé ``(dx, dy)``.

    Returns
    -------
    tuple (np.ndarray, float) or None
        Déplacement ``(dx, dy)`` et score ZNCC associé. ``None`` si le patch est invalide.
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
    """Créer un prédicteur par pondération inverse de la distance."""

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
    """Interpoler un champ de déplacement lisse à partir de correspondances éparses.

    Parameters
    ----------
    pts_ref, pts_def : np.ndarray
        Points correspondants ``(x, y)`` dans les images de référence et déformée.
    image_shape : tuple of int
        Taille de l'image de référence ``(H, W)``. Utilisée uniquement pour
        l'échelle, l'interpolation renvoyant un prédicteur continu.
    method : {"rbf", "tps"}
        Type d'interpolant demandé. ``tps`` s'appuie sur une RBF à plaque mince.
    smooth : float, optional
        Paramètre de lissage passé au solveur RBF. ``None`` revient au défaut.

    Returns
    -------
    callable
        Fonction ``u(xy)`` retournant un tableau ``(m, 2)`` de déplacements
        pour les positions ``xy`` fournies.
    """
    pts_ref = np.asarray(pts_ref, dtype=np.float64)
    pts_def = np.asarray(pts_def, dtype=np.float64)
    disp = pts_def - pts_ref
    n = pts_ref.shape[0]

    if n == 0:
        return lambda xy: np.zeros((np.atleast_2d(xy).shape[0], 2), dtype=np.float64)

    # Cas dégénérés : trop peu de points pour un modèle riche -> déplacement moyen.
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
                # Rbf extrapole en douceur en dehors de l'enveloppe convexe, mais on
                # reste prudent en limitant la croissance numérique.
                dx = rbf_x(q[:, 0], q[:, 1])
                dy = rbf_y(q[:, 0], q[:, 1])
                res = np.vstack([dx, dy]).T
                return res.astype(np.float64, copy=False)

            return predict
        except Exception:
            # On retombe sur un schéma local si l'ajustement RBF échoue (matrice singulière).
            pass

    # Fallback sans SciPy ou en cas d'échec : interpolation par IDW simple.
    idw = _inverse_distance_weighting(pts_ref, disp)
    return idw


def make_displacement_predictor(
    pts_ref: np.ndarray,
    pts_def: np.ndarray,
    image_shape: tuple[int, int],
    method: str = "rbf",
) -> Callable[[np.ndarray], np.ndarray]:
    """Construire un prédicteur de déplacement continu à partir de matches épars.

    Parameters
    ----------
    pts_ref, pts_def : np.ndarray
        Points correspondants ``(x, y)`` dans les images de référence et déformée.
    image_shape : tuple of int
        Taille de l'image de référence ``(H, W)``.
    method : str, optional
        Méthode d'interpolation à utiliser (``"rbf"`` ou ``"tps"``).

    Returns
    -------
    callable
        Fonction ``pred(p)`` qui renvoie ``(dx, dy)`` pour un point ou un lot de points.
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
        # Trop peu de points : prédiction constante robuste.
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
    """Obtenir des correspondances éparses robustes adaptées aux grands déplacements.

    Parameters
    ----------
    I0, I1 : np.ndarray
        Images de référence et déformée (2D ou 3D).
    K : int, optional
        Nombre ciblé de patchs valides.
    win : int, optional
        Taille (impair) des patchs comparés par ZNCC.
    search : int, optional
        Demi-largeur de la zone de recherche pour chaque patch.
    interp_method : {"rbf", "tps", None}, optional
        Méthode d'interpolation pour améliorer la prédiction locale du déplacement
        à partir des matches déjà trouvés. ``None`` désactive cette amélioration.
    centers : np.ndarray, optional
        Centres explicites ``(x, y)`` où lancer une comparaison de patchs. Si
        ``None`` (défaut), les centres sont sélectionnés automatiquement.

    Returns
    -------
    tuple of np.ndarray
        ``(pts_ref, pts_def, scores)`` où les tableaux ont la forme ``(n, 2)``.
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
            # Dès que suffisamment de matches sont disponibles, on interpole un champ
            # lisse pour mieux guider les patchs suivants (déplacements non affines).
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
