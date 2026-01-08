from __future__ import annotations

from typing import Tuple

import numpy as np

from .types import Array
from .mesh_assets import PixelAssets


def bilinear_sample_numpy(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    coords = np.asarray(coords)
    x = coords[:, 0]
    y = coords[:, 1]

    x_cont = x - 0.5
    y_cont = y - 0.5
    j0 = np.floor(x_cont).astype(int)
    i0 = np.floor(y_cont).astype(int)
    fx = x_cont - j0
    fy = y_cont - i0

    H, W = image.shape
    valid = (i0 >= 0) & (i0 < H - 1) & (j0 >= 0) & (j0 < W - 1)
    values = np.full(coords.shape[0], np.nan, dtype=float)
    if not np.any(valid):
        return values

    i0v = i0[valid]
    j0v = j0[valid]
    fxv = fx[valid]
    fyv = fy[valid]
    i1v = i0v + 1
    j1v = j0v + 1

    v00 = image[i0v, j0v]
    v01 = image[i0v, j1v]
    v10 = image[i1v, j0v]
    v11 = image[i1v, j1v]
    values[valid] = (
        v00 * (1.0 - fxv) * (1.0 - fyv)
        + v01 * fxv * (1.0 - fyv)
        + v10 * (1.0 - fxv) * fyv
        + v11 * fxv * fyv
    )
    return values


def compute_discrepancy_map_ref(
    *,
    ref_image: Array,
    def_image: Array,
    u_nodal: Array,
    pixel_assets: PixelAssets,
) -> Tuple[np.ndarray, float]:
    """
    Compute a reference-domain discrepancy map: `I_def(x + u(x)) - I_ref(x)`.

    Returns `(map_ref, rms)` where `map_ref` has shape `(H, W)` and is NaN outside the ROI.
    """
    ref = np.asarray(ref_image)
    deformed = np.asarray(def_image)
    if ref.ndim != 2 or deformed.ndim != 2:
        raise ValueError("ref_image and def_image must be 2D arrays.")

    pixel_coords_ref = np.asarray(pixel_assets.pixel_coords_ref)
    pixel_nodes = np.asarray(pixel_assets.pixel_nodes, dtype=int)
    pixel_shapeN = np.asarray(pixel_assets.pixel_shapeN)
    roi_flat = np.asarray(pixel_assets.roi_mask_flat, dtype=np.int64)
    H, W = pixel_assets.image_shape

    u = np.asarray(u_nodal)
    node_values = u[pixel_nodes]
    pixel_disp = np.sum(pixel_shapeN[..., None] * node_values, axis=1)
    pixel_coords_def = pixel_coords_ref + pixel_disp

    i_ref = bilinear_sample_numpy(ref, pixel_coords_ref)
    i_def = bilinear_sample_numpy(deformed, pixel_coords_def)
    residuals = i_def - i_ref

    valid = np.isfinite(residuals)
    rms = float(np.sqrt(np.mean((residuals[valid]) ** 2))) if np.any(valid) else float("nan")

    flat = np.full(H * W, np.nan, dtype=float)
    if roi_flat.size == 0:
        return flat.reshape(H, W), rms

    idx = roi_flat[valid]
    if idx.size == 0:
        return flat.reshape(H, W), rms

    vals = residuals[valid].astype(float, copy=False)
    accum = np.bincount(idx, weights=vals, minlength=H * W)
    counts = np.bincount(idx, minlength=H * W)
    out = accum / np.maximum(counts, 1)
    mask = counts == 0
    out = out.astype(float, copy=False)
    out[mask] = np.nan
    return out.reshape(H, W), rms
