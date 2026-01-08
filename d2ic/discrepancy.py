from __future__ import annotations

from typing import Tuple

import numpy as np
import jax.numpy as jnp

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


def bilinear_sample_jax(image: Array, coords: Array) -> Array:
    image = jnp.asarray(image)
    coords = jnp.asarray(coords)
    x = coords[:, 0]
    y = coords[:, 1]

    x_cont = x - 0.5
    y_cont = y - 0.5
    j0 = jnp.floor(x_cont).astype(jnp.int32)
    i0 = jnp.floor(y_cont).astype(jnp.int32)
    fx = x_cont - j0.astype(image.dtype)
    fy = y_cont - i0.astype(image.dtype)

    H, W = image.shape
    valid = (i0 >= 0) & (i0 < H - 1) & (j0 >= 0) & (j0 < W - 1)

    i0v = jnp.clip(i0, 0, H - 2)
    j0v = jnp.clip(j0, 0, W - 2)
    i1v = i0v + 1
    j1v = j0v + 1

    v00 = image[i0v, j0v]
    v01 = image[i0v, j1v]
    v10 = image[i1v, j0v]
    v11 = image[i1v, j1v]
    values = (
        v00 * (1.0 - fx) * (1.0 - fy)
        + v01 * fx * (1.0 - fy)
        + v10 * (1.0 - fx) * fy
        + v11 * fx * fy
    )
    values = jnp.where(valid, values, jnp.nan)
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
    ref = jnp.asarray(ref_image)
    deformed = jnp.asarray(def_image)
    if ref.ndim != 2 or deformed.ndim != 2:
        raise ValueError("ref_image and def_image must be 2D arrays.")

    pixel_coords_ref = jnp.asarray(pixel_assets.pixel_coords_ref)
    pixel_nodes = jnp.asarray(pixel_assets.pixel_nodes, dtype=jnp.int32)
    pixel_shapeN = jnp.asarray(pixel_assets.pixel_shapeN)
    roi_flat = jnp.asarray(pixel_assets.roi_mask_flat, dtype=jnp.int32)
    H, W = pixel_assets.image_shape

    u = jnp.asarray(u_nodal)
    node_values = u[pixel_nodes]
    pixel_disp = np.sum(pixel_shapeN[..., None] * node_values, axis=1)
    pixel_coords_def = pixel_coords_ref + pixel_disp

    i_ref = bilinear_sample_jax(ref, pixel_coords_ref)
    i_def = bilinear_sample_jax(deformed, pixel_coords_def)
    residuals = i_def - i_ref

    valid = jnp.isfinite(residuals)
    valid_f = valid.astype(residuals.dtype)
    denom = jnp.maximum(jnp.sum(valid_f), jnp.asarray(1.0, dtype=residuals.dtype))
    rms = jnp.sqrt(jnp.sum((residuals * valid_f) ** 2) / denom)

    accum = jnp.bincount(roi_flat, weights=residuals * valid_f, minlength=H * W)
    counts = jnp.bincount(roi_flat, weights=valid_f, minlength=H * W)
    out = accum / jnp.maximum(counts, 1.0)
    out = jnp.where(counts > 0, out, jnp.nan)
    return out.reshape(H, W), float(rms)
