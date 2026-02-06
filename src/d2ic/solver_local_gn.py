from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from functools import partial
import warnings

try:  # pragma: no cover - optional dependency for fast interpolation
    from dm_pix import flat_nd_linear_interpolate
except Exception:  # pragma: no cover
    flat_nd_linear_interpolate = None
try:  # pragma: no cover - optional cubic interpolation
    from im_jax import flat_nd_cubic_interpolate
except Exception:  # pragma: no cover
    flat_nd_cubic_interpolate = None

from .solver_base import SolverBase
from .types import Array
from .mesh_assets import MeshAssets, PixelAssets


@dataclass(frozen=True)
class LocalGNResult:
    u_nodal: Array
    strain: Array
    history: Array | None = None


class LocalGaussNewtonSolver(SolverBase):
    """
    Local Gauss-Newton solver for mesh DIC (e.g., nodal local updates).

    Notes
    -----
    This solver expects `MeshAssets.pixel_data` to be populated with `PixelAssets`
    (see :func:`d2ic.pixel_assets.build_pixel_assets`).
    """

    def __init__(
        self,
        lam: float = 1e-3,
        max_step: float = 0.2,
        omega: float = 0.5,
        interpolation: str = "cubic",
        use_map_coordinates: bool | None = None,
    ) -> None:
        self._compiled = False
        self._solve_jit = None
        self._lam = float(lam)
        self._max_step = float(max_step)
        self._omega = float(omega)
        if use_map_coordinates is not None:
            warnings.warn(
                "use_map_coordinates is kept for backward compatibility; use interpolation='cubic' or 'linear'.",
                DeprecationWarning,
                stacklevel=2,
            )
            interpolation = "cubic" if use_map_coordinates else "linear"
        self._interpolation = _normalize_interpolation(interpolation)

    def compile(self, assets: Any) -> None:
        pixel_data = getattr(assets, "pixel_data", None)
        if pixel_data is None:
            raise ValueError("MeshAssets must provide pixel_data for LocalGaussNewtonSolver.")

        def _local_fn(
            disp0,
            im1_T,
            im2_T,
            pixel_coords_ref,
            pixel_nodes,
            pixel_shapeN,
            node_pixel_index,
            node_N_weight,
            node_pixel_degree,
            node_neighbor_index,
            node_neighbor_degree,
            node_neighbor_weight,
            node_reg_weight,
            gradx2_T,
            grady2_T,
            lam,
            max_step,
            alpha_reg,
            omega,
            n_sweeps,
            interpolation,
        ):
            return _local_sweeps(
                disp0,
                im1_T,
                im2_T,
                pixel_coords_ref,
                pixel_nodes,
                pixel_shapeN,
                node_pixel_index,
                node_N_weight,
                node_pixel_degree,
                node_neighbor_index,
                node_neighbor_degree,
                node_neighbor_weight,
                node_reg_weight,
                gradx2_T,
                grady2_T,
                lam,
                max_step,
                alpha_reg,
                omega,
                n_sweeps,
                interpolation,
            )

        self._solve_jit = jax.jit(
            _local_fn,
            static_argnums=(19, 20),
            donate_argnums=(0,),
        )
        self._compiled = True

    def warmup(self, state: Any) -> None:
        """Compile the JIT once using dummy inputs that match the real shapes."""
        if not self._compiled or self._solve_jit is None:
            raise RuntimeError("LocalGaussNewtonSolver.compile() must be called before warmup().")

        assets: MeshAssets = state.assets
        pix: PixelAssets = assets.pixel_data  # type: ignore[assignment]
        if pix is None:
            raise ValueError("MeshAssets must provide pixel_data for LocalGaussNewtonSolver warmup.")

        ref_im = state.ref_image
        im1_T = jnp.transpose(ref_im, (1, 0))
        im2_T = im1_T

        nodes_xy_device = getattr(state, "nodes_xy_device", None)
        if nodes_xy_device is None:
            nodes_xy_device = jnp.asarray(assets.mesh.nodes_xy)
        disp0 = jnp.zeros_like(nodes_xy_device)

        gradx2_T = jnp.zeros_like(im2_T)
        grady2_T = jnp.zeros_like(im2_T)

        self._solve_jit.lower(
            disp0,
            im1_T,
            im2_T,
            pix.pixel_coords_ref,
            pix.pixel_nodes,
            pix.pixel_shapeN,
            pix.node_pixel_index,
            pix.node_N_weight,
            pix.node_pixel_degree,
            pix.node_neighbor_index,
            pix.node_neighbor_degree,
            pix.node_neighbor_weight,
            pix.node_reg_weight,
            gradx2_T,
            grady2_T,
            float(self._lam),
            float(self._max_step),
            float(state.config.reg_strength),
            float(self._omega),
            int(state.config.max_iters),
            self._interpolation,
        ).compile()

    def solve(self, state: Any, def_image: Array) -> LocalGNResult:
        if not self._compiled or self._solve_jit is None:
            raise RuntimeError("LocalGaussNewtonSolver.compile() must be called before solve().")

        assets: MeshAssets = state.assets
        pix: PixelAssets = assets.pixel_data  # type: ignore[assignment]
        if pix is None:
            raise ValueError("MeshAssets must provide pixel_data for LocalGaussNewtonSolver.")

        ref_im = state.ref_image
        def_im = jnp.asarray(def_image)
        im1_T = jnp.transpose(ref_im, (1, 0))
        im2_T = jnp.transpose(def_im, (1, 0))

        nodes_xy_device = getattr(state, "nodes_xy_device", None)
        if nodes_xy_device is None:
            nodes_xy_device = jnp.asarray(assets.mesh.nodes_xy)
        disp0 = (
            jnp.asarray(state.u0_nodal)
            if state.u0_nodal is not None
            else jnp.zeros_like(nodes_xy_device)
        )
        disp0 = jnp.asarray(disp0)

        gx2, gy2 = _compute_image_gradient_jax(def_im)
        gx2_T = jnp.transpose(gx2, (1, 0))
        gy2_T = jnp.transpose(gy2, (1, 0))

        disp_sol, history = self._solve_jit(
            disp0,
            im1_T,
            im2_T,
            pix.pixel_coords_ref,
            pix.pixel_nodes,
            pix.pixel_shapeN,
            pix.node_pixel_index,
            pix.node_N_weight,
            pix.node_pixel_degree,
            pix.node_neighbor_index,
            pix.node_neighbor_degree,
            pix.node_neighbor_weight,
            pix.node_reg_weight,
            gx2_T,
            gy2_T,
            float(self._lam),
            float(self._max_step),
            float(state.config.reg_strength),
            float(self._omega),
            int(state.config.max_iters),
            self._interpolation,
        )

        strain = jnp.zeros((disp_sol.shape[0], 3), dtype=disp_sol.dtype)
        save_history = bool(getattr(state.config, "save_history", False))
        return LocalGNResult(u_nodal=disp_sol, strain=strain, history=history if save_history else None)


# ---------------------------------------------------------------------
# Core kernels (ported from previous implementation)
# ---------------------------------------------------------------------

def _compute_image_gradient_np(im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    im = np.asarray(im, dtype=float)
    gx = np.zeros_like(im)
    gy = np.zeros_like(im)
    gx[:, 1:-1] = 0.5 * (im[:, 2:] - im[:, :-2])
    gy[1:-1, :] = 0.5 * (im[2:, :] - im[:-2, :])
    return gx, gy


def _compute_image_gradient_jax(im: Array) -> Tuple[Array, Array]:
    """Central-difference gradient on device (edges set to zero)."""
    im = jnp.asarray(im)
    gx = jnp.zeros_like(im)
    gy = jnp.zeros_like(im)
    gx = gx.at[:, 1:-1].set(0.5 * (im[:, 2:] - im[:, :-2]))
    gy = gy.at[1:-1, :].set(0.5 * (im[2:, :] - im[:-2, :]))
    return gx, gy


@partial(jax.jit, static_argnames=("interpolation",))
def _pixel_state(
    displacement,
    im1_T,
    im2_T,
    pixel_coords,
    pixel_nodes,
    pixel_shapeN,
    gradx2_T,
    grady2_T,
    interpolation="cubic",
):
    shapeN = pixel_shapeN[..., None]
    disp_local = displacement[pixel_nodes]
    u_pix = (shapeN * disp_local).sum(axis=1)
    x_ref = pixel_coords
    x_def = x_ref + u_pix

    if interpolation == "cubic":
        I1 = _cubic_sample(im1_T, x_ref)
        I2 = _cubic_sample(im2_T, x_def)
        gx_def = _cubic_sample(gradx2_T, x_def)
        gy_def = _cubic_sample(grady2_T, x_def)
    elif interpolation == "linear":
        if flat_nd_linear_interpolate is not None:
            I1 = flat_nd_linear_interpolate(im1_T, x_ref.T)
            I2 = flat_nd_linear_interpolate(im2_T, x_def.T)
            gx_def = flat_nd_linear_interpolate(gradx2_T, x_def.T)
            gy_def = flat_nd_linear_interpolate(grady2_T, x_def.T)
        else:
            I1 = _bilinear_sample(im1_T, x_ref)
            I2 = _bilinear_sample(im2_T, x_def)
            gx_def = _bilinear_sample(gradx2_T, x_def)
            gy_def = _bilinear_sample(grady2_T, x_def)
    else:
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    r = I2 - I1
    return r, x_def, gx_def, gy_def


@partial(jax.jit, static_argnames=("interpolation", "n_sweeps"))
def _local_sweeps(
    displacement,
    im1_T,
    im2_T,
    pixel_coords,
    pixel_nodes,
    pixel_shapeN,
    node_pixel_index,
    node_N_weight,
    node_pixel_degree,
    node_neighbor_index,
    node_neighbor_degree,
    node_neighbor_weight,
    node_reg_weight,
    gradx2_T,
    grady2_T,
    lam,
    max_step,
    alpha_reg,
    omega,
    n_sweeps,
    interpolation,
):
    hist0 = jnp.full((int(n_sweeps), 2), jnp.nan, dtype=displacement.dtype)

    def body_fun(k, carry):
        disp, hist = carry
        r, _x_def, gx_def, gy_def = _pixel_state(
            disp,
            im1_T,
            im2_T,
            pixel_coords,
            pixel_nodes,
            pixel_shapeN,
            gradx2_T,
            grady2_T,
            interpolation=interpolation,
        )
        disp_next = jacobi_nodal_step_spring(
            disp,
            r,
            gx_def,
            gy_def,
            node_pixel_index,
            node_N_weight,
            node_pixel_degree,
            node_neighbor_index,
            node_neighbor_degree,
            node_neighbor_weight,
            node_reg_weight,
            lam=lam,
            max_step=max_step,
            alpha_reg=alpha_reg,
            omega=omega,
        )
        # History: per-sweep RMS residual and RMS displacement update.
        r_rms = jnp.sqrt(jnp.mean(r * r))
        step = disp_next - disp
        step_rms = jnp.sqrt(jnp.mean(step * step))
        hist = hist.at[k].set(jnp.asarray([r_rms, step_rms], dtype=hist.dtype))
        return disp_next, hist

    disp_final, hist_final = lax.fori_loop(0, n_sweeps, body_fun, (displacement, hist0))
    return disp_final, hist_final


@jax.jit
def jacobi_nodal_step_spring(
    displacement,
    r,
    gx_def,
    gy_def,
    node_pixel_index,
    node_N_weight,
    node_degree,
    node_neighbor_index,
    node_neighbor_degree,
    node_neighbor_weight,
    node_reg_weight,
    lam=0.1,
    max_step=0.2,
    alpha_reg=0.0,
    omega=0.5,
):
    """Relaxed Jacobi update mixing pixel residuals and spring regularization."""
    disp0 = jnp.asarray(displacement)
    dtype = disp0.dtype
    alpha_reg = jnp.asarray(alpha_reg, dtype=dtype)
    lam = jnp.asarray(lam, dtype=dtype)
    max_step = jnp.asarray(max_step, dtype=dtype)
    omega = jnp.asarray(omega, dtype=dtype)

    Nnodes, max_deg_pix = node_pixel_index.shape
    _, max_deg_neigh = node_neighbor_index.shape

    node_reg_weight = jnp.asarray(node_reg_weight, dtype=dtype)
    deg_pix = jnp.asarray(node_degree)
    deg_neigh = jnp.asarray(node_neighbor_degree)

    # ----------------------------
    # Image term (vectorized over nodes)
    # ----------------------------
    pix_mask = jnp.arange(max_deg_pix)[None, :] < deg_pix[:, None]
    pix_idx = jnp.where(pix_mask, node_pixel_index, 0)
    Ni = jnp.where(pix_mask, jnp.asarray(node_N_weight, dtype=dtype), jnp.asarray(0.0, dtype=dtype))

    ri = r[pix_idx]
    gxi = gx_def[pix_idx]
    gyi = gy_def[pix_idx]

    Jx = gxi * Ni
    Jy = gyi * Ni

    g0_img = jnp.sum(ri * Jx, axis=1)
    g1_img = jnp.sum(ri * Jy, axis=1)
    g_img = jnp.stack([g0_img, g1_img], axis=1)  # (Nnodes, 2)

    H00 = jnp.sum(Jx * Jx, axis=1)
    H01 = jnp.sum(Jx * Jy, axis=1)
    H11 = jnp.sum(Jy * Jy, axis=1)

    traceH = H00 + H11 + jnp.asarray(1e-12, dtype=dtype)
    lam_loc = lam * traceH
    H_img = jnp.stack(
        [
            jnp.stack([H00 + lam_loc, H01], axis=1),
            jnp.stack([H01, H11 + lam_loc], axis=1),
        ],
        axis=1,
    )  # (Nnodes, 2, 2)

    # ----------------------------
    # Regularization term (vectorized over nodes)
    # ----------------------------
    neigh_mask = jnp.arange(max_deg_neigh)[None, :] < deg_neigh[:, None]
    neigh_ids = jnp.where(neigh_mask, node_neighbor_index, 0)
    w = jnp.where(neigh_mask, jnp.asarray(node_neighbor_weight, dtype=dtype), jnp.asarray(0.0, dtype=dtype))

    u_i = disp0[:, None, :]  # (Nnodes, 1, 2)
    u_neigh = disp0[neigh_ids]  # (Nnodes, max_deg_neigh, 2)
    u_neigh = jnp.where(neigh_mask[..., None], u_neigh, jnp.asarray(0.0, dtype=dtype))

    diff = u_i - u_neigh
    alpha_loc = alpha_reg * node_reg_weight  # (Nnodes,)
    g_reg = alpha_loc[:, None] * jnp.sum(w[..., None] * diff, axis=1)  # (Nnodes, 2)

    sumw = jnp.sum(w, axis=1)  # (Nnodes,)
    eye2 = jnp.eye(2, dtype=dtype)[None, :, :]
    H_reg = (alpha_loc * sumw)[:, None, None] * eye2  # (Nnodes, 2, 2)

    # ----------------------------
    # Combine, solve, clip, and relax
    # ----------------------------
    has_reg = jnp.logical_and(alpha_reg != jnp.asarray(0.0, dtype=dtype), deg_neigh > 0)
    has_pix = deg_pix > 0
    has_update = jnp.logical_or(has_pix, has_reg)

    g_loc = g_img + g_reg
    H = H_img + H_reg + jnp.asarray(1e-8, dtype=dtype) * eye2
    delta = -jnp.linalg.solve(H, g_loc[..., None])[..., 0]  # (Nnodes, 2)

    # If a node has neither pixels nor neighbors, keep delta at 0.
    delta = jnp.where(has_update[:, None], delta, jnp.asarray(0.0, dtype=dtype))

    norm_delta = jnp.linalg.norm(delta, axis=1)
    factor = jnp.minimum(jnp.asarray(1.0, dtype=dtype), max_step / (norm_delta + jnp.asarray(1e-12, dtype=dtype)))
    delta = delta * factor[:, None]

    return disp0 + omega * delta


@jax.jit
def _bilinear_sample(image_T: Array, coords: Array) -> Array:
    x = coords[:, 0]
    y = coords[:, 1]
    x0 = jnp.floor(x - 0.5).astype(jnp.int32)
    y0 = jnp.floor(y - 0.5).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = (x - 0.5) - x0.astype(jnp.float32)
    fy = (y - 0.5) - y0.astype(jnp.float32)

    def safe_get(xx, yy):
        h, w = image_T.shape
        xx = jnp.clip(xx, 0, w - 1)
        yy = jnp.clip(yy, 0, h - 1)
        return image_T[xx, yy]

    I00 = safe_get(x0, y0)
    I10 = safe_get(x1, y0)
    I01 = safe_get(x0, y1)
    I11 = safe_get(x1, y1)

    val = (
        (1 - fx) * (1 - fy) * I00
        + fx * (1 - fy) * I10
        + (1 - fx) * fy * I01
        + fx * fy * I11
    )
    return val


def _normalize_interpolation(interpolation: str) -> str:
    value = str(interpolation).lower()
    if value == "bilinear":
        value = "linear"
    if value not in ("cubic", "linear"):
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    return value


def _cubic_sample(image_T: Array, coords: Array) -> Array:
    if flat_nd_cubic_interpolate is None:
        raise RuntimeError("im_jax.flat_nd_cubic_interpolate is unavailable.")
    x = coords[:, 0] - 0.5
    y = coords[:, 1] - 0.5
    sample_coords = jnp.stack([x, y], axis=0)
    return flat_nd_cubic_interpolate(image_T, sample_coords, mode="nearest", cval=0.0, layout="HW")
