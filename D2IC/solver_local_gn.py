from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from functools import partial

try:  # pragma: no cover - optional dependency for fast interpolation
    from dm_pix import flat_nd_linear_interpolate
except Exception:  # pragma: no cover
    flat_nd_linear_interpolate = None
try:  # pragma: no cover - optional scipy-based interpolation
    from jax.scipy.ndimage import map_coordinates as _map_coordinates
except Exception:  # pragma: no cover
    _map_coordinates = None

from .solver_base import SolverBase
from .types import Array
from .mesh_assets import MeshAssets, PixelAssets


@dataclass(frozen=True)
class LocalGNResult:
    u_nodal: Array
    strain: Array


class LocalGaussNewtonSolver(SolverBase):
    """
    Local Gauss-Newton solver for mesh DIC (e.g., nodal local updates).
    Stage-1: placeholders only.
    Stage-2: migrate your current local "gauss-newton / spring" logic here.
    """

    def __init__(
        self,
        lam: float = 1e-3,
        max_step: float = 0.2,
        omega: float = 0.5,
        use_map_coordinates: bool = True,
    ) -> None:
        self._compiled = False
        self._solve_jit = None
        self._lam = float(lam)
        self._max_step = float(max_step)
        self._omega = float(omega)
        self._use_map_coordinates = bool(use_map_coordinates)

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
            use_map_coordinates,
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
                use_map_coordinates,
            )

        self._solve_jit = jax.jit(
            _local_fn,
            static_argnums=(19, 20),
            donate_argnums=(0,),
        )
        self._compiled = True

    def solve(self, state: Any, def_image: Array) -> LocalGNResult:
        if not self._compiled or self._solve_jit is None:
            raise RuntimeError("LocalGaussNewtonSolver.compile() must be called before solve().")

        assets: MeshAssets = state.assets
        pix: PixelAssets = assets.pixel_data  # type: ignore[assignment]
        if pix is None:
            raise ValueError("MeshAssets must provide pixel_data for LocalGaussNewtonSolver.")

        ref_im = jnp.asarray(state.ref_image)
        def_im = jnp.asarray(def_image)
        im1_T = jnp.transpose(ref_im, (1, 0))
        im2_T = jnp.transpose(def_im, (1, 0))

        disp0 = jnp.asarray(state.u0_nodal) if state.u0_nodal is not None else jnp.zeros_like(assets.mesh.nodes_xy)
        disp0 = jnp.asarray(disp0)

        gx2_np, gy2_np = _compute_image_gradient_np(np.asarray(def_image))
        gx2_T = jnp.asarray(gx2_np.T, dtype=def_im.dtype)
        gy2_T = jnp.asarray(gy2_np.T, dtype=def_im.dtype)

        disp_sol = self._solve_jit(
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
            bool(self._use_map_coordinates),
        )

        strain = jnp.zeros((disp_sol.shape[0], 3), dtype=disp_sol.dtype)
        return LocalGNResult(u_nodal=disp_sol, strain=strain)


# ---------------------------------------------------------------------
# Core kernels (ported from legacy)
# ---------------------------------------------------------------------

def _compute_image_gradient_np(im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    im = np.asarray(im, dtype=float)
    gx = np.zeros_like(im)
    gy = np.zeros_like(im)
    gx[:, 1:-1] = 0.5 * (im[:, 2:] - im[:, :-2])
    gy[1:-1, :] = 0.5 * (im[2:, :] - im[:-2, :])
    return gx, gy


@partial(jax.jit, static_argnames=("use_map_coordinates",))
def _pixel_state(
    displacement,
    im1_T,
    im2_T,
    pixel_coords,
    pixel_nodes,
    pixel_shapeN,
    gradx2_T,
    grady2_T,
    use_map_coordinates=False,
):
    shapeN = pixel_shapeN[..., None]
    disp_local = displacement[pixel_nodes]
    u_pix = (shapeN * disp_local).sum(axis=1)
    x_ref = pixel_coords
    x_def = x_ref + u_pix

    w, h = im2_T.shape
    valid_ref = (x_ref[:, 0] >= 0.5) & (x_ref[:, 0] <= w - 0.5) & (x_ref[:, 1] >= 0.5) & (x_ref[:, 1] <= h - 0.5)
    valid_def = (x_def[:, 0] >= 0.5) & (x_def[:, 0] <= w - 0.5) & (x_def[:, 1] >= 0.5) & (x_def[:, 1] <= h - 0.5)
    valid = valid_ref & valid_def

    if use_map_coordinates:
        if _map_coordinates is None:
            raise RuntimeError("jax.scipy.ndimage.map_coordinates is unavailable.")
        I1 = _map_coordinates_sample(im1_T, x_ref)
        I2 = _map_coordinates_sample(im2_T, x_def)
        gx_def = _map_coordinates_sample(gradx2_T, x_def)
        gy_def = _map_coordinates_sample(grady2_T, x_def)
    elif flat_nd_linear_interpolate is not None:
        I1 = flat_nd_linear_interpolate(im1_T, x_ref.T)
        I2 = flat_nd_linear_interpolate(im2_T, x_def.T)
        gx_def = flat_nd_linear_interpolate(gradx2_T, x_def.T)
        gy_def = flat_nd_linear_interpolate(grady2_T, x_def.T)
    else:
        I1 = _bilinear_sample(im1_T, x_ref)
        I2 = _bilinear_sample(im2_T, x_def)
        gx_def = _bilinear_sample(gradx2_T, x_def)
        gy_def = _bilinear_sample(grady2_T, x_def)

    r = jnp.where(valid, I2 - I1, jnp.asarray(0.0, dtype=I2.dtype))
    gx_def = jnp.where(valid, gx_def, jnp.asarray(0.0, dtype=gx_def.dtype))
    gy_def = jnp.where(valid, gy_def, jnp.asarray(0.0, dtype=gy_def.dtype))
    return r, x_def, gx_def, gy_def


@partial(jax.jit, static_argnames=("use_map_coordinates",))
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
    use_map_coordinates,
):
    def body_fun(_, disp):
        r, _x_def, gx_def, gy_def = _pixel_state(
            disp,
            im1_T,
            im2_T,
            pixel_coords,
            pixel_nodes,
            pixel_shapeN,
            gradx2_T,
            grady2_T,
            use_map_coordinates=use_map_coordinates,
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
        return disp_next

    return lax.fori_loop(0, n_sweeps, body_fun, displacement)


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
    disp0 = displacement
    dtype = disp0.dtype
    Nnodes, max_deg_pix = node_pixel_index.shape
    _, max_deg_neigh = node_neighbor_index.shape
    node_reg_weight = jnp.asarray(node_reg_weight, dtype=dtype)

    def body_fun(i, delta_acc):
        deg_pix = node_degree[i]
        deg_neigh = node_neighbor_degree[i]

        def no_pixel_case(delta_acc_inner):
            def only_return_delta(_):
                return delta_acc_inner

            def apply_reg_only(_):
                u_i = disp0[i]

                idx_range_n = jnp.arange(max_deg_neigh)
                mask_n = idx_range_n < deg_neigh

                neigh_ids_all = node_neighbor_index[i]
                w_all = node_neighbor_weight[i]

                neigh_ids = jnp.where(mask_n, neigh_ids_all, 0)
                w = jnp.where(mask_n, w_all, jnp.asarray(0.0, dtype=dtype))

                u_neigh = disp0[neigh_ids]
                u_neigh = jnp.where(mask_n[:, None], u_neigh, jnp.asarray(0.0, dtype=dtype))

                diff = u_i[None, :] - u_neigh
                alpha_loc = alpha_reg * node_reg_weight[i]
                g_reg = alpha_loc * jnp.sum(w[:, None] * diff, axis=0)
                H_reg = alpha_loc * jnp.sum(w) * jnp.eye(2, dtype=dtype)

                g_loc = g_reg
                H = H_reg + jnp.asarray(1e-8, dtype=dtype) * jnp.eye(2, dtype=dtype)
                delta_i = -jnp.linalg.solve(H, g_loc)

                norm_delta = jnp.linalg.norm(delta_i)
                factor = jnp.minimum(1.0, max_step / (norm_delta + 1e-12))
                delta_i = delta_i * factor

                return delta_acc_inner.at[i].set(delta_i)

            cond_reg = jnp.logical_and(alpha_reg != 0.0, deg_neigh > 0)
            return lax.cond(cond_reg, apply_reg_only, only_return_delta, operand=None)

        def update_case(delta_acc_inner):
            idx_all = node_pixel_index[i]
            Ni_all = node_N_weight[i]
            idx_range = jnp.arange(max_deg_pix)
            mask_pix = idx_range < deg_pix

            idx = jnp.where(mask_pix, idx_all, 0)
            Ni = jnp.where(mask_pix, Ni_all, jnp.asarray(0.0, dtype=dtype))

            ri = r[idx]
            gxi = gx_def[idx]
            gyi = gy_def[idx]

            Jx = gxi * Ni
            Jy = gyi * Ni

            g0_img = jnp.sum(ri * Jx)
            g1_img = jnp.sum(ri * Jy)
            g_img = jnp.array([g0_img, g1_img], dtype=dtype)

            H00 = jnp.sum(Jx * Jx)
            H01 = jnp.sum(Jx * Jy)
            H11 = jnp.sum(Jy * Jy)

            traceH = H00 + H11 + 1e-12
            lam_loc = lam * traceH
            H_img = jnp.array(
                [[H00 + lam_loc, H01], [H01, H11 + lam_loc]],
                dtype=dtype,
            )

            u_i = disp0[i]
            idx_range_n = jnp.arange(max_deg_neigh)
            mask_n = idx_range_n < deg_neigh

            neigh_ids_all = node_neighbor_index[i]
            w_all = node_neighbor_weight[i]

            neigh_ids = jnp.where(mask_n, neigh_ids_all, 0)
            w = jnp.where(mask_n, w_all, jnp.asarray(0.0, dtype=dtype))

            u_neigh = disp0[neigh_ids]
            u_neigh = jnp.where(mask_n[:, None], u_neigh, jnp.asarray(0.0, dtype=dtype))

            diff = u_i[None, :] - u_neigh
            alpha_loc = alpha_reg * node_reg_weight[i]
            g_reg = alpha_loc * jnp.sum(w[:, None] * diff, axis=0)
            H_reg = alpha_loc * jnp.sum(w) * jnp.eye(2, dtype=dtype)

            g_loc = g_img + g_reg
            H = H_img + H_reg + jnp.asarray(1e-8, dtype=dtype) * jnp.eye(2, dtype=dtype)

            delta_i = -jnp.linalg.solve(H, g_loc)

            norm_delta = jnp.linalg.norm(delta_i)
            factor = jnp.minimum(1.0, max_step / (norm_delta + 1e-12))
            delta_i = delta_i * factor

            return delta_acc_inner.at[i].set(delta_i)

        return lax.cond(deg_pix == 0, no_pixel_case, update_case, delta_acc)

    delta0 = jnp.zeros_like(displacement)
    delta_all = lax.fori_loop(0, Nnodes, body_fun, delta0)
    displacement_new = displacement + omega * delta_all
    return displacement_new


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


def _map_coordinates_sample(image_T: Array, coords: Array) -> Array:
    if _map_coordinates is None:
        raise RuntimeError("jax.scipy.ndimage.map_coordinates is unavailable.")
    x = coords[:, 0] - 0.5
    y = coords[:, 1] - 0.5
    sample_coords = jnp.stack([x, y], axis=0)
    # JAX currently supports order <= 1; keep a central switch here for future upgrades.
    return _map_coordinates(image_T, sample_coords, order=1, mode="nearest")
