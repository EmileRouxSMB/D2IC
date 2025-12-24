from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from functools import partial

try:  # pragma: no cover - optional legacy dependency
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
class GlobalCGResult:
    u_nodal: Array
    strain: Array


class GlobalCGSolver(SolverBase):
    """
    Global CG-based solver for mesh DIC.
    Stage-1: placeholders only.
    Stage-2: migrate your current global CG (and JAX core calls) into this file.
    """

    def __init__(self, verbose: bool = False, use_map_coordinates: bool = True) -> None:
        self._compiled = False
        self._solve_jit = None
        self._verbose = bool(verbose)
        self._use_map_coordinates = bool(use_map_coordinates)

    def compile(self, assets: Any) -> None:
        pixel_data = getattr(assets, "pixel_data", None)
        if pixel_data is None:
            raise ValueError("MeshAssets must provide pixel_data for GlobalCGSolver.")

        def _cg_fn(
            disp0,
            im1_T,
            im2_T,
            pixel_coords_ref,
            pixel_nodes,
            pixel_shapeN,
            node_neighbor_index,
            node_neighbor_degree,
            node_neighbor_weight,
            node_reg_weight,
            alpha_reg,
            max_iter,
            tol,
            use_map_coordinates,
            verbose,
        ):
            return _cg_solve(
                disp0,
                im1_T,
                im2_T,
                pixel_coords_ref,
                pixel_nodes,
                pixel_shapeN,
                node_neighbor_index,
                node_neighbor_degree,
                node_neighbor_weight,
                node_reg_weight,
                alpha_reg,
                max_iter,
                tol,
                use_map_coordinates,
                verbose,
            )

        self._solve_jit = jax.jit(
            _cg_fn,
            static_argnums=(11, 12, 13, 14),
            donate_argnums=(0,),
        )
        self._compiled = True

    def warmup(self, state: Any) -> None:
        """Compile the JIT once using dummy inputs that match the real shapes."""
        if not self._compiled or self._solve_jit is None:
            raise RuntimeError("GlobalCGSolver.compile() must be called before warmup().")

        assets: MeshAssets = state.assets
        pix: PixelAssets = assets.pixel_data  # type: ignore[assignment]
        if pix is None:
            raise ValueError("MeshAssets must provide pixel_data for GlobalCGSolver warmup.")

        ref_im = jnp.asarray(state.ref_image)
        im1_T = jnp.transpose(ref_im, (1, 0))
        im2_T = im1_T
        disp0 = jnp.zeros_like(assets.mesh.nodes_xy)

        self._solve_jit.lower(
            disp0,
            im1_T,
            im2_T,
            pix.pixel_coords_ref,
            pix.pixel_nodes,
            pix.pixel_shapeN,
            pix.node_neighbor_index,
            pix.node_neighbor_degree,
            pix.node_neighbor_weight,
            pix.node_reg_weight,
            float(state.config.reg_strength),
            int(state.config.max_iters),
            float(state.config.tol),
            bool(self._use_map_coordinates),
            bool(self._verbose),
        ).compile()

    def solve(self, state: Any, def_image: Array) -> GlobalCGResult:
        if not self._compiled or self._solve_jit is None:
            raise RuntimeError("GlobalCGSolver.compile() must be called before solve().")

        assets: MeshAssets = state.assets
        pix: PixelAssets = assets.pixel_data  # type: ignore[assignment]
        ref_im = jnp.asarray(state.ref_image)
        def_im = jnp.asarray(def_image)
        im1_T = jnp.transpose(ref_im, (1, 0))
        im2_T = jnp.transpose(def_im, (1, 0))

        disp0 = jnp.asarray(state.u0_nodal) if state.u0_nodal is not None else jnp.zeros_like(assets.mesh.nodes_xy)
        disp0 = jnp.asarray(disp0)

        disp_sol, _ = self._solve_jit(
            disp0,
            im1_T,
            im2_T,
            pix.pixel_coords_ref,
            pix.pixel_nodes,
            pix.pixel_shapeN,
            pix.node_neighbor_index,
            pix.node_neighbor_degree,
            pix.node_neighbor_weight,
            pix.node_reg_weight,
            float(state.config.reg_strength),
            int(state.config.max_iters),
            float(state.config.tol),
            bool(self._use_map_coordinates),
            bool(self._verbose),
        )

        strain = jnp.zeros_like(disp_sol)
        return GlobalCGResult(u_nodal=disp_sol, strain=strain)


# ---------------------------------------------------------------------
# Core objective functions and CG loop
# ---------------------------------------------------------------------

@partial(jax.jit, static_argnames=("use_map_coordinates",))
def residuals_pixelwise_core(
    displacement,
    im1_T,
    im2_T,
    pixel_coords,
    pixel_nodes,
    pixel_shapeN,
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
    elif flat_nd_linear_interpolate is not None:
        I1 = flat_nd_linear_interpolate(im1_T, x_ref.T)
        I2 = flat_nd_linear_interpolate(im2_T, x_def.T)
    else:
        I1 = _bilinear_sample(im1_T, x_ref)
        I2 = _bilinear_sample(im2_T, x_def)
    r = jnp.where(valid, I2 - I1, jnp.asarray(0.0, dtype=I2.dtype))
    return r, valid


@partial(jax.jit, static_argnames=("use_map_coordinates",))
def J_pixelwise_core(
    displacement,
    im1_T,
    im2_T,
    pixel_coords,
    pixel_nodes,
    pixel_shapeN,
    use_map_coordinates=False,
):
    r, valid = residuals_pixelwise_core(
        displacement,
        im1_T,
        im2_T,
        pixel_coords,
        pixel_nodes,
        pixel_shapeN,
        use_map_coordinates=use_map_coordinates,
    )
    valid_f = valid.astype(r.dtype)
    denom = jnp.maximum(jnp.sum(valid_f), jnp.asarray(1.0, dtype=r.dtype))
    return 0.5 * jnp.sum(r ** 2) / denom


@jax.jit
def reg_energy_spring_global(
    displacement,
    node_neighbor_index,
    node_neighbor_degree,
    node_neighbor_weight,
    node_reg_weight,
):
    displacement = jnp.asarray(displacement)
    dtype = displacement.dtype
    Nnodes, max_deg = node_neighbor_index.shape
    deg = node_neighbor_degree[:, None]
    idx_range = jnp.arange(max_deg)[None, :]
    mask = idx_range < deg
    neigh_ids = jnp.where(mask, node_neighbor_index, 0)
    w = jnp.asarray(node_neighbor_weight, dtype=dtype)
    w = jnp.where(mask, w, jnp.asarray(0.0, dtype=dtype))
    node_reg_weight = jnp.asarray(node_reg_weight, dtype=dtype)
    w_i = node_reg_weight[:, None]
    w_j = node_reg_weight[neigh_ids]
    w = w * 0.5 * (w_i + w_j)
    u_i = displacement[:, None, :]
    u_j = displacement[neigh_ids]
    diff = jnp.where(mask[..., None], u_i - u_j, 0.0)
    sq = jnp.sum(diff ** 2, axis=2)
    return jnp.asarray(0.25, dtype=dtype) * jnp.sum(w * sq)


def J_total(
    displacement,
    im1_T,
    im2_T,
    pixel_coords,
    pixel_nodes,
    pixel_shapeN,
    node_neighbor_index,
    node_neighbor_degree,
    node_neighbor_weight,
    node_reg_weight,
    use_map_coordinates,
    alpha_reg,
):
    J_img = J_pixelwise_core(
        displacement,
        im1_T,
        im2_T,
        pixel_coords,
        pixel_nodes,
        pixel_shapeN,
        use_map_coordinates=use_map_coordinates,
    )
    alpha_reg = jnp.asarray(alpha_reg, dtype=J_img.dtype)
    reg = reg_energy_spring_global(
        displacement,
        node_neighbor_index,
        node_neighbor_degree,
        node_neighbor_weight,
        node_reg_weight,
    ).astype(J_img.dtype)
    return J_img + alpha_reg * reg


_J_TOTAL_VG = jax.value_and_grad(J_total)


def _cg_solve(
    disp0,
    im1_T,
    im2_T,
    pixel_coords_ref,
    pixel_nodes,
    pixel_shapeN,
    node_neighbor_index,
    node_neighbor_degree,
    node_neighbor_weight,
    node_reg_weight,
    alpha_reg,
    max_iter,
    tol,
    use_map_coordinates,
    verbose,
):
    max_iter = int(max_iter)
    disp0 = jnp.asarray(disp0)
    zero = jnp.zeros_like(disp0)

    objective_args = (
        im1_T,
        im2_T,
        pixel_coords_ref,
        pixel_nodes,
        pixel_shapeN,
        node_neighbor_index,
        node_neighbor_degree,
        node_neighbor_weight,
        node_reg_weight,
        use_map_coordinates,
    )

    def cond_fun(state):
        k, disp, direction, grad_prev = state
        del disp, direction, grad_prev
        return k < max_iter

    def body_fun(state):
        k, disp, direction, grad_prev = state
        J_val, grad = _J_TOTAL_VG(
            disp,
            *objective_args,
            alpha_reg,
        )
        grad_norm = jnp.linalg.norm(grad)
        stop = grad_norm < tol

        def stop_branch(_):
            if verbose:
                jax.debug.print(
                    "CG stop {k}: J={J:.3e}, |grad|={g:.3e}, alpha={a:.3e}",
                    k=k,
                    J=J_val,
                    g=grad_norm,
                    a=0.0,
                )
            return max_iter, disp, direction, grad

        def continue_branch(_):
            def dir_first(_):
                return -grad

            def dir_conjugate(_):
                num = jnp.sum(grad * (grad - grad_prev))
                den = jnp.sum(grad_prev * grad_prev) + 1e-16
                beta = jnp.maximum(num / den, 0.0)
                return -grad + beta * direction

            direction_new = jax.lax.cond(
                k == 0,
                dir_first,
                dir_conjugate,
                operand=None,
            )
            gd = jnp.sum(grad * direction_new)
            direction_new = jax.lax.cond(
                gd >= 0,
                lambda _: -grad,
                lambda _: direction_new,
                operand=None,
            )

            c_armijo = 1e-4

            def ls_body(_, carry):
                alpha, J_candidate, done = carry

                def compute_step(_):
                    candidate_disp = disp + alpha * direction_new
                    J_candidate = J_total(
                        candidate_disp,
                        *objective_args,
                        alpha_reg,
                    )
                    satisfies = J_candidate <= J_val + c_armijo * alpha * gd
                    alpha_next = jnp.where(satisfies, alpha, alpha * 0.5)
                    return alpha_next, J_candidate, satisfies

                alpha_next, J_candidate_next, satisfies = jax.lax.cond(
                    done,
                    lambda _: (alpha, J_candidate, jnp.bool_(True)),
                    compute_step,
                    operand=None,
                )
                done_next = jnp.logical_or(done, satisfies)
                return alpha_next, J_candidate_next, done_next

            alpha_init = 1.0
            alpha, J_trial, _ = jax.lax.fori_loop(
                0,
                10,
                ls_body,
                (alpha_init, J_val, jnp.bool_(False)),
            )
            if verbose:
                jax.debug.print(
                    "CG iter {k}: J={J:.3e}, |grad|={g:.3e}, alpha={a:.3e}",
                    k=k,
                    J=J_trial,
                    g=grad_norm,
                    a=alpha,
                )
            disp_next = disp + alpha * direction_new
            return k + 1, disp_next, direction_new, grad

        return jax.lax.cond(stop, stop_branch, continue_branch, operand=None)

    k0 = jnp.array(0)
    direction0 = zero
    grad_prev0 = zero
    final_state = jax.lax.while_loop(cond_fun, body_fun, (k0, disp0, direction0, grad_prev0))
    return final_state[1], final_state[0]


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
