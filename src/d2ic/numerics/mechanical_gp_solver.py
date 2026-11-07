"""Global DIC solver with uniform or GP-adaptive equilibrium regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .adaptive_mechanical_gp import (
    assemble_linear_elastic_residual_q4,
    gp_inducing_prior,
    gp_log_variance,
    laplace_log_variance_uncertainty,
    mechanical_confidence,
    mechanical_residual_nll,
)
from .base import MeshSolverState, SolverBase
from .elastic_regularization import q4_reference_gradients
from .global_cg import J_pixelwise_core
from ..core.types import Array
from ..mesh.assets import MeshAssets, PixelAssets


@dataclass(frozen=True)
class MechanicalGPResult:
    """DIC solution and nodal diagnostics of mechanical-model confidence."""

    u_nodal: Array
    strain: Array
    log_mechanical_variance: Array
    mechanical_confidence: Array
    log_variance_std: Array
    mechanical_residual: Array
    inducing_values: Array
    adaptive: bool
    history: Array | None = None
    n_iters: Array | None = None


def plane_stress_elasticity_tensor(poisson: float = 0.3) -> Array:
    """Return the dimensionless isotropic plane-stress tensor ``C[i,j,k,l]``.

    Young's modulus is one because the mechanical residual is divided by the
    explicit ``force_scale``. Changing Young's modulus would otherwise be
    indistinguishable from changing that statistical scale.
    """
    nu = float(poisson)
    if not (-1.0 < nu < 0.5):
        raise ValueError("poisson must lie in (-1, 0.5)")
    lame = nu / (1.0 - nu * nu)
    shear = 1.0 / (2.0 * (1.0 + nu))
    eye = jnp.eye(2)
    return lame * jnp.einsum("ij,kl->ijkl", eye, eye) + shear * (
        jnp.einsum("ik,jl->ijkl", eye, eye)
        + jnp.einsum("il,jk->ijkl", eye, eye)
    )


class MechanicalGPSolver(SolverBase):
    """Jointly identify displacement and spatial mechanical-model confidence.

    The image term is the usual grey-level MSE. The mechanical observation is
    the assembled weak-equilibrium residual of a small-strain, plane-stress Q4
    model. Boundary nodes are excluded because their reactions/tractions are
    unknown. In ``adaptive=True`` mode, ``log(s**2)`` is represented by an RBF
    GP with a small set of inducing variables and inferred jointly with the
    displacement. In uniform mode the same likelihood uses ``log(s**2)=0``.
    """

    def __init__(
        self,
        *,
        adaptive: bool = True,
        n_inducing: int = 32,
        gp_length_scale: float | None = None,
        gp_amplitude: float = 1.0,
        log_variance_mean: float = 0.0,
        gp_prior_weight: float = 1.0,
        force_scale: float = 1.0,
        poisson: float = 0.3,
        interpolation: str = "linear",
    ) -> None:
        if n_inducing < 1:
            raise ValueError("n_inducing must be positive")
        if force_scale <= 0 or gp_amplitude <= 0 or gp_prior_weight <= 0:
            raise ValueError("force_scale, gp_amplitude and gp_prior_weight must be positive")
        if interpolation not in ("linear", "cubic"):
            raise ValueError("interpolation must be 'linear' or 'cubic'")
        self.adaptive = bool(adaptive)
        self.n_inducing = int(n_inducing)
        self.gp_length_scale = gp_length_scale
        self.gp_amplitude = float(gp_amplitude)
        self.log_variance_mean = float(log_variance_mean)
        self.gp_prior_weight = float(gp_prior_weight)
        self.force_scale = float(force_scale)
        self.poisson = float(poisson)
        self.interpolation = interpolation
        self._compiled = False

    def compile(self, assets: MeshAssets) -> None:
        """Prepare static Q4, boundary and inducing-point arrays and JIT the solve."""
        if assets.pixel_data is None:
            raise ValueError("MeshAssets must provide pixel_data for MechanicalGPSolver")
        nodes = np.asarray(assets.mesh.nodes_xy)
        elements = np.asarray(assets.mesh.elements, dtype=np.int32)
        if elements.ndim != 2 or elements.shape[1] != 4:
            raise ValueError("MechanicalGPSolver currently supports Q4 meshes only")
        active = _interior_dofs(elements, nodes.shape[0])
        active_nodes = np.flatnonzero(active[:, 0])
        if active_nodes.size == 0:
            raise ValueError("The mesh has no interior node for equilibrium regularization")
        inducing_ids = _farthest_point_subset(
            nodes, active_nodes, min(self.n_inducing, active_nodes.size)
        )
        extent = np.ptp(nodes, axis=0)
        default_length = 0.2 * float(np.linalg.norm(extent))
        length_scale = default_length if self.gp_length_scale is None else float(self.gp_length_scale)
        if length_scale <= 0:
            raise ValueError("gp_length_scale must be positive")

        grad, normalized_weight = q4_reference_gradients(nodes, elements)
        # q4_reference_gradients normalizes by the complete mesh area for the
        # legacy energy. Equilibrium requires the physical Gauss-point areas.
        mesh_area = _q4_mesh_area(nodes, elements)
        area_weight = normalized_weight * mesh_area
        self._nodes = jnp.asarray(nodes)
        self._elements = jnp.asarray(elements)
        self._active_dofs = jnp.asarray(active)
        self._inducing_points = jnp.asarray(nodes[inducing_ids])
        self._grad = grad
        self._area_weight = area_weight
        self._elasticity = plane_stress_elasticity_tensor(self.poisson)
        self._length_scale = length_scale

        def solve_kernel(disp0, im1_t, im2_t, pixel_coords, pixel_nodes, pixel_shape, alpha, max_iter, tol):
            return _solve_joint_cg(
                disp0, im1_t, im2_t, pixel_coords, pixel_nodes, pixel_shape,
                alpha, max_iter, tol, self._nodes, self._elements, self._grad,
                self._area_weight, self._elasticity, self._active_dofs,
                self._inducing_points, self._length_scale, self.gp_amplitude,
                self.log_variance_mean, self.gp_prior_weight, self.force_scale,
                self.interpolation, self.adaptive,
            )

        self._solve_jit = jax.jit(solve_kernel, static_argnums=(7, 8))
        self._compiled = True

    def warmup(self, state: Any) -> None:
        """Compile the shape-specialized joint optimization kernel."""
        if state.config.reg_strength <= 0:
            raise ValueError("MechanicalGPSolver requires config.reg_strength > 0")
        pix: PixelAssets = state.assets.pixel_data
        disp0 = jnp.zeros_like(state.nodes_xy_device)
        image_t = jnp.transpose(state.ref_image, (1, 0))
        self._solve_jit.lower(
            disp0, image_t, image_t, pix.pixel_coords_ref, pix.pixel_nodes,
            pix.pixel_shapeN, float(state.config.reg_strength),
            int(state.config.max_iters), float(state.config.tol),
        ).compile()

    def solve(self, state: MeshSolverState, def_image: Array) -> MechanicalGPResult:
        """Solve one deformed frame, keeping all diagnostics on the JAX device."""
        if not self._compiled:
            raise RuntimeError("MechanicalGPSolver.compile() must be called before solve()")
        if state.config.reg_strength <= 0:
            raise ValueError("MechanicalGPSolver requires config.reg_strength > 0")
        pix: PixelAssets = state.assets.pixel_data  # type: ignore[assignment]
        disp0 = (jnp.asarray(state.u0_nodal) if state.u0_nodal is not None
                 else jnp.zeros_like(state.nodes_xy_device))
        values = self._solve_jit(
            disp0, jnp.transpose(state.ref_image, (1, 0)),
            jnp.transpose(jnp.asarray(def_image), (1, 0)), pix.pixel_coords_ref,
            pix.pixel_nodes, pix.pixel_shapeN, float(state.config.reg_strength),
            int(state.config.max_iters), float(state.config.tol),
        )
        displacement, inducing, history, n_iters = values
        residual = assemble_linear_elastic_residual_q4(
            displacement, self._elements, self._grad, self._area_weight,
            self._elasticity, self._active_dofs,
        )
        if self.adaptive:
            log_variance = gp_log_variance(
                self._nodes, self._inducing_points, inducing,
                self.log_variance_mean, self._length_scale, self.gp_amplitude,
            )
            variance = laplace_log_variance_uncertainty(
                residual, self._active_dofs, self.force_scale, self._nodes,
                self._inducing_points, inducing, self.log_variance_mean,
                self._length_scale, self.gp_amplitude,
                self.gp_prior_weight,
                float(state.config.reg_strength),
            )
            log_variance_std = jnp.sqrt(jnp.maximum(variance, 0.0))
        else:
            log_variance = jnp.full((self._nodes.shape[0],), self.log_variance_mean)
            log_variance_std = jnp.zeros_like(log_variance)
        return MechanicalGPResult(
            u_nodal=displacement,
            strain=jnp.zeros((displacement.shape[0], 3), dtype=displacement.dtype),
            log_mechanical_variance=log_variance,
            mechanical_confidence=mechanical_confidence(log_variance),
            log_variance_std=log_variance_std,
            mechanical_residual=residual,
            inducing_values=inducing,
            adaptive=self.adaptive,
            history=history if state.config.save_history else None,
            n_iters=n_iters,
        )


def _joint_objective(
    packed, n_nodes, im1_t, im2_t, pixel_coords, pixel_nodes, pixel_shape,
    alpha, nodes, elements, grad, area_weight, elasticity, active_dofs,
    inducing_points, length_scale, amplitude, mean, prior_weight, force_scale,
    interpolation, adaptive,
):
    displacement = packed[: 2 * n_nodes].reshape((n_nodes, 2))
    inducing = packed[2 * n_nodes:]
    image_loss = J_pixelwise_core(
        displacement, im1_t, im2_t, pixel_coords, pixel_nodes, pixel_shape,
        interpolation=interpolation,
    )
    residual = assemble_linear_elastic_residual_q4(
        displacement, elements, grad, area_weight, elasticity, active_dofs,
    )
    if adaptive:
        log_variance = gp_log_variance(
            nodes, inducing_points, inducing, mean, length_scale, amplitude,
        )
        mechanical = mechanical_residual_nll(
            residual, log_variance, active_dofs, force_scale,
        )
        prior = gp_inducing_prior(
            inducing, inducing_points, mean, length_scale, amplitude,
        )
        mechanical = mechanical + prior_weight * prior
    else:
        log_variance = jnp.full((n_nodes,), mean, dtype=packed.dtype)
        mechanical = mechanical_residual_nll(
            residual, log_variance, active_dofs, force_scale,
        )
    return image_loss + jnp.asarray(alpha, image_loss.dtype) * mechanical


def _solve_joint_cg(
    disp0, im1_t, im2_t, pixel_coords, pixel_nodes, pixel_shape, alpha,
    max_iter, tol, nodes, elements, grad, area_weight, elasticity, active_dofs,
    inducing_points, length_scale, amplitude, mean, prior_weight, force_scale,
    interpolation, adaptive,
):
    """Nonlinear Polak--Ribiere CG over displacement and inducing variables."""
    n_nodes = disp0.shape[0]
    inducing0 = (jnp.full((inducing_points.shape[0],), mean, dtype=disp0.dtype)
                 if adaptive else jnp.empty((0,), dtype=disp0.dtype))
    x0 = jnp.concatenate((disp0.reshape(-1), inducing0))
    history0 = jnp.full((max_iter, 3), jnp.nan, dtype=disp0.dtype)
    args = (
        n_nodes, im1_t, im2_t, pixel_coords, pixel_nodes, pixel_shape, alpha,
        nodes, elements, grad, area_weight, elasticity, active_dofs,
        inducing_points, length_scale, amplitude, mean, prior_weight, force_scale,
        interpolation, adaptive,
    )
    value_grad = jax.value_and_grad(_joint_objective)

    def condition(state):
        k, _x, _direction, _previous, done, _history = state
        return (k < max_iter) & ~done

    def iteration(state):
        k, x, direction, previous, _done, history = state
        value, gradient = value_grad(x, *args)
        norm = jnp.linalg.norm(gradient)
        beta = jnp.where(
            k == 0, 0.0,
            jnp.maximum(jnp.vdot(gradient, gradient - previous)
                        / (jnp.vdot(previous, previous) + 1e-16), 0.0),
        )
        proposal = -gradient + beta * direction
        descent = jnp.vdot(gradient, proposal)
        proposal = jnp.where(descent < 0, proposal, -gradient)
        descent = jnp.minimum(jnp.vdot(gradient, proposal), -1e-16)
        initial_step = 1.0 / jnp.maximum(jnp.max(jnp.abs(proposal)), 1.0)

        def line_step(_, carry):
            step, trial, accepted = carry
            candidate = x + step * proposal
            candidate_value = _joint_objective(candidate, *args)
            ok = candidate_value <= value + 1e-4 * step * descent
            return (jnp.where(accepted | ok, step, 0.5 * step),
                    jnp.where(accepted, trial, candidate_value), accepted | ok)

        step, trial, accepted = jax.lax.fori_loop(
            0, 24, line_step, (initial_step, value, jnp.bool_(False))
        )
        step = jnp.where(accepted, step, 0.0)
        trial = jnp.where(accepted, trial, value)
        next_x = x + step * proposal
        history = history.at[k].set(jnp.asarray([trial, norm, step]))
        converged = (norm < tol) | ~accepted
        return k + 1, next_x, proposal, gradient, converged, history

    state = jax.lax.while_loop(
        condition, iteration,
        (jnp.asarray(0), x0, jnp.zeros_like(x0), jnp.zeros_like(x0),
         jnp.bool_(False), history0),
    )
    k, x, _direction, _previous, _done, history = state
    return x[: 2 * n_nodes].reshape((n_nodes, 2)), x[2 * n_nodes:], history, k


def _interior_dofs(elements: np.ndarray, n_nodes: int) -> np.ndarray:
    """Return two active equilibrium DOFs for nodes not on a Q4 boundary."""
    counts: dict[tuple[int, int], int] = {}
    for element in elements:
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            edge = tuple(sorted((int(element[a]), int(element[b]))))
            counts[edge] = counts.get(edge, 0) + 1
    boundary = {node for edge, count in counts.items() if count == 1 for node in edge}
    active = np.ones((n_nodes, 2), dtype=bool)
    active[np.fromiter(boundary, dtype=np.int32), :] = False
    return active


def _farthest_point_subset(nodes: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    """Deterministic geometry-only inducing subset, computed once on the host."""
    selected = [int(candidates[0])]
    distance_sq = np.sum((nodes[candidates] - nodes[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        next_id = int(candidates[int(np.argmax(distance_sq))])
        selected.append(next_id)
        new_distance = np.sum((nodes[candidates] - nodes[next_id]) ** 2, axis=1)
        distance_sq = np.minimum(distance_sq, new_distance)
    return np.asarray(selected, dtype=np.int32)


def _q4_mesh_area(nodes: np.ndarray, elements: np.ndarray) -> float:
    """Total Q4 reference area used to undo legacy quadrature normalization."""
    coordinates = nodes[elements]
    x = coordinates[..., 0]
    y = coordinates[..., 1]
    signed_twice_area = np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
    area = 0.5 * np.sum(signed_twice_area)
    if area <= 0:
        raise ValueError("Q4 mesh must have positive total oriented area")
    return float(area)
