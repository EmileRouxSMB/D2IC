from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from ..core.types import Array
from .elasticity import isotropic_stiffness
from .hardening import flow_stress
from .hill48 import hill48_equivalent_stress
from .state import Hill48Material, MaterialState


def _residual(z: Array, strain_increment: Array, state: MaterialState,
              material: Hill48Material, compliance: Array) -> Array:
    delta_stress = z[:6]
    delta_lambda = z[6]
    stress = state.stress + delta_stress
    direction = jax.grad(hill48_equivalent_stress, argnums=0)(stress, material.hill)
    plastic_residual = compliance @ delta_stress + delta_lambda * direction - strain_increment
    yield_residual = hill48_equivalent_stress(stress, material.hill) - flow_stress(
        state.equivalent_plastic_strain + delta_lambda,
        material.hardening,
        material.hardening_model,
    )
    return jnp.concatenate((plastic_residual, yield_residual[None]))


@partial(jax.jit, static_argnames=("max_iterations",))
def integrate_hill48_increment(
    strain_increment: Array,
    state: MaterialState,
    material: Hill48Material,
    *,
    max_iterations: int = 30,
    tolerance: float = 1e-10,
) -> tuple[MaterialState, Array, Array, Array]:
    """Fully implicit 3D Hill48 return mapping for one strain increment.

    Returns the new state, consistent 6x6 tangent, convergence flag and final
    residual norm. The strain vector uses engineering shear components.
    """
    C = isotropic_stiffness(material.young, material.poisson)
    compliance = jnp.linalg.inv(C)
    trial_stress = state.stress + C @ strain_increment
    trial_f = hill48_equivalent_stress(trial_stress, material.hill) - flow_stress(
        state.equivalent_plastic_strain, material.hardening, material.hardening_model
    )

    def elastic(_):
        new_state = MaterialState(
            stress=trial_stress,
            plastic_strain=state.plastic_strain,
            equivalent_plastic_strain=state.equivalent_plastic_strain,
            orientation=state.orientation,
            log_thickness_strain=state.log_thickness_strain,
        )
        return new_state, C, jnp.asarray(True), jnp.maximum(trial_f, 0.0)

    def plastic(_):
        z0 = jnp.concatenate((C @ strain_increment, jnp.zeros(1, dtype=C.dtype)))

        def iteration(_, carry):
            z, converged = carry
            residual = _residual(z, strain_increment, state, material, compliance)
            jacobian = jax.jacfwd(_residual)(z, strain_increment, state, material, compliance)
            correction = jnp.linalg.solve(jacobian, residual)
            z_candidate = z - correction
            now_converged = jnp.linalg.norm(residual) <= tolerance
            z = jnp.where(converged, z, z_candidate)
            return z, converged | now_converged

        z, _ = lax.fori_loop(0, max_iterations, iteration, (z0, jnp.asarray(False)))
        residual = _residual(z, strain_increment, state, material, compliance)
        residual_norm = jnp.linalg.norm(residual)
        jacobian = jax.jacfwd(_residual)(z, strain_increment, state, material, compliance)
        rhs = jnp.concatenate((jnp.eye(6, dtype=C.dtype), jnp.zeros((1, 6), dtype=C.dtype)), axis=0)
        tangent = jnp.linalg.solve(jacobian, rhs)[:6]
        stress = state.stress + z[:6]
        direction = jax.grad(hill48_equivalent_stress, argnums=0)(stress, material.hill)
        new_state = MaterialState(
            stress=stress,
            plastic_strain=state.plastic_strain + z[6] * direction,
            equivalent_plastic_strain=state.equivalent_plastic_strain + z[6],
            orientation=state.orientation,
            log_thickness_strain=state.log_thickness_strain,
        )
        return new_state, tangent, residual_norm <= tolerance, residual_norm

    return lax.cond(trial_f <= 0.0, elastic, plastic, operand=None)
