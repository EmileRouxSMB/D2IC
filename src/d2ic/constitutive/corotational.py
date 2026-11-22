from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ..core.types import Array
from .plane_stress import integrate_plane_stress_increment
from .state import Hill48Material, MaterialState


def incremental_polar_2d(F_old: Array, F_new: Array) -> tuple[Array, Array]:
    """Return rotation and logarithmic stretch of ``F_new @ inv(F_old)``."""
    delta_F = jnp.linalg.solve(F_old.T, F_new.T).T
    C = delta_F.T @ delta_F
    eigenvalues, eigenvectors = jnp.linalg.eigh(C)
    stretches = jnp.sqrt(jnp.maximum(eigenvalues, jnp.finfo(C.dtype).eps))
    U = (eigenvectors * stretches) @ eigenvectors.T
    log_U = (eigenvectors * jnp.log(stretches)) @ eigenvectors.T
    rotation = jnp.linalg.solve(U.T, delta_F.T).T
    return rotation, log_U


def integrate_corotational_plane_stress(
    F_old: Array,
    F_new: Array,
    state: MaterialState,
    material: Hill48Material,
) -> tuple[MaterialState, dict[str, Array]]:
    """Advance a plane-stress metal using a corotational logarithmic increment."""
    rotation, log_stretch = incremental_polar_2d(F_old, F_new)
    Q_old = state.orientation
    strain_material = Q_old.T @ log_stretch @ Q_old
    de = jnp.asarray(
        [strain_material[0, 0], strain_material[1, 1], 2.0 * strain_material[0, 1]]
    )
    state_local, tangent, converged, residual = integrate_plane_stress_increment(
        de, state, material
    )
    Q_new = rotation @ Q_old
    state_new = MaterialState(
        stress=state_local.stress,
        plastic_strain=state_local.plastic_strain,
        equivalent_plastic_strain=state_local.equivalent_plastic_strain,
        orientation=Q_new,
        log_thickness_strain=state_local.log_thickness_strain,
    )
    stress_material_2d = jnp.asarray(
        [[state_new.stress[0], state_new.stress[3]],
         [state_new.stress[3], state_new.stress[1]]]
    )
    stress_global = Q_new @ stress_material_2d @ Q_new.T
    return state_new, {
        "cauchy_stress": stress_global,
        "plastic_strain": state_new.plastic_strain,
        "equivalent_plastic_strain": state_new.equivalent_plastic_strain,
        "thickness_ratio": jnp.exp(state_new.log_thickness_strain),
        "tangent_material": tangent,
        "converged": converged,
        "residual": residual,
    }


@jax.jit
def integrate_history(
    F_history: Array,
    initial_state: MaterialState,
    material: Hill48Material,
) -> tuple[MaterialState, dict[str, Array]]:
    """Integrate a deformation-gradient history with ``lax.scan``."""
    def step(carry, F_new):
        F_old, state = carry
        state_new, output = integrate_corotational_plane_stress(
            F_old, F_new, state, material
        )
        return (F_new, state_new), output

    (_, final_state), outputs = lax.scan(step, (F_history[0], initial_state), F_history[1:])
    return final_state, outputs
