from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from ..core.types import Array
from .return_mapping import integrate_hill48_increment
from .state import Hill48Material, MaterialState


_IN_PLANE = jnp.asarray([0, 1, 3])


@partial(jax.jit, static_argnames=("max_iterations", "material_iterations"))
def integrate_plane_stress_increment(
    in_plane_strain_increment: Array,
    state: MaterialState,
    material: Hill48Material,
    *,
    max_iterations: int = 12,
    material_iterations: int = 30,
    tolerance: float = 1e-9,
) -> tuple[MaterialState, Array, Array, Array]:
    """Integrate one plane-stress increment while enforcing ``sigma33=0``."""
    initial_e33 = -(in_plane_strain_increment[0] + in_plane_strain_increment[1])

    def evaluate(e33):
        de = jnp.asarray(
            [in_plane_strain_increment[0], in_plane_strain_increment[1], e33,
             in_plane_strain_increment[2], 0.0, 0.0],
            dtype=in_plane_strain_increment.dtype,
        )
        return integrate_hill48_increment(
            de, state, material, max_iterations=material_iterations, tolerance=tolerance
        )

    def iteration(_, carry):
        e33, converged = carry
        trial_state, tangent, _, _ = evaluate(e33)
        correction = trial_state.stress[2] / tangent[2, 2]
        candidate = e33 - correction
        now_converged = jnp.abs(trial_state.stress[2]) <= tolerance * flow_scale(material)
        return jnp.where(converged, e33, candidate), converged | now_converged

    e33, _ = lax.fori_loop(
        0, max_iterations, iteration, (initial_e33, jnp.asarray(False))
    )
    new_state, tangent_3d, material_ok, material_residual = evaluate(e33)
    a = _IN_PLANE
    Caa = tangent_3d[jnp.ix_(a, a)]
    condensed = Caa - jnp.outer(tangent_3d[a, 2], tangent_3d[2, a]) / tangent_3d[2, 2]
    plane_residual = jnp.abs(new_state.stress[2])
    new_state = MaterialState(
        stress=new_state.stress.at[2].set(0.0),
        plastic_strain=new_state.plastic_strain,
        equivalent_plastic_strain=new_state.equivalent_plastic_strain,
        orientation=new_state.orientation,
        log_thickness_strain=state.log_thickness_strain + e33,
    )
    converged = material_ok & (plane_residual <= tolerance * flow_scale(material))
    return new_state, condensed, converged, jnp.maximum(material_residual, plane_residual)


def flow_scale(material: Hill48Material) -> Array:
    """Stress scale used only to normalize plane-stress convergence."""
    return jnp.maximum(jnp.abs(material.hardening[0]), 1.0)
