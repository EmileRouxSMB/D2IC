from __future__ import annotations

import jax
import jax.numpy as jnp

from ..constitutive.corotational import integrate_history
from ..constitutive.state import Hill48Material, MaterialState, initial_material_state
from ..core.types import Array


@jax.jit
def integrate_gauss_point_histories(
    F_history: Array,
    material: Hill48Material,
    material_angle_rad: Array,
) -> tuple[MaterialState, dict[str, Array]]:
    """Integrate all element/Gauss-point histories in parallel."""
    ne, ng = F_history.shape[1:3]
    angles = jnp.broadcast_to(material_angle_rad, (ne, ng)).reshape(-1)
    states = jax.vmap(initial_material_state, in_axes=(None, 0))(F_history.dtype, angles)
    flat_F = F_history.reshape((F_history.shape[0], ne * ng, 2, 2))
    final_state, outputs = jax.vmap(
        integrate_history, in_axes=(1, 0, None), out_axes=(0, 0)
    )(flat_F, states, material)
    final_state = jax.tree_util.tree_map(
        lambda x: x.reshape((ne, ng) + x.shape[1:]), final_state
    )
    outputs = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1).reshape(
            (F_history.shape[0] - 1, ne, ng) + x.shape[2:]
        ),
        outputs,
    )
    return final_state, outputs
