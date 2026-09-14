from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from d2ic.mechanics.equilibrium import (
    assemble_internal_force_q4,
    integrate_boundary_traction,
    select_q4_boundary_face,
)


def test_uniform_stress_has_zero_global_internal_resultant() -> None:
    nodes = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    elements = jnp.asarray([[0, 1, 2, 3]])
    F = jnp.broadcast_to(jnp.eye(2), (1, 4, 2, 2))
    stress = jnp.zeros((1, 4, 2, 2)).at[:, :, 1, 1].set(100.0)
    force = assemble_internal_force_q4(
        nodes, elements, F, stress, jnp.ones((1, 4)), jnp.asarray(0.5), jnp.asarray(1.0)
    )
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-12)


def test_top_face_traction_resultant_components() -> None:
    nodes = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
    top = select_q4_boundary_face(nodes, elements, side="top", tolerance=0.1)
    stress = jnp.zeros((1, 4, 2, 2))
    stress = stress.at[:, :, 0, 1].set(10.0).at[:, :, 1, 0].set(10.0)
    stress = stress.at[:, :, 1, 1].set(100.0)
    resultant = integrate_boundary_traction(
        jnp.asarray(nodes), jnp.zeros((4, 2)), jnp.asarray(top.nodes),
        jnp.asarray(top.elements), jnp.asarray(top.local_edges), stress,
        jnp.ones((1, 4)), jnp.asarray(0.5), jnp.asarray(1.0),
    )
    # Top outward normal is -y: width 2 * thickness 0.5.
    np.testing.assert_allclose(resultant, [-10.0, -100.0], atol=1e-12)
