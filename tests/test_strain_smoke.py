from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from d2ic.mesh_assets import Mesh, build_node_neighbor_tables
from d2ic.strain import compute_green_lagrange_strain_nodes_lsq, green_lagrange_to_voigt


def test_green_lagrange_strain_uniform_extension() -> None:
    nodes = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    elements = jnp.array([[0, 1, 3, 2]], dtype=jnp.int32)
    mesh = Mesh(nodes_xy=nodes, elements=elements)
    neigh_idx, neigh_deg = build_node_neighbor_tables(mesh)

    eps_val = 0.01
    ux = nodes[:, 0] * eps_val
    uy = jnp.zeros_like(nodes[:, 1])
    disp = jnp.stack([ux, uy], axis=1)

    _, E_all = compute_green_lagrange_strain_nodes_lsq(
        displacement=disp,
        nodes_coord=nodes,
        node_neighbor_index=neigh_idx,
        node_neighbor_degree=neigh_deg,
    )
    strain_voigt = np.asarray(green_lagrange_to_voigt(E_all))

    expected_exx = eps_val + 0.5 * eps_val ** 2
    np.testing.assert_allclose(strain_voigt[:, 0], expected_exx, atol=5e-4)
    np.testing.assert_allclose(strain_voigt[:, 1], 0.0, atol=5e-4)
    np.testing.assert_allclose(strain_voigt[:, 2], 0.0, atol=5e-4)
