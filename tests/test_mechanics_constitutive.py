from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from d2ic.mechanics import deformation_gradient_history_q4


def test_q4_affine_deformation_gradient() -> None:
    nodes = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    elements = jnp.asarray([[0, 1, 2, 3]])
    grad_u = jnp.asarray([[0.10, 0.20], [-0.05, 0.03]])
    displacement = nodes @ grad_u.T
    history = jnp.stack((jnp.zeros_like(displacement), displacement))
    F, _, detJ = deformation_gradient_history_q4(nodes, elements, history)
    expected = jnp.eye(2) + grad_u
    np.testing.assert_allclose(F[1, 0], np.broadcast_to(expected, (4, 2, 2)), atol=1e-12)
    assert np.all(np.asarray(detJ) > 0.0)
