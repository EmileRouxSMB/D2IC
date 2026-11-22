from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from d2ic.mechanics.equilibrium_constraints import (
    active_equilibrium_dofs,
    masked_equilibrium_residual,
)


def test_unknown_grip_reactions_are_excluded_from_residual() -> None:
    internal = jnp.arange(12.0).reshape(2, 3, 2)
    active = active_equilibrium_dofs(3, np.array([0, 2]))
    residual = masked_equilibrium_residual(internal, None, jnp.asarray(active))
    np.testing.assert_allclose(residual[:, 0], 0.0)
    np.testing.assert_allclose(residual[:, 2], 0.0)
    np.testing.assert_allclose(residual[:, 1], np.asarray(internal)[:, 1])
