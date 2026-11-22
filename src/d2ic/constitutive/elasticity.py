from __future__ import annotations

import jax.numpy as jnp

from ..core.types import Array


def isotropic_stiffness(young: Array, poisson: Array) -> Array:
    """Return the 3D isotropic stiffness for engineering-strain Voigt vectors."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    C = jnp.zeros((6, 6), dtype=jnp.result_type(young, poisson))
    C = C.at[:3, :3].set(lam)
    C = C.at[jnp.arange(3), jnp.arange(3)].add(2.0 * mu)
    C = C.at[jnp.arange(3, 6), jnp.arange(3, 6)].set(mu)
    return C
