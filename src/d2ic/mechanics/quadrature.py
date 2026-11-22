from __future__ import annotations

import jax.numpy as jnp

from ..core.types import Array


def q4_gauss_rule(dtype=jnp.float64) -> tuple[Array, Array]:
    """Return the standard 2x2 Gauss rule on ``[-1,1]^2``."""
    a = jnp.asarray(1.0 / jnp.sqrt(3.0), dtype=dtype)
    points = jnp.asarray([[-a, -a], [a, -a], [a, a], [-a, a]], dtype=dtype)
    return points, jnp.ones(4, dtype=dtype)


def q4_shape_gradients(parent_coordinates: Array) -> Array:
    """Derivatives ``dN/d(xi,eta)`` for Q4 nodes ordered around the element."""
    xi, eta = parent_coordinates
    return 0.25 * jnp.asarray(
        [[-(1.0 - eta), -(1.0 - xi)],
         [+(1.0 - eta), -(1.0 + xi)],
         [+(1.0 + eta), +(1.0 + xi)],
         [-(1.0 + eta), +(1.0 - xi)]],
        dtype=parent_coordinates.dtype,
    )
