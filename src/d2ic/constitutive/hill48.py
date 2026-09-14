from __future__ import annotations

import jax.numpy as jnp

from ..core.types import Array


def hill48_matrix(coefficients: Array) -> Array:
    """Build the normalized Hill48 quadratic matrix in stress Voigt order."""
    F, G, H, L, M, N = coefficients
    P = jnp.asarray(
        [[G + H, -H, -G, 0, 0, 0],
         [-H, F + H, -F, 0, 0, 0],
         [-G, -F, F + G, 0, 0, 0],
         [0, 0, 0, 2 * N, 0, 0],
         [0, 0, 0, 0, 2 * L, 0],
         [0, 0, 0, 0, 0, 2 * M]],
        dtype=coefficients.dtype,
    )
    return P / (G + H)


def hill48_equivalent_stress(stress: Array, coefficients: Array) -> Array:
    """Return normalized Hill48 equivalent stress in material axes."""
    value = stress @ hill48_matrix(coefficients) @ stress
    return jnp.sqrt(jnp.maximum(value, 0.0))
