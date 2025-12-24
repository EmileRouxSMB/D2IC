"""Cubic convolution kernels."""

from __future__ import annotations

import jax.numpy as jnp


def keys_cubic_kernel(t, a):
    """Keys cubic kernel with support |t| < 2."""
    at = jnp.abs(t)
    at2 = at * at
    at3 = at2 * at
    w0 = (a + 2.0) * at3 - (a + 3.0) * at2 + 1.0
    w1 = a * at3 - 5.0 * a * at2 + 8.0 * a * at - 4.0 * a
    return jnp.where(at < 1.0, w0, jnp.where(at < 2.0, w1, 0.0))


def cubic_weights(frac, a):
    """Return cubic weights for offsets [-1, 0, 1, 2]."""
    offsets = jnp.array([-1.0, 0.0, 1.0, 2.0], dtype=frac.dtype)
    t = frac[..., None] - offsets
    return keys_cubic_kernel(t, a)
