"""Boundary handling utilities for interpolation."""

from __future__ import annotations

import jax.numpy as jnp


def _reflect_index(index, size):
    size = jnp.asarray(size)
    size_minus_one = size - 1
    period = 2 * size_minus_one
    period = jnp.where(period <= 0, 1, period)
    index = jnp.mod(index, period)
    return jnp.where(index < size, index, period - index)


def _wrap_index(index, size):
    size = jnp.asarray(size)
    size = jnp.where(size <= 0, 1, size)
    return jnp.mod(index, size)


def apply_boundary(index, size, mode):
    """Apply boundary condition to indices (reflect-101 for reflect)."""
    if mode == "reflect":
        index_bc = _reflect_index(index, size)
        valid = jnp.ones_like(index_bc, dtype=bool)
    elif mode == "nearest":
        index_bc = jnp.clip(index, 0, size - 1)
        valid = jnp.ones_like(index_bc, dtype=bool)
    elif mode == "wrap":
        index_bc = _wrap_index(index, size)
        valid = jnp.ones_like(index_bc, dtype=bool)
    elif mode == "constant":
        valid = (index >= 0) & (index < size)
        index_bc = jnp.clip(index, 0, size - 1)
    else:
        raise ValueError(f"Unsupported boundary mode: {mode}")
    return index_bc, valid
