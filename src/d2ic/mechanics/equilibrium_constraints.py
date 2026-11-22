from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..core.types import Array


def active_equilibrium_dofs(
    n_nodes: int,
    excluded_nodes: np.ndarray,
) -> np.ndarray:
    """Build a `(Nn,2)` mask for weak-equilibrium equations used in inversion."""
    active = np.ones((int(n_nodes), 2), dtype=bool)
    active[np.asarray(excluded_nodes, dtype=np.int32)] = False
    return active


def masked_equilibrium_residual(
    internal_force: Array,
    external_force: Array | None,
    active_dofs: Array,
) -> Array:
    """Return ``f_int-f_ext`` with unavailable reaction DOFs set to zero."""
    residual = internal_force if external_force is None else internal_force - external_force
    return jnp.where(active_dofs, residual, 0.0)
