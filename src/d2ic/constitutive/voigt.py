from __future__ import annotations

import jax.numpy as jnp

from ..core.types import Array


def strain_tensor_to_voigt(strain: Array) -> Array:
    """Convert a symmetric 3D strain tensor to engineering Voigt notation."""
    return jnp.stack(
        (strain[..., 0, 0], strain[..., 1, 1], strain[..., 2, 2],
         2.0 * strain[..., 0, 1], 2.0 * strain[..., 1, 2],
         2.0 * strain[..., 0, 2]),
        axis=-1,
    )


def stress_voigt_to_tensor(stress: Array) -> Array:
    """Convert ``[s11,s22,s33,s12,s23,s13]`` to a symmetric tensor."""
    s11, s22, s33, s12, s23, s13 = jnp.moveaxis(stress, -1, 0)
    return jnp.stack(
        (jnp.stack((s11, s12, s13), -1),
         jnp.stack((s12, s22, s23), -1),
         jnp.stack((s13, s23, s33), -1)),
        axis=-2,
    )


def stress_tensor_to_voigt(stress: Array) -> Array:
    """Convert a symmetric 3D stress tensor to physical-shear Voigt notation."""
    return jnp.stack(
        (stress[..., 0, 0], stress[..., 1, 1], stress[..., 2, 2],
         stress[..., 0, 1], stress[..., 1, 2], stress[..., 0, 2]),
        axis=-1,
    )
