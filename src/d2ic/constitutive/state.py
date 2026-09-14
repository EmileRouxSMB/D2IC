from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..core.types import Array


@dataclass(frozen=True)
class Hill48Material:
    """Parameters of an isotropic-elastic, Hill48 plastic metal.

    ``hill`` contains ``[F, G, H, L, M, N]``. ``hardening`` contains
    ``[sigma0, K, eps0, exponent]``; the last entry is unused for Voce.
    """

    young: Array
    poisson: Array
    hill: Array
    hardening: Array
    hardening_model: str = "voce"


@dataclass(frozen=True)
class MaterialState:
    """History variables at one material point in material axes."""

    stress: Array
    plastic_strain: Array
    equivalent_plastic_strain: Array
    orientation: Array
    log_thickness_strain: Array


def initial_material_state(dtype=jnp.float64, angle_rad: float = 0.0) -> MaterialState:
    """Create a stress-free state with material axis 1 at ``angle_rad``."""
    c = jnp.cos(jnp.asarray(angle_rad, dtype=dtype))
    s = jnp.sin(jnp.asarray(angle_rad, dtype=dtype))
    orientation = jnp.asarray([[c, -s], [s, c]], dtype=dtype)
    return MaterialState(
        stress=jnp.zeros(6, dtype=dtype),
        plastic_strain=jnp.zeros(6, dtype=dtype),
        equivalent_plastic_strain=jnp.asarray(0.0, dtype=dtype),
        orientation=orientation,
        log_thickness_strain=jnp.asarray(0.0, dtype=dtype),
    )


jax.tree_util.register_dataclass(
    Hill48Material,
    data_fields=("young", "poisson", "hill", "hardening"),
    meta_fields=("hardening_model",),
)
jax.tree_util.register_dataclass(
    MaterialState,
    data_fields=(
        "stress",
        "plastic_strain",
        "equivalent_plastic_strain",
        "orientation",
        "log_thickness_strain",
    ),
    meta_fields=(),
)
