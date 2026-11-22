from __future__ import annotations

import jax.numpy as jnp

from ..core.types import Array


def flow_stress(eqps: Array, parameters: Array, model: str) -> Array:
    """Evaluate an isotropic flow stress law.

    Voce uses ``sigma0 + K*(1-exp(-eqps/eps0))``. Nadai/Swift uses
    ``sigma0 + K*(eps0+eqps)**n``. Tabulated data use rows
    ``[equivalent plastic strain, true yield stress]`` and linear interpolation.
    """
    if model == "tabulated":
        return jnp.interp(eqps, parameters[:, 0], parameters[:, 1])
    sigma0, K, eps0, exponent = parameters
    if model == "voce":
        return sigma0 + K * (1.0 - jnp.exp(-eqps / eps0))
    if model in ("nadai", "swift"):
        return sigma0 + K * (eps0 + eqps) ** exponent
    raise ValueError(f"Unsupported hardening model: {model}")
