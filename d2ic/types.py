from __future__ import annotations

from typing import Any, Tuple
import numpy as np

try:
    import jax.numpy as jnp
    Array = jnp.ndarray
except Exception:  # pragma: no cover
    Array = Any

NPArray = np.ndarray
XY = Tuple[float, float]
