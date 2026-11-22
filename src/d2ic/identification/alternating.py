"""Low-memory building blocks for integrated DIC/material identification."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import Array
from ..mesh.assets import PixelSampling
from ..physics.discrepancy import bilinear_sample_jax


@jax.jit
def photometric_mse(
    displacement: Array,
    reference_image: Array,
    deformed_image: Array,
    sampling: PixelSampling,
) -> Array:
    """Mean squared brightness residual on the reference ROI pixels.

    ``displacement`` has shape ``(Nn, 2)``. Only the pixels retained in
    ``sampling`` are evaluated, so no dense residual image is materialized.
    """
    nodal_values = displacement[sampling.pixel_nodes]
    pixel_displacement = jnp.sum(
        sampling.pixel_shapeN[..., None] * nodal_values, axis=1
    )
    x_ref = sampling.pixel_coords_ref
    reference = bilinear_sample_jax(reference_image, x_ref)
    deformed = bilinear_sample_jax(deformed_image, x_ref + pixel_displacement)
    residual = deformed - reference
    valid = jnp.isfinite(residual)
    safe = jnp.where(valid, residual, 0.0)
    return jnp.sum(safe * safe) / jnp.maximum(jnp.sum(valid), 1)


@jax.jit
def spring_regularization_mse(displacement: Array, sampling: PixelSampling) -> Array:
    """Weighted graph-gradient energy used by the FE-DIC regularization."""
    graph = sampling.regularization_graph
    slots = jnp.arange(graph.index.shape[1])[None, :]
    valid = slots < graph.degree[:, None]
    neighbors = jnp.where(valid, graph.index, 0)
    difference = displacement[:, None, :] - displacement[neighbors]
    weights = jnp.where(valid, graph.weight, 0.0)
    energy = jnp.sum(weights[..., None] * difference * difference)
    return energy / jnp.maximum(2.0 * jnp.sum(weights), 1.0)


@jax.jit
def weak_equilibrium_mse(
    nodal_internal_force: Array,
    virtual_fields: Array,
    force_scale: Array,
) -> Array:
    """Project nodal equilibrium onto virtual fields and return normalized MSE.

    ``virtual_fields`` has shape ``(Nv, Nn, 2)`` and must vanish where unknown
    grip reactions act. Each field is normalized during its construction.
    """
    residual = jnp.einsum("vni,ni->v", virtual_fields, nodal_internal_force)
    return jnp.mean((residual / jnp.maximum(force_scale, 1e-12)) ** 2)


def gaussian_virtual_fields(
    nodes_xy: np.ndarray,
    excluded_nodes: np.ndarray,
    *,
    centers_x: int = 4,
    centers_y: int = 6,
    radius: float | None = None,
) -> np.ndarray:
    """Construct smooth two-component Gaussian virtual fields on mesh nodes.

    The returned array has shape ``(2*centers_x*centers_y, Nn, 2)``. Fields
    vanish on ``excluded_nodes`` and are normalized to unit nodal L2 norm.
    """
    xy = np.asarray(nodes_xy, dtype=float)
    if centers_x < 1 or centers_y < 1:
        raise ValueError("centers_x and centers_y must be positive.")
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi - lo, np.finfo(float).eps)
    margin = 0.1 * span
    xs = np.linspace(lo[0] + margin[0], hi[0] - margin[0], centers_x)
    ys = np.linspace(lo[1] + margin[1], hi[1] - margin[1], centers_y)
    if radius is None:
        radius = 1.5 * max(span[0] / centers_x, span[1] / centers_y)
    fields = []
    excluded = np.asarray(excluded_nodes, dtype=np.int32)
    for y in ys:
        for x in xs:
            amplitude = np.exp(-np.sum((xy - np.asarray([x, y])) ** 2, axis=1) / (2.0 * radius**2))
            amplitude[excluded] = 0.0
            for component in range(2):
                field = np.zeros((xy.shape[0], 2), dtype=xy.dtype)
                field[:, component] = amplitude
                norm = np.linalg.norm(field)
                if norm > 0.0:
                    fields.append(field / norm)
    if not fields:
        raise ValueError("No non-zero virtual field could be constructed.")
    return np.asarray(fields)


def adam_update(
    value: Array,
    gradient: Array,
    first_moment: Array,
    second_moment: Array,
    step: int,
    learning_rate: float,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[Array, Array, Array]:
    """Apply one allocation-light Adam update to a JAX array."""
    first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
    second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
    m_hat = first_moment / (1.0 - beta1**step)
    v_hat = second_moment / (1.0 - beta2**step)
    value = value - learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
    return value, first_moment, second_moment

