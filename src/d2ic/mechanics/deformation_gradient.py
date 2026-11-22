from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core.types import Array
from .quadrature import q4_gauss_rule, q4_shape_gradients


def q4_deformation_gradient(
    element_coordinates: Array,
    element_displacement: Array,
    parent_coordinates: Array,
) -> tuple[Array, Array]:
    """Compute ``F=I+du/dX`` and reference Jacobian at one Q4 Gauss point."""
    dN_dxi = q4_shape_gradients(parent_coordinates)
    jacobian = element_coordinates.T @ dN_dxi
    dN_dX = dN_dxi @ jnp.linalg.inv(jacobian)
    displacement_gradient = element_displacement.T @ dN_dX
    return jnp.eye(2, dtype=element_coordinates.dtype) + displacement_gradient, jnp.linalg.det(jacobian)


@jax.jit
def deformation_gradient_history_q4(
    nodes_xy: Array,
    elements: Array,
    displacement_history: Array,
) -> tuple[Array, Array, Array]:
    """Evaluate a nodal DIC history at all 2x2 Q4 Gauss points.

    Returns ``F_history`` with shape ``(Nt, Ne, 4, 2, 2)``, the four parent
    coordinates, and reference quadrature measures ``detJ`` of shape `(Ne,4)`.
    """
    gauss_points, _ = q4_gauss_rule(nodes_xy.dtype)
    element_coordinates = nodes_xy[elements]
    element_displacements = displacement_history[:, elements]
    point_map = jax.vmap(q4_deformation_gradient, in_axes=(None, None, 0))
    element_map = jax.vmap(point_map, in_axes=(0, 0, None))

    def frame_map(frame_element_displacements):
        return element_map(element_coordinates, frame_element_displacements, gauss_points)

    F_history, detJ_history = jax.vmap(frame_map)(element_displacements)
    return F_history, gauss_points, detJ_history[0]
