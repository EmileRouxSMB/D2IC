from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import Array
from .quadrature import q4_gauss_rule, q4_shape_gradients


@dataclass(frozen=True)
class BoundaryEdges:
    """Oriented exterior Q4 edges and their owning elements."""

    nodes: np.ndarray
    elements: np.ndarray
    local_edges: np.ndarray


def select_q4_boundary_face(
    nodes_xy: np.ndarray,
    elements: np.ndarray,
    side: str = "top",
    tolerance: float | None = None,
) -> BoundaryEdges:
    """Extract one exterior face from a possibly non-perfectly straight mesh.

    ``top`` means minimum y in the image-coordinate convention. When omitted,
    the selection tolerance is the median exterior-edge length.
    """
    nodes_xy = np.asarray(nodes_xy)
    elements = np.asarray(elements, dtype=np.int32)
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("Boundary extraction currently supports Q4 elements only.")
    occurrences: dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]] = {}
    local_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    for element_id, element in enumerate(elements):
        for local_edge, (a, b) in enumerate(local_pairs):
            oriented = (int(element[a]), int(element[b]))
            key = tuple(sorted(oriented))
            occurrences.setdefault(key, []).append((element_id, local_edge, oriented))
    exterior = [items[0] for items in occurrences.values() if len(items) == 1]
    if not exterior:
        raise ValueError("No exterior Q4 edge was found.")
    edge_nodes = np.asarray([item[2] for item in exterior], dtype=np.int32)
    edge_elements = np.asarray([item[0] for item in exterior], dtype=np.int32)
    local_edges = np.asarray([item[1] for item in exterior], dtype=np.int32)
    midpoints = nodes_xy[edge_nodes].mean(axis=1)
    lengths = np.linalg.norm(nodes_xy[edge_nodes[:, 1]] - nodes_xy[edge_nodes[:, 0]], axis=1)
    tolerance = float(np.median(lengths)) if tolerance is None else float(tolerance)
    axis_and_sign = {"top": (1, -1), "bottom": (1, 1), "left": (0, -1), "right": (0, 1)}
    if side not in axis_and_sign:
        raise ValueError(f"Unknown side {side!r}; expected one of {sorted(axis_and_sign)}.")
    axis, sign = axis_and_sign[side]
    target = nodes_xy[:, axis].min() if sign < 0 else nodes_xy[:, axis].max()
    selected = (
        midpoints[:, axis] <= target + tolerance
        if sign < 0 else midpoints[:, axis] >= target - tolerance
    )
    if not np.any(selected):
        raise ValueError(f"No exterior edge selected for face {side!r}.")
    return BoundaryEdges(edge_nodes[selected], edge_elements[selected], local_edges[selected])


def first_piola_from_cauchy(
    cauchy_stress: Array,
    deformation_gradient: Array,
    thickness_ratio: Array,
) -> Array:
    """Convert plane-stress Cauchy stress to in-plane first Piola stress.

    The 3D Jacobian is ``det(F_2D) * thickness_ratio``.
    """
    jacobian_3d = jnp.linalg.det(deformation_gradient) * thickness_ratio
    inverse_transpose = jnp.swapaxes(jnp.linalg.inv(deformation_gradient), -1, -2)
    return jacobian_3d[..., None, None] * cauchy_stress @ inverse_transpose


@jax.jit
def assemble_internal_force_q4(
    nodes_xy: Array,
    elements: Array,
    deformation_gradient: Array,
    cauchy_stress: Array,
    thickness_ratio: Array,
    initial_thickness: Array,
    length_scale: Array,
) -> Array:
    """Assemble the reference-form nodal internal force ``int B0.T P dV0``.

    Coordinates may remain in pixels. ``length_scale`` converts pixel lengths
    to the unit paired with ``initial_thickness``; MPa and mm then produce N.
    """
    gauss_points, gauss_weights = q4_gauss_rule(nodes_xy.dtype)
    element_coordinates = nodes_xy[elements]

    def gradients_one_element(coordinates):
        def at_point(parent):
            dN_dxi = q4_shape_gradients(parent)
            jacobian = coordinates.T @ dN_dxi
            return dN_dxi @ jnp.linalg.inv(jacobian), jnp.linalg.det(jacobian)
        return jax.vmap(at_point)(gauss_points)

    dN_dX, detJ = jax.vmap(gradients_one_element)(element_coordinates)
    P = first_piola_from_cauchy(cauchy_stress, deformation_gradient, thickness_ratio)
    measures = detJ * gauss_weights[None, :] * initial_thickness * length_scale
    element_force = jnp.einsum("egij,egaj,eg->eai", P, dN_dX, measures)
    return jnp.zeros((nodes_xy.shape[0], 2), dtype=P.dtype).at[elements].add(element_force)


@jax.jit
def assemble_internal_force_history_q4(
    nodes_xy: Array,
    elements: Array,
    deformation_gradient_history: Array,
    cauchy_stress_history: Array,
    thickness_ratio_history: Array,
    initial_thickness: Array,
    length_scale: Array,
) -> Array:
    """Vectorize internal-force assembly over all measured frames."""
    return jax.vmap(
        assemble_internal_force_q4,
        in_axes=(None, None, 0, 0, 0, None, None),
    )(
        nodes_xy, elements, deformation_gradient_history, cauchy_stress_history,
        thickness_ratio_history, initial_thickness, length_scale,
    )


def equilibrium_residual(
    internal_force: Array,
    external_force: Array | None = None,
) -> Array:
    """Return the weak equilibrium residual ``f_int - f_ext``."""
    if external_force is None:
        return internal_force
    return internal_force - external_force


@jax.jit
def integrate_boundary_traction(
    nodes_xy: Array,
    displacement: Array,
    boundary_nodes: Array,
    boundary_elements: Array,
    boundary_local_edges: Array,
    cauchy_stress: Array,
    thickness_ratio: Array,
    initial_thickness: Array,
    length_scale: Array,
) -> Array:
    """Integrate ``sigma @ n`` on oriented exterior edges in current geometry."""
    current = (nodes_xy + displacement) * length_scale
    edge_coordinates = current[boundary_nodes]
    tangents = edge_coordinates[:, 1] - edge_coordinates[:, 0]
    lengths = jnp.linalg.norm(tangents, axis=1)
    # Q4 elements are required to have positive reference Jacobians; their
    # oriented boundary has the outward normal on the right of each edge.
    normals = jnp.stack((tangents[:, 1], -tangents[:, 0]), axis=1) / lengths[:, None]
    edge_gauss = jnp.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=jnp.int32)
    gp_ids = edge_gauss[boundary_local_edges]
    stress_on_edges = cauchy_stress[boundary_elements]
    thickness_on_edges = thickness_ratio[boundary_elements]
    gather = jnp.arange(boundary_elements.shape[0])[:, None]
    stress_mean = jnp.mean(stress_on_edges[gather, gp_ids], axis=1)
    thickness_mean = jnp.mean(thickness_on_edges[gather, gp_ids], axis=1)
    edge_forces = jnp.einsum("eij,ej->ei", stress_mean, normals)
    edge_forces *= (lengths * initial_thickness * thickness_mean)[:, None]
    return jnp.sum(edge_forces, axis=0)


@jax.jit
def integrate_boundary_traction_history(
    nodes_xy: Array,
    displacement_history: Array,
    boundary_nodes: Array,
    boundary_elements: Array,
    boundary_local_edges: Array,
    cauchy_stress_history: Array,
    thickness_ratio_history: Array,
    initial_thickness: Array,
    length_scale: Array,
) -> Array:
    """Vectorize a selected-face traction resultant over measured frames."""
    return jax.vmap(
        integrate_boundary_traction,
        in_axes=(None, 0, None, None, None, 0, 0, None, None),
    )(
        nodes_xy, displacement_history, boundary_nodes, boundary_elements,
        boundary_local_edges, cauchy_stress_history, thickness_ratio_history,
        initial_thickness, length_scale,
    )
