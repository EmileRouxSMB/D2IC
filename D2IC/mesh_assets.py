from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import jax.numpy as jnp

from .types import Array


@dataclass(frozen=True)
class Mesh:
    """
    Minimal mesh container.
    Replace/extend with your own mesh model as needed.
    """
    nodes_xy: Array        # (Nn, 2)
    elements: Array        # (Ne, nen) connectivity (indices into nodes)


@dataclass(frozen=True)
class MeshAssets:
    """
    Precomputations derived from Mesh for fast evaluation.
    Keep shapes/dtypes stable for JAX compilation.
    """
    mesh: Mesh
    element_centers_xy: Array  # (Ne, 2)
    node_neighbor_index: Optional[Array] = None
    node_neighbor_degree: Optional[Array] = None
    pixel_data: "PixelAssets" | None = None

    # Pixel-level lookup (pixel->element, shape functions, etc.) is stored in PixelAssets.


@dataclass(frozen=True)
class PixelAssets:
    """Pixel-level caches needed for the JAX CG solver."""

    pixel_coords_ref: Array
    pixel_nodes: Array
    pixel_shapeN: Array
    node_neighbor_index: Array
    node_neighbor_degree: Array
    node_neighbor_weight: Array
    node_reg_weight: Array
    node_pixel_index: Array
    node_N_weight: Array
    node_pixel_degree: Array
    roi_mask_flat: Array
    image_shape: tuple[int, int]

def compute_element_centers(mesh: Mesh) -> Array:
    """
    Compute element centers (Ne, 2) as the mean of node coordinates.
    """
    elem_nodes = mesh.nodes_xy[mesh.elements]  # (Ne, nen, 2)
    return jnp.mean(elem_nodes, axis=1)


def build_node_neighbor_tables(mesh: Mesh, max_degree: int = 16) -> tuple[Array, Array]:
    nodes_xy = np.asarray(mesh.nodes_xy)
    n_nodes = nodes_xy.shape[0]
    elements = np.asarray(mesh.elements, dtype=int)
    neighbors = [set() for _ in range(n_nodes)]
    for elt in elements:
        elt_nodes = [int(i) for i in elt]
        for i, ni in enumerate(elt_nodes):
            for nj in elt_nodes:
                if nj != ni:
                    neighbors[ni].add(int(nj))

    idx = np.zeros((n_nodes, max_degree), dtype=np.int32)
    deg = np.zeros((n_nodes,), dtype=np.int32)
    for i, neigh in enumerate(neighbors):
        sorted_neigh = sorted(neigh)
        deg_i = min(len(sorted_neigh), max_degree)
        deg[i] = deg_i
        if deg_i > 0:
            idx[i, :deg_i] = sorted_neigh[:deg_i]
    return jnp.asarray(idx), jnp.asarray(deg)


def make_mesh_assets(mesh: Mesh, with_neighbors: bool = True, max_degree: int = 16) -> MeshAssets:
    centers = compute_element_centers(mesh)
    node_idx: Optional[Array] = None
    node_deg: Optional[Array] = None
    if with_neighbors:
        node_idx, node_deg = build_node_neighbor_tables(mesh, max_degree=max_degree)
    return MeshAssets(
        mesh=mesh,
        element_centers_xy=centers,
        node_neighbor_index=node_idx,
        node_neighbor_degree=node_deg,
        pixel_data=None,
    )
