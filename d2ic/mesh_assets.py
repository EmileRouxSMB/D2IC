from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import jax.numpy as jnp

from .types import Array


@dataclass(frozen=True)
class Mesh:
    """
    Minimal 2D mesh container.

    Notes
    -----
    Coordinates are expressed in image space: ``x`` is the column axis and ``y``
    is the row axis.
    """
    nodes_xy: Array        # (Nn, 2)
    elements: Array        # (Ne, nen) connectivity (indices into nodes)


@dataclass(frozen=True)
class MeshAssets:
    """
    Precomputations derived from Mesh for fast evaluation.
    Keep shapes/dtypes stable for JAX compilation.

    Attributes
    ----------
    mesh:
        The underlying mesh definition.
    element_centers_xy:
        Element center coordinates with shape ``(Ne, 2)``.
    node_neighbor_index / node_neighbor_degree:
        Optional dense node-neighborhood tables for strain post-processing.
    pixel_data:
        Optional pixel-level caches required by some solvers (e.g. CG / local GN).
    roi_mask:
        Optional ROI mask aligned with the reference image (after binning).
    """
    mesh: Mesh
    element_centers_xy: Array  # (Ne, 2)
    node_neighbor_index: Optional[Array] = None
    node_neighbor_degree: Optional[Array] = None
    pixel_data: "PixelAssets" | None = None
    roi_mask: Optional[Array] = None

    # Pixel-level lookup (pixel->element, shape functions, etc.) is stored in PixelAssets.


@dataclass(frozen=True)
class PixelAssets:
    """
    Pixel-level caches for image-based DIC on a mesh.

    This bundles the mapping from pixels to element nodes, shape function values
    at each sampled pixel, and compact node-wise gather tables used by solvers.
    """

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
    Compute element centers as the mean of element node coordinates.

    Parameters
    ----------
    mesh:
        Mesh definition.

    Returns
    -------
    Array
        Element centers with shape ``(Ne, 2)``.
    """
    elem_nodes = mesh.nodes_xy[mesh.elements]  # (Ne, nen, 2)
    return jnp.mean(elem_nodes, axis=1)


def build_node_neighbor_tables(mesh: Mesh, max_degree: int = 16) -> tuple[Array, Array]:
    """
    Build dense node-neighborhood tables from element connectivity.

    Parameters
    ----------
    mesh:
        Mesh definition.
    max_degree:
        Maximum number of neighbors stored per node. Extra neighbors are dropped.

    Returns
    -------
    (node_neighbor_index, node_neighbor_degree):
        ``node_neighbor_index`` has shape ``(Nn, max_degree)`` and is padded with
        zeros. ``node_neighbor_degree`` has shape ``(Nn,)`` and indicates how many
        entries are valid per node.
    """
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
    """
    Compute `MeshAssets` for a given `Mesh`.

    Parameters
    ----------
    mesh:
        Mesh definition.
    with_neighbors:
        If True, also compute dense node-neighborhood tables used by strain routines.
    max_degree:
        Maximum number of neighbors stored per node when ``with_neighbors=True``.

    Returns
    -------
    MeshAssets
        Precomputed geometric tables. Pixel-level data is not generated here.
    """
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
