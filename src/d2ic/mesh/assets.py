from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import Array


@dataclass(frozen=True)
class Mesh:
    """Two-dimensional mesh in image coordinates ``(x, y)``.

    ``nodes_xy`` has shape ``(Nn, 2)`` and ``elements`` has shape ``(Ne, nen)``.
    """

    nodes_xy: Array
    elements: Array


@dataclass(frozen=True)
class NeighborTable:
    """Dense padded node adjacency used for strain reconstruction."""

    index: Array
    degree: Array


@dataclass(frozen=True)
class WeightedNeighborTable:
    """Dense weighted adjacency used by displacement regularization."""

    index: Array
    degree: Array
    weight: Array
    node_weight: Array


@dataclass(frozen=True)
class NodePixelGather:
    """Dense node-to-pixel gather table used by local solver kernels."""

    index: Array
    shape_weight: Array
    degree: Array


@dataclass(frozen=True)
class MeshTopology:
    """Image-independent precomputations derived from a mesh."""

    element_centers_xy: Array
    strain_neighbors: Optional[NeighborTable] = None


@dataclass(frozen=True)
class PixelSampling:
    """Image-dependent pixel sampling and solver gather tables."""

    pixel_coords_ref: Array
    pixel_nodes: Array
    pixel_shapeN: Array
    regularization_graph: WeightedNeighborTable
    node_pixel_gather: Optional[NodePixelGather]
    roi_mask_flat: Array
    image_shape: tuple[int, int]

    @property
    def node_neighbor_index(self) -> Array:
        return self.regularization_graph.index

    @property
    def node_neighbor_degree(self) -> Array:
        return self.regularization_graph.degree

    @property
    def node_neighbor_weight(self) -> Array:
        return self.regularization_graph.weight

    @property
    def node_reg_weight(self) -> Array:
        return self.regularization_graph.node_weight

    @property
    def node_pixel_index(self) -> Array:
        if self.node_pixel_gather is None:
            raise ValueError("Node-pixel gather was not requested for this sampling.")
        return self.node_pixel_gather.index

    @property
    def node_N_weight(self) -> Array:
        if self.node_pixel_gather is None:
            raise ValueError("Node-pixel gather was not requested for this sampling.")
        return self.node_pixel_gather.shape_weight

    @property
    def node_pixel_degree(self) -> Array:
        if self.node_pixel_gather is None:
            raise ValueError("Node-pixel gather was not requested for this sampling.")
        return self.node_pixel_gather.degree


# Backward-compatible public name. The underlying data model is PixelSampling.
PixelAssets = PixelSampling


@dataclass(frozen=True)
class MeshAssets:
    """Immutable data associated with one mesh.

    ``topology`` is image-independent; ``pixel_data`` depends on the reference
    image and ROI. ``mesh`` is the unique mesh source used by DIC pipelines.
    """

    mesh: Mesh
    topology: MeshTopology
    pixel_data: Optional[PixelSampling] = None
    roi_mask: Optional[Array] = None

    @property
    def element_centers_xy(self) -> Array:
        return self.topology.element_centers_xy

    @property
    def node_neighbor_index(self) -> Optional[Array]:
        table = self.topology.strain_neighbors
        return None if table is None else table.index

    @property
    def node_neighbor_degree(self) -> Optional[Array]:
        table = self.topology.strain_neighbors
        return None if table is None else table.degree


def compute_element_centers(mesh: Mesh) -> Array:
    """Compute element centers with shape ``(Ne, 2)``."""

    return jnp.mean(mesh.nodes_xy[mesh.elements], axis=1)


def build_strain_neighbors(mesh: Mesh, max_degree: int = 16) -> NeighborTable:
    """Build the node adjacency used for nodal strain reconstruction."""

    if max_degree < 1:
        raise ValueError("max_degree must be >= 1")
    n_nodes = int(np.asarray(mesh.nodes_xy).shape[0])
    elements = np.asarray(mesh.elements, dtype=int)
    neighbors = [set() for _ in range(n_nodes)]
    for element in elements:
        element_nodes = [int(node) for node in element]
        for node in element_nodes:
            neighbors[node].update(other for other in element_nodes if other != node)

    index = -np.ones((n_nodes, max_degree), dtype=np.int32)
    degree = np.zeros((n_nodes,), dtype=np.int32)
    for node, adjacent in enumerate(neighbors):
        values = sorted(adjacent)[:max_degree]
        degree[node] = len(values)
        if values:
            index[node, : len(values)] = values
    return NeighborTable(index=jnp.asarray(index), degree=jnp.asarray(degree))


def build_node_neighbor_tables(mesh: Mesh, max_degree: int = 16) -> tuple[Array, Array]:
    """Compatibility wrapper returning strain-neighbor arrays."""

    table = build_strain_neighbors(mesh, max_degree=max_degree)
    return table.index, table.degree


def make_mesh_assets(
    mesh: Mesh,
    with_neighbors: bool = True,
    max_degree: int = 16,
) -> MeshAssets:
    """Build image-independent topology for ``mesh``."""

    neighbors = build_strain_neighbors(mesh, max_degree) if with_neighbors else None
    topology = MeshTopology(
        element_centers_xy=compute_element_centers(mesh),
        strain_neighbors=neighbors,
    )
    return MeshAssets(mesh=mesh, topology=topology)


jax.tree_util.register_dataclass(Mesh, data_fields=("nodes_xy", "elements"), meta_fields=())
jax.tree_util.register_dataclass(NeighborTable, data_fields=("index", "degree"), meta_fields=())
jax.tree_util.register_dataclass(
    WeightedNeighborTable,
    data_fields=("index", "degree", "weight", "node_weight"),
    meta_fields=(),
)
jax.tree_util.register_dataclass(
    NodePixelGather,
    data_fields=("index", "shape_weight", "degree"),
    meta_fields=(),
)
jax.tree_util.register_dataclass(
    MeshTopology,
    data_fields=("element_centers_xy", "strain_neighbors"),
    meta_fields=(),
)
jax.tree_util.register_dataclass(
    PixelSampling,
    data_fields=(
        "pixel_coords_ref",
        "pixel_nodes",
        "pixel_shapeN",
        "regularization_graph",
        "node_pixel_gather",
        "roi_mask_flat",
    ),
    meta_fields=("image_shape",),
)
jax.tree_util.register_dataclass(
    MeshAssets,
    data_fields=("mesh", "topology", "pixel_data", "roi_mask"),
    meta_fields=(),
)
