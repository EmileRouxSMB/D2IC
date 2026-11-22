"""Mesh geometry, topology, pixel sampling and ROI-based generation."""

from .assets import (
    Mesh,
    MeshAssets,
    MeshTopology,
    NeighborTable,
    NodePixelGather,
    PixelAssets,
    PixelSampling,
    WeightedNeighborTable,
    build_node_neighbor_tables,
    build_strain_neighbors,
    compute_element_centers,
    make_mesh_assets,
)
from .pixel_sampling import (
    PixelSamplingConfig,
    build_node_neighbor_dense,
    build_node_pixel_dense,
    build_pixel_assets,
    build_pixel_to_element_mapping_numpy,
    compute_pixel_shape_functions_jax,
    points_in_convex_quad,
    shape_function_gradients_jax,
    shape_functions_jax,
)
from .generation import (
    downsample_mask,
    mask_to_mesh,
    mask_to_mesh_assets,
    mask_to_mesh_assets_gmsh,
    mask_to_mesh_gmsh,
)

__all__ = [
    "Mesh", "MeshAssets", "MeshTopology", "NeighborTable", "NodePixelGather",
    "PixelAssets", "PixelSampling", "PixelSamplingConfig", "WeightedNeighborTable",
    "build_node_neighbor_dense", "build_node_neighbor_tables", "build_node_pixel_dense",
    "build_pixel_assets", "build_pixel_to_element_mapping_numpy", "build_strain_neighbors",
    "compute_element_centers", "compute_pixel_shape_functions_jax", "downsample_mask",
    "make_mesh_assets", "mask_to_mesh", "mask_to_mesh_assets",
    "mask_to_mesh_assets_gmsh", "mask_to_mesh_gmsh", "points_in_convex_quad",
    "shape_function_gradients_jax", "shape_functions_jax",
]
