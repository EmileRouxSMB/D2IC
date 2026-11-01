from __future__ import annotations

import sys as _sys

from .core import (
    Array,
    BatchConfig,
    BatchDiagnostics,
    BatchResult,
    DICBase,
    DICDiagnostics,
    DICResult,
    InitMotionConfig,
    MeshDICConfig,
    MeshDICResult,
)
from .mesh.assets import (
    Mesh,
    MeshAssets,
    MeshTopology,
    NeighborTable,
    NodePixelGather,
    PixelSampling,
    WeightedNeighborTable,
    compute_element_centers,
)
from .mesh.generation import (
    downsample_mask,
    mask_to_mesh,
    mask_to_mesh_assets,
    mask_to_mesh_gmsh,
    mask_to_mesh_assets_gmsh,
)
from .physics.strain import (
    compute_green_lagrange_strain_nodes_lsq,
    green_lagrange_to_voigt,
    small_strain_nodes_lsq,
)
from .methods import DICInitMotion, DICMeshBased
from .batch import BatchBase, BatchMeshBased
from .propagation import (
    ConstantVelocityPropagator,
    DisplacementPropagatorBase,
    PreviousDisplacementPropagator,
)
from .numerics.translation_zncc import TranslationZNCCSolver
from .numerics.global_cg import GlobalCGSolver
from .numerics.local_gauss_newton import LocalGaussNewtonSolver
from .visualization import DICPlotter
from .io import (
    SequenceConfig,
    bin_image,
    bin_mask,
    configure_jax_platform,
    downsample_image,
    imread_gray,
    list_deformed_images,
    load_sequence_config,
    make_roi_mask,
    write_dic_vtu,
    write_image_vti,
    write_pvd,
    prepare_image,
    AbaqusInterpolationMap,
    abaqus_displacement_to_image_px,
    build_abaqus_interpolation_map,
    compare_abaqus_dic_sequence,
    connected_node_mask,
    image_nodes_to_abaqus_mm,
    image_time_from_name,
    interpolate_abaqus_space,
    interpolate_abaqus_time,
)
from .pipelines import run_sequence_from_config

# Backward-compatible module paths for notebooks and older user scripts.  The
# canonical implementations live in thematic subpackages.
from .core import types as _types_module
from .core import dataclasses as _dataclasses_module
from .core import dic_base as _dic_base_module
from .methods import mesh_based as _mesh_based_module
from .methods import init_motion as _init_motion_module
from .batch import base as _batch_base_module
from .batch import mesh_based as _batch_mesh_based_module
from .propagation import base as _propagation_base_module
from .propagation import previous as _propagation_previous_module
from .propagation import constant_velocity as _propagation_constant_velocity_module
from .visualization import plotter as _plotter_module
from .io import app_utils as _app_utils_module
from .io import config as _config_module
from .io import cli as _cli_module
from .mesh import assets as _mesh_assets_module
from .mesh import pixel_sampling as _pixel_assets_module
from .mesh import generation as _mask2mesh_module
from .numerics import base as _solver_base_module
from .numerics import global_cg as _solver_global_cg_module
from .numerics import local_gauss_newton as _solver_local_gn_module
from .numerics import translation_zncc as _solver_translation_zncc_module
from .physics import strain as _strain_module
from .physics import discrepancy as _discrepancy_module

_sys.modules.setdefault(__name__ + ".types", _types_module)
_sys.modules.setdefault(__name__ + ".dataclasses", _dataclasses_module)
_sys.modules.setdefault(__name__ + ".dic_base", _dic_base_module)
_sys.modules.setdefault(__name__ + ".dic_mesh_based", _mesh_based_module)
_sys.modules.setdefault(__name__ + ".dic_init_motion", _init_motion_module)
_sys.modules.setdefault(__name__ + ".batch_base", _batch_base_module)
_sys.modules.setdefault(__name__ + ".batch_mesh_based", _batch_mesh_based_module)
_sys.modules.setdefault(__name__ + ".propagator_base", _propagation_base_module)
_sys.modules.setdefault(__name__ + ".propagator_previous", _propagation_previous_module)
_sys.modules.setdefault(__name__ + ".propagator_constant_velocity", _propagation_constant_velocity_module)
_sys.modules.setdefault(__name__ + ".plotter", _plotter_module)
_sys.modules.setdefault(__name__ + ".app_utils", _app_utils_module)
_sys.modules.setdefault(__name__ + ".config", _config_module)
_sys.modules.setdefault(__name__ + ".cli", _cli_module)
_sys.modules.setdefault(__name__ + ".mesh_assets", _mesh_assets_module)
_sys.modules.setdefault(__name__ + ".pixel_assets", _pixel_assets_module)
_sys.modules.setdefault(__name__ + ".mask2mesh", _mask2mesh_module)
_sys.modules.setdefault(__name__ + ".solver_base", _solver_base_module)
_sys.modules.setdefault(__name__ + ".solver_global_cg", _solver_global_cg_module)
_sys.modules.setdefault(__name__ + ".solver_local_gn", _solver_local_gn_module)
_sys.modules.setdefault(__name__ + ".solver_translation_zncc", _solver_translation_zncc_module)
_sys.modules.setdefault(__name__ + ".strain", _strain_module)
_sys.modules.setdefault(__name__ + ".discrepancy", _discrepancy_module)

# Also expose module objects for `from d2ic import strain` style imports.
mesh_assets = _mesh_assets_module
pixel_assets = _pixel_assets_module
mask2mesh = _mask2mesh_module
solver_base = _solver_base_module
solver_global_cg = _solver_global_cg_module
solver_local_gn = _solver_local_gn_module
solver_translation_zncc = _solver_translation_zncc_module
strain = _strain_module
discrepancy = _discrepancy_module

__all__ = [
    "Array",
    "InitMotionConfig",
    "MeshDICConfig",
    "DICResult",
    "MeshDICResult",
    "DICDiagnostics",
    "BatchConfig",
    "BatchResult",
    "BatchDiagnostics",
    "DICBase",
    "Mesh",
    "MeshAssets",
    "MeshTopology",
    "NeighborTable",
    "WeightedNeighborTable",
    "NodePixelGather",
    "PixelSampling",
    "compute_element_centers",
    "downsample_mask",
    "mask_to_mesh",
    "mask_to_mesh_assets",
    "mask_to_mesh_gmsh",
    "mask_to_mesh_assets_gmsh",
    "compute_green_lagrange_strain_nodes_lsq",
    "green_lagrange_to_voigt",
    "small_strain_nodes_lsq",
    "DICInitMotion",
    "DICMeshBased",
    "BatchBase",
    "BatchMeshBased",
    "DisplacementPropagatorBase",
    "PreviousDisplacementPropagator",
    "ConstantVelocityPropagator",
    "TranslationZNCCSolver",
    "GlobalCGSolver",
    "LocalGaussNewtonSolver",
    "DICPlotter",
    "bin_image",
    "bin_mask",
    "configure_jax_platform",
    "list_deformed_images",
    "prepare_image",
    "imread_gray",
    "downsample_image",
    "make_roi_mask",
    "SequenceConfig",
    "load_sequence_config",
    "run_sequence_from_config",
    "write_dic_vtu",
    "write_image_vti",
    "write_pvd",
    "AbaqusInterpolationMap",
    "abaqus_displacement_to_image_px",
    "build_abaqus_interpolation_map",
    "compare_abaqus_dic_sequence",
    "connected_node_mask",
    "image_nodes_to_abaqus_mm",
    "image_time_from_name",
    "interpolate_abaqus_space",
    "interpolate_abaqus_time",
]
