"""Photometric residual diagnostics and nodal strain reconstruction."""

from .discrepancy import bilinear_sample_jax, bilinear_sample_numpy, compute_discrepancy_map_ref
from .strain import (
    compute_green_lagrange_strain_nodes_lsq,
    green_lagrange_to_voigt,
    small_strain_nodes_lsq,
)

__all__ = [
    "bilinear_sample_jax", "bilinear_sample_numpy", "compute_discrepancy_map_ref",
    "compute_green_lagrange_strain_nodes_lsq", "green_lagrange_to_voigt",
    "small_strain_nodes_lsq",
]
