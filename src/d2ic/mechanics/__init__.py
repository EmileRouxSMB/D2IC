"""Finite-element kinematics linking measured DIC fields to local materials."""

from .deformation_gradient import deformation_gradient_history_q4, q4_deformation_gradient
from .history import integrate_gauss_point_histories
from .equilibrium_constraints import active_equilibrium_dofs, masked_equilibrium_residual
from .equilibrium import (
    BoundaryEdges, assemble_internal_force_history_q4, assemble_internal_force_q4,
    equilibrium_residual, first_piola_from_cauchy, integrate_boundary_traction,
    integrate_boundary_traction_history, select_q4_boundary_face,
)
from .quadrature import q4_gauss_rule, q4_shape_gradients

__all__ = [
    "q4_gauss_rule", "q4_shape_gradients", "q4_deformation_gradient",
    "deformation_gradient_history_q4", "integrate_gauss_point_histories",
    "active_equilibrium_dofs", "masked_equilibrium_residual",
    "BoundaryEdges", "first_piola_from_cauchy", "assemble_internal_force_q4",
    "assemble_internal_force_history_q4", "equilibrium_residual",
    "select_q4_boundary_face", "integrate_boundary_traction",
    "integrate_boundary_traction_history",
]
