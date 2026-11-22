"""Differentiable local integration of anisotropic metallic constitutive laws."""

from .corotational import integrate_corotational_plane_stress, integrate_history
from .elasticity import isotropic_stiffness
from .hardening import flow_stress
from .hill48 import hill48_equivalent_stress, hill48_matrix
from .plane_stress import integrate_plane_stress_increment
from .return_mapping import integrate_hill48_increment
from .state import Hill48Material, MaterialState, initial_material_state

__all__ = [
    "Hill48Material", "MaterialState", "initial_material_state",
    "isotropic_stiffness", "flow_stress", "hill48_matrix",
    "hill48_equivalent_stress", "integrate_hill48_increment",
    "integrate_plane_stress_increment", "integrate_corotational_plane_stress",
    "integrate_history",
]
