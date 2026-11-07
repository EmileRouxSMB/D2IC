"""JAX solvers for initialization and mesh-based DIC identification."""

from .base import MeshSolverState, SolverBase, SolverState
from .adaptive_mechanical_gp import (
    assemble_linear_elastic_residual_q4, gp_inducing_prior, gp_log_variance,
    laplace_log_variance_uncertainty, mechanical_confidence, mechanical_residual_nll,
)
from .global_cg import GlobalCGResult, GlobalCGSolver
from .local_gauss_newton import LocalGNResult, LocalGaussNewtonSolver
from .mechanical_gp_solver import (
    MechanicalGPResult, MechanicalGPSolver, plane_stress_elasticity_tensor,
)
from .translation_zncc import TranslationZNCCResult, TranslationZNCCSolver

__all__ = [
    "GlobalCGResult", "GlobalCGSolver", "LocalGNResult", "LocalGaussNewtonSolver",
    "MeshSolverState", "SolverBase", "SolverState", "TranslationZNCCResult",
    "TranslationZNCCSolver", "assemble_linear_elastic_residual_q4",
    "gp_inducing_prior", "gp_log_variance", "laplace_log_variance_uncertainty",
    "mechanical_confidence", "mechanical_residual_nll",
    "MechanicalGPResult", "MechanicalGPSolver", "plane_stress_elasticity_tensor",
]
