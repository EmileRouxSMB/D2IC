from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .dic_base import DICBase
from .mesh_assets import MeshAssets
from .dataclasses import InitMotionConfig, DICResult, DICDiagnostics
from .types import Array
from .solver_base import SolverBase


@dataclass
class InitMotionState:
    """
    State for coarse motion estimation.
    """
    ref_image: Array
    assets: MeshAssets


class DICInitMotion(DICBase):
    """
    Coarse initialization using translation-only ZNCC matches at element centers.

    Stage-2 placeholder: extrapolate the sparse center displacements to nodal
    values via inverse-distance weighting (IDW). Later revisions can swap in
    RBF or least-squares projections.
    """

    def __init__(self, config: InitMotionConfig, solver: SolverBase) -> None:
        self.config = config
        self.solver = solver
        self._state: InitMotionState | None = None

    def prepare(self, ref_image: Array, assets: MeshAssets) -> None:
        self._state = InitMotionState(ref_image=ref_image, assets=assets)
        self.solver.compile(assets)

    def run(self, def_image: Array) -> DICResult:
        if self._state is None:
            raise RuntimeError("DICInitMotion.prepare() must be called before run().")

        sol = self.solver.solve(self._state, def_image)

        u_nodal = _project_centers_to_nodes_idw(
            self._state.assets,
            self._state.assets.element_centers_xy,
            sol.u_centers,
        )

        strain = np.zeros_like(u_nodal)
        diag = DICDiagnostics(
            info={
                "stage": "init_motion",
                "matches": int(np.sum(~np.isnan(sol.scores))),
                "note": "IDW projection placeholder",
            }
        )
        return DICResult(u_nodal=u_nodal, strain=strain, diagnostics=diag, history=None)


def _project_centers_to_nodes_idw(
    assets: MeshAssets,
    centers_xy: Array,
    disp_centers: Array,
    power: float = 2.0,
) -> Array:
    nodes_xy = np.asarray(assets.mesh.nodes_xy)
    centers = np.asarray(centers_xy)
    disp = np.asarray(disp_centers)

    valid = ~np.isnan(disp).any(axis=1)
    centers = centers[valid]
    disp = disp[valid]

    if centers.size == 0:
        return np.zeros_like(nodes_xy)

    out = np.zeros_like(nodes_xy)
    for idx, node in enumerate(nodes_xy):
        dist = np.linalg.norm(centers - node, axis=1)
        close = dist < 1e-6
        if np.any(close):
            out[idx] = disp[close][0]
            continue
        w = 1.0 / np.power(dist + 1e-12, power)
        w /= w.sum()
        out[idx] = (w[:, None] * disp).sum(axis=0)
    return out
