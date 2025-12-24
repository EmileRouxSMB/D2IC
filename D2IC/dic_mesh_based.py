from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import jax.numpy as jnp

from .dic_base import DICBase
from .mesh_assets import (
    Mesh,
    MeshAssets,
    build_node_neighbor_tables,
    make_mesh_assets,
)
from .dataclasses import MeshDICConfig, DICResult, DICDiagnostics
from .types import Array
from .solver_base import SolverBase
from .strain import (
    compute_green_lagrange_strain_nodes_lsq,
    green_lagrange_to_voigt,
)

try:  # pragma: no cover - optional dependency until fully migrated
    from .pixel_assets import build_pixel_assets
except Exception:  # pragma: no cover
    build_pixel_assets = None


@dataclass
class MeshDICState:
    """State for pure mesh-based DIC."""

    ref_image: Array
    assets: MeshAssets
    config: MeshDICConfig
    u0_nodal: Array | None = None


class DICMeshBased(DICBase):
    """Pure DIC pipeline using the new solver stack."""

    def __init__(self, mesh: Mesh, solver: SolverBase, config: MeshDICConfig) -> None:
        self.mesh = mesh
        self.solver = solver
        self.config = config
        self._state: MeshDICState | None = None

    def prepare(self, ref_image: Array, assets: Optional[MeshAssets] = None) -> None:
        if assets is None:
            assets = make_mesh_assets(self.mesh, with_neighbors=True)
        else:
            need_neighbors = assets.node_neighbor_index is None or assets.node_neighbor_degree is None
            if need_neighbors:
                idx, deg = build_node_neighbor_tables(assets.mesh)
                assets = replace(
                    assets,
                    node_neighbor_index=idx,
                    node_neighbor_degree=deg,
                )

        if assets.pixel_data is None:
            if build_pixel_assets is None:
                raise RuntimeError("Pixel asset generator is unavailable; supply MeshAssets with pixel_data.")
            pixel_data = build_pixel_assets(
                mesh=assets.mesh,
                ref_image=ref_image,
                binning=1.0,
            )
            assets = replace(assets, pixel_data=pixel_data)

        self._state = MeshDICState(ref_image=ref_image, assets=assets, config=self.config, u0_nodal=None)
        self.solver.compile(assets)
        if hasattr(self.solver, "warmup"):
            self.solver.warmup(self._state)

    def set_initial_guess(self, u0_nodal: Array) -> None:
        if self._state is None:
            raise RuntimeError("Call prepare() before set_initial_guess().")
        self._state.u0_nodal = u0_nodal

    def run(self, def_image: Array) -> DICResult:
        if self._state is None:
            raise RuntimeError("DICMeshBased.prepare() must be called before run().")

        sol = self.solver.solve(self._state, def_image)

        u_nodal = sol.u_nodal
        strain = sol.strain
        assets = self._state.assets
        diag_info = {"stage": "mesh_based", "note": "stage-2 placeholder"}

        has_neighbors = assets.node_neighbor_index is not None and assets.node_neighbor_degree is not None
        if has_neighbors:
            try:
                _, E_all = compute_green_lagrange_strain_nodes_lsq(
                    displacement=u_nodal,
                    nodes_coord=assets.mesh.nodes_xy,
                    node_neighbor_index=assets.node_neighbor_index,
                    node_neighbor_degree=assets.node_neighbor_degree,
                    gauge_length=self.config.strain_gauge_length,
                    eps=self.config.strain_eps,
                )
                strain = green_lagrange_to_voigt(E_all)
                diag_info["strain"] = "green_lagrange"
            except Exception as exc:  # pragma: no cover - defensive guard
                strain = jnp.zeros((u_nodal.shape[0], 3), dtype=u_nodal.dtype)
                diag_info["strain"] = "failed_green_lagrange"
                diag_info["strain_error"] = str(exc)
        else:
            strain = jnp.zeros((u_nodal.shape[0], 3), dtype=u_nodal.dtype)
            diag_info["strain"] = "skipped_no_neighbors"

        diag = DICDiagnostics(info=diag_info)
        return DICResult(u_nodal=u_nodal, strain=strain, diagnostics=diag)
