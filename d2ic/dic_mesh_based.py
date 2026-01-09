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
from .discrepancy import compute_discrepancy_map_ref

try:  # pragma: no cover - optional dependency
    from .pixel_assets import build_pixel_assets
except Exception:  # pragma: no cover
    build_pixel_assets = None


@dataclass
class MeshDICState:
    """State for pure mesh-based DIC."""

    ref_image: Array
    nodes_xy_device: Array
    assets: MeshAssets
    config: MeshDICConfig
    u0_nodal: Array | None = None


class DICMeshBased(DICBase):
    """
    Mesh-based DIC pipeline.

    This class orchestrates:
    - asset preparation (neighbors and pixel-level caches),
    - solver compilation/warmup,
    - per-frame solving through the injected `SolverBase` implementation,
    - optional strain post-processing from nodal displacements.
    """

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
                roi_mask=getattr(assets, "roi_mask", None),
            )
            assets = replace(assets, pixel_data=pixel_data)

        ref_image_dev = jnp.asarray(ref_image)
        nodes_xy_dev = jnp.asarray(assets.mesh.nodes_xy)
        self._state = MeshDICState(
            ref_image=ref_image_dev,
            nodes_xy_device=nodes_xy_dev,
            assets=assets,
            config=self.config,
            u0_nodal=None,
        )
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
        diag_info = {"stage": "mesh_based"}
        pixel_maps = {}
        if getattr(sol, "n_iters", None) is not None:
            diag_info["n_iters"] = int(sol.n_iters)
        if getattr(sol, "history", None) is not None:
            diag_info["history"] = "attached"

        has_neighbors = assets.node_neighbor_index is not None and assets.node_neighbor_degree is not None
        if has_neighbors:
            try:
                _, E_all = compute_green_lagrange_strain_nodes_lsq(
                    displacement=u_nodal,
                    nodes_coord=self._state.nodes_xy_device,
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

        if bool(getattr(self.config, "compute_discrepancy_map", False)):
            if assets.pixel_data is None:
                diag_info["discrepancy"] = "skipped_no_pixel_assets"
            else:
                disc_map, disc_rms = compute_discrepancy_map_ref(
                    ref_image=self._state.ref_image,
                    def_image=def_image,
                    u_nodal=u_nodal,
                    pixel_assets=assets.pixel_data,
                )
                # Keep the map on the current JAX backend; converting to NumPy here
                # forces a device->host sync and can dominate per-frame time on GPU.
                pixel_maps["discrepancy_ref"] = jnp.asarray(disc_map, dtype=jnp.float32)
                diag_info["discrepancy"] = "discrepancy_ref"
                diag_info["discrepancy_rms"] = float(disc_rms)

        diag = DICDiagnostics(info=diag_info)
        history = getattr(sol, "history", None)
        return DICResult(u_nodal=u_nodal, strain=strain, diagnostics=diag, history=history, pixel_maps=pixel_maps)
