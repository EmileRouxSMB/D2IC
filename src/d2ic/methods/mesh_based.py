from __future__ import annotations

from dataclasses import dataclass, replace
import jax.numpy as jnp

from ..core.dic_base import DICBase
from ..mesh.assets import MeshAssets, PixelAssets, NodePixelGather, build_strain_neighbors
from ..core.dataclasses import MeshDICConfig, DICResult, MeshDICResult, DICDiagnostics
from ..core.types import Array
from ..numerics.base import SolverBase
from ..numerics.local_gauss_newton import LocalGaussNewtonSolver
from ..physics.strain import (
    compute_green_lagrange_strain_nodes_lsq,
    green_lagrange_to_voigt,
)
from ..physics.discrepancy import compute_discrepancy_map_ref

try:  # pragma: no cover - optional dependency
    from ..mesh.pixel_sampling import build_pixel_assets, build_node_pixel_dense
except Exception:  # pragma: no cover
    build_pixel_assets = None
    build_node_pixel_dense = None


def _pixel_assets_to_device(pix: PixelAssets, device) -> PixelAssets:
    """Move the complete pixel-sampling PyTree to ``device``."""
    import jax
    return jax.device_put(pix, device)


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

    def __init__(self, solver: SolverBase, config: MeshDICConfig) -> None:
        self.solver = solver
        self.config = config
        self._state: MeshDICState | None = None

    def prepare(self, ref_image: Array, assets: MeshAssets) -> None:
        if assets.topology.strain_neighbors is None:
            topology = replace(
                assets.topology,
                strain_neighbors=build_strain_neighbors(assets.mesh),
            )
            assets = replace(assets, topology=topology)

        # Decide the target device once; used for all device-resident arrays.
        ref_image_dev = jnp.asarray(ref_image)
        nodes_xy_dev = jnp.asarray(assets.mesh.nodes_xy)
        _dev_attr = getattr(nodes_xy_dev, "device", None)
        target_device = _dev_attr() if callable(_dev_attr) else _dev_attr

        needs_gather = (
            isinstance(self.solver, LocalGaussNewtonSolver)
            and self.config.regularization == "spring"
        )
        if assets.pixel_data is None:
            if build_pixel_assets is None:
                raise RuntimeError("Pixel asset generator is unavailable; supply MeshAssets with pixel_data.")
            # Pixel-assets construction mixes NumPy + small JAX kernels; doing it on GPU
            # can incur extra compilation and device overhead. Build on CPU when possible,
            # then transfer once to the target device for the actual solve.
            try:  # pragma: no cover - device selection depends on runtime
                import jax

                cpu_dev = next((d for d in jax.devices() if d.platform == "cpu"), None)
            except Exception:  # pragma: no cover
                cpu_dev = None

            if cpu_dev is not None:
                import jax

                with jax.default_device(cpu_dev):
                    pixel_data_cpu = build_pixel_assets(
                        mesh=assets.mesh,
                        ref_image=ref_image,
                        binning=1.0,
                        roi_mask=getattr(assets, "roi_mask", None),
                        with_node_pixel_gather=needs_gather,
                    )
                pixel_data = (
                    _pixel_assets_to_device(pixel_data_cpu, target_device)
                    if target_device is not None
                    else pixel_data_cpu
                )
            else:
                pixel_data = build_pixel_assets(
                    mesh=assets.mesh,
                    ref_image=ref_image,
                    binning=1.0,
                    roi_mask=getattr(assets, "roi_mask", None),
                    with_node_pixel_gather=needs_gather,
                )
            assets = replace(assets, pixel_data=pixel_data)
        elif needs_gather and assets.pixel_data.node_pixel_gather is None:
            if build_node_pixel_dense is None:
                raise RuntimeError("Node-pixel gather builder is unavailable.")
            pix = assets.pixel_data
            index, weight, degree = build_node_pixel_dense(
                pix.pixel_nodes, pix.pixel_shapeN, assets.mesh.nodes_xy.shape[0]
            )
            pix = replace(
                pix, node_pixel_gather=NodePixelGather(index, weight, degree)
            )
            assets = replace(assets, pixel_data=pix)

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

    @property
    def prepared_assets(self) -> MeshAssets:
        """Assets reused by the prepared solver and subsequent pipeline stages."""
        if self._state is None:
            raise RuntimeError("Call prepare() before accessing prepared_assets.")
        return self._state.assets

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
            diag_info["n_iters"] = sol.n_iters
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
                diag_info["discrepancy_rms"] = disc_rms

        diag = DICDiagnostics(info=diag_info)
        history = getattr(sol, "history", None)
        return MeshDICResult(
            displacement=u_nodal,
            strain=strain,
            diagnostics=diag,
            mesh=assets.mesh,
            history=history,
            pixel_maps=pixel_maps,
        )
