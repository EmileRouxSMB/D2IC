from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import jax.numpy as jnp

from .dic_base import DICBase
from .mesh_assets import MeshAssets
from .dataclasses import MeshDICConfig, DICResult, DICDiagnostics
from .types import Array

try:  # pragma: no cover - import availability depends on repo layout
    from ._legacy.dic import Dic as _LegacyDic
except Exception:  # pragma: no cover
    _LegacyDic = None


@dataclass
class _LegacyState:
    ref_image: Array
    assets: MeshAssets
    legacy: object
    u0_nodal: Optional[Array] = None


class DICMeshBasedLegacy(DICBase):
    """
    Adapter around the previous ``Dic`` implementation so it fits the stage-1 API.

    Stage-1 responsibilities:
      - call ``precompute_pixel_data`` once during ``prepare``
      - forward ``run`` calls to the requested solver flavor (CG or local GN)
      - wrap the outputs inside ``DICResult`` with placeholder strain

    TODO(stage-2): progressively migrate the numerical kernels into the structured
    solvers introduced in the new architecture.
    """

    def __init__(self, mesh_path: str, solver_mode: str, config: MeshDICConfig) -> None:
        warnings.warn(
            "DICMeshBasedLegacy is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.mesh_path = mesh_path
        self.solver_mode = solver_mode
        self.config = config
        self._state: Optional[_LegacyState] = None

    def prepare(self, ref_image: Array, assets: MeshAssets) -> None:
        if _LegacyDic is None:
            raise RuntimeError("legacy Dic implementation is not available for import")
        legacy = _LegacyDic(mesh_path=self.mesh_path)
        ref_proc = self._apply_binning(ref_image, legacy.binning)
        legacy.precompute_pixel_data(ref_proc)
        self._state = _LegacyState(ref_image=ref_image, assets=assets, legacy=legacy)

    def set_initial_guess(self, u_nodal: Array) -> None:
        if self._state is None:
            raise RuntimeError("prepare() must be called before set_initial_guess().")
        self._state.u0_nodal = u_nodal

    def run(self, def_image: Array) -> DICResult:
        if self._state is None:
            raise RuntimeError("DICMeshBasedLegacy.prepare() must be called before run().")

        legacy = self._state.legacy
        ref_proc = self._apply_binning(self._state.ref_image, legacy.binning)
        def_proc = self._apply_binning(def_image, legacy.binning)
        disp_init = self._state.u0_nodal

        if disp_init is not None:
            disp_init = jnp.asarray(disp_init)

        if self.solver_mode == "cg_global":
            disp_sol, history = legacy.run_dic(
                ref_proc,
                def_proc,
                disp_guess=disp_init,
                max_iter=self.config.max_iters,
                tol=self.config.tol,
                alpha_reg=self.config.reg_strength,
                save_history=self.config.save_history,
            )
            diagnostics = {
                "solver": "cg_global",
                "history_available": bool(history),
            }
        elif self.solver_mode == "local_gn":
            disp_sol = legacy.run_dic_nodal(
                ref_proc,
                def_proc,
                disp_init=disp_init,
                n_sweeps=self.config.max_iters,
                alpha_reg=self.config.reg_strength,
            )
            diagnostics = {"solver": "local_gn"}
        else:
            raise ValueError(f"Unsupported solver_mode '{self.solver_mode}'.")

        u_nodal = jnp.asarray(disp_sol)
        strain = jnp.zeros_like(u_nodal)
        diag = DICDiagnostics(info={
            "stage": "legacy_adapter",
            "solver_mode": self.solver_mode,
            **diagnostics,
        })
        out_history = None
        if self.solver_mode == "cg_global" and self.config.save_history:
            out_history = jnp.asarray(history) if history is not None else None
        return DICResult(u_nodal=u_nodal, strain=strain, diagnostics=diag, history=out_history)

    @staticmethod
    def _apply_binning(image: Array, binning: float | int) -> Array:
        """
        Downsample ``image`` by the binning factor used by the previous solver.

        TODO(stage-2): replace with the real binning/averaging logic from the
        prior implementation.
        """
        b = int(round(float(binning))) if binning else 1
        if b <= 1:
            return image
        return image[::b, ::b]
