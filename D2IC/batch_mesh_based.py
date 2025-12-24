from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np

from .batch_base import BatchBase
from .dic_init_motion import DICInitMotion
from .dic_mesh_based import DICMeshBased
from .mesh_assets import MeshAssets
from .types import Array
from .dataclasses import BatchConfig, BatchResult, BatchDiagnostics, DICResult
from .propagator_base import DisplacementPropagatorBase


@dataclass
class BatchState:
    ref_image: Array
    assets: MeshAssets
    is_prepared: bool = False


class BatchMeshBased(BatchBase):
    """
    Concrete batch runner for mesh-based DIC.

    Stage-1:
    - Uses injected DICInitMotion and DICMeshBased instances.
    - before(): prepares both pipelines with ref_image/assets
    - sequence(): runs per-frame DICMeshBased.run(def_image), optionally using init motion
    - end(): placeholder post-processing

    Notes:
    - This class assumes ref_image and assets are fixed for the whole batch.
    """

    def __init__(
        self,
        ref_image: Array,
        assets: MeshAssets,
        dic_mesh: DICMeshBased,
        batch_config: BatchConfig,
        dic_init: Optional[DICInitMotion] = None,
        propagator: Optional[DisplacementPropagatorBase] = None,
    ) -> None:
        super().__init__()
        self.ref_image = ref_image
        self.assets = assets
        self.dic_mesh = dic_mesh
        self.dic_init = dic_init
        self.config = batch_config
        self.propagator = propagator
        self._state = BatchState(ref_image=ref_image, assets=assets, is_prepared=False)

    def before(self, images: Sequence[Array]) -> None:
        # Prepare mesh-based DIC pipeline
        self.dic_mesh.prepare(self.ref_image, self.assets)

        # Prepare init motion pipeline if enabled and provided
        if self.config.use_init_motion and self.dic_init is not None:
            self.dic_init.prepare(self.ref_image, self.assets)

        self._state.is_prepared = True

    def sequence(self, images: Sequence[Array]) -> BatchResult:
        if not self._state.is_prepared:
            raise RuntimeError("BatchMeshBased.before() must be called before sequence().")

        per_frame: list[DICResult] = []
        u_prev = None
        u_prevprev = None
        use_init_each = self.config.init_motion_every_frame
        use_init_first_only = self.config.init_motion_first_frame_only
        if use_init_each and use_init_first_only:
            raise ValueError("BatchConfig cannot enable both init_motion_every_frame and init_motion_first_frame_only.")
        prefer_init = self.config.prefer_init_motion_over_propagation

        for k, Idef in enumerate(images):
            use_init = False
            if self.dic_init is not None and self.config.use_init_motion:
                if use_init_each:
                    use_init = True
                elif use_init_first_only and k == 0:
                    use_init = True
                elif prefer_init:
                    use_init = True

            if use_init and self.dic_init is not None:
                init_res = self.dic_init.run(Idef)
                self.dic_mesh.set_initial_guess(init_res.u_nodal)
                init_disp = init_res.u_nodal
            else:
                if self.propagator is not None:
                    u_warm = self.propagator.propagate(u_prev=u_prev, u_prevprev=u_prevprev)
                else:
                    u_warm = u_prev if self.config.warm_start_from_previous else None

                if u_warm is not None:
                    self.dic_mesh.set_initial_guess(u_warm)
                init_disp = u_warm

            res = self.dic_mesh.run(Idef)
            per_frame.append(res)

            # Update warm-start history for next frame
            u_prevprev = u_prev
            u_prev = np.asarray(res.u_nodal)

        diag = BatchDiagnostics(
            info={
                "stage": "batch_mesh_based",
                "n_frames": len(per_frame),
                "use_init_motion": self.config.use_init_motion,
                "warm_start_from_previous": self.config.warm_start_from_previous,
                "note": "stage-1 placeholder",
            }
        )
        return BatchResult(results=per_frame, diagnostics=diag)

    def end(self, result: BatchResult) -> BatchResult:
        # Stage-1 placeholder for post-processing (e.g. stacking, saving, summary stats).
        # Keep as identity; update diagnostics as needed.
        updated = BatchDiagnostics(info={**result.diagnostics.info, "post": "end() placeholder"})
        return BatchResult(results=result.results, diagnostics=updated)
