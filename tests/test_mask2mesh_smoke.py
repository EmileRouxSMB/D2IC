from __future__ import annotations

import numpy as np

from d2ic.mask2mesh import mask_to_mesh


def test_mask_to_mesh_filters_small_islands() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:48, 10:50] = True
    mask[2, 2] = True  # tiny island

    mesh = mask_to_mesh(
        mask,
        element_size_px=8,
        binning=1,
        remove_islands=True,
        min_island_area_px=10,
    )

    assert mesh.nodes_xy.shape[1] == 2 and mesh.nodes_xy.shape[0] > 0
    assert mesh.elements.shape[1] == 4 and mesh.elements.shape[0] > 0

    centers = (mesh.nodes_xy[mesh.elements].mean(axis=1)).astype(int)
    assert not np.any((centers[:, 0] == 2) & (centers[:, 1] == 2))
