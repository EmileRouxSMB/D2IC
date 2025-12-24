from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from d2ic import mask_to_mesh_assets_gmsh


def test_mask_to_mesh_platehole_gmsh() -> None:
    pytest.importorskip("gmsh")
    pytest.importorskip("meshio")
    root = Path(__file__).resolve().parents[1]
    mask_path = root / "doc" / "img" / "PlateHole" / "roi.tif"
    mask_rgba = imageio.imread(mask_path)
    mask_gray = mask_rgba[..., :3].mean(axis=2).astype(np.float32)
    mask_bool = mask_gray > 0.5

    mesh, assets = mask_to_mesh_assets_gmsh(
        mask=mask_bool,
        element_size_px=20.0,
        binning=1,
        remove_islands=True,
        min_island_area_px=64,
    )

    assert mesh.elements.shape[0] > 0
    assert mesh.nodes_xy.shape[0] > 0
    assert assets.mesh.nodes_xy.shape == mesh.nodes_xy.shape
