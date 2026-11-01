from __future__ import annotations

import numpy as np
import pytest

import d2ic.mask2mesh as mask2mesh_module

from d2ic.mesh.generation import mask_to_mesh, _mesh_from_q4_connectivity


def test_gmsh_import_failure_is_fatal(monkeypatch) -> None:
    import_error = ModuleNotFoundError("No module named gmsh")
    monkeypatch.setattr(mask2mesh_module, "gmsh", None)
    monkeypatch.setattr(mask2mesh_module, "_GMSH_IMPORT_ERROR", import_error)

    with pytest.raises(RuntimeError, match="Gmsh Python API is unavailable") as caught:
        mask2mesh_module.mask_to_mesh_gmsh(
            np.ones((8, 8), dtype=bool), element_size_px=2.0
        )

    assert caught.value.__cause__ is import_error


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
    assert np.unique(np.asarray(mesh.elements)).size == len(mesh.nodes_xy)

    centers = (mesh.nodes_xy[mesh.elements].mean(axis=1)).astype(int)
    assert not np.any((centers[:, 0] == 2) & (centers[:, 1] == 2))


def test_compaction_preserves_quadrilateral_geometry() -> None:
    nodes = np.array([[99., 99.], [0., 0.], [2., 0.],
                      [2., 2.], [0., 2.]], dtype=np.float32)
    elements = np.array([[1, 2, 3, 4]], dtype=np.int32)

    mesh = _mesh_from_q4_connectivity(nodes, elements)

    assert len(mesh.nodes_xy) == 4
    np.testing.assert_array_equal(np.asarray(mesh.nodes_xy)[np.asarray(mesh.elements)],
                                  nodes[elements])
    np.testing.assert_array_equal(np.unique(np.asarray(mesh.elements)), np.arange(4))
