from __future__ import annotations

from dataclasses import replace
import inspect

import jax
import jax.numpy as jnp
import numpy as np

from d2ic.dataclasses import MeshDICConfig
from d2ic.dic_mesh_based import DICMeshBased
from d2ic.mesh.assets import Mesh, PixelSampling, make_mesh_assets
from d2ic.mesh.pixel_sampling import build_pixel_assets
from d2ic.numerics.global_cg import GlobalCGSolver


def _unit_quad() -> Mesh:
    nodes = jnp.asarray(
        [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
        dtype=jnp.float32,
    )
    return Mesh(nodes_xy=nodes, elements=jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32))


def test_mesh_assets_separate_strain_and_regularization_topologies() -> None:
    mesh = _unit_quad()
    assets = make_mesh_assets(mesh)
    sampling = build_pixel_assets(mesh, np.zeros((5, 5), dtype=np.float32), binning=1.0)
    assets = replace(assets, pixel_data=sampling)

    strain_graph = assets.topology.strain_neighbors
    regularization_graph = sampling.regularization_graph

    assert strain_graph is not None
    assert strain_graph.index.shape[1] == 16
    assert regularization_graph.index.shape[1] == 3
    assert np.all(np.asarray(strain_graph.index)[:, 3:] == -1)
    assert regularization_graph.weight.shape == regularization_graph.index.shape


def test_mesh_data_are_jax_pytrees() -> None:
    mesh = _unit_quad()
    sampling = build_pixel_assets(mesh, np.zeros((5, 5), dtype=np.float32), binning=1.0)
    assets = replace(make_mesh_assets(mesh), pixel_data=sampling)

    moved = jax.device_put(assets)
    assert isinstance(moved.pixel_data, PixelSampling)
    centers = jax.jit(lambda value: value.topology.element_centers_xy)(moved)
    np.testing.assert_allclose(np.asarray(centers), [[2.0, 2.0]])


def test_dic_pipeline_has_no_independent_mesh_constructor_argument() -> None:
    parameters = inspect.signature(DICMeshBased).parameters
    assert "mesh" not in parameters

    pipeline = DICMeshBased(solver=GlobalCGSolver(), config=MeshDICConfig())
    assert not hasattr(pipeline, "mesh")
