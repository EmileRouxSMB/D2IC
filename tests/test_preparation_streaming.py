"""Geometry preparation and bounded-memory sequence outputs."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from d2ic.core.dataclasses import MeshDICConfig
from d2ic.mesh.assets import Mesh, make_mesh_assets
from d2ic.mesh.pixel_sampling import (
    build_node_pixel_dense,
    build_pixel_assets,
    build_pixel_to_element_mapping_numpy,
)
from d2ic.methods.mesh_based import DICMeshBased
from d2ic.numerics.local_gauss_newton import LocalGaussNewtonSolver
from d2ic.pipelines.sequence_platehole import _stack_frame_npz


def test_spatial_mapping_preserves_first_containing_element():
    x, y = np.meshgrid(np.arange(7), np.arange(7))
    nodes = np.stack((x.ravel(), y.ravel()), axis=1).astype(float)
    elements = []
    for row in range(6):
        for col in range(6):
            a = row * 7 + col
            elements.append([a, a + 1, a + 8, a + 7])
    elements = np.asarray(elements, dtype=np.int32)
    elements = np.vstack((elements[:1], elements))  # overlap selects first index
    rng = np.random.default_rng(4)
    points = rng.uniform(-0.2, 6.2, size=(300, 2))
    points = np.vstack((points, [[0.5, 0.5], [1.0, 0.5], [6.1, 6.1]]))

    actual, pixel_nodes = build_pixel_to_element_mapping_numpy(
        points, nodes, elements, chunk_size=7
    )
    expected = np.full(len(points), -1, dtype=np.int32)
    for elt, connectivity in enumerate(elements):
        quad = nodes[connectivity]
        edges = np.roll(quad, -1, axis=0) - quad
        vectors = points[:, None, :] - quad[None, :, :]
        cross = edges[None, :, 0] * vectors[:, :, 1] - edges[None, :, 1] * vectors[:, :, 0]
        inside = np.all(cross >= -1e-6, axis=1)
        inside &= np.all((points >= quad.min(axis=0)) & (points <= quad.max(axis=0)), axis=1)
        expected[(expected < 0) & inside] = elt

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(pixel_nodes[actual >= 0], elements[actual[actual >= 0]])
    assert actual[-3] == 0
    assert actual[-1] == -1

    reversed_elements = elements[:1, ::-1]
    reversed_ids, _ = build_pixel_to_element_mapping_numpy(
        points[-3:], nodes, reversed_elements, chunk_size=1
    )
    np.testing.assert_array_equal(reversed_ids, [0, 0, -1])


def test_node_pixel_gather_is_optional_and_stable():
    mesh = Mesh(
        nodes_xy=jnp.asarray([[0., 0.], [4., 0.], [4., 4.], [0., 4.]]),
        elements=jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32),
    )
    image = np.zeros((5, 5), dtype=np.float32)
    sampling = build_pixel_assets(mesh, image, binning=1)
    assert sampling.node_pixel_gather is None

    with_gather = build_pixel_assets(mesh, image, binning=1, with_node_pixel_gather=True)
    assert with_gather.node_pixel_gather is not None
    nodes = np.asarray(with_gather.pixel_nodes)
    weights = np.asarray(with_gather.pixel_shapeN)
    index, shape_weight, degree = build_node_pixel_dense(nodes, weights, 4)
    for node in range(4):
        expected = [(p, weights[p, local]) for p in range(len(nodes))
                    for local in range(4) if nodes[p, local] == node]
        assert int(degree[node]) == len(expected)
        np.testing.assert_array_equal(index[node, :len(expected)], [p for p, _ in expected])
        np.testing.assert_allclose(shape_weight[node, :len(expected)], [w for _, w in expected])

    local = DICMeshBased(
        LocalGaussNewtonSolver(), MeshDICConfig(max_iters=1, regularization="spring")
    )
    local.prepare(image, make_mesh_assets(mesh))
    assert local.prepared_assets.pixel_data.node_pixel_gather is not None


def test_global_fields_reassemble_from_per_frame_npz(tmp_path: Path):
    frame_paths = []
    for k in range(3):
        path = tmp_path / f"frame_{k:04d}.npz"
        np.savez_compressed(
            path,
            u_nodal=np.full((4, 2), k, dtype=np.float32),
            strain=np.full((4, 3), k + 1, dtype=np.float32),
            discrepancy_ref=np.full((5, 6), k + 2, dtype=np.float32),
        )
        frame_paths.append(path)
    stacked = _stack_frame_npz(frame_paths, tmp_path)
    assert all(isinstance(array, np.memmap) for array in stacked)
    out = tmp_path / "sequence.npz"
    np.savez_compressed(
        out, u_nodal=stacked[0], strain=stacked[1], discrepancy_ref=stacked[2]
    )
    with np.load(out) as archive:
        np.testing.assert_array_equal(archive["u_nodal"][:, 0, 0], [0, 1, 2])
        np.testing.assert_array_equal(archive["strain"][:, 0, 0], [1, 2, 3])
        np.testing.assert_array_equal(archive["discrepancy_ref"][:, 0, 0], [2, 3, 4])
