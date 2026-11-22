from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from d2ic.physics.discrepancy import compute_discrepancy_map_ref
from d2ic.mesh.assets import NodePixelGather, PixelSampling, WeightedNeighborTable
from d2ic.numerics.global_cg import J_pixelwise_core, _bilinear_sample as global_bilinear_sample, residuals_pixelwise_core
from d2ic.numerics.local_gauss_newton import _bilinear_sample as local_bilinear_sample, _pixel_state


def test_bilinear_sampling_rectangular_image() -> None:
    image = jnp.arange(15, dtype=jnp.float32).reshape(3, 5)
    image_T = image.T
    coords = jnp.asarray([[4.5, 1.5], [3.0, 1.5]], dtype=jnp.float32)
    expected = np.asarray([9.0, 7.5], dtype=np.float32)
    np.testing.assert_allclose(global_bilinear_sample(image_T, coords), expected)
    np.testing.assert_allclose(local_bilinear_sample(image_T, coords), expected)


def test_global_objective_excludes_out_of_image_pixels() -> None:
    image = jnp.arange(15, dtype=jnp.float32).reshape(3, 5)
    image_T = image.T
    coords = jnp.asarray([[4.5, 1.5], [0.5, 0.5]], dtype=jnp.float32)
    pixel_nodes = jnp.zeros((2, 1), dtype=jnp.int32)
    shape = jnp.ones((2, 1), dtype=jnp.float32)
    displacement = jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)
    residuals, valid = residuals_pixelwise_core(
        displacement, image_T, 2.0 * image_T, coords, pixel_nodes, shape,
        interpolation="linear",
    )
    np.testing.assert_array_equal(np.asarray(valid), [False, True])
    objective = J_pixelwise_core(
        displacement, image_T, 2.0 * image_T, coords, pixel_nodes, shape,
        interpolation="linear",
    )
    expected_valid_residual = float(residuals[1])
    np.testing.assert_allclose(objective, 0.5 * expected_valid_residual**2)
    assert np.isfinite(float(objective))


def test_local_state_zeroes_invalid_residual_and_gradient() -> None:
    image = jnp.arange(15, dtype=jnp.float32).reshape(3, 5)
    image_T = image.T
    coords = jnp.asarray([[4.5, 1.5]], dtype=jnp.float32)
    pixel_nodes = jnp.zeros((1, 1), dtype=jnp.int32)
    shape = jnp.ones((1, 1), dtype=jnp.float32)
    displacement = jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)
    residual, _, gx, gy, valid = _pixel_state(
        displacement, image_T, image_T, coords, pixel_nodes, shape,
        jnp.ones_like(image_T), jnp.ones_like(image_T), interpolation="linear",
    )
    np.testing.assert_array_equal(np.asarray(valid), [False])
    np.testing.assert_allclose(residual, 0.0)
    np.testing.assert_allclose(gx, 0.0)
    np.testing.assert_allclose(gy, 0.0)


def test_discrepancy_all_invalid_is_finite_and_device_resident() -> None:
    graph = WeightedNeighborTable(
        index=jnp.empty((1, 0), dtype=jnp.int32), degree=jnp.zeros((1,), dtype=jnp.int32),
        weight=jnp.empty((1, 0), dtype=jnp.float32), node_weight=jnp.ones((1,), dtype=jnp.float32),
    )
    gather = NodePixelGather(
        index=jnp.asarray([[0, 1]], dtype=jnp.int32), shape_weight=jnp.ones((1, 2), dtype=jnp.float32),
        degree=jnp.asarray([2], dtype=jnp.int32),
    )
    pixels = PixelSampling(
        pixel_coords_ref=jnp.asarray([[0.5, 0.5], [4.5, 2.5]], dtype=jnp.float32),
        pixel_nodes=jnp.zeros((2, 1), dtype=jnp.int32), pixel_shapeN=jnp.ones((2, 1), dtype=jnp.float32),
        regularization_graph=graph, node_pixel_gather=gather,
        roi_mask_flat=jnp.asarray([0, 14], dtype=jnp.int32), image_shape=(3, 5),
    )
    image = jnp.arange(15, dtype=jnp.float32).reshape(3, 5)
    discrepancy, rms = compute_discrepancy_map_ref(
        ref_image=image, def_image=image,
        u_nodal=jnp.asarray([[100.0, 0.0]], dtype=jnp.float32), pixel_assets=pixels,
    )
    assert isinstance(rms, jax.Array)
    assert float(rms) == 0.0
    assert np.isnan(np.asarray(discrepancy)).all()

