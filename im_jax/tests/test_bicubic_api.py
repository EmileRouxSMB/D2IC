import jax.numpy as jnp
import pytest

from im_jax import flat_nd_cubic_interpolate


def test_flat_nd_cubic_interpolate_shapes_and_dtypes():
    image = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    locations = jnp.array(
        [[0.1, 1.2, 2.0, 0.0, 1.5], [0.2, 2.4, 1.0, 3.0, 0.5]],
        dtype=jnp.float32,
    )
    out = flat_nd_cubic_interpolate(image, locations)
    assert out.shape == (locations.shape[1],)
    assert out.dtype == jnp.result_type(image, locations)

    image_c = jnp.stack([image, image + 1.0], axis=0)
    out_c = flat_nd_cubic_interpolate(image_c, locations)
    assert out_c.shape == (image_c.shape[0], locations.shape[1])
    assert out_c.dtype == jnp.result_type(image_c, locations)


@pytest.mark.parametrize("mode", ["reflect", "nearest", "wrap", "constant"])
def test_flat_nd_cubic_interpolate_modes(mode):
    image = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
    locations = jnp.array([[0.2, 1.8], [0.5, 2.2]], dtype=jnp.float32)
    out = flat_nd_cubic_interpolate(image, locations, mode=mode, cval=2.0)
    assert out.shape == (2,)
