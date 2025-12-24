import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from im_jax import flat_nd_cubic_interpolate


def _run_reference_test(dtype, atol):
    key = jax.random.PRNGKey(0)
    image = jax.random.normal(key, (64, 48), dtype=dtype)
    key, sub = jax.random.split(key)
    x = jax.random.uniform(sub, (1, 256), minval=0.0, maxval=63.0, dtype=dtype)
    key, sub = jax.random.split(key)
    y = jax.random.uniform(sub, (1, 256), minval=0.0, maxval=47.0, dtype=dtype)
    locations = jnp.concatenate([x, y], axis=0)

    out = flat_nd_cubic_interpolate(image, locations, mode="reflect", a=-0.5)
    ref = map_coordinates(
        image, locations, order=3, mode="reflect", cval=0.0
    )
    max_err = jnp.max(jnp.abs(out - ref))
    assert max_err < atol


def test_reference_float64():
    _run_reference_test(jnp.float64, atol=1e-6)


def test_reference_float32():
    _run_reference_test(jnp.float32, atol=1e-3)
