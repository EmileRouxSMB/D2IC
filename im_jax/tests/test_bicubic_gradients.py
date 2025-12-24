import jax
import jax.numpy as jnp

from im_jax import flat_nd_cubic_interpolate


def test_gradients_wrt_image_and_locations():
    key = jax.random.PRNGKey(42)
    image = jax.random.normal(key, (8, 7), dtype=jnp.float64)
    key, sub = jax.random.split(key)
    locations = jax.random.uniform(sub, (2, 10), minval=0.0, maxval=6.0, dtype=jnp.float64)

    def loss_img(img):
        return jnp.sum(flat_nd_cubic_interpolate(img, locations))

    grad_img = jax.grad(loss_img)(image)
    assert jnp.all(jnp.isfinite(grad_img))

    def loss_loc(loc):
        return jnp.sum(flat_nd_cubic_interpolate(image, loc))

    grad_loc = jax.grad(loss_loc)(locations)
    assert jnp.all(jnp.isfinite(grad_loc))

    eps = 1e-5
    for i, j in [(0, 0), (1, 1), (0, 2)]:
        shift = jnp.zeros_like(locations).at[i, j].set(eps)
        fd = (loss_loc(locations + shift) - loss_loc(locations - shift)) / (2 * eps)
        assert jnp.allclose(grad_loc[i, j], fd, rtol=1e-3, atol=1e-3)
