import jax
import jax.numpy as jnp
import numpy as np

from d2ic.identification import gaussian_virtual_fields, weak_equilibrium_mse


def test_gaussian_virtual_fields_vanish_on_excluded_nodes() -> None:
    nodes = np.asarray([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [.5, .5]])
    fields = gaussian_virtual_fields(nodes, np.asarray([0, 1]), centers_x=2, centers_y=2)
    assert fields.shape == (8, 5, 2)
    np.testing.assert_allclose(fields[:, :2], 0.0)
    np.testing.assert_allclose(np.linalg.norm(fields, axis=(1, 2)), 1.0)


def test_weak_equilibrium_loss_is_jit_and_differentiable() -> None:
    fields = jnp.asarray([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]]])

    @jax.jit
    def loss(force):
        return weak_equilibrium_mse(force, fields, jnp.asarray(2.0))

    force = jnp.asarray([[2., 0.], [0., 4.]])
    np.testing.assert_allclose(loss(force), 2.5)
    gradient = jax.grad(loss)(force)
    assert gradient.shape == force.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
