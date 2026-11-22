"""Physical checks for the default isotropic Q4 regularization."""

import jax
import jax.numpy as jnp
import numpy as np

from d2ic.numerics.elastic_regularization import elastic_energy, q4_reference_gradients


def test_rigid_motion_has_zero_energy_and_gradient():
    xy = jnp.array([[0., 0.], [90., 0.], [90., 90.], [0., 90.]])
    elements = jnp.array([[0, 1, 2, 3]])
    grad, weight = q4_reference_gradients(xy, elements)
    theta = 0.025
    rotation = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                          [jnp.sin(theta), jnp.cos(theta)]])
    displacement = xy @ rotation.T - xy + jnp.array([12., -7.])
    energy = jax.jit(elastic_energy)(displacement, elements, grad, weight)
    gradient = jax.jit(jax.grad(elastic_energy))(displacement, elements, grad, weight)
    assert float(energy) < 1e-9
    assert float(jnp.max(jnp.abs(gradient))) < 1e-6


def test_uniaxial_stretch_matches_isotropic_energy():
    xy = jnp.array([[0., 0.], [90., 0.], [90., 90.], [0., 90.]])
    elements = jnp.array([[0, 1, 2, 3]])
    grad, weight = q4_reference_gradients(xy, elements)
    displacement = jnp.stack((0.001 * xy[:, 0], jnp.zeros(4)), axis=1)
    strain = 0.001 + 0.5 * 0.001**2
    expected = 1.75 * strain**2
    np.testing.assert_allclose(float(elastic_energy(displacement, elements, grad, weight)), expected, rtol=1e-4)
