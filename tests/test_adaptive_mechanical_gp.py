import jax
import jax.numpy as jnp
import numpy as np

from d2ic.numerics.adaptive_mechanical_gp import (
    assemble_linear_elastic_residual_q4,
    gp_inducing_prior,
    gp_log_variance,
    laplace_log_variance_uncertainty,
    mechanical_confidence,
    mechanical_residual_nll,
)
from d2ic.numerics.elastic_regularization import q4_reference_gradients
from d2ic.numerics.mechanical_gp_solver import _q4_mesh_area


def _patch():
    nodes = jnp.asarray([[x, y] for y in range(3) for x in range(3)], dtype=jnp.float32)
    elements = jnp.asarray([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    grad, normalized = q4_reference_gradients(nodes, elements)
    weights = normalized * 4.0  # area of the 2 x 2 square, [length²]
    lame, mu = 2.0, 3.0
    eye = jnp.eye(2)
    C = lame * jnp.einsum("ij,kl->ijkl", eye, eye) + mu * (
        jnp.einsum("ik,jl->ijkl", eye, eye) + jnp.einsum("il,jk->ijkl", eye, eye)
    )
    active = jnp.zeros((9, 2), dtype=bool).at[4].set(True)
    return nodes, elements, grad, weights, C, active


def test_uniform_affine_patch_has_zero_interior_residual():
    nodes, elements, grad, weights, C, active = _patch()
    translation = jnp.broadcast_to(jnp.asarray([2.5, -1.0]), (9, 2))
    affine = nodes @ jnp.asarray([[0.01, 0.02], [-0.03, 0.04]]) + translation
    for u in (translation, affine):
        residual = assemble_linear_elastic_residual_q4(u, elements, grad, weights, C, active)
        np.testing.assert_allclose(residual, 0.0, atol=2e-6)
    assert _q4_mesh_area(np.asarray(nodes), np.asarray(elements)) == 4.0


def test_local_residual_increases_nll_and_normalization_penalizes_large_variance():
    residual = jnp.zeros((3, 2)).at[1, 0].set(2.0)
    mask = jnp.ones((3, 2), dtype=bool)
    g = jnp.zeros(3)
    base = mechanical_residual_nll(jnp.zeros_like(residual), g, mask, 1.0)
    defective = mechanical_residual_nll(residual, g, mask, 1.0)
    assert defective > base
    # For a fixed nonzero residual, likelihood tends to +infinity as g grows.
    low = mechanical_residual_nll(residual, jnp.full((3,), 3.0), mask, 1.0)
    high = mechanical_residual_nll(residual, jnp.full((3,), 10.0), mask, 1.0)
    assert high > low


def test_gp_kernels_jit_gradient_and_laplace_shapes():
    x = jnp.asarray([[0., 0.], [1., 0.], [0., 1.]])
    z = x[:2]
    values = jnp.asarray([0.1, -0.2])
    active = jnp.ones((3, 2), dtype=bool)
    residual = jnp.asarray([[0.3, 0.1], [0., 0.2], [0.1, 0.]])
    g = jax.jit(gp_log_variance)(x, z, values, 0.0, 1.0)
    assert g.shape == (3,)
    grad = jax.grad(lambda v: jnp.sum(gp_log_variance(x, z, v, 0.0, 1.0)))(values)
    assert np.all(np.isfinite(grad))
    prior = gp_inducing_prior(values, z, 0.0, 1.0)
    assert prior > 0
    confidence = mechanical_confidence(jnp.asarray([-100., 100.]))
    np.testing.assert_allclose(confidence, [np.exp(20), np.exp(-20)], rtol=1e-6)
    variance = jax.jit(laplace_log_variance_uncertainty)(
        residual, active, 1.0, x, z, values, 0.0, 1.0
    )
    assert variance.shape == (3,)
    assert np.all(np.asarray(variance) >= 0.0)
    constant = gp_log_variance(x, z, jnp.full((2,), 2.5), 2.5, 1.0)
    np.testing.assert_allclose(constant, 2.5, rtol=1e-6)
