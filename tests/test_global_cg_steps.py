"""Photometric CG must remain stable for 16-bit image intensities."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from d2ic.numerics.global_cg import _cg_solve, J_pixelwise_core


@pytest.mark.parametrize("intensity", [255.0, 65535.0])
def test_translation_with_large_intensities(intensity):
    """Recover a known subpixel translation using a single translation DOF pair."""
    x, y = jnp.meshgrid(jnp.arange(40) + 0.5, jnp.arange(36) + 0.5, indexing="ij")
    def texture(x, y):
        return intensity * (0.5 + 0.2 * jnp.sin(x * 0.3) + 0.2 * jnp.cos(y * 0.4))
    target = jnp.array([0.6, -0.35])
    ref = texture(x, y)
    deformed = texture(x - target[0], y - target[1])
    coords = jnp.stack((x[5:-5, 5:-5].ravel(), y[5:-5, 5:-5].ravel()), axis=1)
    nodes = jnp.zeros((len(coords), 1), dtype=jnp.int32)
    weights = jnp.ones((len(coords), 1))
    args = (ref, deformed, coords, nodes, weights,
            jnp.zeros((1, 1), dtype=jnp.int32), jnp.zeros(1, dtype=jnp.int32),
            jnp.zeros((1, 1)), jnp.ones(1), 0.0, 100, 1e-3, "linear", False, False)
    solve = jax.jit(lambda u: _cg_solve(u, *args))
    displacement, history, _ = solve(jnp.zeros((1, 2)))
    np.testing.assert_allclose(displacement[0], target, atol=0.025)
    energies = np.asarray(history)[:, 0]
    energies = energies[np.isfinite(energies)]
    assert np.all(np.diff(energies) <= 1e-6 * np.maximum(energies[:-1], 1))


def test_no_overlap_is_not_a_photometric_minimum():
    image = jnp.ones((10, 10))
    energy = J_pixelwise_core(jnp.array([[100.0, 0.0]]), image, image,
                             jnp.array([[5.5, 5.5]]), jnp.array([[0]]),
                             jnp.ones((1, 1)), interpolation="linear")
    assert np.isinf(float(energy))
