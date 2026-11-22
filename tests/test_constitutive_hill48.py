from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from d2ic.constitutive import (
    Hill48Material,
    hill48_equivalent_stress,
    initial_material_state,
    integrate_corotational_plane_stress,
    integrate_plane_stress_increment,
)

jax.config.update("jax_enable_x64", True)


def _material() -> Hill48Material:
    return Hill48Material(
        jnp.asarray(210_000.0), jnp.asarray(0.3),
        jnp.asarray([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),
        jnp.asarray([200.0, 300.0, 0.1, 1.0]), "voce",
    )


def test_hill48_von_mises_normalization() -> None:
    stress = jnp.asarray([321.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(hill48_equivalent_stress(stress, _material().hill), 321.0)


def test_plane_stress_elastic_uniaxial_strain() -> None:
    state, _, converged, _ = integrate_plane_stress_increment(
        jnp.asarray([1e-5, 0.0, 0.0]), initial_material_state(), _material()
    )
    assert bool(converged)
    np.testing.assert_allclose(state.stress[2], 0.0, atol=1e-9)
    np.testing.assert_allclose(state.stress[0], 210_000.0 / (1.0 - 0.3**2) * 1e-5, rtol=1e-9)


def test_plastic_increment_converges_and_is_differentiable() -> None:
    material = _material()
    state, _, converged, residual = integrate_plane_stress_increment(
        jnp.asarray([0.003, -0.0004, 0.0002]), initial_material_state(), material
    )
    assert bool(converged), float(residual)
    assert float(state.equivalent_plastic_strain) > 0.0

    def final_eqps(sigma0):
        hardening = material.hardening.at[0].set(sigma0)
        candidate = Hill48Material(material.young, material.poisson, material.hill, hardening, "voce")
        result, _, _, _ = integrate_plane_stress_increment(
            jnp.asarray([0.003, -0.0004, 0.0002]), initial_material_state(), candidate
        )
        return result.equivalent_plastic_strain

    assert np.isfinite(float(jax.grad(final_eqps)(material.hardening[0])))


def test_rigid_rotation_is_objective() -> None:
    angle = 0.4
    rotation = jnp.asarray([[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]])
    state, output = integrate_corotational_plane_stress(
        jnp.eye(2), rotation, initial_material_state(), _material()
    )
    np.testing.assert_allclose(output["cauchy_stress"], 0.0, atol=1e-8)
    np.testing.assert_allclose(state.equivalent_plastic_strain, 0.0, atol=1e-12)
