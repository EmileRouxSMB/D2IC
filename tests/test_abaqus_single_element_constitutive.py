"""Full loading-history regression against Abaqus 2022 Job-1Elt CPS4R."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from d2ic.constitutive.hardening import flow_stress
from examples.validate_abaqus_single_element import (
    REFERENCE, calculate, comparison_fields, reference_array,
)


jax.config.update("jax_enable_x64", True)


def test_tabulated_hardening_is_linear_and_jittable():
    table = jnp.asarray([[0.0, 414.0], [0.006, 443.683], [0.011, 490.649]])
    law = jax.jit(lambda p: flow_stress(p, table, "tabulated"))
    np.testing.assert_allclose(law(0.003), 0.5 * (414.0 + 443.683), atol=1e-12)
    np.testing.assert_allclose(jax.grad(law)(0.003), (443.683 - 414.0) / 0.006)


def test_entire_abaqus_loading_history():
    reference = json.loads(REFERENCE.read_text())
    assert len(reference["frames"]) == 25
    F, result = calculate(reference)
    assert np.all(result["converged"])
    assert np.max(result["residual"]) < 1e-9

    # ODB values are float32 and the tabulated implicit return differs from
    # Abaqus by <0.001 MPa. Near-zero transverse stress needs absolute error.
    tolerances = {
        "S11": 0.0012, "S22": 0.0012, "S12": 1e-9, "S33": 1e-12,
        "PE11": 5e-8, "PE22": 2e-7, "PE12": 1e-12,
        "PEEQ": 2e-7, "PEMAG": 2e-7,
        "LE11": 5e-9, "LE22": 5e-9, "LE12": 1e-12,
        "RF2_RP": 0.004,
    }
    for node in (1, 2, 3, 4):
        tolerances["NFORC1_%d" % node] = 0.003
        tolerances["NFORC2_%d" % node] = 0.002
    for node in (1, 2):
        tolerances["RF2_%d" % node] = 0.002
    comparisons = comparison_fields(reference, F, result)
    assert comparisons.keys() == tolerances.keys()
    for name, (predicted, expected) in comparisons.items():
        np.testing.assert_allclose(predicted, expected, rtol=0, atol=tolerances[name],
                                   err_msg=name)

    frames = reference["frames"]
    active_yield = reference_array(frames[1:], "AC YIELD").astype(bool)
    np.testing.assert_array_equal(result["equivalent_plastic_strain"] > 1e-12,
                                  active_yield)
    # The ODB path is almost uniaxial; its small transverse stress gives an
    # O(1e-6) departure from the exact -PEEQ/2 relation.
    np.testing.assert_allclose(result["plastic_strain"][:, 0],
                               -result["equivalent_plastic_strain"] / 2,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.sum(result["plastic_strain"][:, :3], axis=1),
                               0, rtol=0, atol=1e-12)
