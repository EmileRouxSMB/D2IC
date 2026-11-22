"""Compare the D2IC local constitutive response to Abaqus Job-1Elt.

Run ``python examples/validate_abaqus_single_element.py`` from the D2IC root.
The JSON reference can be regenerated with ``tools/export_abaqus_single_element.py``.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from d2ic.constitutive.corotational import integrate_history
from d2ic.constitutive.state import Hill48Material, initial_material_state
from d2ic.mechanics.isotropic_history import forces_from_fields, q4_operators


jax.config.update("jax_enable_x64", True)
REFERENCE = Path(__file__).resolve().parents[1] / "tests/job_1elt_abaqus.json"


def reference_array(frames, name, node=None):
    """Read one integration point or one indexed nodal field from each frame."""
    values = []
    for frame in frames:
        candidates = frame["fields"][name]["values"]
        value = next(v for v in candidates if node is None or v["node"] == node)
        values.append(value["data"])
    return np.asarray(values, dtype=float)


def calculate(reference):
    """Integrate the measured affine Q4 kinematics with Abaqus tabular hardening."""
    frames = reference["frames"]
    u = np.stack([reference_array(frames, "U", node=i) for i in (1, 2, 3, 4)], axis=1)
    d_1 = (u[:, 1] - u[:, 0]) / 5.0
    d_2 = (u[:, 2] - u[:, 0]) / 5.0
    F = np.stack((np.stack((1 + d_1[:, 0], d_2[:, 0]), axis=-1),
                  np.stack((d_1[:, 1], 1 + d_2[:, 1]), axis=-1)), axis=-2)
    material = Hill48Material(
        jnp.asarray(210000.0), jnp.asarray(0.3),
        jnp.asarray([0.5, 0.5, 0.5, 1.5, 1.5, 1.5]),
        jnp.asarray(reference["hardening"]), "tabulated",
    )
    _, result = integrate_history(jnp.asarray(F), initial_material_state(), material)
    nodes = jnp.asarray([reference["nodes"][str(i)] for i in (1, 2, 3, 4)])
    elements = jnp.asarray([[0, 1, 3, 2]])
    gradients, det_jacobian, _, weights = q4_operators(nodes, elements)
    repeated = lambda field: jnp.broadcast_to(field[:, None, None], (len(frames) - 1, 1, 4) + field.shape[1:])
    internal_force = forces_from_fields(
        repeated(jnp.asarray(F[1:])), repeated(result["cauchy_stress"]),
        repeated(result["thickness_ratio"]), elements, gradients,
        det_jacobian * weights, jnp.zeros_like(nodes),
    )
    # The exported Abaqus NFORC equals integral(sigma @ F^-T dV0), i.e. its
    # effective volume factor is one. This is checked independently below;
    # D2IC's current-volume force is retained to quantify the convention gap.
    reference_force = forces_from_fields(
        repeated(jnp.asarray(F[1:])), repeated(result["cauchy_stress"]),
        repeated(1.0 / jnp.linalg.det(jnp.asarray(F[1:]))), elements, gradients,
        det_jacobian * weights, jnp.zeros_like(nodes),
    )
    result["NFORC"] = -reference_force
    result["NFORC_D2IC"] = -internal_force
    result["RF2_RP"] = jnp.sum(reference_force[:, 2:4, 1], axis=1)
    return F, {name: np.asarray(value) for name, value in result.items()}


def comparison_fields(reference, F, result):
    """Pair D2IC predictions with the comparable Abaqus frame fields."""
    frames = reference["frames"][1:]
    _, stretches, principal_axes = np.linalg.svd(F[1:])
    log_strain = np.swapaxes(principal_axes, -1, -2) @ (
        np.log(stretches)[..., None] * principal_axes)
    comparisons = {
        "S11": (result["cauchy_stress"][:, 0, 0], reference_array(frames, "S")[:, 0]),
        "S22": (result["cauchy_stress"][:, 1, 1], reference_array(frames, "S")[:, 1]),
        "S12": (result["cauchy_stress"][:, 0, 1], reference_array(frames, "S")[:, 3]),
        "S33": (np.zeros(len(frames)), reference_array(frames, "S")[:, 2]),
        "PE11": (result["plastic_strain"][:, 0], reference_array(frames, "PE")[:, 0]),
        "PE22": (result["plastic_strain"][:, 1], reference_array(frames, "PE")[:, 1]),
        "PE12": (0.5 * result["plastic_strain"][:, 3], reference_array(frames, "PE")[:, 3]),
        "PEEQ": (result["equivalent_plastic_strain"], reference_array(frames, "PEEQ")),
        "PEMAG": (result["equivalent_plastic_strain"], reference_array(frames, "PEMAG")),
        "LE11": (log_strain[:, 0, 0], reference_array(frames, "LE")[:, 0]),
        "LE22": (log_strain[:, 1, 1], reference_array(frames, "LE")[:, 1]),
        "LE12": (log_strain[:, 0, 1], reference_array(frames, "LE")[:, 3]),
        "RF2_RP": (result["RF2_RP"], reference_array(frames, "RF", node=5)[:, 1]),
    }
    for component in (1, 2):
        for index, node in enumerate((1, 2, 3, 4)):
            comparisons["NFORC%d_%d" % (component, node)] = (
                result["NFORC"][:, index, component - 1],
                reference_array(frames, "NFORC%d" % component, node=node),
            )
    for node in (1, 2):
        comparisons["RF2_%d" % node] = (
            -result["NFORC"][:, node - 1, 1],
            reference_array(frames, "RF", node=node)[:, 1],
        )
    return comparisons


def report(reference, F, result):
    """Print maximum absolute errors for fields with matching 2D conventions."""
    frames = reference["frames"][1:]
    comparisons = comparison_fields(reference, F, result)
    for name, (predicted, expected) in comparisons.items():
        print("%-7s max_abs=%10.4g final=(%12.7g,%12.7g)" %
              (name, np.max(np.abs(predicted - expected)), predicted[-1], expected[-1]))
    print("converged", np.all(result["converged"]), "max_residual", np.max(result["residual"]))
    print("D2IC_volume_force_gap_N", np.max(np.abs(
        result["NFORC_D2IC"][:, :, 1] -
        np.stack([reference_array(frames, "NFORC2", node=node)
                  for node in (1, 2, 3, 4)], axis=1))))


if __name__ == "__main__":
    reference = json.loads(REFERENCE.read_text())
    F, result = calculate(reference)
    report(reference, F, result)
