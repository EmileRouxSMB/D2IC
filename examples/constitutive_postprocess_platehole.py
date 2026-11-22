"""Total Hill48/Voce post-processing of the PlateHole DIC NPZ history.

The bundled PlateHole data are a software demonstrator; the illustrative
metal parameters below are not a physical calibration of these images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from d2ic.constitutive import Hill48Material, initial_material_state, integrate_history
from d2ic.io.mechanical_npz import load_dic_history_npz, save_constitutive_npz
from d2ic.io.constitutive_paraview import write_constitutive_sequence
from d2ic.mesh import Mesh
from d2ic.mechanics import deformation_gradient_history_q4, integrate_gauss_point_histories
from d2ic.mechanics.equilibrium_constraints import (
    active_equilibrium_dofs, masked_equilibrium_residual,
)
from d2ic.mechanics.equilibrium import (
    assemble_internal_force_history_q4,
    integrate_boundary_traction_history, select_q4_boundary_face,
)

jax.config.update("jax_enable_x64", True)

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLE_DIR / "outputs" / "sequential_dic_platehole" / "npz" / "fields_sequence.npz"
DEFAULT_OUTPUT = EXAMPLE_DIR / "outputs" / "constitutive_platehole" / "constitutive_fields.npz"


def make_example_material(hardening=None) -> Hill48Material:
    """Return illustrative Hill48/Voce parameters in MPa."""
    hardening = jnp.asarray([250.0, 350.0, 0.08, 1.0]) if hardening is None else hardening
    return Hill48Material(
        young=jnp.asarray(210_000.0),
        poisson=jnp.asarray(0.30),
        hill=jnp.asarray([0.55, 0.45, 0.50, 1.50, 1.50, 1.65]),
        hardening=hardening,
        hardening_model="voce",
    )


def run_postprocess(
    input_path: Path, output_path: Path, angle_deg: float,
    initial_thickness_mm: float, length_scale_mm_per_pixel: float,
) -> None:
    """Integrate every DIC frame and save Gauss-point constitutive fields."""
    data = load_dic_history_npz(input_path, prepend_reference=True)
    F_history, gauss_points, detJ = deformation_gradient_history_q4(
        jnp.asarray(data.nodes_xy), jnp.asarray(data.elements), jnp.asarray(data.displacement)
    )
    if bool(jnp.any(detJ <= 0.0)):
        raise ValueError("The Q4 mesh contains a non-positive reference Jacobian.")

    angle = jnp.deg2rad(jnp.asarray(angle_deg))
    material = make_example_material()
    _, fields = integrate_gauss_point_histories(F_history, material, angle)
    top_face = select_q4_boundary_face(data.nodes_xy, data.elements, side="top")
    bottom_face = select_q4_boundary_face(data.nodes_xy, data.elements, side="bottom")
    grip_node_ids = np.unique(np.concatenate((top_face.nodes.ravel(), bottom_face.nodes.ravel())))
    active_dofs = active_equilibrium_dofs(data.nodes_xy.shape[0], grip_node_ids)
    h0 = jnp.asarray(initial_thickness_mm, dtype=F_history.dtype)
    scale = jnp.asarray(length_scale_mm_per_pixel, dtype=F_history.dtype)
    internal_force = assemble_internal_force_history_q4(
        jnp.asarray(data.nodes_xy), jnp.asarray(data.elements), F_history[1:],
        fields["cauchy_stress"], fields["thickness_ratio"], h0, scale,
    )
    residual_equilibrium = masked_equilibrium_residual(
        internal_force, None, jnp.asarray(active_dofs),
    )
    top_traction_force = integrate_boundary_traction_history(
        jnp.asarray(data.nodes_xy), jnp.asarray(data.displacement[1:]),
        jnp.asarray(top_face.nodes), jnp.asarray(top_face.elements),
        jnp.asarray(top_face.local_edges), fields["cauchy_stress"],
        fields["thickness_ratio"], h0, scale,
    )
    top_node_ids = np.unique(top_face.nodes)
    top_internal_force = jnp.sum(internal_force[:, top_node_ids], axis=1)
    equilibrium_residual_rms = jnp.sqrt(
        jnp.sum(residual_equilibrium**2, axis=(1, 2)) / jnp.sum(jnp.asarray(active_dofs))
    )
    relative_equilibrium_residual = equilibrium_residual_rms / jnp.maximum(
        jnp.linalg.norm(top_traction_force, axis=1), jnp.finfo(F_history.dtype).eps
    )
    fields = dict(fields)
    fields["internal_force"] = internal_force
    fields["equilibrium_residual"] = residual_equilibrium

    # Differentiate one representative Gauss-point history to keep the example light.
    def final_eqps_at_first_gauss_point(hardening):
        initial = initial_material_state(F_history.dtype, angle)
        final, _ = integrate_history(
            F_history[:, 0, 0], initial, make_example_material(hardening)
        )
        return final.equivalent_plastic_strain

    sensitivity = jax.grad(final_eqps_at_first_gauss_point)(material.hardening)
    save_constitutive_npz(
        output_path,
        cauchy_stress=fields["cauchy_stress"],
        equivalent_plastic_strain=fields["equivalent_plastic_strain"],
        thickness_ratio=fields["thickness_ratio"],
        converged=fields["converged"], residual=fields["residual"],
        deformation_gradient=F_history[1:], gauss_parent_coordinates=gauss_points,
        internal_force=internal_force, equilibrium_residual=residual_equilibrium,
        equilibrium_residual_rms=equilibrium_residual_rms,
        relative_equilibrium_residual=relative_equilibrium_residual,
        top_traction_force=top_traction_force, top_internal_force=top_internal_force,
        top_boundary_nodes=top_face.nodes, top_boundary_elements=top_face.elements,
        top_boundary_local_edges=top_face.local_edges,
        equilibrium_active_dofs=active_dofs, excluded_grip_nodes=grip_node_ids,
        reference_detJ=detJ, frame_names=data.frame_names[1:],
        material_angle_deg=angle_deg, young=material.young, poisson=material.poisson,
        initial_thickness_mm=initial_thickness_mm,
        length_scale_mm_per_pixel=length_scale_mm_per_pixel, force_unit="N",
        hill_coefficients=material.hill, hardening_parameters=material.hardening,
        mean_final_eqps_gradient_hardening=sensitivity, stress_unit="MPa",
        voigt_convention="stress physical shear; strain engineering shear",
    )
    pvd_path = write_constitutive_sequence(
        output_path.parent / "paraview",
        Mesh(nodes_xy=data.nodes_xy, elements=data.elements),
        data.displacement[1:],
        fields,
        np.asarray(detJ),
    )
    converged = np.asarray(fields["converged"])
    print(f"Saved constitutive fields to {output_path}")
    print(f"Saved synchronized ParaView collection to {pvd_path}")
    print(f"Final top traction force [Fx, Fy] N: {np.asarray(top_traction_force[-1])}")
    print(f"Final top internal resultant [Fx, Fy] N: {np.asarray(top_internal_force[-1])}")
    print(f"Final equilibrium residual RMS [N]: {float(equilibrium_residual_rms[-1])}")
    print(f"Final relative equilibrium residual: {float(relative_equilibrium_residual[-1])}")
    print(f"Non-converged local increments: {converged.size - np.count_nonzero(converged)}")
    print(f"d(mean final eqps)/d(hardening) = {np.asarray(sensitivity)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--material-angle-deg", type=float, default=0.0)
    parser.add_argument("--initial-thickness-mm", type=float, default=1.0)
    parser.add_argument("--length-scale-mm-per-pixel", type=float, default=1.0)
    args = parser.parse_args()
    run_postprocess(
        args.input, args.output, args.material_angle_deg,
        args.initial_thickness_mm, args.length_scale_mm_per_pixel,
    )


if __name__ == "__main__":
    main()
