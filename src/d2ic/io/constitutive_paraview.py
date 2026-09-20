"""ParaView exports for constitutive fields evaluated at Gauss points."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import meshio
import numpy as np

from ..mesh.assets import Mesh
from .paraview import _meshio_cell_type, _validate_mesh, write_pvd


def _weighted_cell_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average Gauss-point values into one ParaView cell value per element."""
    values = np.asarray(values)
    weights = np.asarray(weights)
    if values.shape[:2] != weights.shape:
        raise ValueError(
            f"Gauss fields and weights must start with the same (Ne,Ng) shape; "
            f"got {values.shape[:2]} and {weights.shape}."
        )
    normalized = weights / np.sum(weights, axis=1, keepdims=True)
    return np.sum(values * normalized[(...,) + (None,) * (values.ndim - 2)], axis=1)


def write_constitutive_vtu(
    path: str | Path,
    mesh: Mesh,
    displacement: np.ndarray,
    *,
    cauchy_stress: np.ndarray,
    equivalent_plastic_strain: np.ndarray,
    thickness_ratio: np.ndarray,
    converged: np.ndarray,
    residual: np.ndarray,
    gauss_weights: np.ndarray,
    equivalent_stress: np.ndarray | None = None,
    internal_force: np.ndarray | None = None,
    equilibrium_residual: np.ndarray | None = None,
) -> Path:
    """Write one constitutive state using element-centered weighted GP fields.

    Gauss arrays have shape ``(Ne,Ng,...)``. They are averaged with the
    reference quadrature measures for visualization; the lossless values stay
    in the constitutive NPZ archive.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes_xy = np.asarray(mesh.nodes_xy)
    elements = np.asarray(mesh.elements, dtype=np.int32)
    _validate_mesh(nodes_xy, elements)
    displacement = np.asarray(displacement)
    if displacement.shape != nodes_xy.shape:
        raise ValueError(f"displacement must have shape {nodes_xy.shape}; got {displacement.shape}.")
    ne = elements.shape[0]
    weights = np.asarray(gauss_weights)
    if weights.shape[0] != ne:
        raise ValueError("gauss_weights must have one row per element.")

    cauchy_stress = np.asarray(cauchy_stress)
    stress = _weighted_cell_average(cauchy_stress, weights)
    if equivalent_stress is None:
        equivalent_stress = np.sqrt(
            cauchy_stress[..., 0, 0] ** 2
            - cauchy_stress[..., 0, 0] * cauchy_stress[..., 1, 1]
            + cauchy_stress[..., 1, 1] ** 2
            + 3.0 * cauchy_stress[..., 0, 1] ** 2
        )
    equivalent_stress = _weighted_cell_average(equivalent_stress, weights)
    eqps = _weighted_cell_average(equivalent_plastic_strain, weights)
    thickness = _weighted_cell_average(thickness_ratio, weights)
    residual_max = np.max(np.asarray(residual), axis=1)
    converged_cell = np.all(np.asarray(converged), axis=1).astype(np.uint8)
    zero_cell = np.zeros(ne, dtype=stress.dtype)
    stress_tensor = np.column_stack(
        (stress[:, 0, 0], stress[:, 0, 1], zero_cell,
         stress[:, 1, 0], stress[:, 1, 1], zero_cell,
         zero_cell, zero_cell, zero_cell)
    )
    zero_node = np.zeros(nodes_xy.shape[0], dtype=np.result_type(nodes_xy, displacement))
    points = np.column_stack((nodes_xy, zero_node))
    displacement_3d = np.column_stack((displacement, zero_node))
    point_data = {
        "Displacement": displacement_3d,
        "Deformed_position": points + displacement_3d,
    }
    if internal_force is not None:
        internal_force = np.asarray(internal_force)
        if internal_force.shape != displacement.shape:
            raise ValueError(
                f"internal_force must have shape {displacement.shape}; got {internal_force.shape}."
            )
        internal_force_3d = np.column_stack((internal_force, zero_node))
        point_data.update(
            {
                "Nodal_internal_force": internal_force_3d,
            }
        )
    if equilibrium_residual is not None:
        equilibrium_residual = np.asarray(equilibrium_residual)
        if equilibrium_residual.shape != displacement.shape:
            raise ValueError(
                "equilibrium_residual must have shape "
                f"{displacement.shape}; got {equilibrium_residual.shape}."
            )
        point_data["Equilibrium_residual"] = np.column_stack(
            (equilibrium_residual, zero_node)
        )
    cell_fields: Mapping[str, np.ndarray] = {
        "Cauchy_stress": stress_tensor,
        "Equivalent_stress": equivalent_stress,
        "Equivalent_plastic_strain": eqps,
        "Thickness_ratio": thickness,
        "Converged": converged_cell,
        "Local_residual_max": residual_max,
    }
    cell_data = {name: [values] for name, values in cell_fields.items()}
    vtk_mesh = meshio.Mesh(
        points=points,
        cells=[(_meshio_cell_type(elements.shape[1]), elements)],
        point_data=point_data,
        cell_data=cell_data,
    )
    meshio.write(path, vtk_mesh, file_format="vtu")
    return path


def write_constitutive_sequence(
    output_dir: str | Path,
    mesh: Mesh,
    displacement_history: np.ndarray,
    fields: Mapping[str, np.ndarray],
    gauss_weights: np.ndarray,
    *,
    pvd_filename: str = "constitutive_sequence.pvd",
    vtu_prefix: str = "constitutive",
    time_values: np.ndarray | None = None,
) -> Path:
    """Write synchronized constitutive VTU frames and their PVD collection."""
    output_dir = Path(output_dir)
    required = {
        "cauchy_stress", "equivalent_plastic_strain", "thickness_ratio",
        "converged", "residual",
    }
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"Missing constitutive fields: {sorted(missing)}")
    n_frames = int(np.asarray(fields["cauchy_stress"]).shape[0])
    if np.asarray(displacement_history).shape[0] != n_frames:
        raise ValueError("displacement_history and constitutive fields must have the same frame count.")
    if time_values is None:
        time_values = np.arange(n_frames, dtype=float)
    else:
        time_values = np.asarray(time_values, dtype=float)
        if time_values.shape != (n_frames,):
            raise ValueError("time_values must have one value per frame.")

    datasets = []
    for frame in range(n_frames):
        vtu_path = output_dir / f"{vtu_prefix}_{frame:04d}.vtu"
        write_constitutive_vtu(
            vtu_path, mesh, np.asarray(displacement_history)[frame],
            cauchy_stress=np.asarray(fields["cauchy_stress"])[frame],
            equivalent_plastic_strain=np.asarray(fields["equivalent_plastic_strain"])[frame],
            thickness_ratio=np.asarray(fields["thickness_ratio"])[frame],
            converged=np.asarray(fields["converged"])[frame],
            residual=np.asarray(fields["residual"])[frame],
            gauss_weights=gauss_weights,
            equivalent_stress=None if "equivalent_stress" not in fields else np.asarray(fields["equivalent_stress"])[frame],
            internal_force=None if "internal_force" not in fields else np.asarray(fields["internal_force"])[frame],
            equilibrium_residual=None if "equilibrium_residual" not in fields else np.asarray(fields["equilibrium_residual"])[frame],
        )
        datasets.append((float(time_values[frame]), vtu_path))
    return write_pvd(output_dir / pvd_filename, datasets)
