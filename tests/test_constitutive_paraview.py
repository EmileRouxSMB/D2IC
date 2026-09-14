from __future__ import annotations

import xml.etree.ElementTree as ET

import meshio
import numpy as np

from d2ic.io.constitutive_paraview import write_constitutive_sequence
from d2ic.mesh import Mesh


def test_constitutive_sequence_exports_synchronized_cell_fields(tmp_path) -> None:
    mesh = Mesh(
        nodes_xy=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]),
        elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
    )
    displacement = np.zeros((2, 4, 2))
    stress = np.zeros((2, 1, 4, 2, 2))
    stress[:, :, :, 0, 0] = np.array([10.0, 20.0])[:, None, None]
    fields = {
        "cauchy_stress": stress,
        "equivalent_plastic_strain": np.full((2, 1, 4), 0.02),
        "thickness_ratio": np.full((2, 1, 4), 0.98),
        "converged": np.ones((2, 1, 4), dtype=bool),
        "residual": np.zeros((2, 1, 4)),
        "internal_force": np.ones((2, 4, 2)),
        "equilibrium_residual": 2.0 * np.ones((2, 4, 2)),
    }
    pvd = write_constitutive_sequence(
        tmp_path, mesh, displacement, fields, np.ones((1, 4))
    )
    datasets = ET.parse(pvd).findall("./Collection/DataSet")
    assert [item.attrib["timestep"] for item in datasets] == ["0", "1"]
    assert [item.attrib["file"] for item in datasets] == [
        "constitutive_0000.vtu", "constitutive_0001.vtu"
    ]
    exported = meshio.read(tmp_path / "constitutive_0001.vtu")
    np.testing.assert_allclose(exported.cell_data_dict["S11"]["quad"], 20.0)
    np.testing.assert_allclose(exported.cell_data_dict["Equivalent_plastic_strain"]["quad"], 0.02)
    assert "Displacement" in exported.point_data
    assert {"Internal_force", "Equilibrium_residual"} <= exported.point_data.keys()
