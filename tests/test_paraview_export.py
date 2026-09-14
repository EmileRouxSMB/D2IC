from __future__ import annotations

import xml.etree.ElementTree as ET
import base64
import struct

import meshio
import numpy as np

from d2ic import DICResult, Mesh, write_dic_vtu, write_image_vti, write_pvd


def test_vtu_contains_mesh_and_kinematic_point_fields(tmp_path) -> None:
    mesh = Mesh(
        nodes_xy=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]),
        elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
    )
    displacement = np.array([[0.0, 0.0], [0.2, 0.0], [0.2, -0.1], [0.0, -0.1]])
    strain = np.tile(np.array([0.1, -0.1, 0.02]), (4, 1))
    result = DICResult(displacement=displacement, strain=strain, fields={"quality": np.arange(4.0)})

    path = write_dic_vtu(tmp_path / "frame_0000.vtu", mesh, result)
    exported = meshio.read(path)

    np.testing.assert_array_equal(exported.cells_dict["quad"], mesh.elements)
    np.testing.assert_allclose(exported.point_data["Displacement"][:, :2], displacement)
    np.testing.assert_allclose(exported.point_data["Strain"][:, [0, 4, 1]], strain)
    assert "Ux" not in exported.point_data and "Uy" not in exported.point_data
    assert {"Displacement", "E11", "E22", "E12", "quality"} <= exported.point_data.keys()


def test_pvd_references_vtu_frames_with_timesteps(tmp_path) -> None:
    pvd = write_pvd(
        tmp_path / "dic_sequence.pvd",
        [(0.0, tmp_path / "frame_0000.vtu"), (2.5, tmp_path / "frame_0001.vtu")],
    )
    datasets = ET.parse(pvd).findall("./Collection/DataSet")

    assert [item.attrib["file"] for item in datasets] == ["frame_0000.vtu", "frame_0001.vtu"]
    assert [item.attrib["timestep"] for item in datasets] == ["0", "2.5"]


def test_vti_image_sequence_uses_an_independent_pvd(tmp_path) -> None:
    image_path = write_image_vti(tmp_path / "image_0000.vti", np.arange(6).reshape(2, 3))
    pvd = write_pvd(
        tmp_path / "sequence.pvd",
        [(0.0, image_path)],
    )
    datasets = ET.parse(pvd).findall("./Collection/DataSet")
    assert [item.attrib["file"] for item in datasets] == ["image_0000.vti"]
    image_tree = ET.parse(image_path)
    array = image_tree.find("./ImageData/Piece/PointData/DataArray")
    raw = base64.b64decode(array.text)
    nbytes = struct.unpack("<Q", raw[:8])[0]
    assert nbytes == 6 * 4
    np.testing.assert_allclose(np.frombuffer(raw[8:], dtype="<f4"), np.arange(6))
