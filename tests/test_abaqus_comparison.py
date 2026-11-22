from __future__ import annotations

import json

import meshio
import numpy as np

from d2ic import DICResult, Mesh
from d2ic.io.abaqus_comparison import (
    abaqus_displacement_to_image_px,
    build_abaqus_interpolation_map,
    compare_abaqus_dic_sequence,
    image_nodes_to_abaqus_mm,
    interpolate_abaqus_space,
    interpolate_abaqus_time,
)


def test_image_and_displacement_axes_follow_camera_conventions() -> None:
    camera = {
        "width_px": 3, "height_px": 3, "scale_px_per_unit": 2.0,
        "center_xy": [10.0, 20.0], "rotation_deg": 0,
    }
    physical, mm_per_pixel = image_nodes_to_abaqus_mm(
        np.array([[0.0, 0.0], [2.0, 2.0]]), camera,
    )
    np.testing.assert_allclose(physical, [[9.5, 20.5], [10.5, 19.5]])
    assert mm_per_pixel == 0.5
    np.testing.assert_allclose(
        abaqus_displacement_to_image_px(np.array([[1.0, 2.0]]), mm_per_pixel),
        [[2.0, -4.0]],
    )


def test_temporal_interpolation_uses_exact_image_times() -> None:
    times = np.array([0.0, 1.0, 3.0])
    history = times[:, None, None] * np.array([[[2.0, -1.0]]])
    sampled = interpolate_abaqus_time(times, history, np.array([0.25, 2.0]))
    np.testing.assert_allclose(sampled[:, 0], [[0.5, -0.25], [4.0, -2.0]])


def test_affine_field_is_exact_on_t3_and_q4() -> None:
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0],
    ])
    connectivity = np.array([[0, 1, 2, 0], [3, 4, 5, 6]])
    counts = np.array([3, 4])
    targets = np.array([[0.2, 0.3], [2.25, 0.7]])
    interpolation = build_abaqus_interpolation_map(
        nodes, connectivity, counts, targets, max_extrapolation_mm=0.0,
    )
    affine = np.column_stack((2.0 * nodes[:, 0] - nodes[:, 1] + 3.0,
                              -0.5 * nodes[:, 0] + 4.0 * nodes[:, 1] - 2.0))
    expected = np.column_stack((2.0 * targets[:, 0] - targets[:, 1] + 3.0,
                                -0.5 * targets[:, 0] + 4.0 * targets[:, 1] - 2.0))
    np.testing.assert_array_equal(interpolation.valid, True)
    np.testing.assert_allclose(interpolate_abaqus_space(affine, interpolation), expected)


def test_edge_extrapolation_is_bounded_by_one_physical_pixel() -> None:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2, 3]])
    targets = np.array([[1.09, 0.5], [1.11, 0.5]])
    interpolation = build_abaqus_interpolation_map(
        nodes, cells, np.array([4]), targets, max_extrapolation_mm=0.1,
    )
    np.testing.assert_array_equal(interpolation.valid, [True, False])
    np.testing.assert_allclose(interpolation.extrapolation_distance_mm[0], 0.09)


def test_sequence_report_excludes_isolated_nodes_and_exports_fields(tmp_path) -> None:
    mesh = Mesh(
        nodes_xy=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [9.0, 9.0]]),
        elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
    )
    # The camera maps the four connected image nodes to the unit Abaqus square.
    metadata = {"camera": {
        "width_px": 2, "height_px": 2, "scale_px_per_unit": 1.0,
        "center_xy": [0.5, 0.5], "rotation_deg": 0,
    }}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    coordinates = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    abaqus_u = np.zeros((2, 4, 2))
    abaqus_u[1, :, 0] = 1.0
    np.savez(
        tmp_path / "abaqus.npz", times=[0.0, 1.0], coordinates=coordinates,
        connectivity=[[0, 1, 2, 3]], element_node_count=[4], U=abaqus_u,
    )
    dic_u = np.zeros((5, 2))
    dic_u[:4, 0] = 1.5
    dic_u[4] = 1000.0  # must not enter metrics
    result = DICResult(displacement=dic_u, strain=np.zeros((5, 3)))
    report = compare_abaqus_dic_sequence(
        mesh, [result], ["frame_0001_t_1.000000.tiff"],
        abaqus_npz=tmp_path / "abaqus.npz", metadata_json=tmp_path / "metadata.json",
        output_dir=tmp_path / "report", vtu_dir=tmp_path / "vtu",
    )
    frame = report["frames"][0]
    assert frame["metric_node_count"] == 4
    assert report["isolated_node_count"] == 1
    np.testing.assert_allclose(frame["rmse_u1_px"], 0.5)
    exported = meshio.read(tmp_path / "vtu/frame_0000.vtu")
    assert {
        "Abaqus_Displacement", "Abaqus_Displacement_magnitude",
        "DIC_minus_Abaqus_Displacement", "DIC_Abaqus_error_magnitude",
        "DIC_connected_node", "Abaqus_interpolation_valid",
    } <= exported.point_data.keys()
    assert (tmp_path / "report/dic_abaqus_comparison.json").is_file()
    assert (tmp_path / "report/dic_abaqus_comparison.csv").is_file()
