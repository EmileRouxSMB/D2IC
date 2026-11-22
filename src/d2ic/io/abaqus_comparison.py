"""Nodal comparison of image-coordinate DIC and Abaqus kinematics."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.dataclasses import DICResult
from ..mesh.assets import Mesh
from .paraview import write_dic_vtu, write_pvd


_TIME_PATTERN = re.compile(r"_t_([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


@dataclass(frozen=True)
class AbaqusInterpolationMap:
    """Fixed T3/Q4 interpolation map from an Abaqus mesh to target nodes."""

    node_ids: np.ndarray
    weights: np.ndarray
    valid: np.ndarray
    extrapolation_distance_mm: np.ndarray


def image_time_from_name(path: str | Path) -> float:
    """Extract the physical time following ``_t_`` in an image filename."""
    match = _TIME_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"Image filename does not contain a '_t_<time>' field: {path}")
    return float(match.group(1))


def image_nodes_to_abaqus_mm(
    nodes_xy_px: np.ndarray, camera: dict, *, image_binning: int = 1,
) -> tuple[np.ndarray, float]:
    """Map image nodes ``(+x right, +y down)`` to Abaqus coordinates in mm."""
    if int(camera.get("rotation_deg", 0)) != 0:
        raise ValueError("Abaqus comparison requires camera.rotation_deg = 0.")
    scale_px_per_mm = float(camera["scale_px_per_unit"])
    if scale_px_per_mm <= 0.0 or image_binning < 1:
        raise ValueError("Camera scale and image_binning must be positive.")
    mm_per_pixel = float(image_binning) / scale_px_per_mm
    width = float(camera["width_px"]) / image_binning
    height = float(camera["height_px"]) / image_binning
    center_x, center_y = map(float, camera["center_xy"])
    nodes = np.asarray(nodes_xy_px, dtype=float)
    physical = np.empty_like(nodes)
    physical[:, 0] = center_x + (nodes[:, 0] - (width - 1.0) / 2.0) * mm_per_pixel
    physical[:, 1] = center_y - (nodes[:, 1] - (height - 1.0) / 2.0) * mm_per_pixel
    return physical, mm_per_pixel


def abaqus_displacement_to_image_px(displacement_mm: np.ndarray, mm_per_pixel: float) -> np.ndarray:
    """Convert Abaqus ``(U1,U2)`` in mm to image ``(+x right,+y down)`` pixels."""
    displacement = np.asarray(displacement_mm, dtype=float) / float(mm_per_pixel)
    displacement = displacement.copy()
    displacement[..., 1] *= -1.0
    return displacement


def interpolate_abaqus_time(
    times: np.ndarray, displacement_history: np.ndarray, target_times: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate nodal Abaqus displacements at exact image times."""
    times = np.asarray(times, dtype=float)
    targets = np.asarray(target_times, dtype=float)
    values = np.asarray(displacement_history, dtype=float)
    if times.ndim != 1 or values.shape[0] != times.size:
        raise ValueError("Abaqus U history must have shape (n_times, n_nodes, 2).")
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("Abaqus times must be strictly increasing.")
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(times)))
    if np.any(targets < times[0] - tolerance) or np.any(targets > times[-1] + tolerance):
        raise ValueError("An image time lies outside the Abaqus time interval.")
    targets = np.clip(targets, times[0], times[-1])
    upper = np.searchsorted(times, targets, side="right")
    upper = np.clip(upper, 1, times.size - 1)
    lower = upper - 1
    alpha = (targets - times[lower]) / (times[upper] - times[lower])
    return values[lower] + alpha[:, None, None] * (values[upper] - values[lower])


def build_abaqus_interpolation_map(
    source_nodes_mm: np.ndarray,
    connectivity: np.ndarray,
    element_node_count: np.ndarray,
    target_nodes_mm: np.ndarray,
    *,
    max_extrapolation_mm: float,
) -> AbaqusInterpolationMap:
    """Locate target nodes in a mixed T3/Q4 mesh, with bounded edge extrapolation."""
    nodes = np.asarray(source_nodes_mm, dtype=float)
    cells = np.asarray(connectivity, dtype=int)
    counts = np.asarray(element_node_count, dtype=int)
    targets = np.asarray(target_nodes_mm, dtype=float)
    if max_extrapolation_mm < 0.0:
        raise ValueError("max_extrapolation_mm must be non-negative.")
    if cells.ndim != 2 or counts.shape != (cells.shape[0],):
        raise ValueError("Invalid mixed-mesh connectivity arrays.")
    if np.any((counts != 3) & (counts != 4)):
        raise ValueError("Abaqus interpolation supports only T3 and Q4 elements.")

    vertices = [nodes[cells[e, : counts[e]]] for e in range(len(cells))]
    bounds_min = np.asarray([value.min(axis=0) for value in vertices])
    bounds_max = np.asarray([value.max(axis=0) for value in vertices])
    node_ids = np.zeros((len(targets), 4), dtype=np.int32)
    weights = np.zeros((len(targets), 4), dtype=float)
    valid = np.zeros(len(targets), dtype=bool)
    distances = np.full(len(targets), np.inf, dtype=float)

    for target_index, point in enumerate(targets):
        candidates = np.flatnonzero(np.all(
            (point >= bounds_min - max_extrapolation_mm)
            & (point <= bounds_max + max_extrapolation_mm), axis=1,
        ))
        best = None
        for element in candidates:
            polygon = vertices[element]
            count = counts[element]
            try:
                shape, inside = _element_weights(point, polygon)
            except np.linalg.LinAlgError:
                continue
            distance = 0.0 if inside else _distance_to_polygon(point, polygon)
            if distance <= max_extrapolation_mm + 1e-12:
                candidate = (distance, int(element), shape)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                    if inside:
                        break
        if best is None:
            continue
        distance, element, shape = best
        count = counts[element]
        ids = cells[element, :count]
        node_ids[target_index, :count] = ids
        if count == 3:
            node_ids[target_index, 3] = ids[-1]
        weights[target_index, :count] = shape
        valid[target_index] = True
        distances[target_index] = distance
    return AbaqusInterpolationMap(node_ids, weights, valid, distances)


def interpolate_abaqus_space(
    nodal_values: np.ndarray, interpolation: AbaqusInterpolationMap,
) -> np.ndarray:
    """Apply a fixed interpolation map to one or several nodal fields."""
    values = np.asarray(nodal_values)
    sampled = np.einsum(
        "nk,...nkd->...nd", interpolation.weights, values[..., interpolation.node_ids, :]
    )
    sampled = np.asarray(sampled, dtype=np.result_type(values, float))
    sampled[..., ~interpolation.valid, :] = np.nan
    return sampled


def connected_node_mask(n_nodes: int, elements: np.ndarray) -> np.ndarray:
    """Return nodes referenced by at least one DIC element."""
    connected = np.zeros(int(n_nodes), dtype=bool)
    connected[np.unique(np.asarray(elements, dtype=int))] = True
    return connected


def compare_abaqus_dic_sequence(
    mesh: Mesh,
    results: Sequence[DICResult],
    image_paths: Sequence[str | Path],
    *,
    abaqus_npz: str | Path,
    metadata_json: str | Path,
    output_dir: str | Path,
    vtu_dir: str | Path,
    vtu_prefix: str = "frame",
    pvd_path: str | Path | None = None,
    image_binning: int = 1,
    max_extrapolation_px: float = 1.0,
) -> dict:
    """Compare every DIC frame to temporally/spatially interpolated Abaqus ``U``."""
    if len(results) != len(image_paths):
        raise ValueError("results and image_paths must have identical lengths.")
    metadata_path = Path(metadata_json)
    metadata = json.loads(metadata_path.read_text())
    camera = metadata["camera"]
    target_nodes, mm_per_pixel = image_nodes_to_abaqus_mm(
        np.asarray(mesh.nodes_xy), camera, image_binning=image_binning,
    )
    frame_times = np.asarray([image_time_from_name(path) for path in image_paths])
    with np.load(abaqus_npz, allow_pickle=False) as archive:
        abaqus_times = np.asarray(archive["times"], dtype=float)
        source_nodes = np.asarray(archive["coordinates"][:, :2], dtype=float)
        connectivity = np.asarray(archive["connectivity"], dtype=int)
        node_counts = np.asarray(archive["element_node_count"], dtype=int)
        temporal_u = interpolate_abaqus_time(abaqus_times, archive["U"], frame_times)

    interpolation = build_abaqus_interpolation_map(
        source_nodes, connectivity, node_counts, target_nodes,
        max_extrapolation_mm=float(max_extrapolation_px) * mm_per_pixel,
    )
    abaqus_u_mm = interpolate_abaqus_space(temporal_u, interpolation)
    abaqus_u_px = abaqus_displacement_to_image_px(abaqus_u_mm, mm_per_pixel)
    connected = connected_node_mask(len(target_nodes), np.asarray(mesh.elements))
    metric_mask = connected & interpolation.valid

    output_dir = Path(output_dir)
    vtu_dir = Path(vtu_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtu_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    datasets = []
    for frame, (time_s, image_path, result) in enumerate(zip(frame_times, image_paths, results)):
        dic_u = np.asarray(result.displacement, dtype=float)
        error = dic_u - abaqus_u_px[frame]
        error_magnitude = np.linalg.norm(error, axis=1)
        fields = dict(getattr(result, "fields", {}))
        fields.update({
            "Abaqus_Displacement": _vtk_vector(abaqus_u_px[frame]),
            "Abaqus_Displacement_magnitude": np.linalg.norm(abaqus_u_px[frame], axis=1),
            "DIC_minus_Abaqus_Displacement": _vtk_vector(error),
            "DIC_Abaqus_error_magnitude": error_magnitude,
            "DIC_connected_node": connected.astype(np.uint8),
            "Abaqus_interpolation_valid": interpolation.valid.astype(np.uint8),
        })
        augmented = DICResult(
            displacement=result.displacement, strain=result.strain,
            diagnostics=result.diagnostics, points_xy=result.points_xy,
            geometry=result.geometry, history=result.history, fields=fields,
            pixel_maps=result.pixel_maps,
        )
        vtu_path = vtu_dir / f"{vtu_prefix}_{frame:04d}.vtu"
        write_dic_vtu(vtu_path, mesh, augmented)
        datasets.append((float(time_s), vtu_path))
        reports.append(_frame_metrics(
            frame, Path(image_path).name, float(time_s), error, metric_mask,
            mm_per_pixel, connected, interpolation.valid,
        ))

    if pvd_path is None:
        pvd_path = vtu_dir / "dic_sequence.pvd"
    write_pvd(pvd_path, datasets)
    report = {
        "conventions": {
            "abaqus_axes": "+X right, +Y up",
            "dic_axes": "+x right, +y down",
            "displacement_conversion": "u_DIC=(U1/mm_per_pixel,-U2/mm_per_pixel)",
            "displacement_unit_vtu": "pixel",
        },
        "camera_px_per_mm": float(camera["scale_px_per_unit"]) / image_binning,
        "mm_per_pixel": mm_per_pixel,
        "max_extrapolation_px": float(max_extrapolation_px),
        "connected_node_count": int(np.count_nonzero(connected)),
        "isolated_node_count": int(np.count_nonzero(~connected)),
        "abaqus_interpolation_valid_count": int(np.count_nonzero(interpolation.valid)),
        "abaqus_extrapolated_node_count": int(np.count_nonzero(
            interpolation.valid & (interpolation.extrapolation_distance_mm > 0.0)
        )),
        "maximum_extrapolation_mm": float(np.max(
            interpolation.extrapolation_distance_mm[interpolation.valid], initial=0.0
        )),
        "frames": reports,
    }
    json_path = output_dir / "dic_abaqus_comparison.json"
    csv_path = output_dir / "dic_abaqus_comparison.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    _write_report_csv(csv_path, reports)
    report["json"] = str(json_path)
    report["csv"] = str(csv_path)
    return report


def _element_weights(point: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, bool]:
    if len(vertices) == 3:
        matrix = np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0]))
        local = np.linalg.solve(matrix, point - vertices[0])
        weights = np.asarray([1.0 - local.sum(), local[0], local[1]])
        return weights, bool(np.min(weights) >= -1e-10)
    natural = np.zeros(2)
    signs = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    for _ in range(20):
        xi, eta = natural
        shape = 0.25 * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta)
        derivatives = np.column_stack((
            0.25 * signs[:, 0] * (1.0 + signs[:, 1] * eta),
            0.25 * signs[:, 1] * (1.0 + signs[:, 0] * xi),
        ))
        correction = np.linalg.solve(vertices.T @ derivatives, shape @ vertices - point)
        natural -= correction
        if np.linalg.norm(correction, ord=np.inf) <= 1e-12:
            break
    xi, eta = natural
    shape = 0.25 * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta)
    return shape, bool(np.max(np.abs(natural)) <= 1.0 + 1e-10)


def _distance_to_polygon(point: np.ndarray, vertices: np.ndarray) -> float:
    starts = vertices
    segments = np.roll(vertices, -1, axis=0) - starts
    denominator = np.sum(segments * segments, axis=1)
    alpha = np.sum((point - starts) * segments, axis=1) / denominator
    projection = starts + np.clip(alpha, 0.0, 1.0)[:, None] * segments
    return float(np.min(np.linalg.norm(projection - point, axis=1)))


def _vtk_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.column_stack((values, np.zeros(len(values), dtype=values.dtype)))


def _frame_metrics(
    frame: int, image_name: str, time_s: float, error: np.ndarray, mask: np.ndarray,
    mm_per_pixel: float, connected: np.ndarray, valid: np.ndarray,
) -> dict:
    selected = error[mask]
    if selected.size == 0:
        raise ValueError(f"No connected node has valid Abaqus interpolation for {image_name}.")
    magnitude = np.linalg.norm(selected, axis=1)
    rmse = np.sqrt(np.mean(selected * selected, axis=0))
    vector_rmse = float(np.sqrt(np.mean(magnitude * magnitude)))
    maximum = float(np.max(magnitude))
    percentile_95 = float(np.percentile(magnitude, 95.0))
    return {
        "frame": int(frame), "image": image_name, "time_s": time_s,
        "metric_node_count": int(np.count_nonzero(mask)),
        "connected_node_count": int(np.count_nonzero(connected)),
        "valid_interpolation_count": int(np.count_nonzero(valid)),
        "rmse_u1_px": float(rmse[0]), "rmse_u2_px": float(rmse[1]),
        "rmse_vector_px": vector_rmse, "maximum_vector_px": maximum,
        "percentile_95_vector_px": percentile_95,
        "rmse_u1_mm": float(rmse[0] * mm_per_pixel),
        "rmse_u2_mm": float(rmse[1] * mm_per_pixel),
        "rmse_vector_mm": vector_rmse * mm_per_pixel,
        "maximum_vector_mm": maximum * mm_per_pixel,
        "percentile_95_vector_mm": percentile_95 * mm_per_pixel,
    }


def _write_report_csv(path: Path, reports: list[dict]) -> None:
    if not reports:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
