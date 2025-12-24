from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Sequence
import os
import tempfile

import numpy as np

try:  # pragma: no cover
    import jax.numpy as jnp
    _ARRAY_LIB = jnp
except Exception:  # pragma: no cover
    _ARRAY_LIB = np

from .mesh_assets import Mesh, MeshAssets, compute_element_centers, make_mesh_assets


def downsample_mask(mask: np.ndarray, binning: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if binning <= 1:
        return mask
    h, w = mask.shape
    pad_h = (-h) % binning
    pad_w = (-w) % binning
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    new_h = mask.shape[0] // binning
    new_w = mask.shape[1] // binning
    reshaped = mask.reshape(new_h, binning, new_w, binning)
    return np.any(reshaped, axis=(1, 3))


def mask_to_mesh(
    mask: np.ndarray,
    element_size_px: float,
    binning: int = 1,
    origin_xy: Tuple[float, float] = (0.0, 0.0),
    enforce_quads: bool = True,
    remove_islands: bool = True,
    min_island_area_px: int = 64,
) -> Mesh:
    if not enforce_quads:
        raise NotImplementedError("Only quadrilateral meshes are supported at the moment.")
    if element_size_px <= 0:
        raise ValueError("element_size_px must be positive")

    mask_bool = np.asarray(mask, dtype=bool)
    if remove_islands:
        mask_bool = _remove_small_components(mask_bool, min_island_area_px)
    mask_ds = downsample_mask(mask_bool, binning)
    if not np.any(mask_ds):
        raise ValueError("mask_to_mesh: no ROI pixels remain after preprocessing.")

    step = max(1, int(round(element_size_px / max(1, binning))))
    h, w = mask_ds.shape
    grid_x = _build_axis_coords(w, step)
    grid_y = _build_axis_coords(h, step)
    nx = grid_x.size
    ny = grid_y.size

    meshgrid = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).reshape(-1, 2)
    node_coords = meshgrid.astype(np.float64)
    node_coords[:, 0] = origin_xy[0] + node_coords[:, 0] * binning
    node_coords[:, 1] = origin_xy[1] + node_coords[:, 1] * binning

    elements = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            cx = 0.5 * (grid_x[ix] + grid_x[ix + 1])
            cy = 0.5 * (grid_y[iy] + grid_y[iy + 1])
            mx = min(int(round(cx)), w - 1)
            my = min(int(round(cy)), h - 1)
            if not mask_ds[my, mx]:
                continue
            n0 = iy * nx + ix
            n1 = n0 + 1
            n2 = n0 + nx + 1
            n3 = n0 + nx
            elements.append([n0, n1, n2, n3])

    if not elements:
        raise ValueError("mask_to_mesh: no elements found inside the ROI.")

    nodes_array = _ARRAY_LIB.asarray(node_coords)
    elements_array = _ARRAY_LIB.asarray(np.asarray(elements, dtype=np.int32))
    _sanity_check_mesh(nodes_array, elements_array)
    return Mesh(nodes_xy=nodes_array, elements=elements_array)


def mask_to_mesh_assets(**kwargs) -> tuple[Mesh, MeshAssets]:
    mesh = mask_to_mesh(**kwargs)
    centers = compute_element_centers(mesh)
    assets = MeshAssets(mesh=mesh, element_centers_xy=centers, pixel_data=None)
    return mesh, assets


def mask_to_mesh_gmsh(
    mask: np.ndarray,
    element_size_px: float,
    binning: int = 1,
    origin_xy: Tuple[float, float] = (0.0, 0.0),
    contour_step_px: float = 2.0,
    remove_islands: bool = True,
    min_island_area_px: int = 64,
) -> Mesh:
    """
    Generate a quad mesh with Gmsh from a binary ROI mask.

    Parameters mirror ``mask_to_mesh`` but the meshing is performed via the
    Gmsh Python API followed by ``meshio`` import, enabling more robust
    boundary handling in stage-2 workflows.
    """
    try:
        import gmsh  # type: ignore
        import meshio  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mask_to_mesh_gmsh requires gmsh and meshio packages.") from exc

    mask_bool = np.asarray(mask, dtype=bool)
    if remove_islands:
        mask_bool = _remove_small_components(mask_bool, min_island_area_px)
    mask_ds = downsample_mask(mask_bool, binning)
    if not np.any(mask_ds):
        raise ValueError("mask_to_mesh_gmsh: no ROI pixels remain after preprocessing.")

    contours = _extract_contours(mask_ds, contour_step_px=contour_step_px, binning=binning)
    if not contours:
        ys, xs = np.nonzero(mask_ds)
        if ys.size == 0:
            raise ValueError("mask_to_mesh_gmsh: could not extract ROI contours from mask.")
        y0 = ys.min()
        y1 = ys.max()
        x0 = xs.min()
        x1 = xs.max()
        rect = np.array(
            [
                [x0 * binning, y0 * binning],
                [x1 * binning, y0 * binning],
                [x1 * binning, y1 * binning],
                [x0 * binning, y1 * binning],
                [x0 * binning, y0 * binning],
            ],
            dtype=float,
        )
        contours = [rect]

    tmp_path = None
    gmsh.initialize([])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model_name = "mask_roi"
        gmsh.model.add(model_name)
        curve_loops: list[int] = []
        major_loop: int | None = None
        for idx, contour in enumerate(contours):
            if contour.shape[0] < 4:
                continue
            coords = np.column_stack(
                [
                    origin_xy[0] + contour[:, 0],
                    origin_xy[1] + contour[:, 1],
                ]
            )
            area = _signed_area(coords)
            if np.isclose(area, 0.0):
                continue
            if area < 0.0:
                coords = coords[::-1]

            point_tags: list[int] = []
            for (x, y) in coords:
                tag = gmsh.model.geo.addPoint(float(x), float(y), 0.0, element_size_px)
                point_tags.append(tag)
            if point_tags[0] != point_tags[-1]:
                point_tags.append(point_tags[0])
            spline = gmsh.model.geo.addSpline(point_tags)
            loop = gmsh.model.geo.addCurveLoop([spline])
            if idx == 0:
                major_loop = loop
            else:
                curve_loops.append(loop)

        if major_loop is None:
            raise ValueError("mask_to_mesh_gmsh: unable to build outer loop from mask contours.")

        surface = gmsh.model.geo.addPlaneSurface([major_loop] + [-loop for loop in curve_loops])
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", element_size_px)
        gmsh.option.setNumber("Mesh.MeshSizeMax", element_size_px)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.setRecombine(2, surface)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

        fd, tmp_path = tempfile.mkstemp(suffix=".msh")
        os.close(fd)
        gmsh.write(tmp_path)
    finally:
        gmsh.finalize()

    try:
        gmsh_mesh = meshio.read(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if "quad" not in gmsh_mesh.cells_dict:
        raise ValueError(
            "mask_to_mesh_gmsh: gmsh did not produce quadrilateral elements. "
            "Ensure Mesh.RecombineAll=1 succeeds for the ROI."
        )

    nodes_xy = gmsh_mesh.points[:, :2]
    elements = gmsh_mesh.cells_dict["quad"].astype(np.int32)
    nodes_array = _ARRAY_LIB.asarray(nodes_xy)
    elements_array = _ARRAY_LIB.asarray(elements)
    return Mesh(nodes_xy=nodes_array, elements=elements_array)


def mask_to_mesh_assets_gmsh(**kwargs) -> tuple[Mesh, MeshAssets]:
    mesh = mask_to_mesh_gmsh(**kwargs)
    assets = make_mesh_assets(mesh, with_neighbors=True)
    return mesh, assets


def _build_axis_coords(size: int, step: int) -> np.ndarray:
    count = max(2, int(np.floor((size - 1) / step)) + 1)
    coords = np.arange(count, dtype=np.float64) * step
    last_val = coords[-1]
    if last_val < size - 1:
        coords = np.append(coords, size - 1)
    return coords


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = -np.ones_like(mask, dtype=np.int32)
    area = []
    label = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != -1:
                continue
            stack = [(y, x)]
            labels[y, x] = label
            count = 0
            while stack:
                cy, cx = stack.pop()
                count += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == -1:
                        labels[ny, nx] = label
                        stack.append((ny, nx))
            area.append(count)
            label += 1

    if label == 0:
        return mask

    area = np.asarray(area)
    keep = area >= max(1, min_area)
    result = np.zeros_like(mask)
    for idx, flag in enumerate(keep):
        if flag:
            result |= labels == idx
    return result


def _sanity_check_mesh(nodes: Array, elements: Array) -> None:
    if np.isnan(np.asarray(nodes)).any():
        raise ValueError("Mesh nodes contain NaNs")
    max_index = nodes.shape[0] - 1
    elem = np.asarray(elements)
    if elem.size == 0 or elem.shape[1] != 4:
        raise ValueError("Elements must be quad connectivity")
    if np.any(elem < 0) or np.any(elem > max_index):
        raise ValueError("Element connectivity out of bounds")


def _extract_contours(mask: np.ndarray, contour_step_px: float, binning: int) -> list[np.ndarray]:
    try:
        from skimage import measure  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mask_to_mesh_gmsh requires scikit-image for contour extraction.") from exc

    mask_float = mask.astype(float)
    contours = measure.find_contours(mask_float, 0.5)
    if not contours:
        return []
    stride = max(1, int(round(max(1.0, contour_step_px) / max(1, binning))))
    processed: list[np.ndarray] = []
    for contour in contours:
        pts = contour[::stride]
        if pts.shape[0] < 4:
            continue
        # convert (row, col) -> (x, y) with scaling by binning
        xy = np.column_stack([pts[:, 1] * binning, pts[:, 0] * binning])
        processed.append(xy)
    processed.sort(key=lambda arr: abs(_signed_area(arr)), reverse=True)
    return processed


def _signed_area(coords: np.ndarray) -> float:
    if len(coords) < 3:
        return 0.0
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
