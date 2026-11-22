"""ParaView exports for mesh-based DIC results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping
import base64
import struct
import xml.etree.ElementTree as ET

import meshio
import numpy as np

from ..core.dataclasses import DICResult
from ..mesh.assets import Mesh


def write_dic_vtu(path: str | Path, mesh: Mesh, result: DICResult) -> Path:
    """Write one DIC state as an unstructured VTK mesh readable by ParaView.

    Nodal displacement is exported both as a 3-component vector and as scalar
    components. Strain ``[E11, E22, E12]`` is exported as scalar components and
    as a symmetric 3D tensor, with zero out-of-plane terms.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes_xy = np.asarray(mesh.nodes_xy)
    elements = np.asarray(mesh.elements, dtype=np.int32)
    _validate_mesh(nodes_xy, elements)

    displacement_xy = np.asarray(result.displacement)
    if displacement_xy.shape != nodes_xy.shape:
        raise ValueError(
            "displacement must have shape (n_nodes, 2); "
            f"got {displacement_xy.shape} for {nodes_xy.shape[0]} nodes."
        )

    zero = np.zeros(nodes_xy.shape[0], dtype=np.result_type(nodes_xy, displacement_xy))
    points = np.column_stack((nodes_xy, zero))
    displacement = np.column_stack((displacement_xy, zero))
    point_data: dict[str, np.ndarray] = {
        "Displacement": displacement,
        "Displacement_magnitude": np.linalg.norm(displacement_xy, axis=1),
        "Deformed_position": points + displacement,
    }

    if result.strain is not None:
        strain = np.asarray(result.strain)
        if strain.shape != (nodes_xy.shape[0], 3):
            raise ValueError(
                "strain must have shape (n_nodes, 3) in [E11, E22, E12] order; "
                f"got {strain.shape}."
            )
        e11, e22, e12 = strain.T
        strain_tensor = np.column_stack(
            (e11, e12, zero, e12, e22, zero, zero, zero, zero)
        )
        point_data.update({"Strain": strain_tensor, "E11": e11, "E22": e22, "E12": e12})

    for name, values in getattr(result, "fields", {}).items():
        data = np.asarray(values)
        if data.shape[0] != nodes_xy.shape[0]:
            raise ValueError(f"field {name!r} must have one value per mesh node.")
        point_data[str(name)] = data

    vtk_mesh = meshio.Mesh(
        points=points,
        cells=[(_meshio_cell_type(elements.shape[1]), elements)],
        point_data=point_data,
    )
    meshio.write(path, vtk_mesh, file_format="vtu")
    return path


def write_image_vti(path: str | Path, image: np.ndarray, *, field_name: str = "Image") -> Path:
    """Write a 2D grayscale image as VTK ImageData in pixel coordinates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(image, dtype="<f4")
    if values.ndim != 2:
        raise ValueError(f"image must have shape (height, width); got {values.shape}.")
    height, width = values.shape
    payload = struct.pack("<Q", values.nbytes) + values.tobytes(order="C")
    encoded = base64.b64encode(payload).decode("ascii")
    extent = f"0 {width - 1} 0 {height - 1} 0 0"
    vtk_file = ET.Element(
        "VTKFile", type="ImageData", version="1.0",
        byte_order="LittleEndian", header_type="UInt64",
    )
    image_data = ET.SubElement(
        vtk_file, "ImageData", WholeExtent=extent,
        Origin="0 0 0", Spacing="1 1 1",
    )
    piece = ET.SubElement(image_data, "Piece", Extent=extent)
    point_data = ET.SubElement(piece, "PointData", Scalars=field_name)
    array = ET.SubElement(
        point_data, "DataArray", type="Float32", Name=field_name, format="binary"
    )
    array.text = encoded
    ET.SubElement(piece, "CellData")
    ET.ElementTree(vtk_file).write(path, encoding="utf-8", xml_declaration=True)
    return path


def write_pvd(
    path: str | Path,
    datasets: Iterable[tuple[float, str | Path] | tuple[float, str | Path, int]],
) -> Path:
    """Write a ParaView collection of time, path and optional part index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vtk_file = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(vtk_file, "Collection")
    for dataset in datasets:
        time_value, dataset_path = dataset[:2]
        part = int(dataset[2]) if len(dataset) == 3 else 0
        dataset_path = Path(dataset_path)
        try:
            filename = dataset_path.relative_to(path.parent).as_posix()
        except ValueError:
            filename = dataset_path.as_posix()
        ET.SubElement(
            collection, "DataSet", timestep=f"{float(time_value):.16g}",
            group="", part=str(part), file=filename,
        )
    ET.ElementTree(vtk_file).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _meshio_cell_type(nodes_per_element: int) -> str:
    cell_types: Mapping[int, str] = {3: "triangle", 4: "quad"}
    try:
        return cell_types[nodes_per_element]
    except KeyError as exc:
        raise ValueError("VTU export supports only 3-node triangles and 4-node quadrilaterals.") from exc


def _validate_mesh(nodes_xy: np.ndarray, elements: np.ndarray) -> None:
    if nodes_xy.ndim != 2 or nodes_xy.shape[1] != 2:
        raise ValueError(f"mesh.nodes_xy must have shape (n_nodes, 2); got {nodes_xy.shape}.")
    if elements.ndim != 2 or elements.shape[1] not in (3, 4):
        raise ValueError(f"mesh.elements must contain triangles or quadrilaterals; got {elements.shape}.")
    if elements.size and (elements.min() < 0 or elements.max() >= nodes_xy.shape[0]):
        raise ValueError("mesh.elements contains an invalid node index.")
