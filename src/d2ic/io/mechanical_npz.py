from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DICHistoryData:
    """Minimal immutable transfer object from DIC to mechanical post-processing."""

    nodes_xy: np.ndarray
    elements: np.ndarray
    displacement: np.ndarray
    frame_names: np.ndarray


def load_dic_history_npz(path: str | Path, prepend_reference: bool = True) -> DICHistoryData:
    """Load and validate a D2IC displacement history archive."""
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = {"nodes_xy", "elements"}.difference(archive.files)
        if missing:
            raise ValueError(f"Missing NPZ fields: {sorted(missing)}")
        displacement_key = "u_nodal" if "u_nodal" in archive.files else "displacement"
        if displacement_key not in archive.files:
            raise ValueError("The NPZ must contain 'u_nodal' or 'displacement'.")
        nodes = np.asarray(archive["nodes_xy"], dtype=np.float64)
        elements = np.asarray(archive["elements"], dtype=np.int32)
        displacement = np.asarray(archive[displacement_key], dtype=np.float64)
        names = np.asarray(
            archive["def_paths"] if "def_paths" in archive.files else np.arange(displacement.shape[0])
        )

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes_xy must have shape (Nn,2), got {nodes.shape}")
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError(f"Only Q4 elements are supported, got {elements.shape}")
    if displacement.ndim != 3 or displacement.shape[1:] != (nodes.shape[0], 2):
        raise ValueError(f"displacement must have shape (Nt,Nn,2), got {displacement.shape}")
    if prepend_reference:
        displacement = np.concatenate((np.zeros_like(displacement[:1]), displacement), axis=0)
        names = np.concatenate((np.asarray(["reference"]), names.astype(str)))
    return DICHistoryData(nodes, elements, displacement, names)


def save_constitutive_npz(path: str | Path, **fields: np.ndarray) -> None:
    """Save constitutive Gauss-point fields in a portable compressed archive."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{key: np.asarray(value) for key, value in fields.items()})
