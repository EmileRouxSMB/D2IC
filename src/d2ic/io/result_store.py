"""Versioned, frame-oriented persistence for sequential FE-DIC results."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from tempfile import TemporaryDirectory

import numpy as np

SCHEMA_VERSION = 1


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class ResultStore:
    """Store DIC fields one completed frame at a time.

    A store directory contains ``metadata.json``, ``progress.json``,
    ``frames/frame_NNNN.npz`` and optionally ``checkpoint.npz``. Displacement
    and strain use nodal shapes ``(Nn,2)`` and ``(Nn,3)``; E12 is tensorial.
    """

    def __init__(self, directory: str | Path, *, metadata: dict[str, Any] | None = None):
        self.path = Path(directory)
        self.frames_path = self.path / "frames"
        self.path.mkdir(parents=True, exist_ok=True)
        self.frames_path.mkdir(parents=True, exist_ok=True)
        meta_path = self.path / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if self.metadata.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Unsupported result schema: {self.metadata.get('schema_version')}")
            if self.metadata.get("strain_shear_convention", "tensorial_E12") != "tensorial_E12":
                raise ValueError("Result schema requires tensorial_E12 strain shear.")
            if metadata:
                for key, value in metadata.items():
                    if key in self.metadata and self.metadata[key] != value:
                        raise ValueError(f"Result metadata mismatch for {key!r}.")
                    self.metadata[key] = value
        else:
            self.metadata = {"schema_version": SCHEMA_VERSION, "strain_shear_convention": "tensorial_E12"}
            self.metadata.update(metadata or {})
            _atomic_json(meta_path, self.metadata)
        if self.metadata.get("strain_shear_convention") != "tensorial_E12":
            raise ValueError("Result schema requires tensorial_E12 strain shear.")
        progress_path = self.path / "progress.json"
        if not progress_path.exists():
            self.update_progress(status="running", total_frames=None)

    @property
    def frame_count(self) -> int:
        files = {int(path.stem.split("_")[-1]) for path in self.frames_path.glob("frame_*.npz")}
        contiguous = 0
        while contiguous in files:
            contiguous += 1
        committed = int(self.progress.get("completed_frames", contiguous))
        checkpoint_path = self.path / "checkpoint.npz"
        if checkpoint_path.exists():
            with np.load(checkpoint_path, allow_pickle=False) as archive:
                committed = min(committed, int(archive["completed_frames"]))
        return min(contiguous, committed)

    @property
    def progress(self) -> dict[str, Any]:
        path = self.path / "progress.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def write_mesh(self, nodes_xy: np.ndarray, elements: np.ndarray) -> None:
        nodes = np.asarray(nodes_xy)
        elems = np.asarray(elements)
        validate_mesh(nodes, elems)
        mesh_path = self.path / "mesh.npz"
        if mesh_path.exists():
            existing = self.read_mesh()
            same_topology = np.array_equal(existing["elements"], elems)
            same_coordinates = existing["nodes_xy"].shape == nodes.shape and np.allclose(
                existing["nodes_xy"], nodes, rtol=0.0, atol=1e-10
            )
            if not (same_topology and same_coordinates):
                raise ValueError(
                    "Regenerated mesh differs from the stored mesh (node coordinates "
                    "must match within 1e-10 and connectivity must match exactly); "
                    "refusing to reuse displacement checkpoints."
                )
            return
        _atomic_npz(mesh_path, nodes_xy=nodes, elements=elems)

    def write_roi_mask(self, roi_mask: np.ndarray) -> None:
        """Persist the reference-aligned ROI mask used to build the mesh."""
        mask = np.asarray(roi_mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("roi_mask must have shape (H,W).")
        _atomic_npz(self.path / "roi.npz", roi_mask=mask)

    def write_frame(self, index: int, *, u_nodal: np.ndarray, strain: np.ndarray,
                    discrepancy_ref: np.ndarray | None = None) -> None:
        u, e = np.asarray(u_nodal), np.asarray(strain)
        if index < 0 or u.ndim != 2 or u.shape[-1] != 2:
            raise ValueError("u_nodal must have shape (Nn,2) and index must be nonnegative.")
        if e.shape != (u.shape[0], 3):
            raise ValueError(f"strain must have shape ({u.shape[0]},3), got {e.shape}.")
        payload = {"u_nodal": u, "strain": e}
        if discrepancy_ref is not None:
            payload["discrepancy_ref"] = np.asarray(discrepancy_ref)
        if not np.issubdtype(u.dtype, np.number) or not np.issubdtype(e.dtype, np.number):
            raise ValueError("u_nodal and strain must have numeric dtypes.")
        _atomic_npz(self.frames_path / f"frame_{index:04d}.npz", **payload)

    def read_frame(self, index: int) -> dict[str, np.ndarray]:
        count = self.frame_count
        if index < 0 or index >= count:
            raise IndexError(f"Frame index {index} is outside [0, {count}).")
        return self._read_committed_frame(index)

    def _read_committed_frame(self, index: int) -> dict[str, np.ndarray]:
        # Caller checked the index against one progress/checkpoint snapshot.
        path = self.frames_path / f"frame_{int(index):04d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            frame = {key: np.asarray(archive[key]) for key in archive.files}
        if "u_nodal" not in frame or "strain" not in frame:
            raise ValueError(f"Frame {index} is missing required fields.")
        u, strain = frame["u_nodal"], frame["strain"]
        if u.ndim != 2 or u.shape[-1] != 2 or strain.shape != (u.shape[0], 3):
            raise ValueError(f"Frame {index} violates the nodal result shapes.")
        if not np.issubdtype(u.dtype, np.number) or not np.issubdtype(strain.dtype, np.number):
            raise ValueError(f"Frame {index} fields must have numeric dtypes.")
        mesh_path = self.path / "mesh.npz"
        if mesh_path.exists():
            with np.load(mesh_path, allow_pickle=False) as mesh:
                if u.shape[0] != mesh["nodes_xy"].shape[0]:
                    raise ValueError(f"Frame {index} node count does not match mesh.npz.")
        if "discrepancy_ref" in frame and frame["discrepancy_ref"].ndim != 2:
            raise ValueError(f"Frame {index} discrepancy_ref must have shape (H,W).")
        return frame

    def read_mesh(self) -> dict[str, np.ndarray]:
        with np.load(self.path / "mesh.npz", allow_pickle=False) as archive:
            mesh = {key: np.asarray(archive[key]) for key in archive.files}
        validate_mesh(mesh["nodes_xy"], mesh["elements"])
        return mesh

    def read_history(self, field: str = "u_nodal") -> np.ndarray:
        count = self.frame_count
        if count == 0:
            raise ValueError("No result frames are stored.")
        return np.stack([self._read_committed_frame(k)[field] for k in range(count)])

    def read_node_history(self, node: int, field: str = "u_nodal") -> np.ndarray:
        """Read one nodal value through time without stacking whole fields."""
        count = self.frame_count
        if count == 0:
            raise ValueError('No result frames are stored.')
        first = self._read_committed_frame(0)[field]
        if node < 0 or node >= first.shape[0]:
            raise IndexError(f"Node index {node} is outside [0, {first.shape[0]}).")
        return np.stack([self._read_committed_frame(k)[field][node] for k in range(count)])

    def save_checkpoint(self, *, fingerprint: str, u_prev: np.ndarray | None,
                        u_prevprev: np.ndarray | None, completed_frames: int) -> None:
        arrays: dict[str, Any] = {"fingerprint": np.asarray(fingerprint),
                                  "completed_frames": np.asarray(completed_frames, dtype=np.int64)}
        if u_prev is not None:
            arrays["u_prev"] = np.asarray(u_prev)
        if u_prevprev is not None:
            arrays["u_prevprev"] = np.asarray(u_prevprev)
        _atomic_npz(self.path / "checkpoint.npz", **arrays)

    def load_checkpoint(self, fingerprint: str) -> dict[str, Any] | None:
        path = self.path / "checkpoint.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["fingerprint"].item()) != fingerprint:
                return None
            checkpoint = {key: np.asarray(archive[key]) for key in archive.files}
        completed = int(checkpoint["completed_frames"])
        if completed < 0:
            raise ValueError("Checkpoint completed_frames must be nonnegative.")
        if completed:
            mesh = self.read_mesh()
            expected = (mesh["nodes_xy"].shape[0], 2)
            for key in ("u_prev", "u_prevprev"):
                if key in checkpoint and (checkpoint[key].shape != expected or
                                          not np.issubdtype(checkpoint[key].dtype, np.number)):
                    raise ValueError(f"Checkpoint {key} must be numeric with shape {expected}.")
            if "u_prev" not in checkpoint:
                raise ValueError("A nonempty checkpoint requires u_prev.")
        return checkpoint

    def update_progress(self, *, status: str, total_frames: int | None = None,
                        completed_frames: int | None = None, last_frame: int | None = None,
                        error: str | None = None) -> None:
        old = self.progress
        value = {"schema_version": SCHEMA_VERSION, "status": status,
                 "completed_frames": old.get("completed_frames", 0) if completed_frames is None else int(completed_frames),
                 "total_frames": total_frames if total_frames is not None else old.get("total_frames"),
                 "last_frame": old.get("last_frame") if last_frame is None else int(last_frame),
                 "updated_at": datetime.now(timezone.utc).isoformat()}
        if error:
            value["error"] = error
        _atomic_json(self.path / "progress.json", value)

    def export_npz(self, path: str | Path) -> None:
        """Build the legacy monolithic NPZ using disk-backed temporary stacks."""
        mesh = self.read_mesh()
        count = self.frame_count
        if count == 0:
            raise ValueError("No committed result frames to export.")
        first = self._read_committed_frame(0)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent) as temporary:
            temporary_path = Path(temporary)
            u_stack = np.lib.format.open_memmap(
                temporary_path / "u_nodal.npy", mode="w+", dtype=first["u_nodal"].dtype,
                shape=(count, *first["u_nodal"].shape))
            strain_stack = np.lib.format.open_memmap(
                temporary_path / "strain.npy", mode="w+", dtype=first["strain"].dtype,
                shape=(count, *first["strain"].shape))
            has_discrepancy = all("discrepancy_ref" in self._read_committed_frame(k) for k in range(count))
            discrepancy_stack = None
            if has_discrepancy:
                d0 = first["discrepancy_ref"]
                discrepancy_stack = np.lib.format.open_memmap(
                    temporary_path / "discrepancy.npy", mode="w+", dtype=d0.dtype,
                    shape=(count, *d0.shape))
            for k in range(count):
                frame = self._read_committed_frame(k)
                u_stack[k] = frame["u_nodal"]
                strain_stack[k] = frame["strain"]
                if discrepancy_stack is not None:
                    discrepancy_stack[k] = frame["discrepancy_ref"]
            payload: dict[str, Any] = {**mesh, "u_nodal": u_stack, "strain": strain_stack}
            roi_path = self.path / "roi.npz"
            if roi_path.exists():
                with np.load(roi_path, allow_pickle=False) as archive:
                    payload["roi_mask"] = np.asarray(archive["roi_mask"], dtype=bool)
            payload.update({key: np.asarray(value) for key, value in self.metadata.items()
                            if key != "schema_version" and isinstance(value, (str, int, float, list, np.ndarray))})
            if discrepancy_stack is not None:
                payload["discrepancy_ref"] = discrepancy_stack
            fd, temporary_name = tempfile.mkstemp(dir=target.parent, suffix=".npz")
            os.close(fd)
            try:
                np.savez_compressed(temporary_name, **payload)
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)


class LegacyNPZReader:
    """Read interface over the former monolithic result archive."""

    def __init__(self, path: Path):
        self.path = path
        with np.load(path, allow_pickle=False) as archive:
            self._metadata = {key: np.asarray(archive[key]) for key in archive.files
                              if key not in ("u_nodal", "strain", "discrepancy_ref")}
            self._count = int(archive["u_nodal"].shape[0])

    @property
    def metadata(self):
        return self._metadata

    @property
    def frame_count(self):
        return self._count

    def read_frame(self, index: int) -> dict[str, np.ndarray]:
        if index < 0 or index >= self._count:
            raise IndexError(f"Frame index {index} is outside [0, {self._count}).")
        with np.load(self.path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key][index]) for key in ("u_nodal", "strain", "discrepancy_ref") if key in archive.files}

    def read_history(self, field: str = "u_nodal") -> np.ndarray:
        with np.load(self.path, allow_pickle=False) as archive:
            return np.asarray(archive[field])

    def read_node_history(self, node: int, field: str = "u_nodal") -> np.ndarray:
        with np.load(self.path, allow_pickle=False) as archive:
            values = archive[field]
            if node < 0 or node >= values.shape[1]:
                raise IndexError(f"Node index {node} is outside [0, {values.shape[1]}).")
            return np.asarray(values[:, node])

    def read_mesh(self) -> dict[str, np.ndarray]:
        with np.load(self.path, allow_pickle=False) as archive:
            result = {key: np.asarray(archive[key]) for key in ("nodes_xy", "elements")}
        validate_mesh(result["nodes_xy"], result["elements"])
        return result


def validate_mesh(nodes_xy: np.ndarray, elements: np.ndarray) -> None:
    if nodes_xy.ndim != 2 or nodes_xy.shape[1] != 2:
        raise ValueError(f"nodes_xy must have shape (Nn,2), got {nodes_xy.shape}.")
    if not np.issubdtype(nodes_xy.dtype, np.number):
        raise ValueError("nodes_xy must have a numeric dtype.")
    if elements.ndim != 2 or elements.shape[1] not in (3, 4):
        raise ValueError(f"elements must have shape (Ne,3|4), got {elements.shape}.")
    if not np.issubdtype(elements.dtype, np.integer):
        raise ValueError("elements must have an integer dtype.")
    if elements.size and (elements.min() < 0 or elements.max() >= nodes_xy.shape[0]):
        raise ValueError("elements contain node indices outside nodes_xy.")


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def open_result_store(path: str | Path) -> ResultStore | LegacyNPZReader:
    """Open a versioned store directory or a legacy monolithic NPZ."""
    target = Path(path)
    if target.is_dir():
        return ResultStore(target)
    return LegacyNPZReader(target)
