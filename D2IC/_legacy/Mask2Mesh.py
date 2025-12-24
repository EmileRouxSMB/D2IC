"""Turn binary ROI masks into quad meshes through the Gmsh Python API.

We read an ROI, extract white regions with OpenCV, rebuild the curve network,
and let Gmsh recombine triangles into quads when requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import gmsh
import numpy as np


@dataclass
class RoiMeshConfig:
    """Holds the geometry/meshing knobs used by :class:`RoiMeshGenerator`.

    Attributes
    ----------
    image_path:
        ROI image where white pixels mark the area to mesh.
    element_size:
        Target spacing assigned to all generated Gmsh points.
    threshold:
        Grayscale cutoff that separates foreground/background.
    approx_tolerance:
        Relative tolerance passed to ``cv2.approxPolyDP``.
    recombine:
        Let Gmsh recombine triangles into quads when ``True``.
    corner_angle_deg:
        Minimum deviation from a straight angle to treat a point as a corner.
    line_deviation_factor:
        Allowed offset from the best fitting line, relative to ``element_size``.
    arc_fit_tolerance:
        Max RMS misfit between sampled points and their fitted circle (fraction of radius).
    min_arc_angle_deg:
        Smallest arc span we keep when deciding between arcs and splines.
    """

    image_path: Path
    element_size: float
    threshold: int = 127
    approx_tolerance: float = 10.0e-3
    recombine: bool = True
    corner_angle_deg: float = 25.0
    line_deviation_factor: float = 0.15
    arc_fit_tolerance: float = 0.02
    min_arc_angle_deg: float = 10.0


class _PointManager:
    """Caches Gmsh point tags so shared vertices stay consistent."""

    def __init__(self, element_size: float) -> None:
        self.element_size = element_size
        self._cache: Dict[Tuple[float, float], int] = {}

    def tag(self, point: np.ndarray | Iterable[float]) -> int:
        coords = tuple(float(coord) for coord in point)
        if len(coords) != 2:
            raise ValueError("2D points are required to build the mesh edges.")
        x, y = coords
        key = (x, y)
        if key not in self._cache:
            self._cache[key] = gmsh.model.geo.addPoint(x, y, 0.0, self.element_size)
        return self._cache[key]

    def auxiliary(self, point: np.ndarray | Iterable[float]) -> int:
        coords = tuple(float(coord) for coord in point)
        if len(coords) != 2:
            raise ValueError("2D points are required to build the mesh edges.")
        x, y = coords
        return gmsh.model.geo.addPoint(x, y, 0.0, self.element_size)


class RoiMeshGenerator:
    """Builds a conforming quad-dominant mesh from a binary ROI mask."""

    def __init__(self, config: RoiMeshConfig) -> None:
        self.config = config
        self._curve_loops: Dict[int, int] = {}
        self._surface_tags: List[int] = []

    def generate(self, msh_path: Optional[Path] = None) -> Optional[Path]:
        """Run the full pipeline and optionally dump a ``.msh`` file."""

        gmsh.initialize()
        try:
            gmsh.model.add("roi_mesh")
            self._build_geometry()
            gmsh.model.geo.synchronize()

            if self.config.recombine:
                gmsh.option.setNumber("Mesh.RecombineAll", 1)

            gmsh.model.mesh.generate(2)
            gmsh.model.addPhysicalGroup(2, self._surface_tags, 1)
            gmsh.model.setPhysicalName(2, 1, "ROI")

            if msh_path is not None:
                gmsh.write(str(msh_path))
                return msh_path
            return None
        finally:
            gmsh.finalize()

    # Geometry assembly --------------------------------------------------
    def _build_geometry(self) -> None:
        self._curve_loops.clear()
        self._surface_tags.clear()

        mask = self._load_binary_mask()
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No white regions were found in the provided ROI image.")
        if hierarchy is None:
            raise ValueError("Contour hierarchy is missing; cannot build curve nesting.")

        hierarchy = hierarchy[0]
        depth_cache: Dict[int, int] = {}

        for idx, contour in enumerate(contours):
            loop_tag = self._add_curve_loop(self._simplify(contour))
            self._curve_loops[idx] = loop_tag
            depth_cache[idx] = self._depth(idx, hierarchy, depth_cache)

        for idx, loop_tag in self._curve_loops.items():
            if depth_cache[idx] % 2 != 0:
                continue
            hole_loops = [self._curve_loops[ch] for ch in self._children(idx, hierarchy)]
            surface = gmsh.model.geo.addPlaneSurface([loop_tag, *hole_loops])
            self._surface_tags.append(surface)

    def _load_binary_mask(self) -> np.ndarray:
        image = cv2.imread(str(self.config.image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Cannot read ROI image at {self.config.image_path}")

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(image, self.config.threshold, 255, cv2.THRESH_BINARY)
        return mask

    def _simplify(self, contour: np.ndarray) -> np.ndarray:
        perimeter = cv2.arcLength(contour, True)
        epsilon = self.config.approx_tolerance * perimeter
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if simplified.shape[0] < 3:
            raise ValueError("Encountered a degenerate contour while simplifying the ROI.")
        return simplified.reshape(-1, 2).astype(float)

    def _segment_points(self, points: np.ndarray) -> List[np.ndarray]:
        if len(points) < 2:
            raise ValueError("Contours must contain at least two points to form a loop.")

        n_points = len(points)
        raw_corners = self._detect_corners(points)
        if len(raw_corners) < 2:
            step = max(1, n_points // 4)
            raw_corners = list(range(0, n_points, step))
        if not raw_corners:
            raw_corners = [0]

        corner_indices = sorted({idx % n_points for idx in raw_corners})
        if 0 not in corner_indices:
            corner_indices.insert(0, 0)
        if len(corner_indices) < 2:
            second = (corner_indices[0] + max(1, n_points // 3)) % n_points
            if second == corner_indices[0]:
                second = (corner_indices[0] + 1) % n_points
            corner_indices.append(second)

        wrap_indices = corner_indices + [corner_indices[0] + n_points]
        segments: List[np.ndarray] = []
        for start, end in zip(wrap_indices[:-1], wrap_indices[1:]):
            length = end - start
            if length <= 0:
                continue
            segment_points = [points[(start + offset) % n_points] for offset in range(length + 1)]
            segments.append(np.array(segment_points, dtype=float))
        return segments

    def _detect_corners(self, points: np.ndarray) -> List[int]:
        n_points = len(points)
        if n_points < 3:
            return list(range(n_points))

        corners: List[int] = []
        angle_threshold = np.deg2rad(self.config.corner_angle_deg)
        for idx in range(n_points):
            prev_pt = points[idx - 1]
            curr_pt = points[idx]
            next_pt = points[(idx + 1) % n_points]
            v1 = prev_pt - curr_pt
            v2 = next_pt - curr_pt
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1.0e-9 or norm2 < 1.0e-9:
                continue
            cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            deviation = abs(np.pi - angle)
            if deviation >= angle_threshold:
                corners.append(idx)
        return corners

    def _build_curves_for_segment(
        self, segment: np.ndarray, point_manager: _PointManager
    ) -> List[int]:
        clean_segment = self._deduplicate_segment(segment)
        if len(clean_segment) < 2:
            return []

        if self._is_line(clean_segment):
            start_tag = point_manager.tag(clean_segment[0])
            end_tag = point_manager.tag(clean_segment[-1])
            return [gmsh.model.geo.addLine(start_tag, end_tag)]

        arc_center = self._fit_arc(clean_segment)
        if arc_center is not None:
            start_tag = point_manager.tag(clean_segment[0])
            end_tag = point_manager.tag(clean_segment[-1])
            center_tag = point_manager.auxiliary(arc_center)
            return [gmsh.model.geo.addCircleArc(start_tag, center_tag, end_tag)]

        point_tags = [point_manager.tag(point) for point in clean_segment]
        return [gmsh.model.geo.addSpline(point_tags)]

    def _deduplicate_segment(self, segment: np.ndarray) -> np.ndarray:
        unique_points = [segment[0]]
        for point in segment[1:]:
            if np.linalg.norm(point - unique_points[-1]) > 1.0e-9:
                unique_points.append(point)
        if len(unique_points) > 1 and np.linalg.norm(unique_points[0] - unique_points[-1]) < 1.0e-9:
            unique_points.pop()
        return np.array(unique_points, dtype=float)

    def _is_line(self, segment: np.ndarray) -> bool:
        if len(segment) <= 2:
            return True
        start = segment[0]
        end = segment[-1]
        vec = end - start
        length = np.linalg.norm(vec)
        if length < 1.0e-9:
            return False
        normal = np.array([-vec[1], vec[0]]) / length
        deviations = np.abs(np.dot(segment - start, normal))
        max_allowed = self.config.line_deviation_factor * self.config.element_size
        return float(np.max(deviations)) <= max_allowed

    def _fit_arc(self, segment: np.ndarray) -> Optional[np.ndarray]:
        if len(segment) < 3:
            return None
        circle = self._fit_circle(segment)
        if circle is None:
            return None
        center, radius, residual = circle
        if radius < 1.0e-6:
            return None
        if residual > self.config.arc_fit_tolerance * radius:
            return None
        start_vec = segment[0] - center
        end_vec = segment[-1] - center
        start_norm = np.linalg.norm(start_vec)
        end_norm = np.linalg.norm(end_vec)
        if start_norm < 1.0e-9 or end_norm < 1.0e-9:
            return None
        cos_angle = np.clip(np.dot(start_vec, end_vec) / (start_norm * end_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < np.deg2rad(self.config.min_arc_angle_deg):
            return None
        return center

    def _fit_circle(self, segment: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
        x = segment[:, 0]
        y = segment[:, 1]
        a_matrix = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
        b_vector = x ** 2 + y ** 2
        try:
            solution, *_ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
        except np.linalg.LinAlgError:
            return None
        cx, cy, c = solution
        radius_sq = cx ** 2 + cy ** 2 + c
        if radius_sq <= 0:
            return None
        center = np.array([cx, cy])
        radius = float(np.sqrt(radius_sq))
        distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        residual = float(np.sqrt(np.mean((distances - radius) ** 2)))
        return center, radius, residual

    def _add_curve_loop(self, points: np.ndarray) -> int:
        point_manager = _PointManager(self.config.element_size)
        segments = self._segment_points(points)
        curves: List[int] = []
        for segment in segments:
            curves.extend(self._build_curves_for_segment(segment, point_manager))
        if not curves:
            raise ValueError("Failed to generate curves for the provided contour.")
        return gmsh.model.geo.addCurveLoop(curves)

    def _children(self, idx: int, hierarchy: np.ndarray) -> Iterable[int]:
        child = hierarchy[idx][2]
        while child != -1:
            yield child
            child = hierarchy[child][0]

    def _depth(
        self, idx: int, hierarchy: np.ndarray, cache: Dict[int, int]
    ) -> int:
        parent = hierarchy[idx][3]
        if parent == -1:
            cache[idx] = 0
            return 0
        if parent in cache:
            cache[idx] = cache[parent] + 1
            return cache[idx]
        cache[parent] = self._depth(parent, hierarchy, cache)
        cache[idx] = cache[parent] + 1
        return cache[idx]


def generate_roi_mesh(
    image_path: str | Path,
    element_size: float,
    msh_path: Optional[str] = None,
    approx_tolerance: float = 10.0e-3,
) -> Optional[Path]:
    """One-shot helper: read ``image_path``, mesh white pixels, optionally save ``msh_path``.

    ``element_size`` sets the target spacing; ``approx_tolerance`` goes to ``cv2.approxPolyDP``.
    """

    config = RoiMeshConfig(
        image_path=Path(image_path),
        element_size=element_size,
        approx_tolerance=approx_tolerance,
    )
    generator = RoiMeshGenerator(config)
    return generator.generate(Path(msh_path) if msh_path else None)


if __name__ == "__main__":
    generate_roi_mesh(Path("../../SandBox/roiB.tif"), element_size=40.0, msh_path="../../SandBox/roi.msh")
