"""Visualization utilities for 2D DIC displacement and strain fields."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class DICPlotter:
    """Plot displacement and strain fields using pixel-wise Q1 interpolation."""

    def __init__(
        self,
        background_image: np.ndarray,
        displacement: np.ndarray,
        strain_fields: Optional[Union[Dict[str, np.ndarray], np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None,
        dic_object: Optional[Any] = None,
    ) -> None:
        """
        Initialize the plotter with image, mesh, displacement, and optional strain fields.

        Parameters
        ----------
        background_image:
            2D array representing the background image (typically the deformed frame).
        displacement:
            Array of nodal displacements with shape (Nnodes, 2), columns ordered as [Ux, Uy].
        strain_fields:
            Optional mapping from strain field names to one-dimensional nodal arrays.
            It can also be either:
              - ``E_all`` tensor of shape (Nnodes, 2, 2) returned by ``compute_green_lagrange_strain_nodes``.
              - A tuple ``(F_all, E_all)`` as returned by ``compute_green_lagrange_strain_nodes``.
        dic_object:
            DIC/FEM study object exposing ``compute_q1_pixel_displacement_field`` and
            ``get_mesh_as_polyCollection``. The object must already have pixel data
            precomputed via ``precompute_pixel_data``.
        """
        self.background_image = background_image
        self.displacement = displacement
        if dic_object is None:
            raise ValueError("dic_object must be provided to DICPlotter.")
        self._dic_object = dic_object
        self._ensure_pixel_data_ready()
        self._image_shape = tuple(self._dic_object._image_shape)
        self._precompute_pixel_projection()
        self._scalar_field_cache: Dict[str, np.ndarray] = {}
        self.strain_fields: Dict[str, np.ndarray] = self._normalize_strain_input(strain_fields)
        self._validate_inputs()
        self._quad_mesh_available = hasattr(self._dic_object, "get_mesh_as_polyCollection")

    @staticmethod
    def _tensor_to_fields(tensor: np.ndarray, prefix: str) -> Dict[str, np.ndarray]:
        if tensor.ndim != 3 or tensor.shape[1:] != (2, 2):
            raise ValueError(f"{prefix}_all tensor must have shape (Nnodes, 2, 2).")
        return {
            f"{prefix}xx": tensor[:, 0, 0],
            f"{prefix}xy": tensor[:, 0, 1],
            f"{prefix}yx": tensor[:, 1, 0],
            f"{prefix}yy": tensor[:, 1, 1],
        }

    def _ensure_pixel_data_ready(self) -> None:
        required_attrs = [
            "pixel_nodes",
            "pixel_shapeN",
            "pixel_coords_ref",
            "_roi_flat_indices",
            "_image_shape",
        ]
        for attr in required_attrs:
            if not hasattr(self._dic_object, attr):
                raise RuntimeError(
                    "dic_object must have pixel data available. Call 'precompute_pixel_data' before plotting."
                )

    def _precompute_pixel_projection(self) -> None:
        pixel_nodes = np.asarray(self._dic_object.pixel_nodes)
        pixel_shape = np.asarray(self._dic_object.pixel_shapeN)
        pixel_coords_ref = np.asarray(self._dic_object.pixel_coords_ref)
        nodal_disp = np.asarray(self.displacement)

        node_values = nodal_disp[pixel_nodes]
        pixel_disp = np.sum(pixel_shape[..., None] * node_values, axis=1)
        pixel_coords_def = pixel_coords_ref + pixel_disp

        self._pixel_nodes = pixel_nodes
        self._pixel_shape = pixel_shape
        self._pixel_coords_ref = pixel_coords_ref
        self._pixel_coords_def = pixel_coords_def
        self._pixel_displacement = pixel_disp

        x_cont = pixel_coords_def[:, 0] - 0.5
        y_cont = pixel_coords_def[:, 1] - 0.5
        self._pixel_j0 = np.floor(x_cont).astype(int)
        self._pixel_i0 = np.floor(y_cont).astype(int)
        self._pixel_fx = x_cont - self._pixel_j0
        self._pixel_fy = y_cont - self._pixel_i0

        self._ux_map = self._scatter_pixel_values(pixel_disp[:, 0])
        self._uy_map = self._scatter_pixel_values(pixel_disp[:, 1])

    def _normalize_strain_input(
        self,
        strain_input: Optional[Union[Dict[str, np.ndarray], np.ndarray, Tuple[np.ndarray, np.ndarray]]],
    ) -> Dict[str, np.ndarray]:
        if strain_input is None:
            return {}
        if isinstance(strain_input, dict):
            return {key: np.asarray(val) for key, val in strain_input.items()}
        if isinstance(strain_input, (tuple, list)) and len(strain_input) == 2:
            F_all, E_all = strain_input
            fields = {}
            fields.update(self._tensor_to_fields(np.asarray(F_all), prefix="F"))
            fields.update(self._tensor_to_fields(np.asarray(E_all), prefix="E"))
            return fields
        if isinstance(strain_input, np.ndarray):
            return self._tensor_to_fields(np.asarray(strain_input), prefix="E")
        raise TypeError("strain_fields must be a dict, a strain tensor array, or a (F_all, E_all) tuple.")

    def _validate_inputs(self) -> None:
        if self.background_image.ndim != 2:
            raise ValueError("background_image must be a 2D array.")

        n_nodes = int(self.displacement.shape[0])
        if self.displacement.ndim != 2 or self.displacement.shape[1] != 2:
            raise ValueError("displacement must have shape (Nnodes, 2).")

        for name, values in self.strain_fields.items():
            if values.shape[0] != n_nodes:
                raise ValueError(f"strain field '{name}' must have length Nnodes.")
            if values.ndim != 1:
                raise ValueError(f"strain field '{name}' must be one-dimensional.")

    @staticmethod
    def _validate_component(component: str) -> str:
        normalized = component.strip().lower()
        if normalized not in {"ux", "uy"}:
            raise ValueError("component must be 'Ux' or 'Uy'.")
        return normalized

    def _get_strain_field(self, field_name: str) -> np.ndarray:
        if field_name not in self.strain_fields:
            raise ValueError(f"strain field '{field_name}' is not available.")
        return self.strain_fields[field_name]

    def _quad_mesh_collection(self):
        if not self._quad_mesh_available:
            return None
        return self._dic_object.get_mesh_as_polyCollection(
            displacement=np.asarray(self.displacement),
            facecolors="none",
            edgecolors="k",
            linewidths=0.5,
        )

    def _get_component_map(self, component: str) -> np.ndarray:
        normalized = self._validate_component(component)
        return self._ux_map if normalized == "ux" else self._uy_map

    def _interpolate_scalar_field(self, nodal_values: np.ndarray) -> np.ndarray:
        values = np.asarray(nodal_values)
        expected = int(self.displacement.shape[0])
        if values.ndim != 1 or values.shape[0] != expected:
            raise ValueError(f"Scalar field must have shape ({expected},).")

        pixel_vals = np.sum(self._pixel_shape * values[self._pixel_nodes], axis=1)
        return self._scatter_pixel_values(pixel_vals)

    def _get_scalar_field_map(self, field_name: str) -> np.ndarray:
        if field_name not in self._scalar_field_cache:
            self._scalar_field_cache[field_name] = self._interpolate_scalar_field(self._get_strain_field(field_name))
        return self._scalar_field_cache[field_name]

    def _scatter_pixel_values(self, pixel_values: np.ndarray) -> np.ndarray:
        pixel_values = np.asarray(pixel_values)
        H, W = self._image_shape
        grid = np.full((H, W), np.nan, dtype=pixel_values.dtype)
        if pixel_values.size == 0:
            return grid

        accum = np.zeros((H, W), dtype=pixel_values.dtype)
        weights = np.zeros((H, W), dtype=np.float64)

        def add(di: int, dj: int, w: np.ndarray) -> None:
            i_idx = self._pixel_i0 + di
            j_idx = self._pixel_j0 + dj
            mask = (
                (i_idx >= 0)
                & (i_idx < H)
                & (j_idx >= 0)
                & (j_idx < W)
                & (w > 0)
            )
            if not np.any(mask):
                return
            np.add.at(accum, (i_idx[mask], j_idx[mask]), pixel_values[mask] * w[mask])
            np.add.at(weights, (i_idx[mask], j_idx[mask]), w[mask])

        fx = self._pixel_fx
        fy = self._pixel_fy
        add(0, 0, (1.0 - fx) * (1.0 - fy))
        add(0, 1, fx * (1.0 - fy))
        add(1, 0, (1.0 - fx) * fy)
        add(1, 1, fx * fy)

        mask = weights > 0
        grid[mask] = accum[mask] / weights[mask]
        return grid

    def _plot_field_overlay(
        self,
        field_map: np.ndarray,
        label: str,
        cmap: str,
        image_alpha: float,
        figsize: Tuple[float, float],
        plotmesh: bool,
    ) -> Tuple[Figure, Axes]:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.background_image, cmap="gray", origin="lower", alpha=1.0)
        masked = np.ma.array(field_map, mask=~np.isfinite(field_map))
        mesh = ax.imshow(masked, cmap=cmap, origin="lower", alpha=image_alpha)
        if plotmesh:
            quad_mesh = self._quad_mesh_collection()
            if quad_mesh is not None:
                ax.add_collection(quad_mesh)
        ax.set_aspect("equal")
        ax.set_title(label)
        fig.colorbar(mesh, ax=ax, label=label)
        return fig, ax

    @staticmethod
    def _latex_label(name: str, field_type: str) -> str:
        if field_type == "displacement":
            mapping = {"ux": "$U_x$", "uy": "$U_y$"}
            return mapping.get(name.strip().lower(), f"${name}$")

        if field_type == "strain":
            trimmed = name.strip()
            if len(trimmed) == 3 and trimmed[0].lower() in {"e", "f"}:
                index_map = {"x": "1", "y": "2"}
                second = index_map.get(trimmed[1].lower())
                third = index_map.get(trimmed[2].lower())
                if second and third:
                    prefix = trimmed[0].upper()
                    return f"${prefix}_{{{second}{third}}}$"
            return f"${trimmed}$"

        return f"${name}$"

    def plot_displacement_component(
        self,
        component: str = "Ux",
        image_alpha: float = 0.6,
        cmap: str = "jet",
        figsize: Tuple[float, float] = (6.0, 6.0),
        plotmesh: bool = True,
    ) -> Tuple[Figure, Axes]:
        """
        Plot a single displacement component overlaid on the background image.

        Parameters
        ----------
        component:
            Either "Ux" or "Uy" to select the component to display.
        image_alpha:
            Transparency factor for the displacement field overlay.
        cmap:
            Matplotlib colormap name for the field.
        figsize:
            Figure size in inches as (width, height).
        """
        label = self._latex_label(component, "displacement")
        field_map = self._get_component_map(component)
        return self._plot_field_overlay(field_map, label, cmap, image_alpha, figsize, plotmesh)

    def plot_strain_component(
        self,
        field_name: str,
        image_alpha: float = 0.6,
        cmap: str = "jet",
        figsize: Tuple[float, float] = (6.0, 6.0),
        plotmesh: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot a scalar strain component using Q1 interpolation over the ROI."""

        label = self._latex_label(field_name, "strain")
        field_map = self._get_scalar_field_map(field_name)
        return self._plot_field_overlay(field_map, label, cmap, image_alpha, figsize, plotmesh)
