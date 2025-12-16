"""Visualization utilities for 2D DIC displacement and strain fields."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation


class DICPlotter:
    """Plot displacement and strain fields on a triangulated mesh."""

    def __init__(
        self,
        background_image: np.ndarray,
        displacement: np.ndarray,
        triangulation: Optional[Triangulation] = None,
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
        triangulation:
            Triangulation describing the mesh coordinates. Optional if ``dic_object`` is provided.
        strain_fields:
            Optional mapping from strain field names to one-dimensional nodal arrays.
            It can also be either:
              - ``E_all`` tensor of shape (Nnodes, 2, 2) returned by ``compute_green_lagrange_strain_nodes``.
              - A tuple ``(F_all, E_all)`` as returned by ``compute_green_lagrange_strain_nodes``.
        dic_object:
            Optional DIC/FEM study object exposing ``get_mesh_as_triangulation`` and
            ``get_mesh_as_polyCollection``. If provided, the triangulation and quad mesh
            overlays are built directly from this object using the given displacement.
        """
        self.background_image = background_image
        self.displacement = displacement
        self._dic_object = dic_object
        self.triangulation = triangulation or self._build_triangulation_from_dic()
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

    def _build_triangulation_from_dic(self) -> Triangulation:
        if self._dic_object is None:
            raise ValueError("Either 'triangulation' or 'dic_object' must be provided.")
        if not hasattr(self._dic_object, "get_mesh_as_triangulation"):
            raise ValueError("dic_object must implement 'get_mesh_as_triangulation'.")
        return self._dic_object.get_mesh_as_triangulation(displacement=np.asarray(self.displacement))

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

        n_nodes = self.triangulation.x.size
        if self.displacement.ndim != 2 or self.displacement.shape[1] != 2:
            raise ValueError("displacement must have shape (Nnodes, 2).")
        if self.displacement.shape[0] != n_nodes:
            raise ValueError("displacement and triangulation must have the same number of nodes.")

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

    def _get_displacement_component(self, component: str) -> np.ndarray:
        normalized = self._validate_component(component)
        idx = 0 if normalized == "ux" else 1
        return self.displacement[:, idx]

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
        values = self._get_displacement_component(component)
        label = self._latex_label(component, "displacement")

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.background_image, cmap="gray", origin="lower", alpha=1.0)
        mesh = ax.tripcolor(
            self.triangulation,
            values,
            shading="gouraud",
            cmap=cmap,
            alpha=image_alpha,
            edgecolors="none",
            linewidth=0.0,
        )
        mesh.set_edgecolors("none")
        mesh.set_linewidth(0.0)
        mesh.set_antialiaseds(False)
        quad_mesh = self._quad_mesh_collection()
        if quad_mesh is not None:
            ax.add_collection(quad_mesh)
        ax.set_aspect("equal")
        ax.set_title(label)
        fig.colorbar(mesh, ax=ax, label=label)
        return fig, ax
