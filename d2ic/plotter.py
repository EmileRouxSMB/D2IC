"""Matplotlib-only plotting utilities for DIC results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection

from .dataclasses import DICResult
from .mesh_assets import Mesh, PixelAssets
from .pixel_assets import build_pixel_assets


@dataclass(frozen=True)
class PlotField:
    key: str
    label: str


class DICPlotter:
    """
    Plot DIC nodal fields over the deformed image with optional mesh overlay.

    Parameters
    ----------
    result:
        DICResult providing nodal displacement and strain.
    mesh:
        Mesh used for the DIC solve.
    def_image:
        Deformed image (2D) to use as background.
    ref_image:
        Optional reference image (2D) used to compute discrepancy maps.
    binning:
        Binning factor used when mapping mesh coordinates to pixels.
    """

    def __init__(
        self,
        result: DICResult,
        mesh: Mesh,
        def_image: np.ndarray,
        ref_image: Optional[np.ndarray] = None,
        binning: float = 1.0,
        pixel_assets: Optional[PixelAssets] = None,
        project_on_deformed: bool | str = True,
    ) -> None:
        self._result = result
        self._mesh = mesh
        self._def_image = np.asarray(def_image)
        self._ref_image = None if ref_image is None else np.asarray(ref_image)
        self._binning = float(binning)
        self._projection_mode = self._normalize_projection_mode(project_on_deformed)
        self._project_on_deformed = self._projection_mode in ("direct", "fast")

        self._validate_inputs()
        self._strain_fields = self._normalize_strain(result.strain)
        self._u_nodal = np.asarray(self._result.u_nodal)
        self._user_scalar_fields: Dict[str, np.ndarray] = {}
        self._user_scalar_labels: Dict[str, str] = {}
        self._user_scalar_names: Dict[str, str] = {}

        pixel_ref = self._ref_image if self._ref_image is not None else self._def_image

        pix_ref = pixel_assets
        if pix_ref is None:
            pix_ref = build_pixel_assets(mesh=mesh, ref_image=pixel_ref, binning=self._binning)

        if self._projection_mode == "direct":
            nodes_def = np.asarray(mesh.nodes_xy) + self._u_nodal
            mesh_def = Mesh(nodes_xy=nodes_def, elements=mesh.elements)
            pix_proj = build_pixel_assets(mesh=mesh_def, ref_image=self._def_image, binning=self._binning)
        else:
            pix_proj = pix_ref
            if self._projection_mode == "fast":
                self._projection_mode = "fast"
            else:
                self._projection_mode = "splat"

        self._pixel_coords_ref = np.asarray(pix_ref.pixel_coords_ref)
        self._pixel_nodes_ref = np.asarray(pix_ref.pixel_nodes, dtype=int)
        self._pixel_shape_ref = np.asarray(pix_ref.pixel_shapeN)
        self._image_shape_ref = pix_ref.image_shape
        self._roi_flat_ref = np.asarray(pix_ref.roi_mask_flat, dtype=np.int64)

        self._pixel_coords_proj = np.asarray(pix_proj.pixel_coords_ref)
        self._pixel_nodes_proj = np.asarray(pix_proj.pixel_nodes, dtype=int)
        self._pixel_shape_proj = np.asarray(pix_proj.pixel_shapeN)
        self._image_shape_proj = pix_proj.image_shape
        self._roi_flat_proj = np.asarray(pix_proj.roi_mask_flat, dtype=np.int64)

        self._mesh_nodes_plot = np.asarray(mesh.nodes_xy) / self._binning
        if self._project_on_deformed:
            self._mesh_nodes_plot = (np.asarray(mesh.nodes_xy) + self._u_nodal) / self._binning

        self._precompute_reference_projection()
        if self._projection_mode == "direct":
            self._precompute_deformed_projection()
        elif self._projection_mode == "fast":
            self._precompute_deformed_projection_fast()
        else:
            self._ux_map = self._scatter_pixel_values(self._pixel_displacement_ref[:, 0])
            self._uy_map = self._scatter_pixel_values(self._pixel_displacement_ref[:, 1])
        self._scalar_field_cache: Dict[str, np.ndarray] = {}
        self._load_user_fields()

        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None
        self._overlay_im = None
        self._colorbar = None
        self._mesh_collection: Optional[PolyCollection] = None

    @staticmethod
    def _normalize_field_key(field: str) -> str:
        key = field.strip().lower()
        key = re.sub(r"[\s$\\{}()]", "", key).replace("_", "")
        return key

    def _load_user_fields(self) -> None:
        fields: Dict[str, object] = {}
        result_fields = getattr(self._result, "fields", None)
        if isinstance(result_fields, dict):
            fields.update(result_fields)

        diagnostics_fields = None
        if hasattr(self._result, "diagnostics") and hasattr(self._result.diagnostics, "info"):
            diagnostics_fields = self._result.diagnostics.info.get("fields")
        if isinstance(diagnostics_fields, dict):
            for name, values in diagnostics_fields.items():
                fields.setdefault(name, values)

        for name, values in fields.items():
            try:
                self.register_scalar_field(name, values, overwrite=False)
            except ValueError:
                # Ignore invalid user fields at plotter construction time.
                continue

    def register_scalar_field(
        self,
        name: str,
        nodal_values,
        *,
        label: Optional[str] = None,
        overwrite: bool = True,
    ) -> str:
        """
        Register a user-defined scalar nodal field for later plotting.

        Parameters
        ----------
        name:
            Field identifier. Normalization ignores case, underscores and common LaTeX
            wrappers (e.g. ``"$\\sigma_{E_{22}}$"``).
        nodal_values:
            1D nodal scalar values with shape ``(Nn,)``.
        label:
            Optional display label used for colorbar and title (defaults to `name`).
        overwrite:
            If True, overwrite an existing user field with the same normalized key.

        Returns
        -------
        str
            Normalized field key used internally.
        """
        key = self._normalize_field_key(name)
        values = np.asarray(nodal_values)
        expected = int(self._result.u_nodal.shape[0])
        if values.ndim != 1 or values.shape[0] != expected:
            raise ValueError(f"User field '{name}' must have shape ({expected},).")
        if not overwrite and key in self._user_scalar_fields:
            return key
        self._user_scalar_fields[key] = values
        self._user_scalar_names[key] = str(name)
        self._user_scalar_labels[key] = str(name) if label is None else str(label)
        return key

    def _validate_inputs(self) -> None:
        if self._def_image.ndim != 2:
            raise ValueError("def_image must be a 2D array.")
        u_nodal = np.asarray(self._result.u_nodal)
        if u_nodal.ndim != 2 or u_nodal.shape[1] != 2:
            raise ValueError("result.u_nodal must have shape (Nnodes, 2).")
        if self._ref_image is not None and self._ref_image.ndim != 2:
            raise ValueError("ref_image must be a 2D array when provided.")

    @staticmethod
    def _normalize_projection_mode(project_on_deformed: bool | str) -> str:
        if isinstance(project_on_deformed, str):
            mode = project_on_deformed.strip().lower()
            if mode in ("fast", "deformed_fast", "fast_deformed"):
                return "fast"
            if mode in ("exact", "direct", "true", "deformed"):
                return "direct"
            if mode in ("false", "none", "ref", "reference"):
                return "splat"
            raise ValueError(f"Unsupported project_on_deformed mode: {project_on_deformed}")
        return "direct" if project_on_deformed else "splat"

    @staticmethod
    def _normalize_strain(strain: np.ndarray) -> Dict[str, np.ndarray]:
        if strain is None:
            return {}
        arr = np.asarray(strain)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return {
                "e11": arr[:, 0],
                "e22": arr[:, 1],
                "e12": arr[:, 2],
            }
        if arr.ndim == 3 and arr.shape[1:] == (2, 2):
            return {
                "e11": arr[:, 0, 0],
                "e22": arr[:, 1, 1],
                "e12": arr[:, 0, 1],
            }
        raise ValueError("result.strain must have shape (Nnodes, 3) or (Nnodes, 2, 2).")

    def _precompute_reference_projection(self) -> None:
        node_values = self._u_nodal[self._pixel_nodes_ref]
        pixel_disp = np.sum(self._pixel_shape_ref[..., None] * node_values, axis=1)
        pixel_coords_def = self._pixel_coords_ref + pixel_disp

        self._pixel_coords_def_ref = pixel_coords_def
        self._pixel_displacement_ref = pixel_disp

        x_cont = pixel_coords_def[:, 0] - 0.5
        y_cont = pixel_coords_def[:, 1] - 0.5
        i0 = np.floor(y_cont).astype(int)
        j0 = np.floor(x_cont).astype(int)
        fx = x_cont - j0
        fy = y_cont - i0

        H, W = self._image_shape_ref
        contribs = []
        for w, di, dj in (
            ((1.0 - fx) * (1.0 - fy), 0, 0),
            (fx * (1.0 - fy), 0, 1),
            ((1.0 - fx) * fy, 1, 0),
            (fx * fy, 1, 1),
        ):
            ii = i0 + di
            jj = j0 + dj
            mask = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W) & (w > 0)
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                contribs.append({"pixel_idx": idx, "idx_flat": idx, "weights": w[:0]})
                continue
            idx_flat = (ii[mask] * W + jj[mask]).astype(np.int64, copy=False)
            contribs.append(
                {
                    "pixel_idx": idx,
                    "idx_flat": idx_flat,
                    "weights": w[mask],
                }
            )

        self._scatter_contribs_ref = contribs

    def _precompute_deformed_projection(self) -> None:
        node_values = self._u_nodal[self._pixel_nodes_proj]
        pixel_disp = np.sum(self._pixel_shape_proj[..., None] * node_values, axis=1)
        self._pixel_displacement_proj = pixel_disp
        self._ux_map = self._scatter_pixel_values_direct(pixel_disp[:, 0])
        self._uy_map = self._scatter_pixel_values_direct(pixel_disp[:, 1])

    def _precompute_deformed_projection_fast(self) -> None:
        """Fast projection on the deformed image using reference pixel assets."""
        self._image_shape_def = self._def_image.shape[:2]
        self._scatter_contribs_def = self._build_scatter_contribs(
            self._pixel_coords_def_ref,
            self._image_shape_def,
        )
        self._ux_map = self._scatter_pixel_values_def(self._pixel_displacement_ref[:, 0])
        self._uy_map = self._scatter_pixel_values_def(self._pixel_displacement_ref[:, 1])

    @staticmethod
    def _normalize_field_name(field: str) -> PlotField:
        key = DICPlotter._normalize_field_key(field)
        aliases = {
            "u1": PlotField("u1", "$U_1$"),
            "ux": PlotField("u1", "$U_1$"),
            "u2": PlotField("u2", "$U_2$"),
            "uy": PlotField("u2", "$U_2$"),
            "e11": PlotField("e11", "$E_{11}$"),
            "exx": PlotField("e11", "$E_{11}$"),
            "e22": PlotField("e22", "$E_{22}$"),
            "eyy": PlotField("e22", "$E_{22}$"),
            "e12": PlotField("e12", "$E_{12}$"),
            "e21": PlotField("e12", "$E_{12}$"),
            "exy": PlotField("e12", "$E_{12}$"),
            "eyx": PlotField("e12", "$E_{12}$"),
            "discrepancy": PlotField("discrepancy", "Discrepancy"),
            "residual": PlotField("discrepancy", "Discrepancy"),
            "discrep": PlotField("discrepancy", "Discrepancy"),
            "dyscepancy": PlotField("discrepancy", "Discrepancy"),
        }
        return aliases.get(key, PlotField(key, field))

    def _scatter_pixel_values(self, pixel_values: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self._scatter_pixel_values_with(
            pixel_values,
            self._scatter_contribs_ref,
            self._image_shape_ref,
            valid_mask=valid_mask,
        )

    def _scatter_pixel_values_def(self, pixel_values: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self._scatter_pixel_values_with(
            pixel_values,
            self._scatter_contribs_def,
            self._image_shape_def,
            valid_mask=valid_mask,
        )

    @staticmethod
    def _scatter_pixel_values_with(
        pixel_values: np.ndarray,
        scatter_contribs,
        image_shape,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        pixel_values = np.asarray(pixel_values)
        H, W = image_shape
        grid = np.full((H, W), np.nan, dtype=float)
        if pixel_values.size == 0:
            return grid

        accum_flat = np.zeros(H * W, dtype=float)
        weight_flat = np.zeros(H * W, dtype=float)

        for contrib in scatter_contribs:
            pixel_idx = contrib["pixel_idx"]
            if pixel_idx.size == 0:
                continue
            if valid_mask is not None:
                pixel_idx = pixel_idx[valid_mask[pixel_idx]]
                if pixel_idx.size == 0:
                    continue
                mask = np.isin(contrib["pixel_idx"], pixel_idx)
                idx_flat = contrib["idx_flat"][mask]
                weights = contrib["weights"][mask]
            else:
                idx_flat = contrib["idx_flat"]
                weights = contrib["weights"]
            vals = pixel_values[pixel_idx] * weights
            accum_flat += np.bincount(idx_flat, weights=vals, minlength=H * W)
            weight_flat += np.bincount(idx_flat, weights=weights, minlength=H * W)

        mask = weight_flat > 0
        grid.flat[mask] = accum_flat[mask] / weight_flat[mask]
        return grid

    def _scatter_pixel_values_direct(
        self,
        pixel_values: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        pixel_values = np.asarray(pixel_values)
        H, W = self._image_shape_proj
        grid = np.full((H, W), np.nan, dtype=float)
        if pixel_values.size == 0:
            return grid

        idx = self._roi_flat_proj
        if valid_mask is not None:
            idx = idx[valid_mask]
            pixel_values = pixel_values[valid_mask]
        if idx.size == 0:
            return grid
        accum = np.bincount(idx, weights=pixel_values, minlength=H * W)
        counts = np.bincount(idx, weights=np.ones_like(pixel_values, dtype=float), minlength=H * W)
        mask = counts > 0
        grid.flat[mask] = accum[mask] / counts[mask]
        return grid

    def _interpolate_scalar_field(self, nodal_values: np.ndarray) -> np.ndarray:
        values = np.asarray(nodal_values)
        expected = int(self._result.u_nodal.shape[0])
        if values.ndim != 1 or values.shape[0] != expected:
            raise ValueError(f"Scalar field must have shape ({expected},).")
        if self._projection_mode == "direct":
            pixel_vals = np.sum(self._pixel_shape_proj * values[self._pixel_nodes_proj], axis=1)
            return self._scatter_pixel_values_direct(pixel_vals)
        if self._projection_mode == "fast":
            pixel_vals = np.sum(self._pixel_shape_ref * values[self._pixel_nodes_ref], axis=1)
            return self._scatter_pixel_values_def(pixel_vals)
        pixel_vals = np.sum(self._pixel_shape_ref * values[self._pixel_nodes_ref], axis=1)
        return self._scatter_pixel_values(pixel_vals)

    def _get_strain_map(self, key: str) -> np.ndarray:
        if key not in self._strain_fields:
            raise ValueError(f"Strain field '{key}' is not available.")
        if key not in self._scalar_field_cache:
            self._scalar_field_cache[key] = self._interpolate_scalar_field(self._strain_fields[key])
        return self._scalar_field_cache[key]

    def _get_user_field_map(self, key: str) -> np.ndarray:
        if key not in self._user_scalar_fields:
            raise ValueError(f"User field '{key}' is not available.")
        cache_key = f"user:{key}"
        if cache_key not in self._scalar_field_cache:
            self._scalar_field_cache[cache_key] = self._interpolate_scalar_field(self._user_scalar_fields[key])
        return self._scalar_field_cache[cache_key]

    @staticmethod
    def _bilinear_sample(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
        H, W = image.shape
        x = coords[:, 0] - 0.5
        y = coords[:, 1] - 0.5
        i0 = np.floor(y).astype(int)
        j0 = np.floor(x).astype(int)
        fx = x - j0
        fy = y - i0

        valid = (i0 >= 0) & (i0 < H - 1) & (j0 >= 0) & (j0 < W - 1)
        values = np.full(coords.shape[0], np.nan, dtype=float)
        if not np.any(valid):
            return values

        i0v = i0[valid]
        j0v = j0[valid]
        fxv = fx[valid]
        fyv = fy[valid]

        i1v = i0v + 1
        j1v = j0v + 1
        v00 = image[i0v, j0v]
        v01 = image[i0v, j1v]
        v10 = image[i1v, j0v]
        v11 = image[i1v, j1v]
        values[valid] = (
            v00 * (1.0 - fxv) * (1.0 - fyv)
            + v01 * fxv * (1.0 - fyv)
            + v10 * (1.0 - fxv) * fyv
            + v11 * fxv * fyv
        )
        return values

    def _discrepancy_map(self) -> np.ndarray:
        if self._ref_image is None:
            raise ValueError("ref_image is required to compute discrepancy maps.")
        i1 = self._bilinear_sample(self._ref_image, self._pixel_coords_ref)
        i2 = self._bilinear_sample(self._def_image, self._pixel_coords_def_ref)
        residuals = i2 - i1
        valid = np.isfinite(residuals)
        if self._projection_mode == "direct":
            return self._scatter_pixel_values_direct(residuals, valid_mask=valid)
        if self._projection_mode == "fast":
            return self._scatter_pixel_values_def(residuals, valid_mask=valid)
        return self._scatter_pixel_values(residuals, valid_mask=valid)

    @staticmethod
    def _build_scatter_contribs(pixel_coords: np.ndarray, image_shape: tuple[int, int]):
        x_cont = pixel_coords[:, 0] - 0.5
        y_cont = pixel_coords[:, 1] - 0.5
        i0 = np.floor(y_cont).astype(int)
        j0 = np.floor(x_cont).astype(int)
        fx = x_cont - j0
        fy = y_cont - i0

        H, W = image_shape
        contribs = []
        for w, di, dj in (
            ((1.0 - fx) * (1.0 - fy), 0, 0),
            (fx * (1.0 - fy), 0, 1),
            ((1.0 - fx) * fy, 1, 0),
            (fx * fy, 1, 1),
        ):
            ii = i0 + di
            jj = j0 + dj
            mask = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W) & (w > 0)
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                contribs.append({"pixel_idx": idx, "idx_flat": idx, "weights": w[:0]})
                continue
            idx_flat = (ii[mask] * W + jj[mask]).astype(np.int64, copy=False)
            contribs.append(
                {
                    "pixel_idx": idx,
                    "idx_flat": idx_flat,
                    "weights": w[mask],
                }
            )
        return contribs

    def _quad_mesh_collection(self) -> Optional[PolyCollection]:
        nodes = np.asarray(self._mesh_nodes_plot)
        elements = np.asarray(self._mesh.elements, dtype=int)
        if elements.size == 0:
            return None
        verts = nodes[elements]
        return PolyCollection(
            verts,
            facecolors="none",
            edgecolors="k",
            linewidths=0.5,
        )

    def _init_figure_template(
        self,
        figsize: Tuple[float, float],
        cmap: str,
        image_alpha: float,
        plotmesh: bool,
    ) -> None:
        if self._fig is not None:
            self._activate_figure()
            if self._fig is not None:
                return
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self._def_image, cmap="gray", origin="lower", alpha=1.0)
        placeholder = np.zeros_like(self._def_image, dtype=float)
        overlay = ax.imshow(
            placeholder,
            cmap=cmap,
            origin="lower",
            alpha=image_alpha,
        )
        colorbar = fig.colorbar(overlay, ax=ax, label="")
        mesh_collection = None
        if plotmesh:
            mesh_collection = self._quad_mesh_collection()
            if mesh_collection is not None:
                ax.add_collection(mesh_collection)
        ax.set_aspect("equal")
        self._fig = fig
        self._ax = ax
        self._overlay_im = overlay
        self._colorbar = colorbar
        self._mesh_collection = mesh_collection

    def _activate_figure(self) -> None:
        if self._fig is None or self._ax is None:
            return
        if self._fig.number not in plt.get_fignums():
            self._fig = None
            self._ax = None
            self._overlay_im = None
            self._colorbar = None
            self._mesh_collection = None
            return
        plt.figure(self._fig.number)
        plt.sca(self._ax)

    def _set_mesh_visibility(self, plotmesh: bool) -> None:
        if plotmesh:
            if self._mesh_collection is None:
                self._mesh_collection = self._quad_mesh_collection()
                if self._mesh_collection is not None and self._ax is not None:
                    self._ax.add_collection(self._mesh_collection)
            if self._mesh_collection is not None:
                self._mesh_collection.set_visible(True)
        else:
            if self._mesh_collection is not None:
                self._mesh_collection.set_visible(False)

    def _resolve_field_map(self, field: str) -> tuple[PlotField, np.ndarray]:
        selected = self._normalize_field_name(field)

        if selected.key == "u1":
            field_map = self._ux_map
        elif selected.key == "u2":
            field_map = self._uy_map
        elif selected.key in ("e11", "e22", "e12"):
            field_map = self._get_strain_map(selected.key)
        elif selected.key == "discrepancy":
            field_map = self._discrepancy_map()
        elif selected.key in self._user_scalar_fields:
            field_map = self._get_user_field_map(selected.key)
            stored_label = self._user_scalar_labels.get(selected.key, selected.label)
            stored_name = self._user_scalar_names.get(selected.key)
            label = stored_label
            if stored_name is not None and stored_label == stored_name and selected.label != stored_name:
                label = selected.label
            selected = PlotField(selected.key, label)
        else:
            user_fields = ", ".join(sorted(self._user_scalar_labels.values()))
            hint = "" if not user_fields else f" Available user fields: {user_fields}."
            raise ValueError(f"Unsupported field '{field}'.{hint}")

        return selected, field_map

    def plot_into(
        self,
        ax: Axes,
        field: str,
        image_alpha: float = 0.75,
        cmap: str = "jet",
        plotmesh: bool = True,
        add_colorbar: bool = True,
    ) -> tuple[Axes, Optional[Colorbar]]:
        """
        Plot a scalar field into an existing Matplotlib axis.

        This helper enables layouts such as subplots. Unlike `plot(...)`, it does
        not reuse or mutate the internal figure state stored on the plotter.
        """
        selected, field_map = self._resolve_field_map(field)

        ax.imshow(self._def_image, cmap="gray", origin="lower", alpha=1.0)
        placeholder = np.zeros_like(self._def_image, dtype=float)
        overlay = ax.imshow(
            placeholder,
            cmap=cmap,
            origin="lower",
            alpha=image_alpha,
        )
        colorbar = None
        if add_colorbar:
            colorbar = ax.figure.colorbar(overlay, ax=ax, label="")
        if plotmesh:
            mesh_collection = self._quad_mesh_collection()
            if mesh_collection is not None:
                ax.add_collection(mesh_collection)

        masked = np.ma.array(field_map, mask=~np.isfinite(field_map))
        overlay.set_data(masked)
        overlay.set_alpha(image_alpha)
        overlay.set_cmap(cmap)
        if masked.count() > 0:
            vmin = masked.min()
            vmax = masked.max()
            if vmin != vmax:
                overlay.set_clim(vmin=float(vmin), vmax=float(vmax))
        if colorbar is not None:
            colorbar.set_label(selected.label)
            colorbar.update_normal(overlay)
        ax.set_title(selected.label)
        ax.set_aspect("equal")
        return ax, colorbar

    def plot(
        self,
        field: str,
        image_alpha: float = 0.75,
        cmap: str = "jet",
        figsize: Tuple[float, float] = (6.0, 6.0),
        plotmesh: bool = True,
    ) -> Tuple[Figure, Axes]:
        """
        Plot a scalar field over the deformed image.

        Supported fields: U1/U2/E11/E22/E12/discrepancy and user-defined fields
        registered via `register_scalar_field(...)` or provided in `DICResult.fields`.
        """
        self._init_figure_template(figsize, cmap, image_alpha, plotmesh)
        self._activate_figure()
        self._set_mesh_visibility(plotmesh)

        selected, field_map = self._resolve_field_map(field)

        masked = np.ma.array(field_map, mask=~np.isfinite(field_map))
        self._overlay_im.set_data(masked)
        self._overlay_im.set_alpha(image_alpha)
        self._overlay_im.set_cmap(cmap)
        if masked.count() > 0:
            vmin = masked.min()
            vmax = masked.max()
            if vmin != vmax:
                self._overlay_im.set_clim(vmin=float(vmin), vmax=float(vmax))
        self._colorbar.set_label(selected.label)
        self._colorbar.update_normal(self._overlay_im)
        self._ax.set_title(selected.label)
        return self._fig, self._ax
