from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import tomllib


@dataclass(frozen=True)
class SequenceConfig:
    """Parsed configuration for the sequential DIC pipeline."""

    img_dir: Path
    ref_image_name: str
    mask_filename: str
    image_pattern: str
    out_dir: Path

    mesh_element_size_px: float
    mesh_remove_islands: bool
    mesh_min_island_area_px: int
    mesh_gmsh_optimize: bool
    mesh_gmsh_verbose: bool
    mesh_gmsh_contour_step_px: float

    max_iters: int
    tol: float
    reg_strength: float
    local_sweeps: int
    use_velocity: bool
    strain_gauge_length: float
    interpolation: str
    verbose: bool

    image_binning: int

    export_png: bool
    export_frames: Sequence[int] | None
    plot_fields: Sequence[str]
    plot_include_discrepancy: bool
    plot_cmap: str
    plot_alpha: float
    plot_mesh: bool
    plot_dpi: int
    plot_binning: float
    plot_projection: str | bool
    keep_results: bool
    save_npz: bool
    save_mesh: bool

    jax_preferred: str
    jax_fallback: str
    jax_enable_x64: bool
    jax_matmul_precision: str | None


def load_sequence_config(cfg_path: Path) -> SequenceConfig:
    """Load a sequential DIC config from TOML."""
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("rb") as handle:
        data = tomllib.load(handle)

    cfg_dir = cfg_path.parent

    paths = _require_table(data, "paths")
    img_dir = _resolve_path(_require_str(paths, "img_dir"), base=cfg_dir)
    out_dir = _resolve_path(_require_str(paths, "out_dir"), base=cfg_dir)

    mesh = _require_table(data, "mesh")
    dic = _require_table(data, "dic")
    output = _require_table(data, "output")
    runtime = _require_table(data, "runtime")

    export_frames = output.get("export_frames")
    if export_frames is None:
        parsed_frames = None
    else:
        parsed_frames = _parse_int_sequence(export_frames, "output.export_frames")

    plot_fields = output.get("plot_fields", ("u1", "u2", "e11", "e22", "e12"))
    plot_fields = _parse_str_sequence(plot_fields, "output.plot_fields")

    image_binning = int(dic.get("image_binning", 1))
    if image_binning < 1:
        raise ValueError("dic.image_binning must be >= 1.")

    return SequenceConfig(
        img_dir=img_dir,
        ref_image_name=_require_str(paths, "ref_image_name"),
        mask_filename=_require_str(paths, "mask_filename"),
        image_pattern=_require_str(paths, "image_pattern"),
        out_dir=out_dir,
        mesh_element_size_px=float(_require_number(mesh, "element_size_px")),
        mesh_remove_islands=bool(mesh.get("remove_islands", True)),
        mesh_min_island_area_px=int(mesh.get("min_island_area_px", 64)),
        mesh_gmsh_optimize=bool(mesh.get("gmsh_optimize", True)),
        mesh_gmsh_verbose=bool(mesh.get("gmsh_verbose", False)),
        mesh_gmsh_contour_step_px=float(mesh.get("gmsh_contour_step_px", 2.0)),
        max_iters=int(_require_number(dic, "max_iters")),
        tol=float(_require_number(dic, "tol")),
        reg_strength=float(_require_number(dic, "reg_strength")),
        local_sweeps=int(dic.get("local_sweeps", 0)),
        use_velocity=bool(dic.get("use_velocity", True)),
        strain_gauge_length=float(_require_number(dic, "strain_gauge_length")),
        interpolation=str(dic.get("interpolation", "cubic")),
        verbose=bool(dic.get("verbose", False)),
        image_binning=image_binning,
        export_png=bool(output.get("export_png", True)),
        export_frames=parsed_frames,
        plot_fields=plot_fields,
        plot_include_discrepancy=bool(output.get("plot_include_discrepancy", False)),
        plot_cmap=str(output.get("plot_cmap", "jet")),
        plot_alpha=float(output.get("plot_alpha", 0.6)),
        plot_mesh=bool(output.get("plot_mesh", True)),
        plot_dpi=int(output.get("plot_dpi", 200)),
        plot_binning=float(output.get("plot_binning", image_binning)),
        plot_projection=output.get("plot_projection", "fast"),
        keep_results=bool(output.get("keep_results", True)),
        save_npz=bool(output.get("save_npz", True)),
        save_mesh=bool(output.get("save_mesh", True)),
        jax_preferred=str(runtime.get("jax_preferred", "gpu")),
        jax_fallback=str(runtime.get("jax_fallback", "cpu")),
        jax_enable_x64=bool(runtime.get("jax_enable_x64", False)),
        jax_matmul_precision=runtime.get("jax_matmul_precision"),
    )


def _require_table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if value is None or not isinstance(value, dict):
        raise ValueError(f"Missing required config table: {name}")
    return value


def _require_str(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if value is None:
        raise ValueError(f"Missing required config key: {name}")
    if not isinstance(value, str):
        raise ValueError(f"Config key {name} must be a string.")
    return value


def _require_number(data: dict[str, Any], name: str) -> float | int:
    value = data.get(name)
    if value is None:
        raise ValueError(f"Missing required config key: {name}")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Config key {name} must be a number.")
    return value


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _parse_int_sequence(value: Any, name: str) -> Sequence[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
        return [int(v) for v in items]
    raise ValueError(f"Config key {name} must be a list or comma-separated string.")


def _parse_str_sequence(value: Any, name: str) -> Sequence[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
        return items
    raise ValueError(f"Config key {name} must be a list or comma-separated string.")
