"""
Generic, copy-anywhere script to run the D2IC sequential pipeline on any image study.

Usage example (from a study folder):
    python dic_sequence_template.py --img-dir ./img/StudyA --ref-image ref.tif --mask roi.tif
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Sequence

import jax
import matplotlib
import numpy as np


def _ensure_d2ic_importable() -> None:
    """Allow running the script from any folder by locating the D2IC package."""
    if importlib.util.find_spec("D2IC") is not None:
        return

    candidates = []
    env_root = os.environ.get("D2IC_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    script_dir = Path(__file__).resolve().parent
    candidates.extend(script_dir.parents)
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    seen: set[str] = set()
    for root in candidates:
        root_str = str(root)
        if root_str in seen:
            continue
        seen.add(root_str)
        pkg_dir = root / "D2IC"
        if pkg_dir.exists() and (pkg_dir / "__init__.py").is_file():
            sys.path.insert(0, root_str)
            if importlib.util.find_spec("D2IC") is not None:
                return

    raise ModuleNotFoundError(
        "Unable to locate the D2IC package. Install it (pip install -e .) or set D2IC_ROOT"
    )


_ensure_d2ic_importable()

from D2IC.app_utils import run_pipeline_sequence as run_pipeline_sequence_app

# Non-interactive backend so figures can be saved without a display.
matplotlib.use("Agg")


def _configure_jax_platform(preferred: str = "gpu", fallback: str = "cpu") -> None:
    """Force a backend when available, otherwise fall back to CPU to avoid crashes."""
    try:
        devices = jax.devices(preferred)
    except RuntimeError:
        devices = []
    if devices:
        jax.config.update("jax_platform_name", preferred)
        print(f"JAX backend: {preferred} ({len(devices)} device(s) detected)")
    else:
        jax.config.update("jax_platform_name", fallback)
        print(f"JAX backend: {preferred} unavailable, falling back to {fallback}.")


_configure_jax_platform()
jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------- #
#                          DEFAULT PARAMETERS BLOCK                           #
# --------------------------------------------------------------------------- #
PWD = Path.cwd().resolve()

DEFAULT_PARAMS = {
    "img_dir": PWD / "img" / "PlateHole",
    "ref_image_name": "ohtcfrp_00.tif",
    "mask_filename": "roi.tif",
    "image_pattern": "ohtcfrp_*.tif",
    "out_dir": PWD / "_outputs" / "sequence_result",
    "mesh_element_size_px": 20.0,
    "dic_max_iter": 4000,
    "dic_tol": 1e-3,
    "dic_reg_type": "spring",
    "dic_alpha_reg": 0.1,
    "local_sweeps": 10,
    "use_velocity": True,
    "vel_smoothing": 0.5,
    "strain_k_ring": 2,
    "strain_gauge_length": 40.0,
    "frames_to_plot": None,
    "plot_cmap": "jet",
    "plot_alpha": 0.6,
    "enable_initial_guess": True,
    "image_binning": 1,
}


def _parse_frames_option(value: str) -> Sequence[int] | None:
    """Parse comma-separated frame indices, 'all', or 'none'."""
    text = value.strip().lower()
    if text in {"", "all"}:
        return None
    if text in {"none", "off"}:
        return []
    frame_indices = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            frame_indices.append(int(chunk))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "frames must be a comma-separated list of integers, 'all', or 'none'"
            ) from exc
    if not frame_indices:
        return []
    return frame_indices


def _build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step-by-step DIC pipeline executable from any study folder."
    )
    parser.add_argument("--img-dir", type=Path, default=defaults["img_dir"], help="Folder with the sequence.")
    parser.add_argument("--ref-image", default=defaults["ref_image_name"], help="Reference image filename.")
    parser.add_argument("--mask", default=defaults["mask_filename"], help="ROI mask filename.")
    parser.add_argument("--image-pattern", default=defaults["image_pattern"], help="Pattern for deformed images.")
    parser.add_argument("--out-dir", type=Path, default=defaults["out_dir"], help="Results folder.")
    parser.add_argument("--mesh-element-size", type=float, default=defaults["mesh_element_size_px"], help="Target ROI mesh size in pixels.")
    parser.add_argument("--dic-max-iter", type=int, default=defaults["dic_max_iter"], help="Maximum DIC iterations.")
    parser.add_argument("--dic-tol", type=float, default=defaults["dic_tol"], help="Convergence tolerance.")
    parser.add_argument("--dic-reg-type", default=defaults["dic_reg_type"], help="Regularization type.")
    parser.add_argument("--dic-alpha-reg", type=float, default=defaults["dic_alpha_reg"], help="Regularization weight.")
    parser.add_argument("--local-sweeps", type=int, default=defaults["local_sweeps"], help="Local refinement sweeps.")
    parser.add_argument(
        "--use-velocity",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_velocity"],
        help="Enable velocity extrapolation between frames.",
    )
    parser.add_argument(
        "--vel-smoothing",
        type=float,
        default=defaults["vel_smoothing"],
        help="Velocity smoothing factor in [0, 1].",
    )
    parser.add_argument("--strain-k-ring", type=int, default=defaults["strain_k_ring"], help="Neighborhood ring for strain.")
    parser.add_argument("--strain-gauge-length", type=float, default=defaults["strain_gauge_length"], help="Gauge length for strain computation.")
    parser.add_argument(
        "--frames-to-plot",
        type=_parse_frames_option,
        default=defaults["frames_to_plot"],
        help="Comma-separated indices, 'all' for every frame, or 'none' to skip exports.",
    )
    parser.add_argument("--plot-cmap", default=defaults["plot_cmap"], help="Matplotlib colormap for overlays.")
    parser.add_argument("--plot-alpha", type=float, default=defaults["plot_alpha"], help="Alpha of the background image.")
    parser.add_argument(
        "--enable-initial-guess",
        action=argparse.BooleanOptionalAction,
        default=defaults["enable_initial_guess"],
        help="Run sparse-match initialization for the first frame (default: on).",
    )
    parser.add_argument(
        "--image-binning",
        type=int,
        default=defaults["image_binning"],
        help="Downsample factor applied to reference/deformed images (must be >=1).",
    )
    return parser


def _resolve_path(path_value: Path) -> Path:
    return path_value.expanduser().resolve()


def _format_frames_for_log(frames: Sequence[int] | None) -> str:
    if frames is None:
        return "all"
    if len(frames) == 0:
        return "none"
    return ",".join(str(idx) for idx in frames)


def _print_configuration(params: dict[str, object]) -> None:
    fields = [
        "img_dir",
        "ref_image_name",
        "mask_filename",
        "image_pattern",
        "out_dir",
        "mesh_element_size_px",
        "dic_max_iter",
        "dic_tol",
        "dic_reg_type",
        "dic_alpha_reg",
        "local_sweeps",
        "use_velocity",
        "vel_smoothing",
        "strain_k_ring",
        "strain_gauge_length",
        "frames_to_plot",
        "plot_cmap",
        "plot_alpha",
        "enable_initial_guess",
        "image_binning",
    ]
    print("Running DIC pipeline with the following configuration:")
    for key in fields:
        value = params[key]
        if isinstance(value, Path):
            value = str(value)
        if key == "frames_to_plot":
            value = _format_frames_for_log(value)  # type: ignore[arg-type]
        print(f"  - {key}: {value}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser(DEFAULT_PARAMS)
    args = parser.parse_args(argv)
    kwargs = {
        "img_dir": _resolve_path(args.img_dir),
        "ref_image_name": args.ref_image,
        "mask_filename": args.mask,
        "image_pattern": args.image_pattern,
        "out_dir": _resolve_path(args.out_dir),
        "mesh_element_size_px": args.mesh_element_size,
        "dic_max_iter": args.dic_max_iter,
        "dic_tol": args.dic_tol,
        "dic_reg_type": args.dic_reg_type,
        "dic_alpha_reg": args.dic_alpha_reg,
        "local_sweeps": args.local_sweeps,
        "use_velocity": args.use_velocity,
        "vel_smoothing": args.vel_smoothing,
        "strain_k_ring": args.strain_k_ring,
        "strain_gauge_length": args.strain_gauge_length,
        "frames_to_plot": args.frames_to_plot,
        "plot_cmap": args.plot_cmap,
        "plot_alpha": args.plot_alpha,
        "enable_initial_guess": args.enable_initial_guess,
        "image_binning": args.image_binning,
    }
    _print_configuration(kwargs)
    disp_all, E_all = run_pipeline_sequence_app(**kwargs)
    print(f"DIC run complete. Displacements: {np.asarray(disp_all).shape}, strains: {np.asarray(E_all).shape}")


if __name__ == "__main__":
    main()
