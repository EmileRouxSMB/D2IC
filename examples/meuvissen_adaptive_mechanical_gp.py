"""Compare unregularized, uniform-mechanical and GP-adaptive DIC on Meuvissen.

The default run uses one mid-sequence frame and a binned image so it can serve
as a reproducible validation before launching a full 39-frame study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from d2ic.core.dataclasses import MeshDICConfig
from d2ic.io.app_utils import imread_gray, make_roi_mask, prepare_image
from d2ic.mesh.generation import downsample_mask, mask_to_mesh_assets_gmsh
from d2ic.methods.mesh_based import DICMeshBased
from d2ic.numerics import GlobalCGSolver, MechanicalGPSolver


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = ROOT / "virtual-dic" / "results" / "meuvissen"


def _truth_on_dic_nodes(case_dir, nodes_xy, frame_time, binning):
    """Interpolate Abaqus U to DIC nodes and convert mm to binned image pixels."""
    data = np.load(case_dir / "odb_data.npz")
    times = np.asarray(data["times"])
    upper = int(np.searchsorted(times, frame_time, side="right"))
    upper = min(max(upper, 1), len(times) - 1)
    lower = upper - 1
    fraction = (frame_time - times[lower]) / (times[upper] - times[lower])
    displacement_mm = (1.0 - fraction) * data["U"][lower] + fraction * data["U"][upper]

    metadata = json.loads((case_dir / "metadata.json").read_text())
    camera = metadata["camera"]
    scale = float(camera["scale_px_per_unit"])
    center = np.asarray(camera["center_xy"])
    width = float(camera["width_px"])
    height = float(camera["height_px"])
    # A binned pixel is the average of a b x b source block; map its center,
    # consistently with virtual-dic's (size - 1) / 2 camera convention.
    full = np.asarray(nodes_xy) * binning + 0.5 * (binning - 1)
    points_mm = np.column_stack((
        center[0] + (full[:, 0] - 0.5 * (width - 1.0)) / scale,
        center[1] - (full[:, 1] - 0.5 * (height - 1.0)) / scale,
    ))
    truth = LinearNDInterpolator(data["coordinates"][:, :2], displacement_mm)(points_mm)
    truth *= np.asarray([scale / binning, -scale / binning])
    return truth


def run_validation(
    case_dir: Path = DEFAULT_CASE,
    *,
    frame: int = 20,
    binning: int = 4,
    element_size_px: float = 40.0,
    max_iters: int = 30,
    mechanical_weight: float = 1.0e-3,
    force_scale: float = 1.0e-2,
    output: Path | None = None,
):
    """Run the three DIC variants and save truth errors plus GP confidence maps."""
    image_dir = case_dir / "images"
    frames = sorted(image_dir.glob("frame_*.tiff"))
    if not frames or not 1 <= frame < len(frames):
        raise ValueError(f"frame must be between 1 and {len(frames) - 1}")
    reference = prepare_image(frames[0], binning=binning)
    deformed = prepare_image(frames[frame], binning=binning)
    mask = make_roi_mask(imread_gray(image_dir / "roi_mask_initial.png"))
    # Keep mesh and image coordinates in the same binned-pixel system. The
    # generic mesh generator otherwise preserves full-resolution coordinates.
    mask = downsample_mask(mask, binning)
    mesh, assets = mask_to_mesh_assets_gmsh(
        mask=mask, element_size_px=element_size_px / binning, binning=1,
        remove_islands=True, min_island_area_px=64,
    )
    common = dict(max_iters=max_iters, tol=1.0e-6, save_history=True)
    methods = {
        "unregularized": (
            GlobalCGSolver(interpolation="linear"),
            MeshDICConfig(reg_strength=0.0, regularization="elastic", **common),
        ),
        "uniform_mechanical": (
            MechanicalGPSolver(adaptive=False, force_scale=force_scale),
            MeshDICConfig(reg_strength=mechanical_weight, **common),
        ),
        "adaptive_gp": (
            MechanicalGPSolver(
                adaptive=True, n_inducing=32, force_scale=force_scale,
                gp_length_scale=0.2 * np.linalg.norm(np.ptp(mesh.nodes_xy, axis=0)),
            ),
            MeshDICConfig(reg_strength=mechanical_weight, **common),
        ),
    }
    results = {}
    for name, (solver, config) in methods.items():
        method = DICMeshBased(solver=solver, config=config)
        method.prepare(reference, assets)
        results[name] = method.run(deformed)

    frame_time = float(frames[frame].stem.rsplit("_", 1)[-1])
    truth = _truth_on_dic_nodes(case_dir, mesh.nodes_xy, frame_time, binning)
    valid = np.all(np.isfinite(truth), axis=1)
    metrics = {"frame": frame, "time": frame_time, "binning": binning}
    output_arrays = {
        "nodes_xy": np.asarray(mesh.nodes_xy),
        "elements": np.asarray(mesh.elements),
        "truth": truth,
        "valid": valid,
    }
    for name, result in results.items():
        displacement = np.asarray(jax.device_get(result.displacement))
        error = displacement[valid] - truth[valid]
        metrics[f"{name}_rmse_px"] = float(np.sqrt(np.mean(error * error)))
        output_arrays[f"{name}_displacement"] = displacement
        for field_name, values in result.fields.items():
            output_arrays[f"{name}_{field_name}"] = np.asarray(jax.device_get(values))

    output = case_dir / "d2ic_gp_validation.npz" if output is None else output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **output_arrays)
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--frame", type=int, default=20)
    parser.add_argument("--binning", type=int, default=4)
    parser.add_argument("--element-size-px", type=float, default=40.0)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--mechanical-weight", type=float, default=1.0e-3)
    parser.add_argument("--force-scale", type=float, default=1.0e-2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = run_validation(
        args.case_dir, frame=args.frame, binning=args.binning,
        element_size_px=args.element_size_px, max_iters=args.max_iters,
        mechanical_weight=args.mechanical_weight, force_scale=args.force_scale,
        output=args.output,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
