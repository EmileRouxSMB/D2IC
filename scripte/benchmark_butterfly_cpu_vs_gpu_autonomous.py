#!/usr/bin/env python3
"""Autonomous CPU vs GPU benchmark on the Butterfly tutorial sequence."""

from __future__ import annotations

import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
import shutil

# =====================
# SETTINGS
# =====================
DATASET_DIR = Path(os.environ.get("DIC_BENCH_DATASET_DIR", "doc/img/butterFly"))
REF_IMAGE_NAME = "VIS_0000.tif"
MASK_FILENAME = "roi.tif"
IMAGE_PATTERN = "VIS_*.tif"
NFRAMES = 8
DO_WARMUP = True
EXPORT_NPZ = True
EXPORT_FIGURES = False
EXPORT_INDICATOR_FIGURE = True
MAX_ITER = 400
TOL = 1e-2
OUTPUT_DIR = Path("doc") / "benchmark_results"
MESH_ELEMENT_SIZE_PX = 30.
IMAGE_BINNING = 1
INTERPOLATION = "cubic"
DIC_ALPHA_REG = 0.5
LOCAL_SWEEPS = 3
LOCAL_LAM = 0.1
LOCAL_ALPHA_REG = 100.0
LOCAL_MAX_STEP = 0.2
LOCAL_OMEGA = 0.5
USE_VELOCITY = True
VEL_SMOOTHING = 0.5
MAX_EXTRAPOLATION = 5.0
STRAIN_K_RING = 2
STRAIN_GAUGE_LENGTH = 200.0
FEATURE_N_PATCHES = 32
FEATURE_PATCH_WIN = 21
FEATURE_PATCH_SEARCH = 15
FEATURE_REFINE = True
FEATURE_SEARCH_DILATION = 5.0
PLOT_CMAP = "jet"
PLOT_ALPHA = 0.6
FRAMES_TO_PLOT: Optional[List[int]] = None
FIG_COMPONENTS = ("Ux", "Uy", "Exx", "Exy", "Eyy")
RANDOM_SEED = 0

BASE_DIR = Path(__file__).resolve().parents[1]  # repository root
DATASET_PATH = (BASE_DIR / DATASET_DIR).resolve()
OUTPUT_PATH = (BASE_DIR / OUTPUT_DIR).resolve()
STEP_LABELS = [
    ("A_data_load_s", "A: data load"),
    ("B_dic_setup_s", "B: mesh + DIC"),
    ("C_precompute_s", "C: precompute"),
    ("D_warmup_s", "D: warmup"),
    ("E_solve_total_s", "E: run_dic total"),
    ("F_strain_s", "F: strain"),
    ("G_npz_export_s", "G: export NPZ"),
    ("G_figures_export_s", "G: export figures"),
    ("H_total_s", "H: total"),
]
INDICATOR_LABELS = [
    ("indicator_solve_total_s", "Solve total (series)"),
    ("indicator_prep_s", "Prep once (A-D)"),
    ("indicator_frame_mean_s", "Per-frame mean"),
]


class StepTimer:
    """Context manager recording elapsed time into a dict."""

    def __init__(self, sink: Dict[str, float], key: str) -> None:
        self._sink = sink
        self._key = key
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._sink[self._key] = time.perf_counter() - self._start
        return False


def sanitize_label(text: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "-", text or "").strip("-")
    return sanitized.lower() or "unknown"


def get_base_slug() -> str:
    hostname = platform.node() or "host"
    cpu_model = platform.processor() or platform.uname().processor or "cpu"
    return sanitize_label(f"{hostname}-{cpu_model}")


def collect_cpu_info() -> Dict[str, Optional[str]]:
    uname = platform.uname()
    model = platform.processor() or uname.processor or uname.machine
    os_name = f"{uname.system} {uname.release}"
    return {
        "model": model or "unknown",
        "cores": os.cpu_count(),
        "os": os_name,
    }


def collect_gpu_info(jax_module):
    gpu_devices = [dev for dev in jax_module.devices() if dev.platform == "gpu"]
    if not gpu_devices:
        return None
    dev = gpu_devices[0]
    info = {
        "platform": getattr(dev, "platform", "gpu"),
        "device_name": getattr(dev, "device_kind", getattr(dev, "device_name", "unknown")),
    }
    # Best-effort NVIDIA driver query.
    nvsmi = shutil.which("nvidia-smi") if info["platform"] == "gpu" else None
    if nvsmi:
        try:
            cmd = [nvsmi, "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader", "-i", "0"]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
            if out:
                parts = [p.strip() for p in out.split(",")]
                if len(parts) >= 3:
                    info["device_name"] = parts[0]
                    info["driver"] = parts[1]
                    info["cuda_version"] = parts[2]
        except Exception:
            pass
    return info


def limit_extrapolation(guess: np.ndarray, anchor: np.ndarray, threshold: float) -> np.ndarray:
    delta = guess - anchor
    norms = np.linalg.norm(delta, axis=1)
    mask = norms > threshold
    if np.any(mask):
        delta[mask] *= (threshold / (norms[mask] + 1e-12))[:, None]
    return anchor + delta


def run_backend_benchmark(backend: str, base_slug: Optional[str] = None) -> Dict[str, Any]:
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ["JAX_PLATFORM_NAME"] = backend
    import jax

    jax.config.update("jax_platform_name", backend)
    try:
        devices = jax.devices()
    except RuntimeError as err:
        raise RuntimeError(f"Unable to initialize JAX backend '{backend}': {err}") from err
    if backend == "gpu" and not any(dev.platform == "gpu" for dev in devices):
        raise RuntimeError("No GPU detected for backend 'gpu'.")
    from d2ic import (
        mask_to_mesh_assets,
        mask_to_mesh_assets_gmsh,
        MeshDICConfig,
        DICMeshBased,
        GlobalCGSolver,
        LocalGaussNewtonSolver,
        PreviousDisplacementPropagator,
        ConstantVelocityPropagator,
    )
    from d2ic.app_utils import prepare_image, imread_gray, list_deformed_images
    from d2ic.mesh_assets import build_node_neighbor_tables
    from d2ic.pixel_assets import build_pixel_assets
    from d2ic.plotter import DICPlotter

    base_slug_sanitized = sanitize_label(base_slug or os.environ.get("DIC_BENCH_BASE_SLUG") or get_base_slug())
    cpu_info = collect_cpu_info()
    gpu_info = collect_gpu_info(jax)

    if backend == "gpu" and gpu_info is None:
        raise RuntimeError("No GPU detected for backend 'gpu'.")

    if backend == "cpu":
        hw_tag = sanitize_label(cpu_info.get("model", "cpu"))
    else:
        hw_tag = sanitize_label(gpu_info.get("device_name") if gpu_info else backend)
    run_id = f"{base_slug_sanitized}__{hw_tag}__{backend}"

    backend_dir = OUTPUT_PATH / run_id
    backend_dir.mkdir(parents=True, exist_ok=True)

    timings: Dict[str, float] = {}
    per_frame_times: List[float] = []
    overall_start = time.perf_counter()

    # Step A: data loading
    with StepTimer(timings, "A_data_load_s"):
        mask_path = DATASET_PATH / MASK_FILENAME
        ref_path = DATASET_PATH / REF_IMAGE_NAME
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask image not found: {mask_path}")
        im_ref = prepare_image(ref_path, binning=IMAGE_BINNING)
        frame_paths = list_deformed_images(DATASET_PATH, IMAGE_PATTERN, exclude_name=REF_IMAGE_NAME)
        if not frame_paths:
            raise RuntimeError("No deformed frames found for benchmarking.")
        frames = frame_paths[:NFRAMES]
        images_def = [prepare_image(p, binning=IMAGE_BINNING) for p in frames]
        mask = imread_gray(mask_path) > 0.5

    # Step B: mesh + Dic
    with StepTimer(timings, "B_dic_setup_s"):
        try:
            mesh, assets = mask_to_mesh_assets_gmsh(
                mask=mask,
                element_size_px=MESH_ELEMENT_SIZE_PX,
                binning=IMAGE_BINNING,
                remove_islands=True,
                min_island_area_px=64,
            )
        except Exception:
            mesh, assets = mask_to_mesh_assets(
                mask=mask,
                element_size_px=MESH_ELEMENT_SIZE_PX,
                binning=IMAGE_BINNING,
                remove_islands=True,
                min_island_area_px=64,
            )

        mesh_cfg = MeshDICConfig(
            max_iters=MAX_ITER,
            tol=TOL,
            reg_strength=DIC_ALPHA_REG,
            strain_gauge_length=STRAIN_GAUGE_LENGTH,
            save_history=False,
            compute_discrepancy_map=False,
        )
        dic = DICMeshBased(
            mesh=mesh,
            solver=GlobalCGSolver(interpolation=INTERPOLATION, verbose=False),
            config=mesh_cfg,
        )
        dic_local = None
        if LOCAL_SWEEPS > 0:
            local_cfg = MeshDICConfig(
                max_iters=LOCAL_SWEEPS,
                tol=TOL,
                reg_strength=LOCAL_ALPHA_REG,
                strain_gauge_length=STRAIN_GAUGE_LENGTH,
                save_history=False,
                compute_discrepancy_map=True,
            )
            local_solver = LocalGaussNewtonSolver(
                lam=LOCAL_LAM,
                max_step=LOCAL_MAX_STEP,
                omega=LOCAL_OMEGA,
                interpolation=INTERPOLATION,
            )
            dic_local = DICMeshBased(mesh=mesh, solver=local_solver, config=local_cfg)
        propagator = ConstantVelocityPropagator() if USE_VELOCITY else PreviousDisplacementPropagator()

    # Step C: precompute pixel data
    with StepTimer(timings, "C_precompute_s"):
        if assets.node_neighbor_index is None or assets.node_neighbor_degree is None:
            idx, deg = build_node_neighbor_tables(mesh)
            assets = replace(assets, node_neighbor_index=idx, node_neighbor_degree=deg)
        pixel_data = build_pixel_assets(mesh=mesh, ref_image=im_ref, binning=IMAGE_BINNING, roi_mask=assets.roi_mask)
        assets = replace(assets, pixel_data=pixel_data)

    if DO_WARMUP:
        with StepTimer(timings, "D_warmup_s"):
            dic.prepare(im_ref, assets)
            if dic_local is not None:
                dic_local.prepare(im_ref, assets)
    else:
        dic.prepare(im_ref, assets)
        if dic_local is not None:
            dic_local.prepare(im_ref, assets)
        timings["D_warmup_s"] = 0.0

    disp_history: List[np.ndarray] = []
    strain_history: List[np.ndarray] = []
    results_history = []
    frame_names = [p.name for p in frames]
    u_prev = None
    u_prevprev = None

    for idx, im_def in enumerate(images_def):
        frame_start = time.perf_counter()
        u_warm = propagator.propagate(u_prev=u_prev, u_prevprev=u_prevprev) if propagator else None
        if u_warm is not None:
            dic.set_initial_guess(u_warm)
        res = dic.run(im_def)
        if dic_local is not None:
            dic_local.set_initial_guess(res.u_nodal)
            res = dic_local.run(im_def)
        disp_np = np.asarray(res.u_nodal)
        disp_history.append(disp_np)
        strain_history.append(np.asarray(res.strain))
        results_history.append(res)
        u_prevprev = u_prev
        u_prev = disp_np
        per_frame_times.append(time.perf_counter() - frame_start)

    solve_total = float(np.sum(per_frame_times))
    timings["E_solve_total_s"] = solve_total

    timings["F_strain_s"] = 0.0

    npz_time = 0.0
    if EXPORT_NPZ:
        with StepTimer(timings, "G_npz_export_s"):
            np.savez(
                backend_dir / f"fields_{backend}.npz",
                disp=np.asarray(disp_history),
                strain=np.asarray(strain_history),
                frame_names=np.array(frame_names),
            )
    else:
        timings["G_npz_export_s"] = 0.0

    fig_time = 0.0
    if EXPORT_FIGURES:
        frames_to_plot = FRAMES_TO_PLOT if FRAMES_TO_PLOT is not None else list(range(len(disp_history)))
        component_map = {
            "ux": "u1",
            "uy": "u2",
            "exx": "e11",
            "exy": "e12",
            "eyy": "e22",
        }
        with StepTimer(timings, "G_figures_export_s"):
            for idx in frames_to_plot:
                if idx >= len(disp_history):
                    continue
                plotter = DICPlotter(
                    result=results_history[idx],
                    mesh=mesh,
                    def_image=images_def[idx],
                    ref_image=im_ref,
                    binning=IMAGE_BINNING,
                    pixel_assets=assets.pixel_data,
                    project_on_deformed="fast",
                )
                for comp in FIG_COMPONENTS:
                    field_key = component_map.get(comp.lower())
                    if field_key is None:
                        continue
                    fig, ax = plotter.plot(
                        field=field_key,
                        image_alpha=PLOT_ALPHA,
                        cmap=PLOT_CMAP,
                        plotmesh=True,
                    )
                    ax.set_title(f"{field_key.upper()} frame {idx}")
                    fig.savefig(backend_dir / f"{field_key}_frame_{idx:03d}.png", dpi=200)
                    plt.close(fig)
    else:
        timings["G_figures_export_s"] = 0.0

    timings["H_total_s"] = time.perf_counter() - overall_start

    per_frame_mean = statistics.mean(per_frame_times) if per_frame_times else 0.0
    per_frame_std = statistics.pstdev(per_frame_times) if len(per_frame_times) > 1 else 0.0
    prep_total = sum(
        timings.get(key, 0.0) for key in ("A_data_load_s", "B_dic_setup_s", "C_precompute_s", "D_warmup_s")
    )
    indicators = {
        "indicator_solve_total_s": timings.get("E_solve_total_s", solve_total),
        "indicator_prep_s": prep_total,
        "indicator_frame_mean_s": per_frame_mean,
    }

    result = {
        "backend": backend,
        "status": "ok",
        "machine_id": run_id,
        "base_slug": base_slug_sanitized,
        "settings": {
            "DATASET_DIR": str(DATASET_DIR),
            "NFRAMES": NFRAMES,
            "MAX_ITER": MAX_ITER,
            "TOL": TOL,
            "EXPORT_NPZ": EXPORT_NPZ,
            "EXPORT_FIGURES": EXPORT_FIGURES,
            "RANDOM_SEED": RANDOM_SEED,
            "IMAGE_BINNING": IMAGE_BINNING,
            "INTERPOLATION": INTERPOLATION,
        },
        "cpu_info": cpu_info,
        "gpu_info": gpu_info,
        "timings": timings,
        "indicators": indicators,
        "per_frame_s": per_frame_times,
        "per_frame_mean_s": per_frame_mean,
        "per_frame_std_s": per_frame_std,
        "per_frame_total_s": solve_total,
    }
    return result


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def load_latest_result(base_slug: str, backend: str) -> Optional[Dict[str, Any]]:
    pattern = f"bench_{base_slug}__*__{backend}.json"
    candidates = sorted(
        OUTPUT_PATH.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return json.loads(candidates[0].read_text())


def update_summary_csv(result: Dict[str, Any], csv_path: Path) -> None:
    header = ["machine_id", "backend", "status"]
    header += [key for key, _ in STEP_LABELS]
    header += ["per_frame_mean_s", "per_frame_std_s", "per_frame_total_s"]

    def fmt(value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.6f}"
        if value is None:
            return ""
        return str(value)

    existing_rows: List[Dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                existing_rows.append(row)

    key_tuple = (result.get("machine_id"), result.get("backend"))
    new_row = {
        "machine_id": result.get("machine_id", ""),
        "backend": result.get("backend", ""),
        "status": result.get("status", "ok"),
    }
    timings = result.get("timings", {})
    for key, _label in STEP_LABELS:
        new_row[key] = fmt(timings.get(key))
    new_row["per_frame_mean_s"] = fmt(result.get("per_frame_mean_s"))
    new_row["per_frame_std_s"] = fmt(result.get("per_frame_std_s"))
    new_row["per_frame_total_s"] = fmt(result.get("per_frame_total_s"))

    replaced = False
    for idx, row in enumerate(existing_rows):
        if (row.get("machine_id"), row.get("backend")) == key_tuple:
            existing_rows[idx] = new_row
            replaced = True
            break
    if not replaced:
        existing_rows.append(new_row)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)


def run_parent() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    base_slug = get_base_slug()
    cumulative_csv = OUTPUT_PATH / "benchmark_results.csv"
    results: Dict[str, Dict[str, Any]] = {}
    for backend in ("cpu", "gpu"):
        env = os.environ.copy()
        env["DIC_BENCH_BACKEND"] = backend
        env["DIC_BENCH_BASE_SLUG"] = base_slug
        cmd = [sys.executable, str(Path(__file__).resolve())]
        completed = subprocess.run(cmd, env=env)
        if completed.returncode != 0:
            print(f"[WARN] Backend {backend} subprocess exited with code {completed.returncode}", file=sys.stderr)
        res = load_latest_result(base_slug, backend)
        if res:
            results[backend] = res
            update_summary_csv(res, cumulative_csv)
    cpu_res = results.get("cpu")
    gpu_res = results.get("gpu")

    def summarize_hw(label: str, data: Optional[Dict[str, Any]]):
        if not data:
            return f"{label}: unavailable"
        model = data.get("cpu_info", {}).get("model", "unknown")
        cores = data.get("cpu_info", {}).get("cores", "?")
        os_name = data.get("cpu_info", {}).get("os", "?")
        gpu = data.get("gpu_info")
        gpu_str = "none" if not gpu else f"{gpu.get('device_name','?')} ({gpu.get('platform','gpu')})"
        return f"{label}: CPU {model} ({cores} cores, {os_name}); GPU {gpu_str}"

    print(summarize_hw("CPU run", cpu_res))
    print(summarize_hw("GPU run", gpu_res))

    print("\nStep\tCPU (s)\tGPU (s)\tSpeedup")
    rows = []
    for key, label in STEP_LABELS:
        c_val = cpu_res.get("timings", {}).get(key) if cpu_res else None
        g_val = gpu_res.get("timings", {}).get(key) if gpu_res else None
        speedup = (c_val / g_val) if c_val is not None and g_val not in (None, 0) else None
        c_str = f"{c_val:.3f}" if c_val is not None else "-"
        g_str = f"{g_val:.3f}" if g_val is not None else "-"
        s_str = f"{speedup:.2f}x" if speedup is not None else "-"
        print(f"{label}\t{c_str}\t{g_str}\t{s_str}")
        rows.append((label, c_val if c_val is not None else "", g_val if g_val is not None else "", speedup if speedup is not None else ""))

    if cpu_res:
        mean_c = cpu_res.get("per_frame_mean_s", 0.0)
        std_c = cpu_res.get("per_frame_std_s", 0.0)
        print(f"CPU per-frame solve: {mean_c:.3f} ± {std_c:.3f} s")
    if gpu_res and gpu_res.get("status") == "ok":
        mean_g = gpu_res.get("per_frame_mean_s", 0.0)
        std_g = gpu_res.get("per_frame_std_s", 0.0)
        print(f"GPU per-frame solve: {mean_g:.3f} ± {std_g:.3f} s")

    def build_indicators(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if not result or result.get("status") != "ok":
            return None
        indicators = result.get("indicators")
        if indicators:
            return {key: float(indicators.get(key, 0.0)) for key, _ in INDICATOR_LABELS}
        timings = result.get("timings", {})
        prep_total = sum(
            timings.get(key, 0.0) for key in ("A_data_load_s", "B_dic_setup_s", "C_precompute_s", "D_warmup_s")
        )
        return {
            "indicator_solve_total_s": float(timings.get("E_solve_total_s", result.get("per_frame_total_s", 0.0))),
            "indicator_prep_s": float(prep_total),
            "indicator_frame_mean_s": float(result.get("per_frame_mean_s", 0.0)),
        }

    def plot_indicator_histogram(
        cpu_indicators: Dict[str, float],
        gpu_indicators: Dict[str, float],
        out_path: Path,
    ) -> None:
        labels = [label for _key, label in INDICATOR_LABELS]
        cpu_vals = [cpu_indicators.get(key, 0.0) for key, _label in INDICATOR_LABELS]
        gpu_vals = [gpu_indicators.get(key, 0.0) for key, _label in INDICATOR_LABELS]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        bars_cpu = ax.bar(x - width / 2, cpu_vals, width, label="CPU", color="#4c78a8")
        bars_gpu = ax.bar(x + width / 2, gpu_vals, width, label="GPU", color="#f58518")
        ax.set_ylabel("Time (s)")
        ax.set_xticks(x, labels)
        ax.set_title("Benchmark indicators: CPU vs GPU")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bars in (bars_cpu, bars_gpu):
            for rect in bars:
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    summary_csv = OUTPUT_PATH / "benchmark_results_summary.csv"
    with summary_csv.open("w", encoding="utf-8") as fh:
        fh.write("Step,CPU_s,GPU_s,Speedup\n")
        for label, c_val, g_val, speedup in rows:
            c_txt = f"{c_val:.6f}" if isinstance(c_val, (int, float)) else ""
            g_txt = f"{g_val:.6f}" if isinstance(g_val, (int, float)) else ""
            s_txt = f"{speedup:.6f}" if isinstance(speedup, (int, float)) else ""
            fh.write(f"{label},{c_txt},{g_txt},{s_txt}\n")

    cpu_ind = build_indicators(cpu_res)
    gpu_ind = build_indicators(gpu_res)
    if EXPORT_INDICATOR_FIGURE and cpu_ind and gpu_ind:
        indicator_plot = OUTPUT_PATH / f"benchmark_indicators_{base_slug}.png"
        plot_indicator_histogram(cpu_ind, gpu_ind, indicator_plot)
        print(f"\nIndicator histogram saved to {indicator_plot}")
    else:
        print("\nIndicator histogram skipped (missing CPU/GPU results).")


def run_child(backend: str) -> None:
    base_slug = sanitize_label(os.environ.get("DIC_BENCH_BASE_SLUG") or get_base_slug())
    try:
        result = run_backend_benchmark(backend, base_slug)
        result_path = OUTPUT_PATH / f"bench_{result['machine_id']}.json"
        write_json(result_path, result)
    except Exception as exc:
        if backend == "gpu":
            cpu_info = collect_cpu_info()
            fallback_id = f"{base_slug}__{sanitize_label('gpu-unavailable')}__gpu"
            payload = {
                "backend": backend,
                "status": "gpu_unavailable",
                "machine_id": fallback_id,
                "base_slug": base_slug,
                "cpu_info": cpu_info,
                "gpu_info": None,
                "timings": {},
                "per_frame_s": [],
                "per_frame_mean_s": 0.0,
                "per_frame_std_s": 0.0,
                "per_frame_total_s": 0.0,
                "error": str(exc),
            }
            result_path = OUTPUT_PATH / f"bench_{fallback_id}.json"
            write_json(result_path, payload)
        else:
            raise


if __name__ == "__main__":
    backend = os.environ.get("DIC_BENCH_BACKEND")
    if backend:
        run_child(backend)
    else:
        run_parent()
