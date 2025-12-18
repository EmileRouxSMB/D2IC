#!/usr/bin/env python3
"""Autonomous CPU vs GPU benchmark on the Butterfly tutorial sequence."""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import re
import shutil

# =====================
# SETTINGS
# =====================
DATASET_DIR = "doc/img/butterFly"
REF_IMAGE_NAME = "VIS_0000.tif"
MASK_FILENAME = "roi.tif"
IMAGE_PATTERN = "VIS_*.tif"
NFRAMES = 8
DO_WARMUP = True
EXPORT_NPZ = True
EXPORT_FIGURES = False
MAX_ITER = 400
TOL = 1e-2
OUTPUT_DIR = "benchmark_results"
MESH_ELEMENT_SIZE_PX = 40.0
DIC_REG_TYPE = "spring"
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

BASE_DIR = Path(__file__).resolve().parent
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
    os.environ["JAX_PLATFORM_NAME"] = backend
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    jax.config.update("jax_platform_name", backend)
    try:
        devices = jax.devices()
    except RuntimeError as err:
        raise RuntimeError(f"Unable to initialize JAX backend '{backend}': {err}") from err
    if backend == "gpu" and not any(dev.platform == "gpu" for dev in devices):
        raise RuntimeError("No GPU detected for backend 'gpu'.")
    from D2IC import generate_roi_mesh
    from D2IC.dic import Dic
    from D2IC.dic_plotter import DICPlotter

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
        im_ref = imread(DATASET_PATH / REF_IMAGE_NAME).astype(np.float32)
        all_imgs = sorted(DATASET_PATH.glob(IMAGE_PATTERN))
        frame_paths = [p for p in all_imgs if p.name != REF_IMAGE_NAME]
        if not frame_paths:
            raise RuntimeError("No deformed frames found for benchmarking.")
        frames = frame_paths[:NFRAMES]
        images_def = [imread(p).astype(np.float32) for p in frames]

    # Step B: mesh + Dic
    with StepTimer(timings, "B_dic_setup_s"):
        mesh_path = backend_dir / f"roi_mesh_{backend}.msh"
        mesh_generated = generate_roi_mesh(mask_path, element_size=MESH_ELEMENT_SIZE_PX, msh_path=str(mesh_path))
        if mesh_generated is None:
            raise RuntimeError("Mesh generation failed.")
        dic = Dic(mesh_path=str(mesh_path))

    # Step C: precompute pixel data
    with StepTimer(timings, "C_precompute_s"):
        dic.precompute_pixel_data(im_ref)

    n_nodes = int(dic.node_coordinates.shape[0])
    warmup_im = images_def[0]
    if DO_WARMUP:
        with StepTimer(timings, "D_warmup_s"):
            warmup_disp_guess = np.zeros((n_nodes, 2), dtype=np.float32)
            disp_warmup, _ = dic.run_dic(
                im_ref,
                warmup_im,
                disp_guess=warmup_disp_guess,
                max_iter=MAX_ITER,
                tol=TOL,
                reg_type=DIC_REG_TYPE,
                alpha_reg=DIC_ALPHA_REG,
                save_history=False,
            )
            disp_warmup.block_until_ready()
    else:
        timings["D_warmup_s"] = 0.0

    disp_history: List[np.ndarray] = []
    F_all = np.zeros((len(images_def), n_nodes, 2, 2), dtype=np.float32)
    E_all = np.zeros_like(F_all)
    frame_names = [p.name for p in frames]

    for idx, im_def in enumerate(images_def):
        frame_start = time.perf_counter()
        if idx == 0:
            disp_guess, _ = dic.compute_feature_disp_guess_big_motion(
                im_ref,
                im_def,
                n_patches=FEATURE_N_PATCHES,
                patch_win=FEATURE_PATCH_WIN,
                patch_search=FEATURE_PATCH_SEARCH,
                refine=FEATURE_REFINE,
                search_dilation=FEATURE_SEARCH_DILATION,
            )
            disp_guess = np.asarray(disp_guess)
        else:
            disp_guess = np.asarray(disp_history[-1])
            if USE_VELOCITY and idx >= 2:
                v_prev = disp_history[-1] - disp_history[-2]
                disp_guess = disp_guess + VEL_SMOOTHING * v_prev
                disp_guess = limit_extrapolation(disp_guess, disp_history[-1], MAX_EXTRAPOLATION)
        disp_sol, _ = dic.run_dic(
            im_ref,
            im_def,
            disp_guess=disp_guess,
            max_iter=MAX_ITER,
            tol=TOL,
            reg_type=DIC_REG_TYPE,
            alpha_reg=DIC_ALPHA_REG,
            save_history=False,
        )
        disp_sol.block_until_ready()
        if LOCAL_SWEEPS > 0:
            disp_sol = dic.run_dic_nodal(
                im_ref,
                im_def,
                disp_init=disp_sol,
                n_sweeps=LOCAL_SWEEPS,
                lam=LOCAL_LAM,
                reg_type="spring_jacobi",
                alpha_reg=LOCAL_ALPHA_REG,
                max_step=LOCAL_MAX_STEP,
                omega_local=LOCAL_OMEGA,
            )
            disp_sol.block_until_ready()
        disp_np = np.asarray(disp_sol)
        disp_history.append(disp_np)
        per_frame_times.append(time.perf_counter() - frame_start)

    solve_total = float(np.sum(per_frame_times))
    timings["E_solve_total_s"] = solve_total

    with StepTimer(timings, "F_strain_s"):
        for i, disp_np in enumerate(disp_history):
            F_k, E_k = dic.compute_green_lagrange_strain_nodes(
                disp_np,
                k_ring=STRAIN_K_RING,
                gauge_length=STRAIN_GAUGE_LENGTH,
            )
            F_all[i] = np.asarray(F_k)
            E_all[i] = np.asarray(E_k)

    npz_time = 0.0
    if EXPORT_NPZ:
        with StepTimer(timings, "G_npz_export_s"):
            np.savez(
                backend_dir / f"fields_{backend}.npz",
                disp=np.asarray(disp_history),
                F=F_all,
                E=E_all,
                frame_names=np.array(frame_names),
            )
    else:
        timings["G_npz_export_s"] = 0.0

    fig_time = 0.0
    if EXPORT_FIGURES:
        frames_to_plot = FRAMES_TO_PLOT if FRAMES_TO_PLOT is not None else list(range(len(disp_history)))
        with StepTimer(timings, "G_figures_export_s"):
            for idx in frames_to_plot:
                if idx >= len(disp_history):
                    continue
                plotter = DICPlotter(
                    background_image=images_def[idx],
                    displacement=disp_history[idx],
                    strain_fields=(F_all[idx], E_all[idx]),
                    dic_object=dic,
                )
                for comp in FIG_COMPONENTS:
                    if comp.lower() in {"ux", "uy"}:
                        fig, _ = plotter.plot_displacement_component(comp, cmap=PLOT_CMAP, image_alpha=PLOT_ALPHA)
                    else:
                        fig, _ = plotter.plot_strain_component(comp, cmap=PLOT_CMAP, image_alpha=PLOT_ALPHA)
                    fig.savefig(backend_dir / f"{comp}_frame_{idx:03d}.png", dpi=200)
                    plt.close(fig)
    else:
        timings["G_figures_export_s"] = 0.0

    timings["H_total_s"] = time.perf_counter() - overall_start

    per_frame_mean = statistics.mean(per_frame_times) if per_frame_times else 0.0
    per_frame_std = statistics.pstdev(per_frame_times) if len(per_frame_times) > 1 else 0.0

    result = {
        "backend": backend,
        "status": "ok",
        "machine_id": run_id,
        "base_slug": base_slug_sanitized,
        "settings": {
            "NFRAMES": NFRAMES,
            "MAX_ITER": MAX_ITER,
            "TOL": TOL,
            "EXPORT_NPZ": EXPORT_NPZ,
            "EXPORT_FIGURES": EXPORT_FIGURES,
        },
        "cpu_info": cpu_info,
        "gpu_info": gpu_info,
        "timings": timings,
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

    summary_csv = OUTPUT_PATH / "benchmark_results_summary.csv"
    with summary_csv.open("w", encoding="utf-8") as fh:
        fh.write("Step,CPU_s,GPU_s,Speedup\n")
        for label, c_val, g_val, speedup in rows:
            c_txt = f"{c_val:.6f}" if isinstance(c_val, (int, float)) else ""
            g_txt = f"{g_val:.6f}" if isinstance(g_val, (int, float)) else ""
            s_txt = f"{speedup:.6f}" if isinstance(speedup, (int, float)) else ""
            fh.write(f"{label},{c_txt},{g_txt},{s_txt}\n")


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
