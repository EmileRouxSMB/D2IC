#!/usr/bin/env python3
"""Simplified CPU vs GPU benchmark on the Butterfly tutorial sequence."""

from __future__ import annotations

import json
import os
import platform
import random
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shutil

# =====================
# SETTINGS
# =====================
PWD = Path(__file__).resolve().parent
DATASET_DIR = Path(
    os.environ.get("DIC_BENCH_DATASET_DIR", PWD.parent / "doc" / "img" / "PlateHole")
)
REF_IMAGE_NAME = "ohtcfrp_00.tif"
MASK_FILENAME = "roi.tif"
IMAGE_PATTERN = "ohtcfrp_*.tif"

MESH_ELEMENT_SIZE_PX = 15.0
IMAGE_BINNING = 1
INTERPOLATION = "cubic"
MAX_ITERS = 400
TOL = 1e-2
LOCAL_SWEEPS = 3
LOCAL_LAM = 0.1
LOCAL_MAX_STEP = 0.2
LOCAL_OMEGA = 0.5
USE_VELOCITY = True
STRAIN_GAUGE_LENGTH = 200.0

RANDOM_SEED = 0
EXPORT_INDICATOR_FIGURE = True

CPU_JSON = PWD / "benchmark_cpu.json"
GPU_JSON = PWD / "benchmark_gpu.json"
INDICATOR_PNG = PWD / "benchmark_indicators.png"

INDICATOR_LABELS = [
    ("pixel_assets_build_s", "Pixel-assets build"),
    ("cg_compile_s", "CG compile/warmup"),
    ("local_compile_s", "Local compile/warmup"),
    ("prep_s", "Warmup/Prep (total)"),
    ("dic_solve_s", "Résolution DIC"),
    ("total_s", "Total"),
]


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


def run_backend_benchmark(backend: str) -> Dict[str, Any]:
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ["JAX_PLATFORM_NAME"] = backend

    import jax
    import jax.numpy as jnp

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
    from d2ic.mesh_assets import make_mesh_assets
    from d2ic.pixel_assets import build_pixel_assets

    cpu_info = collect_cpu_info()
    gpu_info = collect_gpu_info(jax)

    prep_start = time.perf_counter()

    mask_path = DATASET_DIR / MASK_FILENAME
    ref_path = DATASET_DIR / REF_IMAGE_NAME
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")

    frame_paths = list_deformed_images(DATASET_DIR, IMAGE_PATTERN, exclude_name=REF_IMAGE_NAME)
    if not frame_paths:
        raise RuntimeError("No deformed frames found for benchmarking.")
    frames = frame_paths

    im_ref = prepare_image(ref_path, binning=IMAGE_BINNING)
    images_def = [prepare_image(p, binning=IMAGE_BINNING) for p in frames]
    mask = imread_gray(mask_path) > 0.5

    try:
        mesh, assets = mask_to_mesh_assets_gmsh(
            mask=mask,
            element_size_px=MESH_ELEMENT_SIZE_PX,
            binning=IMAGE_BINNING,
            remove_islands=True,
            min_island_area_px=64,
        )
    except Exception:
        mesh, _ = mask_to_mesh_assets(
            mask=mask,
            element_size_px=MESH_ELEMENT_SIZE_PX,
            binning=IMAGE_BINNING,
            remove_islands=True,
            min_island_area_px=64,
        )
        assets = make_mesh_assets(mesh, with_neighbors=True)

    roi_mask = getattr(assets, "roi_mask", None)
    if roi_mask is not None and hasattr(roi_mask, "shape") and roi_mask.shape != im_ref.shape:
        roi_mask = None

    # Build pixel-assets on CPU to avoid GPU compile/overhead in a geometric precomputation,
    # then transfer once to the target backend device for the solve.
    pixel_assets_build_start = time.perf_counter()
    cpu_dev = None
    try:  # pragma: no cover - depends on installed backends
        cpu_devs = jax.devices("cpu")
        cpu_dev = cpu_devs[0] if cpu_devs else None
    except Exception:
        cpu_dev = None

    if cpu_dev is not None:
        with jax.default_device(cpu_dev):
            pix_cpu = build_pixel_assets(
                mesh=assets.mesh,
                ref_image=im_ref,
                binning=float(IMAGE_BINNING),
                roi_mask=roi_mask,
            )
    else:
        pix_cpu = build_pixel_assets(
            mesh=assets.mesh,
            ref_image=im_ref,
            binning=float(IMAGE_BINNING),
            roi_mask=roi_mask,
        )
    pixel_assets_build_s = time.perf_counter() - pixel_assets_build_start

    def pixel_assets_to_device(pix, device):
        return replace(
            pix,
            pixel_coords_ref=jax.device_put(pix.pixel_coords_ref, device),
            pixel_nodes=jax.device_put(pix.pixel_nodes, device),
            pixel_shapeN=jax.device_put(pix.pixel_shapeN, device),
            node_neighbor_index=jax.device_put(pix.node_neighbor_index, device),
            node_neighbor_degree=jax.device_put(pix.node_neighbor_degree, device),
            node_neighbor_weight=jax.device_put(pix.node_neighbor_weight, device),
            node_reg_weight=jax.device_put(pix.node_reg_weight, device),
            node_pixel_index=jax.device_put(pix.node_pixel_index, device),
            node_N_weight=jax.device_put(pix.node_N_weight, device),
            node_pixel_degree=jax.device_put(pix.node_pixel_degree, device),
            roi_mask_flat=jax.device_put(pix.roi_mask_flat, device),
        )

    target_device = devices[0]
    pix = pix_cpu if target_device.platform == "cpu" else pixel_assets_to_device(pix_cpu, target_device)
    assets = replace(assets, pixel_data=pix)

    mesh_cfg = MeshDICConfig(
        max_iters=MAX_ITERS,
        tol=TOL,
        reg_strength=0.5,
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
            reg_strength=100.0,
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

    cg_compile_start = time.perf_counter()
    dic.prepare(im_ref, assets)
    cg_compile_s = time.perf_counter() - cg_compile_start

    local_compile_s = 0.0
    if dic_local is not None:
        local_compile_start = time.perf_counter()
        dic_local.prepare(im_ref, assets)
        local_compile_s = time.perf_counter() - local_compile_start
    jax.block_until_ready(jnp.asarray(im_ref))

    prep_time = time.perf_counter() - prep_start

    # Dummy frame (unmeasured): forces any remaining autotune/cache population.
    u_prev = None
    u_prevprev = None
    if images_def:
        dummy_def = images_def[0]
        dummy_res = dic.run(dummy_def)
        if dic_local is not None:
            dic_local.set_initial_guess(jnp.copy(jnp.asarray(dummy_res.u_nodal)))
            dummy_res = dic_local.run(dummy_def)
        dummy_res.u_nodal.block_until_ready()
        u_prev = dummy_res.u_nodal
        u_prevprev = None

    dic_solve_times = []
    measured_frames = images_def[1:] if len(images_def) > 1 else []
    for im_def in measured_frames:
        u_warm = propagator.propagate(u_prev=u_prev, u_prevprev=u_prevprev) if propagator else None
        if u_warm is not None:
            # Copy to avoid donated buffers invalidating warm-start history.
            dic.set_initial_guess(jnp.copy(jnp.asarray(u_warm)))
        solve_start = time.perf_counter()
        res = dic.run(im_def)
        if dic_local is not None:
            # Copy to keep CG output valid after donated local solve.
            dic_local.set_initial_guess(jnp.copy(jnp.asarray(res.u_nodal)))
            res = dic_local.run(im_def)
        res.u_nodal.block_until_ready()
        dic_solve_times.append(time.perf_counter() - solve_start)
        u_prevprev = u_prev
        u_prev = res.u_nodal

    total_time = time.perf_counter() - prep_start

    dic_solve_mean = statistics.mean(dic_solve_times) if dic_solve_times else 0.0

    return {
        "backend": backend,
        "status": "ok",
        "frames_total": len(images_def),
        "frames_measured": len(measured_frames),
        "dummy_frame": bool(images_def),
        "cpu": cpu_info,
        "gpu": gpu_info,
        "indicators": {
            "pixel_assets_build_s": float(pixel_assets_build_s),
            "cg_compile_s": float(cg_compile_s),
            "local_compile_s": float(local_compile_s),
            "prep_s": float(prep_time),
            "dic_solve_s": float(dic_solve_mean),
            "total_s": float(total_time),
        },
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


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
    ax.tick_params(axis="x", labelrotation=30)
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


def run_parent() -> None:
    for backend, path in (("cpu", CPU_JSON), ("gpu", GPU_JSON)):
        env = os.environ.copy()
        env["DIC_BENCH_BACKEND"] = backend
        cmd = [os.sys.executable, str(Path(__file__).resolve())]
        completed = subprocess.run(cmd, env=env)
        if completed.returncode != 0:
            print(f"[WARN] Backend {backend} subprocess exited with code {completed.returncode}")

    cpu_res = load_json(CPU_JSON)
    gpu_res = load_json(GPU_JSON)

    def summarize_hw(label: str, data: Optional[Dict[str, Any]]):
        if not data:
            return f"{label}: unavailable"
        cpu = data.get("cpu", {})
        gpu = data.get("gpu")
        cpu_model = cpu.get("model", "unknown")
        cores = cpu.get("cores", "?")
        os_name = cpu.get("os", "?")
        gpu_str = "none" if not gpu else f"{gpu.get('device_name','?')} ({gpu.get('platform','gpu')})"
        return f"{label}: CPU {cpu_model} ({cores} cores, {os_name}); GPU {gpu_str}"

    print(summarize_hw("CPU run", cpu_res))
    print(summarize_hw("GPU run", gpu_res))

    cpu_ind = cpu_res.get("indicators") if cpu_res else None
    gpu_ind = gpu_res.get("indicators") if gpu_res else None
    if EXPORT_INDICATOR_FIGURE and cpu_ind and gpu_ind and gpu_res.get("status") == "ok":
        plot_indicator_histogram(cpu_ind, gpu_ind, INDICATOR_PNG)
        print(f"Indicator histogram saved to {INDICATOR_PNG}")
    else:
        print("Indicator histogram skipped (missing CPU/GPU results).")


def run_child(backend: str) -> None:
    try:
        result = run_backend_benchmark(backend)
        write_json(CPU_JSON if backend == "cpu" else GPU_JSON, result)
    except Exception as exc:
        payload = {
            "backend": backend,
            "status": "gpu_unavailable" if backend == "gpu" else "error",
            "frames": 0,
            "cpu": collect_cpu_info(),
            "gpu": None,
            "indicators": {
                "pixel_assets_build_s": 0.0,
                "cg_compile_s": 0.0,
                "local_compile_s": 0.0,
                "prep_s": 0.0,
                "dic_solve_s": 0.0,
                "total_s": 0.0,
            },
            "error": str(exc),
        }
        write_json(CPU_JSON if backend == "cpu" else GPU_JSON, payload)
        if backend != "gpu":
            raise


if __name__ == "__main__":
    backend = os.environ.get("DIC_BENCH_BACKEND")
    if backend:
        run_child(backend)
    else:
        run_parent()
