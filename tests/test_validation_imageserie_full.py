from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt

from d2ic.pixel_assets import build_pixel_assets
from d2ic.mask2mesh import mask_to_mesh_assets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "doc" / "00_Validation_ImageSerie.py"
DATA_DIR = PROJECT_ROOT / "doc" / "img" / "Sample3"
OUT_DIR = PROJECT_ROOT / "tests" / "_outputs"

matplotlib.use("Agg")


def _load_validation_module():
    spec = importlib.util.spec_from_file_location("validation_imageserie", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load 00_Validation_ImageSerie module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_validation_imageserie_full_error_stats() -> None:
    if not DATA_DIR.exists():
        pytest.skip("Sample3 image sequence not available.")

    module = _load_validation_module()
    module.USE_GMSH_MESH = False

    im_ref, images_def, expected, frame_paths = module._load_sequence()
    disp_history = module._solve_sequence(im_ref, images_def)

    err = disp_history - expected[:, None, :]
    err_mean = err.mean(axis=1)
    err_std = err.std(axis=1)

    assert err_mean.shape == expected.shape
    assert err_std.shape == expected.shape
    assert np.isfinite(err_mean).all()
    assert np.isfinite(err_std).all()

    assert np.all(np.abs(err_mean) < 0.1)
    assert np.all(np.isfinite(err_std))

    # Debug: relate large errors to pixel support (degree per node).
    mask = module._load_mask(DATA_DIR / "roi.tif")
    mesh, _ = mask_to_mesh_assets(
        mask=mask,
        element_size_px=module.MESH_ELEMENT_SIZE_PX,
        remove_islands=True,
        min_island_area_px=64,
    )
    pix = build_pixel_assets(mesh=mesh, ref_image=im_ref, binning=1.0)
    node_pixel_degree = np.asarray(pix.node_pixel_degree)
    max_err_per_node = np.max(np.abs(err), axis=(0, 2))
    zero_support = node_pixel_degree == 0
    max_err_zero = float(max_err_per_node[zero_support].max(initial=0.0))
    max_err_nonzero = float(max_err_per_node[~zero_support].max(initial=0.0))
    worst_idx = int(np.argmax(max_err_per_node))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "validation_sample3_error_stats.npz",
        frame_names=[p.stem for p in frame_paths],
        err_mean=err_mean,
        err_std=err_std,
        expected=expected,
        node_pixel_degree=node_pixel_degree,
        max_err_per_node=max_err_per_node,
        max_err_zero=max_err_zero,
        max_err_nonzero=max_err_nonzero,
        worst_node_index=worst_idx,
    )

    nodes = np.asarray(mesh.nodes_xy)
    fig_nodes, ax_nodes = plt.subplots(figsize=(6.5, 6.0))
    sc = ax_nodes.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c=max_err_per_node,
        cmap="magma",
        s=18,
        edgecolors="none",
    )
    ax_nodes.set_aspect("equal")
    ax_nodes.set_title("Max |error| per node")
    fig_nodes.colorbar(sc, ax=ax_nodes, label="|error| [px]")
    fig_nodes.tight_layout()
    fig_nodes.savefig(OUT_DIR / "validation_sample3_max_error_nodes.png", dpi=300)
    plt.close(fig_nodes)

    x = np.arange(1, err_mean.shape[0] + 1, dtype=int)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.5, 6.5))
    colors = ("#1b9e77", "#d95f02")
    for ax, comp_idx, title, color in zip(axes, (0, 1), ("Ux", "Uy"), colors):
        mean = err_mean[:, comp_idx]
        std = err_std[:, comp_idx]
        ax.plot(x, mean, marker="o", color=color, label=f"Error {title}")
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
            label=f"{title} ±1σ",
        )
        ax.set_ylabel(f"{title} error [px]")
        ax.set_title(f"Error {title} with 1σ uncertainty")
        ax.grid(True)
        ax.legend(loc="upper left", frameon=False)

    axes[-1].set_xlabel("Frame index")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([p.stem.split()[0] for p in frame_paths], rotation=45, ha="right")
    fig.tight_layout()
    fig_path = OUT_DIR / "validation_sample3_error_plot.png"
    fig.savefig(fig_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
