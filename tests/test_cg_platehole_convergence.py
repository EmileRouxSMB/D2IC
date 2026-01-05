from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest
import sys
import importlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from d2ic import (
    MeshDICConfig,
    DICMeshBased,
    GlobalCGSolver,
    mask_to_mesh_assets,
    DICPlotter,
    mask_to_mesh_assets_gmsh,
)
from d2ic.solver_global_cg import _J_TOTAL_VG
from d2ic.app_utils import imread_gray


def _import_legacy_dic():
    sys.modules.setdefault("D2IC", importlib.import_module("d2ic"))
    sys.modules.setdefault("D2IC.motion_init", importlib.import_module("d2ic._legacy.motion_init"))
    sys.modules.setdefault("D2IC.dic_JaxCore", importlib.import_module("d2ic._legacy.dic_JaxCore"))
    sys.modules.setdefault("D2IC.feature_matching", importlib.import_module("d2ic._legacy.feature_matching"))
    return importlib.import_module("d2ic._legacy.dic")


def _crop_with_y(image: np.ndarray, size: int, y0: int) -> tuple[np.ndarray, int, int]:
    h, w = image.shape
    if h < size or w < size:
        raise ValueError("ROI size exceeds image dimensions.")
    i0 = int(y0)
    if i0 < 0 or i0 + size > h:
        raise ValueError("ROI y-range exceeds image dimensions.")
    j0 = (w - size) // 2
    return image[i0 : i0 + size, j0 : j0 + size], i0, j0


def test_cg_platehole_convergence() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "doc" / "img" / "PlateHole"
    ref_path = data_dir / "ohtcfrp_00.tif"
    def_path = data_dir / "ohtcfrp_01.tif"
    if not ref_path.exists() or not def_path.exists():
        raise FileNotFoundError("PlateHole images not found for CG convergence test.")

    ref_full = imread_gray(ref_path)
    def_full = imread_gray(def_path)
    roi_size = 240
    roi_y0 = 700
    ref_image, i0, j0 = _crop_with_y(ref_full, roi_size, roi_y0)
    def_image, _, _ = _crop_with_y(def_full, roi_size, roi_y0)

    mask = np.ones_like(ref_image, dtype=bool)
    mesh, assets = mask_to_mesh_assets(
        mask=mask,
        element_size_px=40.0,
        binning=1,
        remove_islands=False,
    )

    config = MeshDICConfig(
        max_iters=200,
        tol=1e-3,
        reg_strength=0.1,
        strain_gauge_length=40.0,
    )
    dic = DICMeshBased(mesh=mesh, solver=GlobalCGSolver(interpolation="linear"), config=config)
    dic.prepare(ref_image=ref_image, assets=assets)
    result = dic.run(def_image)

    assert dic._state is not None
    pix = dic._state.assets.pixel_data
    assert pix is not None
    im1_T = np.transpose(ref_image, (1, 0))
    im2_T = np.transpose(def_image, (1, 0))
    J_val, grad = _J_TOTAL_VG(
        result.u_nodal,
        im1_T,
        im2_T,
        pix.pixel_coords_ref,
        pix.pixel_nodes,
        pix.pixel_shapeN,
        pix.node_neighbor_index,
        pix.node_neighbor_degree,
        pix.node_neighbor_weight,
        pix.node_reg_weight,
        "linear",
        float(config.reg_strength),
    )
    grad_norm = float(np.linalg.norm(np.asarray(grad)))

    legacy = _import_legacy_dic()
    legacy_J, legacy_grad = legacy._J_TOTAL_VG(
        result.u_nodal,
        im1_T,
        im2_T,
        pix.pixel_coords_ref,
        pix.pixel_nodes,
        pix.pixel_shapeN,
        pix.node_neighbor_index,
        pix.node_neighbor_degree,
        pix.node_neighbor_weight,
        float(config.reg_strength),
    )
    legacy_grad_norm = float(np.linalg.norm(np.asarray(legacy_grad)))
    if legacy_grad_norm < 1e-3:
        assert grad_norm < 1e-3
    else:
        assert np.isclose(grad_norm, legacy_grad_norm, rtol=1e-3, atol=1e-6)
    assert np.isclose(
        float(np.asarray(J_val)),
        float(np.asarray(legacy_J)),
        rtol=1e-4,
        atol=1e-6,
    )

    disp0 = np.zeros_like(np.asarray(result.u_nodal))
    legacy_disp, _, _, _, _ = legacy._cg_solve(
        disp0,
        im1_T,
        im2_T,
        pix.pixel_coords_ref,
        pix.pixel_nodes,
        pix.pixel_shapeN,
        pix.node_neighbor_index,
        pix.node_neighbor_degree,
        pix.node_neighbor_weight,
        float(config.reg_strength),
        int(config.max_iters),
        float(config.tol),
        False,
    )
    diff = np.linalg.norm(np.asarray(result.u_nodal) - np.asarray(legacy_disp))
    denom = np.linalg.norm(np.asarray(legacy_disp)) + 1e-12
    assert diff / denom < 1e-3

    out_dir = Path(__file__).resolve().parent / "_outputs" / "cg_platehole"
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_xy_full = np.asarray(mesh.nodes_xy) + np.array([j0, i0], dtype=float)
    mesh_full = type(mesh)(nodes_xy=nodes_xy_full, elements=mesh.elements)
    plotter = DICPlotter(
        result=result,
        mesh=mesh_full,
        def_image=def_full,
        ref_image=ref_full,
        binning=1.0,
    )
    fig, _ = plotter.plot(field="u1", image_alpha=0.75, cmap="jet", plotmesh=True)
    fig.savefig(out_dir / "u1.png", dpi=200)
    plt.close(fig)


def test_cg_platehole_convergence_gmsh() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "doc" / "img" / "PlateHole"
    ref_path = data_dir / "ohtcfrp_00.tif"
    def_path = data_dir / "ohtcfrp_01.tif"
    if not ref_path.exists() or not def_path.exists():
        raise FileNotFoundError("PlateHole images not found for CG convergence test.")

    ref_full = imread_gray(ref_path)
    def_full = imread_gray(def_path)
    roi_size = 250
    roi_y0 = 700
    ref_image, i0, j0 = _crop_with_y(ref_full, roi_size, roi_y0)
    def_image, _, _ = _crop_with_y(def_full, roi_size, roi_y0)

    mask = np.ones_like(ref_image, dtype=bool)
    try:
        mesh, assets = mask_to_mesh_assets_gmsh(
            mask=mask,
            element_size_px=40.0,
            binning=1,
            remove_islands=False,
        )
    except ImportError as exc:
        pytest.skip(f"gmsh pipeline unavailable: {exc}")

    config = MeshDICConfig(
        max_iters=200,
        tol=1e-3,
        reg_strength=0.1,
        strain_gauge_length=40.0,
    )
    dic = DICMeshBased(mesh=mesh, solver=GlobalCGSolver(interpolation="linear"), config=config)
    dic.prepare(ref_image=ref_image, assets=assets)
    result = dic.run(def_image)

    assert dic._state is not None
    pix = dic._state.assets.pixel_data
    assert pix is not None
    im1_T = np.transpose(ref_image, (1, 0))
    im2_T = np.transpose(def_image, (1, 0))
    J_val, grad = _J_TOTAL_VG(
        result.u_nodal,
        im1_T,
        im2_T,
        pix.pixel_coords_ref,
        pix.pixel_nodes,
        pix.pixel_shapeN,
        pix.node_neighbor_index,
        pix.node_neighbor_degree,
        pix.node_neighbor_weight,
        pix.node_reg_weight,
        "linear",
        float(config.reg_strength),
    )
    grad_norm = float(np.linalg.norm(np.asarray(grad)))

    legacy = _import_legacy_dic()
    legacy_J, legacy_grad = legacy._J_TOTAL_VG(
        result.u_nodal,
        im1_T,
        im2_T,
        pix.pixel_coords_ref,
        pix.pixel_nodes,
        pix.pixel_shapeN,
        pix.node_neighbor_index,
        pix.node_neighbor_degree,
        pix.node_neighbor_weight,
        float(config.reg_strength),
    )
    legacy_grad_norm = float(np.linalg.norm(np.asarray(legacy_grad)))
    if legacy_grad_norm < 1e-3:
        assert grad_norm < 1e-3
    else:
        assert np.isclose(grad_norm, legacy_grad_norm, rtol=1e-3, atol=1e-6)
    assert np.isclose(
        float(np.asarray(J_val)),
        float(np.asarray(legacy_J)),
        rtol=1e-4,
        atol=1e-6,
    )

    out_dir = Path(__file__).resolve().parent / "_outputs" / "cg_platehole_gmsh"
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_xy_full = np.asarray(mesh.nodes_xy) + np.array([j0, i0], dtype=float)
    mesh_full = type(mesh)(nodes_xy=nodes_xy_full, elements=mesh.elements)
    plotter = DICPlotter(
        result=result,
        mesh=mesh_full,
        def_image=def_full,
        ref_image=ref_full,
        binning=1.0,
    )
    fig, _ = plotter.plot(field="u1", image_alpha=0.75, cmap="jet", plotmesh=True)
    fig.savefig(out_dir / "u1.png", dpi=200)
    plt.close(fig)
    fig, _ = plotter.plot(field="u2", image_alpha=0.75, cmap="jet", plotmesh=True)
    fig.savefig(out_dir / "u2.png", dpi=200)
    plt.close(fig)
    fig, _ = plotter.plot(field="u2", image_alpha=0.75, cmap="jet", plotmesh=True)
    fig.savefig(out_dir / "u2.png", dpi=200)
    plt.close(fig)
