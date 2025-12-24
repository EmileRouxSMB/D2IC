"""Helpers shared by tutorials and CLI scripts (plotting and pipelines)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread


def _apply_binning(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample ``image`` by averaging non-overlapping ``factor``×``factor`` blocks."""
    factor = int(factor)
    if factor <= 1:
        return image
    if image.ndim < 2:
        raise ValueError("Binning expects at least 2D images.")
    h, w = image.shape[:2]
    new_h = (h // factor) * factor
    new_w = (w // factor) * factor
    if new_h == 0 or new_w == 0:
        raise ValueError(f"Binning factor {factor} exceeds the image size {image.shape}.")
    cropped = image[:new_h, :new_w, ...]
    if cropped.ndim == 2:
        reshaped = cropped.reshape(new_h // factor, factor, new_w // factor, factor)
        binned = reshaped.mean(axis=(1, 3), dtype=cropped.dtype)
    else:
        c = cropped.shape[2]
        reshaped = cropped.reshape(new_h // factor, factor, new_w // factor, factor, c)
        binned = reshaped.mean(axis=(1, 3), dtype=cropped.dtype)
    return binned.astype(image.dtype, copy=False)


class LazyImageSequence(Sequence[np.ndarray]):
    """Provide list-like access to TIFF frames without loading everything in memory."""

    def __init__(
        self,
        paths: Sequence[Path],
        dtype: np.dtype | type = np.float32,
        binning: int = 1,
    ) -> None:
        self._paths = [Path(p) for p in paths]
        self._dtype = np.dtype(dtype)
        self._cache_idx: int | None = None
        self._cache_img: np.ndarray | None = None
        self._binning = int(binning)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._paths)

    def _load_frame(self, idx: int) -> np.ndarray:
        arr = imread(self._paths[idx]).astype(self._dtype, copy=False)
        if self._binning > 1:
            arr = _apply_binning(arr, self._binning).astype(self._dtype, copy=False)
        # Keep a single-frame cache to avoid re-reading the same image multiple times.
        self._cache_idx = idx
        self._cache_img = arr
        return arr

    def __getitem__(self, item):  # type: ignore[override]
        if isinstance(item, slice):
            return [self[i] for i in range(*item.indices(len(self)))]

        idx = int(item)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if self._cache_idx == idx and self._cache_img is not None:
            return self._cache_img
        return self._load_frame(idx)


from . import generate_roi_mesh
from .dic import Dic
from .dic_plotter import DICPlotter


def plot_sparse_matches(im_ref: np.ndarray, im_def: np.ndarray, extras: dict, out_path: Path) -> None:
    """Display the matched points and the initial displacement vectors."""
    pts_ref = extras["pts_ref"]
    pts_def = extras["pts_def"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    ax = axes[0]
    ax.imshow(im_ref, cmap="gray", origin="lower")
    ax.scatter(pts_ref[:, 0], pts_ref[:, 1], s=10, c="lime", edgecolors="k", linewidths=0.5)
    ax.set_title("Reference + retained points")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(im_def, cmap="gray", origin="lower")
    ax.scatter(pts_def[:, 0], pts_def[:, 1], s=10, c="cyan", edgecolors="k", linewidths=0.5)
    ax.quiver(
        pts_ref[:, 0],
        pts_ref[:, 1],
        pts_def[:, 0] - pts_ref[:, 0],
        pts_def[:, 1] - pts_ref[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="yellow",
        width=0.003,
    )
    ax.set_title("Deformed + matching vectors")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_field(
    plotter: DICPlotter,
    field: str,
    out_path: Path,
    cmap: str = "jet",
    image_alpha: float = 0.6,
) -> None:
    """Render a PNG overlay for ``Ux``, ``Uy``, or any scalar strain component."""
    field_norm = field.strip().lower()
    if field_norm in {"ux", "uy"}:
        fig, _ = plotter.plot_displacement_component(field, image_alpha=image_alpha, cmap=cmap)
    else:
        fig, _ = plotter.plot_strain_component(field, image_alpha=image_alpha, cmap=cmap)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_dic_sequence(
    dic: Dic,
    im_ref: np.ndarray,
    images_def: Sequence[np.ndarray],
    disp_guess_first: np.ndarray | None = None,
    use_velocity: bool = True,
    vel_smoothing: float = 0.5,
    max_iter: int = 400,
    tol: float = 1e-3,
    reg_type: str = "spring",
    alpha_reg: float = 0.1,
    n_sweeps_local: int = 3,
    max_extrapolation: float = 5.0,
    per_frame_callback: Callable[[int, np.ndarray, dict], None] | None = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Sequential DIC solve over ``images_def`` with optional velocity prediction and nodal sweeps.

    Returns stacked displacements plus per-frame history dicts; ``per_frame_callback`` runs right after
    each frame is processed.
    """
    n_frames = len(images_def)
    if n_frames == 0:
        raise ValueError("No deformed image was provided to run_dic_sequence.")

    n_nodes = int(dic.node_coordinates.shape[0])
    disp_all = np.zeros((n_frames, n_nodes, 2))
    history_all: List[dict] = [{} for _ in range(n_frames)]

    def _limit_extrapolation(guess: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """Clamp the extrapolated guess so no node drifts more than ``max_extrapolation`` pixels."""
        delta = guess - anchor
        norms = np.linalg.norm(delta, axis=1)
        mask = norms > max_extrapolation
        if np.any(mask):
            scale = max_extrapolation / (norms[mask] + 1e-12)
            delta[mask] = delta[mask] * scale[:, None]
        return anchor + delta

    # Frame 0: initialize with sparse correspondences.
    disp_guess = disp_guess_first if disp_guess_first is not None else np.zeros((n_nodes, 2))
    print("   [Frame 0] Global DIC with feature-based initial field")
    disp0, hist0 = dic.run_dic(
        im_ref,
        images_def[0],
        disp_guess=disp_guess,
        max_iter=max_iter,
        tol=tol,
        reg_type=reg_type,
        alpha_reg=alpha_reg,
        save_history=True,
    )
    if n_sweeps_local > 0:
        disp0 = dic.run_dic_nodal(
            im_ref,
            images_def[0],
            disp_init=disp0,
            n_sweeps=n_sweeps_local,
            lam=0.1,
            reg_type="spring_jacobi",
            alpha_reg=1.0,
            max_step=0.2,
            omega_local=0.5,
        )
    disp_all[0] = np.asarray(disp0)
    history_all[0] = {"history": hist0}
    if per_frame_callback is not None:
        per_frame_callback(0, disp_all[0], history_all[0])

    # Next frames: propagate previous solution and optionally extrapolate using velocity.
    for k in range(1, n_frames):
        print(f"   [Frame {k}] Propagating the previous displacement")
        disp_guess = np.asarray(disp_all[k - 1])
        if use_velocity and k >= 2:
            v_prev = disp_all[k - 1] - disp_all[k - 2]
            disp_guess = disp_guess + vel_smoothing * v_prev
            disp_guess = _limit_extrapolation(disp_guess, np.asarray(disp_all[k - 1]))

        disp_k, hist_k = dic.run_dic(
            im_ref,
            images_def[k],
            disp_guess=disp_guess,
            max_iter=max_iter,
            tol=tol,
            reg_type=reg_type,
            alpha_reg=alpha_reg,
            save_history=True,
        )
        if n_sweeps_local > 0:
            disp_k = dic.run_dic_nodal(
                im_ref,
                images_def[k],
                disp_init=disp_k,
                n_sweeps=n_sweeps_local,
                lam=0.1,
                reg_type="spring_jacobi",
                alpha_reg=100.0,
                max_step=0.2,
                omega_local=0.5,
            )
        disp_all[k] = np.asarray(disp_k)
        history_all[k] = {"history": hist_k}
        if per_frame_callback is not None:
            per_frame_callback(k, disp_all[k], history_all[k])

    return disp_all, history_all


def run_pipeline_sequence(
    img_dir: Path,
    ref_image_name: str,
    mask_filename: str,
    image_pattern: str,
    out_dir: Path,
    mesh_element_size_px: float,
    dic_max_iter: int,
    dic_tol: float,
    dic_reg_type: str,
    dic_alpha_reg: float,
    local_sweeps: int,
    use_velocity: bool,
    vel_smoothing: float,
    strain_k_ring: int,
    strain_gauge_length: float,
    frames_to_plot: List[int] | np.ndarray | None,
    plot_cmap: str,
    plot_alpha: float,
    enable_initial_guess: bool = True,
    image_binning: int = 1,
    patch_search: int = 15,
    initial_match_mode: str = "translation_zncc",
) -> Tuple[np.ndarray, np.ndarray]:
    """End-to-end pipeline: load data, mesh the ROI, run sequential DIC, and export fields/plots.

    Returns ``(disp_all, E_all_seq)`` with shapes ``(N_frames, N_nodes, 2)`` and ``(N_frames, N_nodes, 2, 2)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1) Loading sequence data")
    mask_path = img_dir / mask_filename
    mesh_path = out_dir / f"roi_mesh_{int(mesh_element_size_px)}px_sequence.msh"
    im_ref_path = img_dir / ref_image_name

    if not im_ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {im_ref_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"ROI mask not found: {mask_path}")

    binning = int(image_binning)
    if binning < 1:
        raise ValueError("image_binning must be >= 1.")

    im_ref = imread(im_ref_path).astype(np.float32, copy=False)
    im_ref = _apply_binning(im_ref, binning).astype(np.float32, copy=False)

    all_imgs = sorted(img_dir.glob(image_pattern))
    im_def_paths = [p for p in all_imgs if p.name != im_ref_path.name]
    if len(im_def_paths) == 0:
        raise FileNotFoundError(f"No deformed image found in {img_dir} (pattern {image_pattern}).")

    images_def = LazyImageSequence(im_def_paths, dtype=im_ref.dtype, binning=binning)
    n_frames = len(images_def)
    print(f"   - Reference image shape: {im_ref.shape}")
    print(f"   - {n_frames} deformed images detected: {[p.name for p in im_def_paths]}")

    print(f"2) Mesh generation from the mask (target size {mesh_element_size_px:.0f} px)")
    mesh_path_generated = generate_roi_mesh(mask_path, element_size=mesh_element_size_px, msh_path=str(mesh_path))
    if mesh_path_generated is None:
        raise RuntimeError("Failed to generate the mesh.")
    mesh_path = mesh_path_generated
    print(f"   - Mesh generated: {mesh_path}")

    print("3) Create the Dic object and precompute pixel data")
    dic = Dic(mesh_path=str(mesh_path))
    dic.binning = float(binning)
    dic.precompute_pixel_data(jnp.asarray(im_ref))
    n_nodes = int(dic.node_coordinates.shape[0])
    print(f"   - Number of nodes: {n_nodes}")

    print("4) Initial displacement (frame 0) from sparse correspondences")
    if enable_initial_guess:
        try:
            disp_guess, extras = dic.compute_feature_disp_guess_big_motion(
                im_ref,
                images_def[0],
                n_patches=64,
                patch_win=21,
                patch_search=patch_search,
                refine=True,
                search_dilation=5.0,
                use_element_centers=False,
                initial_match_mode=initial_match_mode,
            )
        except ValueError as exc:
            raise RuntimeError(
                "Initial sparse matching failed; consider reducing PATCH_SEARCH, increasing IMAGE_BINNING, "
                "or disabling ENABLE_INITIAL_GUESS. "
                f"(binning={binning}, patch_win=21, patch_search={patch_search}, match_mode={initial_match_mode})"
            ) from exc
        print(f"   - {extras['pts_ref'].shape[0]} correspondences kept after RANSAC")
        plot_sparse_matches(im_ref, images_def[0], extras, out_dir / "01_sparse_matches_first.png")
    else:
        disp_guess = np.zeros((n_nodes, 2), dtype=np.float32)
        print("   - Initial sparse search disabled; starting from zero displacement.")

    print("5) Sequential global DIC (pixelwise CG + spring regularization)")
    frames_to_plot_arr = (
        np.arange(n_frames, dtype=int) if frames_to_plot is None else np.unique(np.asarray(frames_to_plot, dtype=int))
    )
    if frames_to_plot_arr.size > 0:
        min_idx = int(frames_to_plot_arr.min())
        max_idx = int(frames_to_plot_arr.max())
        if min_idx < 0 or max_idx >= n_frames:
            raise ValueError(
                f"frames_to_plot contains indices outside 0..{n_frames - 1}: {frames_to_plot_arr}"
            )
    frames_to_plot_set = {int(idx) for idx in frames_to_plot_arr.tolist()}
    print(f"   - Frames selected for visualization: {frames_to_plot_arr}")

    print("6) Progressive post-processing: calculations + exports on the fly")
    F_all_seq = np.zeros((n_frames, n_nodes, 2, 2))
    E_all_seq = np.zeros_like(F_all_seq)
    frame_fields_dir = out_dir / "per_frame_fields"
    frame_fields_dir.mkdir(parents=True, exist_ok=True)

    def _export_frame_results(frame_idx: int, disp_frame: np.ndarray, _history: dict) -> None:
        disp_np = np.asarray(disp_frame)
        F_k, E_k = dic.compute_green_lagrange_strain_nodes(
            disp_np,
            k_ring=strain_k_ring,
            gauge_length=strain_gauge_length,
        )
        F_np = np.asarray(F_k)
        E_np = np.asarray(E_k)
        F_all_seq[frame_idx] = F_np
        E_all_seq[frame_idx] = E_np

        frame_npz_path = frame_fields_dir / f"fields_frame_{frame_idx:03d}.npz"
        np.savez(
            frame_npz_path,
            Ux=disp_np[..., 0],
            Uy=disp_np[..., 1],
            Exx=E_np[..., 0, 0],
            Exy=E_np[..., 0, 1],
            Eyy=E_np[..., 1, 1],
        )

        if frame_idx in frames_to_plot_set:
            tag = f"{frame_idx:03d}"
            plotter = DICPlotter(
                background_image=images_def[frame_idx],
                displacement=disp_np,
                strain_fields=(F_np, E_np),
                dic_object=dic,
            )
            plot_field(
                plotter,
                "Ux",
                out_dir / f"02_displacement_Ux_frame_{tag}.png",
                cmap=plot_cmap,
                image_alpha=plot_alpha,
            )
            plot_field(
                plotter,
                "Uy",
                out_dir / f"03_displacement_Uy_frame_{tag}.png",
                cmap=plot_cmap,
                image_alpha=plot_alpha,
            )
            plot_field(
                plotter,
                "Exx",
                out_dir / f"04_strain_Exx_frame_{tag}.png",
                cmap=plot_cmap,
                image_alpha=plot_alpha,
            )
            plot_field(
                plotter,
                "Exy",
                out_dir / f"05_strain_Exy_frame_{tag}.png",
                cmap=plot_cmap,
                image_alpha=plot_alpha,
            )
            plot_field(
                plotter,
                "Eyy",
                out_dir / f"06_strain_Eyy_frame_{tag}.png",
                cmap=plot_cmap,
                image_alpha=plot_alpha,
            )
            print(f"      ↳ Frame {tag} figures saved.")

    disp_all, history_all = run_dic_sequence(
        dic,
        im_ref,
        images_def,
        disp_guess_first=disp_guess,
        use_velocity=use_velocity,
        vel_smoothing=vel_smoothing,
        max_iter=dic_max_iter,
        tol=dic_tol,
        reg_type=dic_reg_type,
        alpha_reg=dic_alpha_reg,
        n_sweeps_local=local_sweeps,
        per_frame_callback=_export_frame_results,
    )
    print(f"   - Sequence processed: {disp_all.shape[0]} frames")
    last_hist = history_all[-1].get("history")
    if last_hist:
        print(f"   - Last J={last_hist[-1][0]:.3e}, ||grad||={last_hist[-1][1]:.3e}")
    else:
        print("   - Solver history disabled for performance.")
    print(f"   - Fields computed: F_all {F_all_seq.shape}, E_all {E_all_seq.shape}")
    print(f"   - Per-frame saves: {frame_fields_dir}")

    print("7) Final compact save of Ux, Uy, Exx, Exy, Eyy")
    np.savez(
        out_dir / "fields_butterfly_sequence.npz",
        Ux=np.asarray(disp_all[..., 0]),
        Uy=np.asarray(disp_all[..., 1]),
        Exx=np.asarray(E_all_seq[..., 0, 0]),
        Exy=np.asarray(E_all_seq[..., 0, 1]),
        Eyy=np.asarray(E_all_seq[..., 1, 1]),
    )
    print(f"   - Fields saved to {out_dir / 'fields_butterfly_sequence.npz'}")
    print(f"Figures exported progressively to {out_dir}")
    return disp_all, E_all_seq


__all__ = ["plot_sparse_matches", "plot_field", "run_dic_sequence", "run_pipeline_sequence"]
