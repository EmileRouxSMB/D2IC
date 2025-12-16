"""
Fonctions utilitaires pour les tutoriels et scripts CLI (visualisation + pipelines).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

from . import generate_roi_mesh
from .dic import Dic
from .dic_plotter import DICPlotter


def plot_sparse_matches(im_ref: np.ndarray, im_def: np.ndarray, extras: dict, out_path: Path) -> None:
    """Affiche les points appariés et les vecteurs déplacement initiaux."""
    pts_ref = extras["pts_ref"]
    pts_def = extras["pts_def"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    ax = axes[0]
    ax.imshow(im_ref, cmap="gray", origin="lower")
    ax.scatter(pts_ref[:, 0], pts_ref[:, 1], s=10, c="lime", edgecolors="k", linewidths=0.5)
    ax.set_title("Référence + points retenus")
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
    ax.set_title("Déformée + vecteurs d'appariement")
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
    """Réalise un rendu PNG d'un champ (Ux/Uy ou Exx/Exy/Eyy)."""
    field_norm = field.strip().lower()
    if field_norm in {"ux", "uy"}:
        fig, _ = plotter.plot_displacement_component(field, image_alpha=image_alpha, cmap=cmap)
    else:
        values = plotter._get_strain_field(field)  # type: ignore[attr-defined]
        label = plotter._latex_label(field, "strain")  # type: ignore[attr-defined]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(plotter.background_image, cmap="gray", origin="lower", alpha=1.0)
        mesh = ax.tripcolor(
            plotter.triangulation,
            values,
            shading="gouraud",
            cmap=cmap,
            alpha=image_alpha,
            edgecolors="none",
            linewidth=0.0,
        )
        quad_mesh = plotter._quad_mesh_collection()
        if quad_mesh is not None:
            ax.add_collection(quad_mesh)
        ax.set_aspect("equal")
        ax.set_title(label)
        fig.colorbar(mesh, ax=ax, label=label)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_dic_sequence(
    dic: Dic,
    im_ref: np.ndarray,
    images_def: List[np.ndarray],
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
    """
    Corrèle l'image de référence ``im_ref`` vers chaque image de ``images_def``.
    Retourne les déplacements et l'historique solveur pour chaque frame.
    Un callback optionnel peut être exécuté dès qu'une frame est validée.
    """
    n_frames = len(images_def)
    if n_frames == 0:
        raise ValueError("Aucune image déformée fournie à run_dic_sequence.")

    n_nodes = int(dic.node_coordinates.shape[0])
    disp_all = np.zeros((n_frames, n_nodes, 2), dtype=np.float32)
    history_all: List[dict] = [{} for _ in range(n_frames)]

    def _limit_extrapolation(guess: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """Limite la norme du pas d'extrapolation pour rester robuste."""
        delta = guess - anchor
        norms = np.linalg.norm(delta, axis=1)
        mask = norms > max_extrapolation
        if np.any(mask):
            scale = max_extrapolation / (norms[mask] + 1e-12)
            delta[mask] = delta[mask] * scale[:, None]
        return anchor + delta

    # Frame 0 : initialisation par correspondances parcimonieuses
    disp_guess = disp_guess_first if disp_guess_first is not None else np.zeros((n_nodes, 2), dtype=np.float32)
    print("   [Frame 0] DIC global avec champ initial issu des features")
    disp0, hist0 = dic.run_dic(
        im_ref,
        images_def[0],
        disp_guess=disp_guess,
        max_iter=max_iter,
        tol=tol,
        reg_type=reg_type,
        alpha_reg=alpha_reg,
    )
    if n_sweeps_local > 0:
        disp0 = dic.run_dic_nodal(
            im_ref,
            images_def[0],
            disp_init=disp0,
            n_sweeps=n_sweeps_local,
            lam=0.1,
            reg_type="spring_jacobi",
            alpha_reg=100.0,
            max_step=0.2,
            omega_local=0.5,
        )
    disp_all[0] = np.asarray(disp0)
    history_all[0] = {"history": hist0}
    if per_frame_callback is not None:
        per_frame_callback(0, disp_all[0], history_all[0])

    # Frames suivantes : propagation + vitesse optionnelle
    for k in range(1, n_frames):
        print(f"   [Frame {k}] Propagation du déplacement précédent")
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline linéaire : lecture des images, maillage, DIC séquentielle, puis exports.

    Retour :
        - disp_all : déplacements (N_frames, N_nodes, 2)
        - E_all_seq : déformations de Green-Lagrange (N_frames, N_nodes, 2, 2)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1) Chargement des données (séquence)")
    mask_path = img_dir / mask_filename
    mesh_path = out_dir / f"roi_mesh_{int(mesh_element_size_px)}px_sequence.msh"
    im_ref_path = img_dir / ref_image_name

    if not im_ref_path.exists():
        raise FileNotFoundError(f"Image de référence introuvable : {im_ref_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Masque ROI introuvable : {mask_path}")

    im_ref = imread(im_ref_path).astype(np.float32)

    all_imgs = sorted(img_dir.glob(image_pattern))
    im_def_paths = [p for p in all_imgs if p.name != im_ref_path.name]
    if len(im_def_paths) == 0:
        raise FileNotFoundError(f"Aucune image déformée trouvée dans {img_dir} (pattern {image_pattern}).")

    images_def = [imread(path).astype(np.float32) for path in im_def_paths]
    n_frames = len(images_def)
    print(f"   - Image réf : {im_ref.shape}")
    print(f"   - {n_frames} images déformées détectées : {[p.name for p in im_def_paths]}")

    print(f"2) Génération du maillage à partir du masque (taille cible {mesh_element_size_px:.0f} px)")
    mesh_path_generated = generate_roi_mesh(mask_path, element_size=mesh_element_size_px, msh_path=str(mesh_path))
    if mesh_path_generated is None:
        raise RuntimeError("Echec de la génération du maillage.")
    mesh_path = mesh_path_generated
    print(f"   - Maillage généré : {mesh_path}")

    print("3) Création de l'objet Dic et pré-calcul des données pixel")
    dic = Dic(mesh_path=str(mesh_path))
    dic.precompute_pixel_data(jnp.asarray(im_ref))
    n_nodes = int(dic.node_coordinates.shape[0])
    print(f"   - Nombre de nœuds : {n_nodes}")

    print("4) Estimation du déplacement initial (frame 0) par correspondances parcimonieuses")
    disp_guess, extras = dic.compute_feature_disp_guess_big_motion(
        im_ref,
        images_def[0],
        n_patches=32,
        patch_win=21,
        patch_search=15,
        refine=True,
        search_dilation=5.0,
    )
    print(f"   - {extras['pts_ref'].shape[0]} correspondances retenues après RANSAC")
    plot_sparse_matches(im_ref, images_def[0], extras, out_dir / "01_sparse_matches_first.png")

    print("5) DIC globale séquentielle (CG pixelwise + régularisation ressort)")
    frames_to_plot_arr = (
        np.arange(n_frames, dtype=int) if frames_to_plot is None else np.unique(np.asarray(frames_to_plot, dtype=int))
    )
    if frames_to_plot_arr.size > 0:
        min_idx = int(frames_to_plot_arr.min())
        max_idx = int(frames_to_plot_arr.max())
        if min_idx < 0 or max_idx >= n_frames:
            raise ValueError(
                f"frames_to_plot contient des indices hors bornes (0..{n_frames - 1}): {frames_to_plot_arr}"
            )
    frames_to_plot_set = {int(idx) for idx in frames_to_plot_arr.tolist()}
    print(f"   - Frames visualisées : {frames_to_plot_arr}")

    print("6) Post-traitement progressif : calculs + exports effectués à la volée")
    F_all_seq = np.zeros((n_frames, n_nodes, 2, 2), dtype=np.float64)
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
            print(f"      ↳ Figures frame {tag} sauvegardées.")

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
    print(f"   - Séquence traitée : {disp_all.shape[0]} frames")
    print(f"   - Dernier J={history_all[-1]['history'][-1][0]:.3e}, ||grad||={history_all[-1]['history'][-1][1]:.3e}")
    print(f"   - Champs calculés : F_all {F_all_seq.shape}, E_all {E_all_seq.shape}")
    print(f"   - Sauvegardes par frame : {frame_fields_dir}")

    print("7) Sauvegarde compacte finale des champs (Ux, Uy, Exx, Exy, Eyy)")
    np.savez(
        out_dir / "fields_butterfly_sequence.npz",
        Ux=np.asarray(disp_all[..., 0]),
        Uy=np.asarray(disp_all[..., 1]),
        Exx=np.asarray(E_all_seq[..., 0, 0]),
        Exy=np.asarray(E_all_seq[..., 0, 1]),
        Eyy=np.asarray(E_all_seq[..., 1, 1]),
    )
    print(f"   - Champs sauvegardés dans {out_dir / 'fields_butterfly_sequence.npz'}")
    print(f"Figures exportées progressivement dans {out_dir}")
    return disp_all, E_all_seq


__all__ = ["plot_sparse_matches", "plot_field", "run_dic_sequence", "run_pipeline_sequence"]
