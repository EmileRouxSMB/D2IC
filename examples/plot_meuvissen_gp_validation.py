"""Render publication-ready PNG diagnostics from a Meuvissen GP validation NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def _triangulation(nodes, elements):
    triangles = np.concatenate((elements[:, [0, 1, 2]], elements[:, [0, 2, 3]]))
    return mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)


def _field(ax, triangulation, values, title, *, cmap="viridis", symmetric=False):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    kwargs = {}
    if symmetric and finite.size:
        limit = np.nanpercentile(np.abs(finite), 99)
        kwargs.update(vmin=-limit, vmax=limit)
    image = ax.tripcolor(triangulation, values, shading="gouraud", cmap=cmap, **kwargs)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x [px binned]")
    ax.set_ylabel("y [px binned]")
    plt.colorbar(image, ax=ax, shrink=0.82)


def render(npz_path: Path, output_dir: Path | None = None):
    """Create displacement, error and GP-diagnostic PNGs from one validation."""
    data = np.load(npz_path)
    output_dir = npz_path.with_suffix("").parent / "gp_validation_figures" if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    triangulation = _triangulation(data["nodes_xy"], data["elements"])
    truth = data["truth"]
    valid = data["valid"]
    names = ("unregularized", "uniform_mechanical", "adaptive_gp")
    labels = ("Sans régularisation", "Mécanique uniforme", "Mécanique adaptative GP")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for component, symbol in enumerate(("uₓ", "uᵧ")):
        _field(axes[component, 0], triangulation, truth[:, component], f"Vérité Abaqus — {symbol}", cmap="coolwarm", symmetric=True)
        for column, (name, label) in enumerate(zip(names, labels), start=1):
            _field(axes[component, column], triangulation, data[f"{name}_displacement"][:, component], f"{label} — {symbol}", cmap="coolwarm", symmetric=True)
    fig.suptitle("Meuvissen — champs de déplacement [px binned]", fontsize=15)
    fig.savefig(output_dir / "01_displacement_fields.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    rmses = []
    for ax, name, label in zip(axes, names, labels):
        error = np.linalg.norm(data[f"{name}_displacement"] - truth, axis=1)
        error = np.where(valid, error, np.nan)
        rmses.append(float(np.sqrt(np.nanmean(error**2))))
        _field(ax, triangulation, error, f"{label}\nRMSE vectorielle = {rmses[-1]:.3f} px", cmap="magma")
    fig.suptitle("Meuvissen — erreur de déplacement par rapport à Abaqus", fontsize=15)
    fig.savefig(output_dir / "02_displacement_errors.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    _field(axes[0, 0], triangulation, data["adaptive_gp_mechanical_confidence"], "Confiance mécanique λ = exp(-g)", cmap="viridis")
    _field(axes[0, 1], triangulation, data["adaptive_gp_log_mechanical_variance"], "Log-variance mécanique g", cmap="coolwarm", symmetric=True)
    _field(axes[1, 0], triangulation, data["adaptive_gp_log_variance_std"], "Incertitude Laplace σ(g)", cmap="cividis")
    _field(axes[1, 1], triangulation, data["adaptive_gp_mechanical_residual_norm"], "Norme du résidu mécanique", cmap="inferno")
    fig.suptitle("Meuvissen — diagnostics de la régularisation adaptative GP", fontsize=15)
    fig.savefig(output_dir / "03_gp_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(labels, rmses, color=("#777777", "#4c78a8", "#e45756"))
    ax.set_ylabel("RMSE vectorielle [px binned]")
    ax.set_title("Erreur globale de déplacement — frame étudiée")
    ax.tick_params(axis="x", rotation=12)
    for index, value in enumerate(rmses):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom")
    fig.savefig(output_dir / "04_rmse_comparison.png", dpi=180)
    plt.close(fig)

    summary = {label: value for label, value in zip(labels, rmses)}
    (output_dir / "figure_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(render(args.npz, args.output_dir))


if __name__ == "__main__":
    main()
