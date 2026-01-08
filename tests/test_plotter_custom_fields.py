import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from d2ic.dataclasses import DICDiagnostics, DICResult
from d2ic.mesh_assets import Mesh
from d2ic.plotter import DICPlotter


def test_plotter_accepts_user_scalar_fields_from_result() -> None:
    nodes_xy = np.array(
        [
            [0.5, 0.5],
            [5.5, 0.5],
            [5.5, 5.5],
            [0.5, 5.5],
        ],
        dtype=float,
    )
    elements = np.array([[0, 1, 2, 3]], dtype=int)
    mesh = Mesh(nodes_xy=nodes_xy, elements=elements)

    u_nodal = np.zeros((4, 2), dtype=float)
    strain = np.zeros((4, 3), dtype=float)
    result = DICResult(
        u_nodal=u_nodal,
        strain=strain,
        diagnostics=DICDiagnostics(),
        fields={"sigma_e22": np.arange(4, dtype=float)},
    )

    im = np.zeros((8, 8), dtype=float)
    plotter = DICPlotter(result=result, mesh=mesh, def_image=im, project_on_deformed=False)
    fig, ax = plotter.plot("$\\sigma_{E_{22}}$", cmap="Reds")
    assert fig is not None
    assert ax.get_title() == "$\\sigma_{E_{22}}$"


def test_plotter_register_scalar_field_overrides_label() -> None:
    nodes_xy = np.array(
        [
            [0.5, 0.5],
            [5.5, 0.5],
            [5.5, 5.5],
            [0.5, 5.5],
        ],
        dtype=float,
    )
    elements = np.array([[0, 1, 2, 3]], dtype=int)
    mesh = Mesh(nodes_xy=nodes_xy, elements=elements)

    u_nodal = np.zeros((4, 2), dtype=float)
    strain = np.zeros((4, 3), dtype=float)
    result = DICResult(u_nodal=u_nodal, strain=strain, diagnostics=DICDiagnostics())

    im = np.zeros((8, 8), dtype=float)
    plotter = DICPlotter(result=result, mesh=mesh, def_image=im, project_on_deformed=False)
    plotter.register_scalar_field("my_field", np.array([0.0, 1.0, 2.0, 3.0]), label="My Field")
    _, ax = plotter.plot("my_field")
    assert ax.get_title() == "My Field"


def test_plotter_supports_discrepancy_field() -> None:
    assert DICPlotter._normalize_field_name("discrepancy").key == "discrepancy"

    nodes_xy = np.array(
        [
            [0.5, 0.5],
            [5.5, 0.5],
            [5.5, 5.5],
            [0.5, 5.5],
        ],
        dtype=float,
    )
    elements = np.array([[0, 1, 2, 3]], dtype=int)
    mesh = Mesh(nodes_xy=nodes_xy, elements=elements)

    u_nodal = np.zeros((4, 2), dtype=float)
    strain = np.zeros((4, 3), dtype=float)
    result = DICResult(u_nodal=u_nodal, strain=strain, diagnostics=DICDiagnostics())

    ref = np.zeros((8, 8), dtype=float)
    deformed = np.ones((8, 8), dtype=float)
    plotter = DICPlotter(result=result, mesh=mesh, def_image=deformed, ref_image=ref, project_on_deformed=False)
    _, ax = plotter.plot("discrepancy")
    assert ax.get_title() == "Discrepancy"


def test_plotter_plot_makes_figure_current() -> None:
    nodes_xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    elements = np.array([[0, 1, 2, 3]], dtype=int)
    mesh = Mesh(nodes_xy=nodes_xy, elements=elements)

    u_nodal = np.zeros((4, 2), dtype=float)
    strain = np.zeros((4, 3), dtype=float)
    result = DICResult(u_nodal=u_nodal, strain=strain, diagnostics=DICDiagnostics())
    im = np.zeros((4, 4), dtype=float)
    plotter = DICPlotter(result=result, mesh=mesh, def_image=im, project_on_deformed=False)

    empty = plt.figure()
    assert len(empty.axes) == 0

    fig, ax = plotter.plot("u1")
    assert plt.gcf() is fig
    assert ax in fig.axes
