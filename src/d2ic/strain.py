from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .types import Array


@jax.jit
def compute_green_lagrange_strain_nodes_lsq(
    displacement: Array,
    nodes_coord: Array,
    node_neighbor_index: Array,
    node_neighbor_degree: Array,
    gauge_length: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[Array, Array]:
    """
    Weighted LSQ estimate of deformation gradient F and Green–Lagrange strain E.

    Parameters
    ----------
    displacement:
        Nodal displacement field (Nn, 2) with column order (ux, uy).
    nodes_coord:
        Reference nodal coordinates (Nn, 2), [:,0]=x (columns), [:,1]=y (rows).
    node_neighbor_index:
        Dense neighbor table (Nn, max_deg) padded with arbitrary indices.
    node_neighbor_degree:
        Degree per node (Nn,). Only the first ``deg`` entries in the neighbor table are valid.
    gauge_length:
        Optional Gaussian weighting radius (0 disables the weighting).
    eps:
        Diagonal Tikhonov regularization added to the normal matrix.

    Returns
    -------
    (F_all, E_all):
        Tuple of deformation gradients and strains, each with shape (Nn, 2, 2).
    """

    disp = jnp.asarray(displacement)
    X = jnp.asarray(nodes_coord)
    node_neighbor_index = jnp.asarray(node_neighbor_index)
    node_neighbor_degree = jnp.asarray(node_neighbor_degree)

    Nnodes, max_deg = node_neighbor_index.shape
    L = jnp.squeeze(jnp.asarray(gauge_length))

    def one_node(i):
        xi = X[i]
        ui = disp[i]

        idx_all = node_neighbor_index[i]
        deg = node_neighbor_degree[i]
        idx_range = jnp.arange(max_deg)
        mask = idx_range < deg

        neigh_ids = jnp.where(mask, idx_all, 0)
        xj = X[neigh_ids]
        uj = disp[neigh_ids]

        dX = xj - xi[None, :]
        du = uj - ui[None, :]

        r = jnp.linalg.norm(dX, axis=1)
        base_w = mask.astype(jnp.float32)
        w_exp = base_w * jnp.exp(-(r / (L + 1e-12)) ** 2)
        w = jnp.where(L > 0.0, w_exp, base_w)

        w_col = w[:, None]
        Xw = dX * w_col

        A = Xw.T @ dX + eps * jnp.eye(2, dtype=dX.dtype)
        b0 = Xw.T @ du[:, 0]
        b1 = Xw.T @ du[:, 1]

        grad_ux = jnp.linalg.solve(A, b0)
        grad_uy = jnp.linalg.solve(A, b1)
        Grad_u = jnp.stack([grad_ux, grad_uy], axis=0)

        I = jnp.eye(2, dtype=dX.dtype)
        F = I + Grad_u
        C = F.T @ F
        E = 0.5 * (C - I)
        return F, E

    idxs = jnp.arange(Nnodes)
    F_all, E_all = jax.vmap(one_node)(idxs)
    return F_all, E_all


def green_lagrange_to_voigt(E_all: Array) -> Array:
    """
    Convert Green–Lagrange strain tensors to Voigt vectors.

    Parameters
    ----------
    E_all:
        Strain tensor array with shape ``(N, 2, 2)``.

    Returns
    -------
    Array
        Voigt strain array with shape ``(N, 3)`` in ``[E11, E22, E12]`` order.
    """
    E = jnp.asarray(E_all)
    sym = 0.5 * (E + jnp.swapaxes(E, 1, 2))
    exx = sym[:, 0, 0]
    eyy = sym[:, 1, 1]
    exy = sym[:, 0, 1]
    return jnp.stack([exx, eyy, exy], axis=1)


@jax.jit
def small_strain_nodes_lsq(
    displacement: Array,
    nodes_coord: Array,
    node_neighbor_index: Array,
    node_neighbor_degree: Array,
    gauge_length: float = 0.0,
    eps: float = 1e-8,
) -> Array:
    """
    Least-squares small-strain estimate at each node.

    This function computes Green–Lagrange strains via
    :func:`compute_green_lagrange_strain_nodes_lsq` and returns them in Voigt
    form. For sufficiently small displacement gradients, this matches the
    infinitesimal strain approximation.

    Returns
    -------
    eps_voigt: Array
        Small-strain tensor in Voigt form (N, 3).
    """
    _, E_gl = compute_green_lagrange_strain_nodes_lsq(
        displacement,
        nodes_coord,
        node_neighbor_index,
        node_neighbor_degree,
        gauge_length=gauge_length,
        eps=eps,
    )
    # Green–Lagrange strain matches small strain for small displacement gradients.
    return green_lagrange_to_voigt(E_gl)
