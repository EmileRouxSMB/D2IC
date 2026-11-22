"""Objective isotropic membrane regularization for Q4 DIC meshes."""

import jax.numpy as jnp
import numpy as np


def q4_reference_gradients(nodes_xy, elements):
    """Return Q4 shape gradients (Ne, 4 Gauss points, 4 nodes, 2) and area weights."""
    x = np.asarray(nodes_xy)[np.asarray(elements, dtype=np.int32)]
    a = 1.0 / np.sqrt(3.0)
    points = np.array([[-a, -a], [a, -a], [a, a], [-a, a]])
    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    xi, eta = points[:, 0], points[:, 1]
    d_nat = np.stack((
        0.25 * signs[None, :, 0] * (1.0 + eta[:, None] * signs[None, :, 1]),
        0.25 * signs[None, :, 1] * (1.0 + xi[:, None] * signs[None, :, 0]),
    ), axis=-1)
    jac = np.einsum("eni,gnj->egij", x, d_nat)
    det = np.linalg.det(jac)
    if np.any(det <= 0):
        raise ValueError("Q4 elements must have positive Jacobian at every Gauss point")
    grad = np.einsum("gnj,egjk->egnk", d_nat, np.linalg.inv(jac))
    return jnp.asarray(grad), jnp.asarray(det / det.sum())


def elastic_energy(displacement, elements, grad, area_weight, poisson=0.3):
    """Normalized St Venant–Kirchhoff membrane energy, zero for finite rigid motions.

    The shear modulus is one; only the Poisson ratio fixes the isotropic form.
    ``displacement`` is (Nn,2), ``elements`` is (Ne,4), and ``grad`` is
    (Ne,4,4,2). The result has displacement units squared per pixel squared.
    """
    du = displacement[elements]
    H = jnp.einsum("eni,egnj->egij", du, grad)
    F = H + jnp.eye(2, dtype=displacement.dtype)
    E = 0.5 * (jnp.swapaxes(F, -1, -2) @ F - jnp.eye(2, dtype=displacement.dtype))
    trace = jnp.trace(E, axis1=-2, axis2=-1)
    lame = 2.0 * poisson / (1.0 - 2.0 * poisson)
    density = 0.5 * lame * trace**2 + jnp.sum(E * E, axis=(-2, -1))
    return jnp.sum(area_weight * density)
