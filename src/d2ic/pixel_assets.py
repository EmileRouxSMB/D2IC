from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, vmap

from .mesh_assets import Mesh, PixelAssets
from .types import Array


@dataclass(frozen=True)
class PixelSamplingConfig:
    bbox_pad: float = 2.0
    chunk_size: int = 4096


def build_pixel_assets(
    mesh: Mesh,
    ref_image: Array,
    binning: float,
    roi_mask: np.ndarray | None = None,
    bbox_pad: float = 2.0,
    chunk_size: int = 4096,
) -> PixelAssets:
    """
    Build pixel-level assets required by mesh-based image solvers.

    This routine:
    - samples candidate pixels in the mesh bounding box,
    - assigns each pixel to a containing element (if any),
    - computes bilinear shape functions at each retained pixel,
    - builds node-wise gather tables for fast residual/regularization assembly.

    Parameters
    ----------
    mesh:
        Mesh definition (2D quads expected).
    ref_image:
        Reference image used for sizing the ROI sampling domain.
    binning:
        Image downsampling factor used by the solver; mesh coordinates are scaled by ``1/binning``.
    roi_mask:
        Optional boolean mask aligned with the reference image. When provided,
        pixels outside the mask are discarded before element assignment.
    bbox_pad:
        Padding (in pixel units, in the binned coordinate system) added around the mesh bounding box.
    chunk_size:
        Chunk size used by some geometry kernels to control memory.

    Returns
    -------
    PixelAssets
        Pixel-level lookup and interpolation tables.
    """
    nodes = np.asarray(mesh.nodes_xy) / float(binning)
    elements = np.asarray(mesh.elements, dtype=int)
    H, W = ref_image.shape[:2]
    if roi_mask is not None:
        roi_mask = np.asarray(roi_mask, dtype=bool)
        if roi_mask.shape != (H, W):
            raise ValueError(
                f"roi_mask shape {roi_mask.shape} does not match ref_image shape {(H, W)}"
            )
    pixel_coords = _sample_pixels_in_bbox(nodes, H, W, bbox_pad)
    if roi_mask is not None and pixel_coords.size > 0:
        roi_rows = np.clip(np.floor(pixel_coords[:, 1] - 0.5).astype(int), 0, H - 1)
        roi_cols = np.clip(np.floor(pixel_coords[:, 0] - 0.5).astype(int), 0, W - 1)
        keep = roi_mask[roi_rows, roi_cols]
        pixel_coords = pixel_coords[keep]
    pixel_elts, pixel_nodes = build_pixel_to_element_mapping_numpy(
        pixel_coords, nodes, elements, chunk_size=chunk_size
    )
    valid_mask = pixel_elts >= 0
    pixel_coords = pixel_coords[valid_mask]
    pixel_nodes = pixel_nodes[valid_mask]
    pixel_shapeN, _ = compute_pixel_shape_functions_jax(
        jnp.asarray(pixel_coords),
        jnp.asarray(pixel_nodes),
        jnp.asarray(nodes),
    )
    node_neighbor_index, node_neighbor_degree, node_neighbor_weight = build_node_neighbor_dense(
        elements,
        nodes,
        nodes.shape[0],
    )
    node_reg_weight = np.ones_like(np.asarray(node_neighbor_degree), dtype=nodes.dtype)
    node_pixel_index, node_N_weight, node_degree = build_node_pixel_dense(
        pixel_nodes,
        pixel_shapeN,
        nodes.shape[0],
    )
    roi_rows = np.clip(np.floor(pixel_coords[:, 1] - 0.5).astype(int), 0, H - 1)
    roi_cols = np.clip(np.floor(pixel_coords[:, 0] - 0.5).astype(int), 0, W - 1)
    roi_flat = np.ravel_multi_index((roi_rows, roi_cols), (H, W))
    return PixelAssets(
        pixel_coords_ref=jnp.asarray(pixel_coords),
        pixel_nodes=jnp.asarray(pixel_nodes),
        pixel_shapeN=jnp.asarray(pixel_shapeN),
        node_neighbor_index=jnp.asarray(node_neighbor_index),
        node_neighbor_degree=jnp.asarray(node_neighbor_degree),
        node_neighbor_weight=jnp.asarray(node_neighbor_weight),
        node_reg_weight=jnp.asarray(node_reg_weight),
        node_pixel_index=jnp.asarray(node_pixel_index),
        node_N_weight=jnp.asarray(node_N_weight),
        node_pixel_degree=jnp.asarray(node_degree),
        roi_mask_flat=jnp.asarray(roi_flat),
        image_shape=(H, W),
    )


def _sample_pixels_in_bbox(nodes: np.ndarray, H: int, W: int, pad: float) -> np.ndarray:
    """
    Sample pixel centers within the axis-aligned bounding box of the mesh.

    Returns pixel coordinates in ``(x, y)`` order, where ``x`` maps to columns and
    ``y`` maps to rows. Pixel centers are placed at ``(j + 0.5, i + 0.5)``.
    """
    x_min = float(nodes[:, 0].min())
    x_max = float(nodes[:, 0].max())
    y_min = float(nodes[:, 1].min())
    y_max = float(nodes[:, 1].max())
    j0 = max(0, int(np.floor(x_min - pad)))
    j1 = min(W, int(np.ceil(x_max + pad)))
    i0 = max(0, int(np.floor(y_min - pad)))
    i1 = min(H, int(np.ceil(y_max + pad)))
    jj, ii = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1))
    pixel_coords = np.stack([jj.ravel() + 0.5, ii.ravel() + 0.5], axis=1)
    return pixel_coords


def build_pixel_to_element_mapping_numpy(
    pixel_coords,
    nodes_coord,
    elements,
    chunk_size: int = 4096,
):
    """
    Assign pixels to a containing quad element (NumPy implementation).

    Parameters
    ----------
    pixel_coords:
        Pixel center coordinates with shape ``(Np, 2)`` in ``(x, y)`` order.
    nodes_coord:
        Node coordinates with shape ``(Nn, 2)`` in the same coordinate system.
    elements:
        Element connectivity array with shape ``(Ne, 4)``.
    chunk_size:
        Chunk size for the inside-test kernel.

    Returns
    -------
    (pixel_elts, pixel_nodes):
        ``pixel_elts`` is an int array of shape ``(Np,)`` with -1 for pixels outside
        the mesh. ``pixel_nodes`` is an int array of shape ``(Np, 4)`` giving the
        element node indices for each pixel (undefined when ``pixel_elts == -1``).
    """
    pixel_coords = np.asarray(pixel_coords, dtype=float)
    nodes_coord = np.asarray(nodes_coord, dtype=float)
    elements = np.asarray(elements, dtype=np.int32)
    elements_jnp = jnp.asarray(elements)

    Np = int(pixel_coords.shape[0])
    Ne = int(elements.shape[0])
    if Np == 0 or Ne == 0:
        return -np.ones((Np,), dtype=int), np.zeros((Np, 4), dtype=int)

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    quad_nodes = jnp.asarray(nodes_coord[elements])
    edges = jnp.roll(quad_nodes, -1, axis=1) - quad_nodes  # (Ne, 4, 2)
    bbox_min = quad_nodes.min(axis=1)  # (Ne, 2)
    bbox_max = quad_nodes.max(axis=1)  # (Ne, 2)

    quad_next = jnp.roll(quad_nodes, -1, axis=1)
    area2 = jnp.sum(
        quad_nodes[:, :, 0] * quad_next[:, :, 1] - quad_nodes[:, :, 1] * quad_next[:, :, 0],
        axis=1,
    )
    orientation = jnp.where(area2 >= 0.0, 1.0, -1.0)  # (Ne,)

    tol = 1e-6

    @jax.jit
    def _assign_chunk(pix_chunk, active_mask):
        x = pix_chunk[:, 0][:, None]
        y = pix_chunk[:, 1][:, None]
        mask_bbox = (
            (x >= bbox_min[:, 0])
            & (x <= bbox_max[:, 0])
            & (y >= bbox_min[:, 1])
            & (y <= bbox_max[:, 1])
        )  # (chunk_size, Ne)

        vec = pix_chunk[:, None, None, :] - quad_nodes[None, :, :, :]  # (chunk_size, Ne, 4, 2)
        cross = edges[None, :, :, 0] * vec[:, :, :, 1] - edges[None, :, :, 1] * vec[:, :, :, 0]
        inside = jnp.all(orientation[None, :, None] * cross >= -tol, axis=-1)  # (chunk_size, Ne)

        valid = mask_bbox & inside
        valid_any = jnp.any(valid, axis=1)
        first_idx = jnp.argmax(valid.astype(jnp.int32), axis=1)
        elt_idx = jnp.where(valid_any, first_idx, -1)
        safe_idx = jnp.where(elt_idx < 0, 0, elt_idx)
        pix_nodes = jnp.where(
            valid_any[:, None],
            elements_jnp[safe_idx],
            jnp.zeros((pix_chunk.shape[0], 4), dtype=jnp.int32),
        )

        elt_idx = jnp.where(active_mask, elt_idx, -1)
        pix_nodes = jnp.where(active_mask[:, None], pix_nodes, 0)
        return elt_idx, pix_nodes

    pixel_elts = []
    pixel_nodes = []
    for start in range(0, Np, chunk_size):
        end = min(start + chunk_size, Np)
        cur = end - start
        chunk = pixel_coords[start:end]
        if cur < chunk_size:
            pad = np.zeros((chunk_size - cur, 2), dtype=chunk.dtype)
            chunk = np.vstack([chunk, pad])
            active = np.zeros((chunk_size,), dtype=bool)
            active[:cur] = True
        else:
            active = np.ones((chunk_size,), dtype=bool)

        elts_chunk, nodes_chunk = _assign_chunk(jnp.asarray(chunk), jnp.asarray(active))
        pixel_elts.append(np.asarray(elts_chunk)[:cur])
        pixel_nodes.append(np.asarray(nodes_chunk)[:cur])

    return np.concatenate(pixel_elts, axis=0), np.vstack(pixel_nodes)


@jax.jit
def compute_pixel_shape_functions_jax(
    pixel_coords,
    pixel_nodes,
    nodes_coord,
):
    """
    Compute bilinear quad shape functions at each sampled pixel (JAX).

    Parameters
    ----------
    pixel_coords:
        Pixel coordinates with shape ``(Np, 2)`` in ``(x, y)`` order.
    pixel_nodes:
        Node indices per pixel with shape ``(Np, 4)``.
    nodes_coord:
        Node coordinates with shape ``(Nn, 2)``.

    Returns
    -------
    (pixel_N, xi_eta):
        ``pixel_N`` has shape ``(Np, 4)`` and contains the shape function values
        at each pixel. ``xi_eta`` has shape ``(Np, 2)`` and contains the solved
        local coordinates in the reference element.
    """
    Xe_all = nodes_coord[pixel_nodes]
    xi_eta_all = vmap(_newton_quadrature)(pixel_coords, Xe_all)
    xi = xi_eta_all[:, 0]
    eta = xi_eta_all[:, 1]

    def shape_from_xieta(xi, eta):
        return shape_functions_jax(xi, eta)

    pixel_N = vmap(shape_from_xieta)(xi, eta)
    return pixel_N, xi_eta_all


def _newton_quadrature(point, Xe):
    """
    Solve for (xi, eta) such that the bilinear mapping of the quad hits `point`.

    Uses a fixed number of Newton iterations.
    """
    xi = 0.0
    eta = 0.0
    for _ in range(10):
        N = shape_functions_jax(xi, eta)
        dN_dxi, dN_deta = shape_function_gradients_jax(xi, eta)
        x = jnp.dot(N, Xe)
        J = jnp.stack([jnp.dot(dN_dxi, Xe), jnp.dot(dN_deta, Xe)], axis=1)
        F = x - point
        delta = jnp.linalg.solve(J, F)
        xi -= delta[0]
        eta -= delta[1]
    return jnp.array([xi, eta])


def shape_functions_jax(xi, eta):
    """Bilinear shape functions for a 4-node quad at local coordinates (xi, eta)."""
    return jnp.array(
        [
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta),
        ]
    )


def shape_function_gradients_jax(xi, eta):
    """Gradients of bilinear quad shape functions with respect to (xi, eta)."""
    dN_dxi = jnp.array(
        [
            -0.25 * (1 - eta),
            0.25 * (1 - eta),
            0.25 * (1 + eta),
            -0.25 * (1 + eta),
        ]
    )
    dN_deta = jnp.array(
        [
            -0.25 * (1 - xi),
            -0.25 * (1 + xi),
            0.25 * (1 + xi),
            0.25 * (1 - xi),
        ]
    )
    return dN_dxi, dN_deta


def build_node_neighbor_dense(elements, nodes_coord, n_nodes):
    """
    Build dense node-neighborhood tables from element connectivity.

    This variant also computes a simple inverse-distance weight for each edge.
    """
    elements_np = np.asarray(elements, dtype=np.int64)
    nodes_coord_np = np.asarray(nodes_coord)
    n_nodes = int(n_nodes)

    if elements_np.size == 0 or n_nodes == 0:
        zero_idx = jnp.full((n_nodes, 0), -1, dtype=jnp.int32)
        zero_deg = jnp.zeros((n_nodes,), dtype=jnp.int32)
        zero_w = jnp.zeros((n_nodes, 0), dtype=nodes_coord_np.dtype)
        return zero_idx, zero_deg, zero_w

    n_local = int(elements_np.shape[1])
    idx = np.arange(n_local, dtype=np.int64)
    src_idx = np.repeat(idx, n_local)
    dst_idx = np.tile(idx, n_local)
    mask = src_idx != dst_idx
    src_idx = src_idx[mask]
    dst_idx = dst_idx[mask]
    src_flat = elements_np[:, src_idx].reshape(-1)
    dst_flat = elements_np[:, dst_idx].reshape(-1)

    if src_flat.size == 0:
        zero_idx = jnp.full((n_nodes, 0), -1, dtype=jnp.int32)
        zero_deg = jnp.zeros((n_nodes,), dtype=jnp.int32)
        zero_w = jnp.zeros((n_nodes, 0), dtype=nodes_coord_np.dtype)
        return zero_idx, zero_deg, zero_w

    # Deduplicate edges without building an O(n_nodes^2) adjacency matrix.
    edges = np.stack([src_flat, dst_flat], axis=1)
    edges = np.unique(edges, axis=0)
    if edges.size == 0:
        zero_idx = jnp.full((n_nodes, 0), -1, dtype=jnp.int32)
        zero_deg = jnp.zeros((n_nodes,), dtype=jnp.int32)
        zero_w = jnp.zeros((n_nodes, 0), dtype=nodes_coord_np.dtype)
        return zero_idx, zero_deg, zero_w

    src = edges[:, 0].astype(np.int64, copy=False)
    dst = edges[:, 1].astype(np.int64, copy=False)
    order = np.lexsort((dst, src))
    src = src[order]
    dst = dst[order]

    degrees = np.bincount(src, minlength=n_nodes).astype(np.int32, copy=False)
    max_deg = int(degrees.max(initial=0))
    if max_deg == 0:
        zero_idx = jnp.full((n_nodes, 0), -1, dtype=jnp.int32)
        zero_deg = jnp.zeros((n_nodes,), dtype=jnp.int32)
        zero_w = jnp.zeros((n_nodes, 0), dtype=nodes_coord_np.dtype)
        return zero_idx, zero_deg, zero_w

    xi = nodes_coord_np[src, :2]
    xj = nodes_coord_np[dst, :2]
    dist = np.linalg.norm(xj - xi, axis=1)
    weight_vals = (1.0 / (dist + 1e-6)).astype(nodes_coord_np.dtype, copy=False)

    # Vectorized "slot within each src group" for scatter-fill:
    # slot[k] = k - first_index_of_src_group(k)
    m = int(src.shape[0])
    change = np.empty((m,), dtype=bool)
    change[0] = True
    change[1:] = src[1:] != src[:-1]
    group_start = np.where(change, np.arange(m), 0)
    group_start = np.maximum.accumulate(group_start)
    slot = np.arange(m) - group_start

    node_neighbor_index = -np.ones((n_nodes, max_deg), dtype=np.int32)
    node_neighbor_weight = np.zeros((n_nodes, max_deg), dtype=nodes_coord_np.dtype)
    node_neighbor_index[src.astype(np.intp), slot.astype(np.intp)] = dst.astype(np.int32, copy=False)
    node_neighbor_weight[src.astype(np.intp), slot.astype(np.intp)] = weight_vals

    return (
        jnp.asarray(node_neighbor_index),
        jnp.asarray(degrees),
        jnp.asarray(node_neighbor_weight),
    )


@jax.jit(static_argnums=(3, 4))
def _build_node_neighbor_dense_from_edges_jax(
    src_flat,
    dst_flat,
    nodes_coord,
    n_nodes,
    max_deg,
):
    nodes_coord = jnp.asarray(nodes_coord)
    src_flat = jnp.asarray(src_flat, dtype=jnp.int32)
    dst_flat = jnp.asarray(dst_flat, dtype=jnp.int32)
    if src_flat.size == 0 or max_deg == 0:
        node_neighbor_index = jnp.full((n_nodes, 0), -1, dtype=jnp.int32)
        node_neighbor_degree = jnp.zeros((n_nodes,), dtype=jnp.int32)
        node_neighbor_weight = jnp.zeros((n_nodes, 0), dtype=nodes_coord.dtype)
        return node_neighbor_index, node_neighbor_degree, node_neighbor_weight

    adj = jnp.zeros((n_nodes, n_nodes), dtype=jnp.bool_)
    adj = adj.at[src_flat, dst_flat].set(True)

    xi = nodes_coord[src_flat, :2]
    xj = nodes_coord[dst_flat, :2]
    dist = jnp.linalg.norm(xj - xi, axis=1)
    weight_vals = 1.0 / (dist + 1e-6)
    weight_matrix = jnp.zeros((n_nodes, n_nodes), dtype=nodes_coord.dtype)
    weight_matrix = weight_matrix.at[src_flat, dst_flat].set(weight_vals)

    node_neighbor_degree = jnp.sum(adj.astype(jnp.int32), axis=1)

    def process_row(row_mask, row_weight, degree):
        values = jnp.where(row_mask, 1.0, 0.0)
        _, idx = lax.top_k(values, max_deg)
        idx = idx.astype(jnp.int32)
        valid_slots = jnp.arange(max_deg, dtype=jnp.int32) < degree
        gather_idx = jnp.clip(idx, 0)
        weights = row_weight[gather_idx]
        weights = jnp.where(valid_slots, weights, 0.0)
        idx = jnp.where(valid_slots, idx, -1)
        return idx, weights

    node_neighbor_index, node_neighbor_weight = jax.vmap(process_row)(
        adj, weight_matrix, node_neighbor_degree
    )

    return node_neighbor_index, node_neighbor_degree, node_neighbor_weight


def build_node_pixel_dense(pixel_nodes, pixel_shapeN, n_nodes):
    """
    Build dense node->pixel gather tables.

    Parameters
    ----------
    pixel_nodes:
        Node indices per pixel with shape ``(Np, n_local)``.
    pixel_shapeN:
        Shape function values per pixel with shape ``(Np, n_local)``.
    n_nodes:
        Total number of nodes.

    Returns
    -------
    (node_pixel_index, node_N_weight, node_degree):
        Dense gather indices and weights suitable for JAX kernels.
    """
    pixel_nodes_np = np.asarray(pixel_nodes, dtype=int)
    n_nodes = int(n_nodes)
    if pixel_nodes_np.size == 0:
        node_pixel_index = jnp.full((n_nodes, 0), -1, dtype=jnp.int32)
        node_N_weight = jnp.zeros((n_nodes, 0), dtype=jnp.asarray(pixel_shapeN).dtype)
        node_degree = jnp.zeros((n_nodes,), dtype=jnp.int32)
        return node_pixel_index, node_N_weight, node_degree

    flat_nodes = pixel_nodes_np.reshape(-1)
    degrees = np.bincount(flat_nodes, minlength=n_nodes)
    max_deg = int(degrees.max(initial=0))
    if max_deg == 0:
        node_pixel_index = jnp.full((n_nodes, 0), -1, dtype=jnp.int32)
        node_N_weight = jnp.zeros((n_nodes, 0), dtype=jnp.asarray(pixel_shapeN).dtype)
        node_degree = jnp.zeros((n_nodes,), dtype=jnp.int32)
        return node_pixel_index, node_N_weight, node_degree
    return _build_node_pixel_dense_jax(
        pixel_nodes,
        pixel_shapeN,
        n_nodes,
        max_deg,
    )


@jax.jit(static_argnums=(2, 3))
def _build_node_pixel_dense_jax(pixel_nodes, pixel_shapeN, n_nodes, max_deg):
    pixel_nodes = jnp.asarray(pixel_nodes, dtype=jnp.int32)
    pixel_shapeN = jnp.asarray(pixel_shapeN)
    n_nodes = int(n_nodes)
    max_deg = int(max_deg)
    Np = pixel_nodes.shape[0]
    n_local = pixel_nodes.shape[1]
    flat_nodes = pixel_nodes.reshape(-1)
    flat_weights = pixel_shapeN.reshape(-1)
    pixel_ids = jnp.repeat(jnp.arange(Np, dtype=jnp.int32), n_local)

    def body(counts, node):
        idx = counts[node]
        counts = counts.at[node].add(1)
        return counts, idx

    counts0 = jnp.zeros((n_nodes,), dtype=jnp.int32)
    node_degree, slot_idx = lax.scan(body, counts0, flat_nodes)

    node_pixel_index = jnp.full((n_nodes, max_deg), -1, dtype=jnp.int32)
    node_N_weight = jnp.zeros((n_nodes, max_deg), dtype=pixel_shapeN.dtype)
    node_pixel_index = node_pixel_index.at[flat_nodes, slot_idx].set(pixel_ids)
    node_N_weight = node_N_weight.at[flat_nodes, slot_idx].set(flat_weights)

    return node_pixel_index, node_N_weight, node_degree


def points_in_convex_quad(points, quad_nodes, chunk_size=4096, tol=1e-6):
    """
    Test whether points are inside a convex quad (in 2D).

    Parameters
    ----------
    points:
        Array of shape ``(N, 2)`` with ``(x, y)`` coordinates.
    quad_nodes:
        Array of shape ``(4, 2)`` with quad node coordinates.
    chunk_size:
        Chunk size used for the vectorized JAX kernel.
    tol:
        Tolerance applied to the half-plane tests (helps with boundary points).

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(N,)``.
    """
    pts = np.asarray(points, dtype=np.float32)
    quad = np.asarray(quad_nodes, dtype=np.float32)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)

    n = pts.shape[0]
    inside = np.zeros(n, dtype=bool)
    for start in range(0, n, chunk_size):
        chunk = pts[start : start + chunk_size]
        cur = chunk.shape[0]
        if cur < chunk_size:
            pad_chunk = np.pad(chunk, ((0, chunk_size - cur), (0, 0)), mode="edge")
            mask = np.zeros(chunk_size, dtype=bool)
            mask[:cur] = True
        else:
            pad_chunk = chunk
            mask = np.ones(chunk_size, dtype=bool)
        res = _points_in_convex_quad_chunk(
            pad_chunk,
            quad,
            mask,
            tol,
            chunk_size=chunk_size,
        )
        inside[start : start + chunk_size] = np.asarray(res)[:cur]
    return inside


@jax.jit
def _points_in_convex_quad_chunk(points_chunk, quad_nodes, active_mask, tol, chunk_size=4096):
    pts = jnp.asarray(points_chunk)
    quad = jnp.asarray(quad_nodes)
    active_mask = jnp.asarray(active_mask, dtype=jnp.bool_)

    edges = jnp.roll(quad, -1, axis=0) - quad
    quad_next = jnp.roll(quad, -1, axis=0)
    area2 = jnp.sum(quad[:, 0] * quad_next[:, 1] - quad[:, 1] * quad_next[:, 0])
    orientation = jnp.where(area2 >= 0.0, 1.0, -1.0)

    def inside(point):
        vec = point - quad
        cross = edges[:, 0] * vec[:, 1] - edges[:, 1] * vec[:, 0]
        return jnp.all(orientation * cross >= -tol)

    inside_vals = jax.vmap(inside)(pts)
    return jnp.where(active_mask, inside_vals, False)
