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
    bbox_pad: float = 2.0,
    chunk_size: int = 4096,
) -> PixelAssets:
    nodes = np.asarray(mesh.nodes_xy) / float(binning)
    elements = np.asarray(mesh.elements, dtype=int)
    H, W = ref_image.shape[:2]
    pixel_coords = _sample_pixels_in_bbox(nodes, H, W, bbox_pad)
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
    max_deg = int(np.asarray(node_neighbor_degree).max(initial=0))
    border_mask = np.asarray(node_neighbor_degree) < max_deg
    node_reg_weight = np.where(border_mask, 10.0, 1.0).astype(nodes.dtype, copy=False)
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
    Np = pixel_coords.shape[0]
    pixel_elts = -np.ones(Np, dtype=int)
    pixel_nodes = np.zeros((Np, 4), dtype=int)
    elements = np.asarray(elements, dtype=int)
    nodes_coord = np.asarray(nodes_coord, dtype=float)
    if elements.size == 0 or Np == 0:
        return pixel_elts, pixel_nodes

    elt_bboxes = []
    for e in elements:
        Xe = nodes_coord[e]
        xmin, ymin = Xe.min(axis=0)
        xmax, ymax = Xe.max(axis=0)
        elt_bboxes.append([xmin, xmax, ymin, ymax])
    elt_bboxes = np.asarray(elt_bboxes)

    for e, bbox in enumerate(elt_bboxes):
        remaining = pixel_elts < 0
        if not np.any(remaining):
            break
        mask_bbox = (
            remaining
            & (pixel_coords[:, 0] >= bbox[0])
            & (pixel_coords[:, 0] <= bbox[1])
            & (pixel_coords[:, 1] >= bbox[2])
            & (pixel_coords[:, 1] <= bbox[3])
        )
        if not np.any(mask_bbox):
            continue
        idx_candidates = np.nonzero(mask_bbox)[0]
        pts = pixel_coords[idx_candidates]
        quad_nodes = nodes_coord[elements[e]]
        inside = points_in_convex_quad(
            pts,
            quad_nodes,
            chunk_size=chunk_size,
        )
        if not np.any(inside):
            continue
        idx_inside = idx_candidates[inside]
        pixel_elts[idx_inside] = e
        pixel_nodes[idx_inside] = elements[e]
    return pixel_elts, pixel_nodes


@jax.jit
def compute_pixel_shape_functions_jax(
    pixel_coords,
    pixel_nodes,
    nodes_coord,
):
    Xe_all = nodes_coord[pixel_nodes]
    xi_eta_all = vmap(_newton_quadrature)(pixel_coords, Xe_all)
    xi = xi_eta_all[:, 0]
    eta = xi_eta_all[:, 1]

    def shape_from_xieta(xi, eta):
        return shape_functions_jax(xi, eta)

    pixel_N = vmap(shape_from_xieta)(xi, eta)
    return pixel_N, xi_eta_all


def _newton_quadrature(point, Xe):
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
    return jnp.array(
        [
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta),
        ]
    )


def shape_function_gradients_jax(xi, eta):
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
    elements_np = np.asarray(elements, dtype=int)
    nodes_coord_np = np.asarray(nodes_coord)
    if elements_np.size == 0:
        zero_idx = jnp.full((n_nodes, 0), -1, dtype=jnp.int32)
        zero_deg = jnp.zeros((n_nodes,), dtype=jnp.int32)
        zero_w = jnp.zeros((n_nodes, 0), dtype=nodes_coord_np.dtype)
        return zero_idx, zero_deg, zero_w

    n_local = elements_np.shape[1]
    idx = np.arange(n_local, dtype=int)
    src_idx = np.repeat(idx, n_local)
    dst_idx = np.tile(idx, n_local)
    mask = src_idx != dst_idx
    src_idx = src_idx[mask]
    dst_idx = dst_idx[mask]
    src_flat = elements_np[:, src_idx].reshape(-1)
    dst_flat = elements_np[:, dst_idx].reshape(-1)

    adj = np.zeros((n_nodes, n_nodes), dtype=bool)
    adj[src_flat, dst_flat] = True
    degrees = adj.sum(axis=1)
    max_deg = int(degrees.max(initial=0))

    if src_flat.size == 0 or max_deg == 0:
        zero_idx = jnp.full((n_nodes, 0), -1, dtype=jnp.int32)
        zero_deg = jnp.zeros((n_nodes,), dtype=jnp.int32)
        zero_w = jnp.zeros((n_nodes, 0), dtype=nodes_coord_np.dtype)
        return zero_idx, zero_deg, zero_w

    return _build_node_neighbor_dense_from_edges_jax(
        src_flat,
        dst_flat,
        nodes_coord_np,
        n_nodes,
        max_deg,
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
