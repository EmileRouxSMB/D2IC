import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax import jit
import numpy as np
from functools import partial


from  dm_pix import flat_nd_linear_interpolate


@partial(jax.jit, static_argnames=("chunk_size",))
def _points_in_convex_quad_chunk(points_chunk, quad_nodes, active_mask, tol, chunk_size=4096):
    """Return inside mask for a padded chunk of points against a convex quad."""
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


def points_in_convex_quad(points, quad_nodes, chunk_size=4096, tol=1e-6):
    """Vectorized convex-quad point-in-polygon test implemented with JAX."""
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


@partial(jax.jit, static_argnames=("n_nodes", "max_deg"))
def _build_node_neighbor_dense_from_edges_jax(src_flat, dst_flat, nodes_coord, n_nodes, max_deg):
    """JAX-accelerated construction of dense neighbor data from explicit edge lists."""
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


def build_node_neighbor_dense(elements, nodes_coord, n_nodes):
    """Dense neighbor + weight arrays derived from quadrilateral connectivity.

    Returns ``(node_neighbor_index, node_neighbor_degree, node_neighbor_weight)`` with weights ``1 / ||xi - xj||``.
    """
    elements_np = np.asarray(elements, dtype=int)
    nodes_coord_np = np.asarray(nodes_coord)
    n_nodes = int(n_nodes)
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

    return _build_node_neighbor_dense_from_edges_jax(src_flat, dst_flat, nodes_coord_np, n_nodes, max_deg)


def build_k_ring_neighbors(node_neighbor_index, node_neighbor_degree, k=2):
    """Expand first-ring adjacency into a k-ring neighborhood (duplicates removed)."""
    node_neighbor_index = np.asarray(node_neighbor_index)
    node_neighbor_degree = np.asarray(node_neighbor_degree)
    Nnodes, _ = node_neighbor_index.shape

    # Unique sets to avoid duplicates.
    neigh_sets = [set() for _ in range(Nnodes)]
    for i in range(Nnodes):
        deg_i = int(node_neighbor_degree[i])
        for d in range(deg_i):
            j = int(node_neighbor_index[i, d])
            neigh_sets[i].add(j)

    # Propagate over k-1 additional rings.
    for _ in range(max(0, k - 1)):
        new_sets = [set(s) for s in neigh_sets]
        for i in range(Nnodes):
            for j in neigh_sets[i]:
                deg_j = int(node_neighbor_degree[j]) if j < Nnodes else 0
                for d in range(deg_j):
                    jj = int(node_neighbor_index[j, d])
                    new_sets[i].add(jj)
        neigh_sets = new_sets

    degrees = [len(s) for s in neigh_sets]
    max_deg_k = max(degrees) if degrees else 0

    neigh_index_k = np.full((Nnodes, max_deg_k), -1, dtype=int)
    neigh_degree_k = np.asarray(degrees, dtype=int)

    for i in range(Nnodes):
        if degrees[i] == 0:
            continue
        neigh_i = np.asarray(sorted(neigh_sets[i]), dtype=int)
        neigh_index_k[i, :degrees[i]] = neigh_i

    return neigh_index_k, neigh_degree_k


def build_pixel_to_element_mapping_numpy(pixel_coords, nodes_coord, elements, chunk_size=4096):
    """Map each pixel to its containing quadrilateral and return the four local nodes."""
    Np = pixel_coords.shape[0]
    pixel_elts = -np.ones(Np, dtype=int)
    pixel_nodes = np.zeros((Np, 4), dtype=int)

    elements = np.asarray(elements, dtype=int)
    nodes_coord = np.asarray(nodes_coord, dtype=float)
    if elements.size == 0 or Np == 0:
        return pixel_elts, pixel_nodes

    # Precompute bounding boxes for quick rejection.
    elt_bboxes = []
    for e in elements:
        Xe = nodes_coord[e]  # (4,2)
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
        idx = idx_candidates[inside]
        pixel_elts[idx] = e
        pixel_nodes[idx] = elements[e]

        if np.all(pixel_elts >= 0):
            break

    return pixel_elts, pixel_nodes


# ------------------------
# Q4 shape functions
# ------------------------

def shape_functions_jax(xi, eta):
    """Bilinear Q4 shape functions on [0, 1]^2."""
    N1 = (1.0 - xi) * (1.0 - eta)
    N2 = xi * (1.0 - eta)
    N3 = xi * eta
    N4 = (1.0 - xi) * eta
    return jnp.array([N1, N2, N3, N4])


def dN_dxi_jax(xi, eta):
    return jnp.array([
        -(1.0 - eta),
        +(1.0 - eta),
        +eta,
        -eta
    ])


def dN_deta_jax(xi, eta):
    return jnp.array([
        -(1.0 - xi),
        -xi,
        +xi,
        +(1.0 - xi)
    ])


def map_quad_jax(xi_eta, Xe):
    """Evaluate the bilinear Q4 map at ``(xi, eta)`` given node coordinates ``Xe``."""
    xi, eta = xi_eta
    N = shape_functions_jax(xi, eta)  # (4,)
    return N @ Xe  # (2,)


def jacobian_quad_jax(xi_eta, Xe):
    """2×2 Jacobian of the Q4 map (∂x/∂xi, ∂x/∂eta)."""
    xi, eta = xi_eta
    dNxi = dN_dxi_jax(xi, eta)[:, None]   # (4,1)
    dNeta = dN_deta_jax(xi, eta)[:, None] # (4,1)

    dxdxi = (dNxi * Xe).sum(axis=0)   # (2,)
    dxdeta = (dNeta * Xe).sum(axis=0) # (2,)

    return jnp.column_stack([dxdxi, dxdeta])  # (2,2)


# ------------------------
# Newton 2D jittable
# ------------------------

def _newton_one_pixel(xp, Xe, max_iter=20):
    """Invert the Q4 mapping for one pixel via a fixed-number Newton loop (`max_iter`)."""
    def body_fun(i, xi_eta):
        x_map = map_quad_jax(xi_eta, Xe)
        r = xp - x_map
        J = jacobian_quad_jax(xi_eta, Xe)
        delta = jnp.linalg.solve(J, r)
        return xi_eta + delta

    xi_eta0 = jnp.array([0.5, 0.5])
    xi_eta = lax.fori_loop(0, max_iter, body_fun, xi_eta0)
    xi_eta = jnp.clip(xi_eta, 0.0, 1.0)  # safety
    return xi_eta  # (2,)


# Vectorized Newton on all pixels
vmap_newton = jax.jit(
    vmap(_newton_one_pixel, in_axes=(0, 0)),
    static_argnames=()  # max_iter is hard-coded here
)


@jax.jit
def compute_pixel_shape_functions_jax(pixel_coords,
                                      pixel_nodes,
                                      nodes_coord):
    """Invert Q4 elements for every pixel and return shape functions plus ``(xi, eta)``."""
    # Coordinates of the 4 nodes for each pixel (Np,4,2)
    Xe_all = nodes_coord[pixel_nodes]  # advanced JAX indexing OK

    # Vectorized inversion: (Np,2)
    xi_eta_all = vmap_newton(pixel_coords, Xe_all)

    xi = xi_eta_all[:, 0]
    eta = xi_eta_all[:, 1]

    # Shape functions (vectorized over pixels)
    def _shape_from_xieta(xi, eta):
        return shape_functions_jax(xi, eta)

    pixel_N = vmap(_shape_from_xieta)(xi, eta)  # (Np,4)

    return pixel_N, xi_eta_all


@jit
def residuals_pixelwise_core(displacement,
                             im1_T, im2_T,
                             pixel_coords,
                             pixel_nodes,
                             pixel_shapeN):
    """Compute pixelwise residuals ``I2(x+u) - I1(x)`` for the current nodal displacement."""
    disp = jnp.asarray(displacement)
    im1_T = jnp.asarray(im1_T)
    im2_T = jnp.asarray(im2_T)

    # (Np,4,1)
    shapeN = pixel_shapeN[..., None]

    # nodal displacements of the 4 nodes per pixel
    disp_local = disp[pixel_nodes]          # (Np,4,2)

    # u(x_p) = Σ_i N_i * u_i
    u_pix = (shapeN * disp_local).sum(axis=1)  # (Np,2)

    x_ref = pixel_coords                    # (Np,2)
    x_def = x_ref + u_pix                   # (Np,2)

    # Transposes hoisted out of the hot loop for better GPU fusion.
    I1 = flat_nd_linear_interpolate(im1_T, x_ref.T)  # (Np,)
    I2 = flat_nd_linear_interpolate(im2_T, x_def.T)  # (Np,)

    return I2 - I1                           # (Np,)


@jit
def J_pixelwise_core(displacement,
                     im1_T, im2_T,
                     pixel_coords,
                     pixel_nodes,
                     pixel_shapeN):
    r = residuals_pixelwise_core(displacement,
                                 im1_T, im2_T,
                                 pixel_coords,
                                 pixel_nodes,
                                 pixel_shapeN)
    return 0.5 * jnp.mean(r**2)


@partial(jax.jit, static_argnames=("n_nodes", "max_deg"))
def _build_node_pixel_dense_jax(pixel_nodes, pixel_shapeN, n_nodes, max_deg):
    """JAX helper that fills dense node→pixel CSR-like arrays."""
    pixel_nodes = jnp.asarray(pixel_nodes, dtype=jnp.int32)
    pixel_shapeN = jnp.asarray(pixel_shapeN)
    if pixel_nodes.size == 0 or max_deg == 0:
        node_pixel_index = jnp.full((n_nodes, 0), -1, dtype=jnp.int32)
        node_N_weight = jnp.zeros((n_nodes, 0), dtype=pixel_shapeN.dtype)
        node_degree = jnp.zeros((n_nodes,), dtype=jnp.int32)
        return node_pixel_index, node_N_weight, node_degree

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


def build_node_pixel_dense(pixel_nodes, pixel_shapeN, n_nodes):
    """Dense node→pixel lookup storing indices, weights, and degrees."""
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

    return _build_node_pixel_dense_jax(pixel_nodes, pixel_shapeN, n_nodes, max_deg)


@jax.jit
def compute_pixel_state(displacement,
                        im1_T, im2_T,
                        pixel_coords,
                        pixel_nodes,
                        pixel_shapeN,
                        gradx2_T, grady2_T):
    """Return residuals, deformed coords, and image-2 gradients for the given nodal displacement."""
    shapeN = pixel_shapeN[..., None]            # (Np,4,1)
    disp_local = displacement[pixel_nodes]      # (Np,4,2)
    u_pix = (shapeN * disp_local).sum(axis=1)   # (Np,2)

    x_ref = pixel_coords
    x_def = x_ref + u_pix

    I1 = flat_nd_linear_interpolate(im1_T, x_ref.T)
    I2 = flat_nd_linear_interpolate(im2_T, x_def.T)

    # gradients of image 2 interpolated at x_def
    gx_def = flat_nd_linear_interpolate(gradx2_T, x_def.T)
    gy_def = flat_nd_linear_interpolate(grady2_T, x_def.T)

    r = I2 - I1

    return r, x_def, gx_def, gy_def

def compute_image_gradient(im):
    im = np.asarray(im, dtype=float)
    gx = np.zeros_like(im)
    gy = np.zeros_like(im)
    gx[:, 1:-1] = 0.5 * (im[:, 2:] - im[:, :-2])
    gy[1:-1, :] = 0.5 * (im[2:, :] - im[:-2, :])
    return gx, gy

@jax.jit
@jax.jit
def jacobi_nodal_step_spring(displacement,
                             r,
                             gx_def, gy_def,
                             node_pixel_index,
                             node_N_weight,
                             node_degree,
                             node_neighbor_index,
                             node_neighbor_degree,
                             node_neighbor_weight,
                             lam=0.1,
                             max_step=0.2,
                             alpha_reg=0.0,
                             omega=0.5):
    """Relaxed Jacobi update that blends image residuals with spring regularization.

    Computes node increments from ``r``/``gx_def``/``gy_def`` without in-loop updates,
    clips them to ``max_step``, adds the spring contribution scaled by ``alpha_reg``,
    then applies ``u_new = u_old + omega * δu`` to suppress Gauss–Seidel oscillations.
    """
    disp0 = displacement
    Nnodes, max_deg_pix   = node_pixel_index.shape
    _,      max_deg_neigh = node_neighbor_index.shape

    def body_fun(i, delta_acc):
        deg_pix   = node_degree[i]
        deg_neigh = node_neighbor_degree[i]

        # No pixel information: rely solely on springs if available.
        def no_pixel_case(delta_acc_inner):

            def only_return_delta(_):
                return delta_acc_inner

            def apply_reg_only(_):
                u_i = disp0[i]

                idx_range_n = jnp.arange(max_deg_neigh)
                mask_n = idx_range_n < deg_neigh

                neigh_ids_all = node_neighbor_index[i]
                w_all         = node_neighbor_weight[i]

                neigh_ids = jnp.where(mask_n, neigh_ids_all, 0)
                w         = jnp.where(mask_n, w_all,         0.0)

                u_neigh = disp0[neigh_ids]
                u_neigh = jnp.where(mask_n[:, None], u_neigh, 0.0)

                diff = u_i[None, :] - u_neigh
                g_reg = alpha_reg * jnp.sum(w[:, None] * diff, axis=0)
                H_reg = alpha_reg * jnp.sum(w) * jnp.eye(2)

                g_loc = g_reg
                H = H_reg + 1e-8 * jnp.eye(2)
                delta_i = -jnp.linalg.solve(H, g_loc)

                # Clip the update norm.
                norm_delta = jnp.linalg.norm(delta_i)
                factor = jnp.minimum(1.0, max_step / (norm_delta + 1e-12))
                delta_i = delta_i * factor

                return delta_acc_inner.at[i].set(delta_i)

            cond_reg = jnp.logical_and(alpha_reg != 0.0, deg_neigh > 0)
            return lax.cond(cond_reg, apply_reg_only, only_return_delta, operand=None)

        # Node is supported by image pixels: mix data and spring terms.
        def update_case(delta_acc_inner):
            idx_all = node_pixel_index[i]
            Ni_all  = node_N_weight[i]
            idx_range = jnp.arange(max_deg_pix)
            mask_pix = idx_range < deg_pix

            idx = jnp.where(mask_pix, idx_all, 0)
            Ni  = jnp.where(mask_pix, Ni_all, 0.0)

            ri  = r[idx]
            gxi = gx_def[idx]
            gyi = gy_def[idx]

            Jx = gxi * Ni
            Jy = gyi * Ni

            g0_img = jnp.sum(ri * Jx)
            g1_img = jnp.sum(ri * Jy)
            g_img = jnp.array([g0_img, g1_img])

            H00 = jnp.sum(Jx * Jx)
            H01 = jnp.sum(Jx * Jy)
            H11 = jnp.sum(Jy * Jy)

            traceH = H00 + H11 + 1e-12
            lam_loc = lam * traceH
            H_img = jnp.array([[H00 + lam_loc, H01],
                               [H01,          H11 + lam_loc]])

            # Spring contribution.
            u_i = disp0[i]
            idx_range_n = jnp.arange(max_deg_neigh)
            mask_n = idx_range_n < deg_neigh

            neigh_ids_all = node_neighbor_index[i]
            w_all         = node_neighbor_weight[i]

            neigh_ids = jnp.where(mask_n, neigh_ids_all, 0)
            w         = jnp.where(mask_n, w_all,         0.0)

            u_neigh = disp0[neigh_ids]
            u_neigh = jnp.where(mask_n[:, None], u_neigh, 0.0)

            diff = u_i[None, :] - u_neigh
            g_reg = alpha_reg * jnp.sum(w[:, None] * diff, axis=0)
            H_reg = alpha_reg * jnp.sum(w) * jnp.eye(2)

            g_loc = g_img + g_reg
            H = H_img + H_reg + 1e-8 * jnp.eye(2)

            delta_i = -jnp.linalg.solve(H, g_loc)

            norm_delta = jnp.linalg.norm(delta_i)
            factor = jnp.minimum(1.0, max_step / (norm_delta + 1e-12))
            delta_i = delta_i * factor

            return delta_acc_inner.at[i].set(delta_i)

        return lax.cond(deg_pix == 0, no_pixel_case, update_case, delta_acc)

    delta0 = jnp.zeros_like(displacement)
    delta_all = lax.fori_loop(0, Nnodes, body_fun, delta0)

    # Apply all increments at once with under-relaxation.
    displacement_new = displacement + omega * delta_all
    return displacement_new

@jax.jit
def reg_energy_spring_global(displacement,
                             node_neighbor_index,
                             node_neighbor_degree,
                             node_neighbor_weight):
    """Global spring energy ``0.5 Σ w_ij ||u_i - u_j||^2`` using the cached neighbor weights."""
    u = displacement
    Nnodes, max_deg = node_neighbor_index.shape

    deg = node_neighbor_degree[:, None]
    idx_range = jnp.arange(max_deg)[None, :]
    mask = idx_range < deg

    neigh_ids = jnp.where(mask, node_neighbor_index, 0)
    w = jnp.where(mask, node_neighbor_weight, 0.0)        # (Nnodes,max_deg)

    u_i = u[:, None, :]
    u_j = u[neigh_ids]
    diff = jnp.where(mask[..., None], u_i - u_j, 0.0)     # (Nnodes,max_deg,2)
    sq = jnp.sum(diff**2, axis=2)                         # (Nnodes,max_deg)

    # Same expression but weighted.
    J_reg = 0.25 * jnp.sum(w * sq)
    return J_reg

@jax.jit
def compute_green_lagrange_strain_nodes_lsq(
    displacement,
    nodes_coord,
    node_neighbor_index,
    node_neighbor_degree,
    gauge_length=0.0,
    eps=1e-8,
):
    """Weighted LSQ estimate of ``F`` and Green–Lagrange ``E`` at each node.

    ``gauge_length`` applies an optional radial weight, and the result is returned as
    ``(F_all, E_all)`` with shape ``(Nnodes, 2, 2)``.
    """
    disp = jnp.asarray(displacement)
    X = jnp.asarray(nodes_coord)
    node_neighbor_index = jnp.asarray(node_neighbor_index)
    node_neighbor_degree = jnp.asarray(node_neighbor_degree)

    Nnodes, max_deg = node_neighbor_index.shape
    L = jnp.squeeze(jnp.asarray(gauge_length))

    def one_node(i):
        xi = X[i]        # (2,)
        ui = disp[i]     # (2,)

        idx_all = node_neighbor_index[i]   # (max_deg,)
        deg = node_neighbor_degree[i]      # scalaire int
        idx_range = jnp.arange(max_deg)
        mask = idx_range < deg             # (max_deg,) bool

        # Effective neighbors (indices) - 0 used as masked padding.
        neigh_ids = jnp.where(mask, idx_all, 0)
        xj = X[neigh_ids]                  # (max_deg, 2)
        uj = disp[neigh_ids]               # (max_deg, 2)

        dX = xj - xi[None, :]              # (max_deg, 2)
        du = uj - ui[None, :]              # (max_deg, 2)

        # Distances and weights
        r = jnp.linalg.norm(dX, axis=1)    # (max_deg,)
        base_w = mask.astype(float)  # zero weight on inactive padding
        w_exp = base_w * jnp.exp(-(r / (L + 1e-12)) ** 2)
        w = jnp.where(L > 0.0, w_exp, base_w)  # (max_deg,)

        w_col = w[:, None]                 # (max_deg, 1)
        Xw = dX * w_col                    # (max_deg, 2)

        # Normal matrix A and right-hand sides b0/b1
        A = Xw.T @ dX + eps * jnp.eye(2)   # (2, 2)
        b0 = Xw.T @ du[:, 0]              # (2,)
        b1 = Xw.T @ du[:, 1]              # (2,)

        grad_ux = jnp.linalg.solve(A, b0)  # (2,)
        grad_uy = jnp.linalg.solve(A, b1)  # (2,)

        Grad_u = jnp.stack([grad_ux, grad_uy], axis=0)  # (2, 2)

        I = jnp.eye(2)
        F = I + Grad_u
        C = F.T @ F
        E = 0.5 * (C - I)
        return F, E

    idxs = jnp.arange(Nnodes)
    F_all, E_all = jax.vmap(one_node)(idxs)
    return F_all, E_all
