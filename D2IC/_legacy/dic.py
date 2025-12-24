"""Core DIC routines used throughout D2IC."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from matplotlib.collections import PolyCollection
from skimage.transform import AffineTransform, SimilarityTransform

import meshio 


from D2IC.motion_init import big_motion_sparse_matches
from D2IC.dic_JaxCore import (
    build_pixel_to_element_mapping_numpy,
    compute_pixel_shape_functions_jax,
    J_pixelwise_core,
    build_node_pixel_dense,
    compute_image_gradient,
    compute_pixel_state,
    build_node_neighbor_dense,
    build_k_ring_neighbors,
    reg_energy_spring_global,
    jacobi_nodal_step_spring,
    compute_green_lagrange_strain_nodes_lsq,
)
from D2IC.feature_matching import (
    _points_in_mesh_area,
    refine_matches_ncc,
    local_ransac_outlier_filter,
)


def build_weighted_rbf_interpolator(x, u, w, smoothing=1e-3):
    """Thin-plate RBF interpolator where each correspondence carries a non-negative weight."""
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    n = x.shape[0]

    if n == 0:
        return lambda q: np.zeros((np.atleast_2d(q).shape[0], 2), dtype=np.float64)

    weights = np.clip(w, 0.0, None)
    weights = weights - weights.min()
    max_w = weights.max()
    if max_w > 0:
        weights = weights / max_w
    else:
        weights = np.ones_like(weights)

    diff = x[:, None, :] - x[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    eps = 1e-12
    K = (r ** 2) * np.log(r + eps)

    # Weighted normal equations: (K^T W K + λI) α = K^T W u.
    A = K.T @ (weights[:, None] * K) + float(smoothing) * np.eye(n)
    B = K.T @ (weights[:, None] * u)
    alpha = np.linalg.solve(A, B)

    def interp(q):
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        rq = np.linalg.norm(q[:, None, :] - x[None, :, :], axis=2)
        Kq = (rq ** 2) * np.log(rq + eps)
        return Kq @ alpha

    return interp


def J_total(
    disp,
    im1_T,
    im2_T,
    pixel_coords_ref,
    pixel_nodes,
    pixel_shapeN,
    node_neighbor_index,
    node_neighbor_degree,
    node_neighbor_weight,
    alpha_reg,
):
    """Image mismatch energy optionally augmented with the nodal spring penalty."""
    J_img = J_pixelwise_core(
        disp,
        im1_T,
        im2_T,
        pixel_coords_ref,
        pixel_nodes,
        pixel_shapeN,
    )

    def add_reg(J_val):
        J_reg = reg_energy_spring_global(
            disp,
            node_neighbor_index,
            node_neighbor_degree,
            node_neighbor_weight,
        )
        return J_val + alpha_reg * J_reg

    return jax.lax.cond(
        alpha_reg == 0.0,
        lambda J_val: J_val,
        add_reg,
        J_img,
    )

_J_TOTAL_VG = jax.value_and_grad(J_total)


class _CGState(NamedTuple):
    k: jnp.ndarray
    displacement: jnp.ndarray
    direction: jnp.ndarray
    g_prev: jnp.ndarray
    first: jnp.ndarray
    history: jnp.ndarray
    run: jnp.ndarray
    last_J: jnp.ndarray
    last_grad_norm: jnp.ndarray


def _cg_solve(
    disp0,
    im1_T,
    im2_T,
    pixel_coords_ref,
    pixel_nodes,
    pixel_shapeN,
    node_neighbor_index,
    node_neighbor_degree,
    node_neighbor_weight,
    alpha_reg,
    max_iter,
    tol,
    save_history,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Conjugate-gradient loop expressed as a jitted ``lax.while_loop``."""
    save_history = bool(save_history)
    disp0 = jnp.asarray(disp0)
    tol = jnp.asarray(tol)
    alpha_reg = jnp.asarray(alpha_reg)
    max_iter_int = int(max_iter)
    max_iter = jnp.asarray(max_iter_int)

    objective_args = (
        im1_T,
        im2_T,
        pixel_coords_ref,
        pixel_nodes,
        pixel_shapeN,
        node_neighbor_index,
        node_neighbor_degree,
        node_neighbor_weight,
    )

    history = (
        jnp.zeros((max_iter_int, 2))
        if save_history
        else jnp.zeros((0, 2))
    )
    zero_disp = jnp.zeros_like(disp0)
    init_state = _CGState(
        k=jnp.array(0),
        displacement=disp0,
        direction=zero_disp,
        g_prev=zero_disp,
        first=jnp.bool_(True),
        history=history,
        run=jnp.bool_(True),
        last_J=0.0,
        last_grad_norm=jnp.inf,
    )
    c_armijo = 1e-4

    def cond_fun(state):
        return jnp.logical_and(state.k < max_iter, state.run)

    def body_fun(state):
        J_val, grad = _J_TOTAL_VG(
            state.displacement,
            *objective_args,
            alpha_reg,
        )
        grad_norm = jnp.linalg.norm(grad)
        if save_history:
            history = state.history.at[state.k].set(
                jnp.asarray([J_val, grad_norm])
            )
        else:
            history = state.history
        stop = grad_norm < tol

        def stop_branch(_):
            return state._replace(
                k=state.k + 1,
                history=history,
                run=jnp.bool_(False),
                last_J=J_val,
                last_grad_norm=grad_norm,
                g_prev=grad,
            )

        def continue_branch(_):
            def dir_first(_):
                return -grad

            def dir_conjugate(_):
                num = jnp.sum(grad * (grad - state.g_prev))
                den = jnp.sum(state.g_prev * state.g_prev) + 1e-16
                beta = jnp.maximum(num / den, 0.0)
                return -grad + beta * state.direction

            direction = jax.lax.cond(
                state.first,
                dir_first,
                dir_conjugate,
                operand=None,
            )
            gd = jnp.sum(grad * direction)

            def restart_direction(_):
                return -grad, -jnp.sum(grad * grad)

            direction, gd = jax.lax.cond(
                gd >= 0,
                restart_direction,
                lambda _: (direction, gd),
                operand=None,
            )

            def ls_body(i, carry):
                alpha, J_candidate, done = carry

                def compute_step(_):
                    candidate_disp = state.displacement + alpha * direction
                    J_candidate = J_total(
                        candidate_disp,
                        *objective_args,
                        alpha_reg,
                    )
                    satisfies = J_candidate <= J_val + c_armijo * alpha * gd
                    alpha_next = jnp.where(
                        satisfies,
                        alpha,
                        alpha * 0.5,
                    )
                    return alpha_next, J_candidate, satisfies

                alpha_next, J_candidate_next, satisfies = jax.lax.cond(
                    done,
                    lambda _: (alpha, J_candidate, jnp.bool_(True)),
                    compute_step,
                    operand=None,
                )
                done_next = jnp.logical_or(done, satisfies)
                return alpha_next, J_candidate_next, done_next

            alpha_init = 1.0
            alpha_final, _, _ = jax.lax.fori_loop(
                0,
                10,
                ls_body,
                (alpha_init, J_val, jnp.bool_(False)),
            )
            displacement_new = state.displacement + alpha_final * direction
            return state._replace(
                k=state.k + 1,
                displacement=displacement_new,
                direction=direction,
                g_prev=grad,
                first=jnp.bool_(False),
                history=history,
                run=jnp.bool_(True),
                last_J=J_val,
                last_grad_norm=grad_norm,
            )

        new_state = jax.lax.cond(
            stop,
            stop_branch,
            continue_branch,
            operand=None,
        )
        return new_state

    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    history_out = final_state.history if save_history else jnp.zeros((0, 2))
    return (
        final_state.displacement,
        history_out,
        final_state.k,
        final_state.last_J,
        final_state.last_grad_norm,
    )


class Dic():
    """Pixelwise DIC solver operating on a quadrilateral mesh."""

    # API_CANDIDATE
    def __init__(self, mesh_path=""):
        """Load the mesh and prepare basic geometry caches from ``mesh_path``."""
        self.mesh = None
        self.mesh_path = mesh_path
        self._read_mesh()
        self.binning = 1.
        self._image_shape = None
        self._roi_mask = None
        self._roi_flat_indices = None
        # Keep a stable jitted solver to avoid retracing every frame.
        # Donating the displacement buffer limits repeated large allocations.
        self._solve_jit = jax.jit(
            _cg_solve,
            static_argnums=(10, 12),
            donate_argnums=(0,),
        )
    
    def __repr__(self):
        """Return a compact identifier including the mesh path."""
        return f"Dic(mesh_path={self.mesh_path})"

    def precompute_pixel_data(self, im):
        """Build pixel-to-element maps and Q4 shape functions for the reference ROI.

        Takes the (possibly binned) reference image, keeps pixels inside the mesh footprint,
        precomputes ``pixel_nodes``, ``pixel_shapeN``, and CSR node→pixel structures, and
        caches neighbor lists for the spring regularization.
        """
        H, W = im.shape[:2]
        self._image_shape = (H, W)

        nodes_coord_np = np.asarray(self.node_coordinates_binned[:, :2])
        print("   [precompute] Step 1/5: ROI pixel sampling in mesh bounding box")
        # 1) Build pixel coordinates restricted to the mesh bounding box (pads by ~2 px)
        x_min = float(nodes_coord_np[:, 0].min())
        x_max = float(nodes_coord_np[:, 0].max())
        y_min = float(nodes_coord_np[:, 1].min())
        y_max = float(nodes_coord_np[:, 1].max())
        bbox_pad = 2.0
        j0 = max(0, int(np.floor(x_min - bbox_pad)))
        j1 = min(W, int(np.ceil(x_max + bbox_pad)))
        i0 = max(0, int(np.floor(y_min - bbox_pad)))
        i1 = min(H, int(np.ceil(y_max + bbox_pad)))
        if j1 <= j0 or i1 <= i0:
            raise RuntimeError("Mesh bounding box is degenerate.")

        jj, ii = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1))
        pixel_coords = np.stack([jj.ravel() + 0.5, ii.ravel() + 0.5], axis=1)  # (Np_box,2)
        print(f"      - Bounding box size: {(i1 - i0)}x{(j1 - j0)} px ({pixel_coords.shape[0]} candidates)")

        # 2) Mesh data (NumPy + JAX views)
        elements_np = np.asarray(self.element_conectivity)
        nodes_coord_jax = jnp.asarray(nodes_coord_np)

        print("   [precompute] Step 2/5: Pixel→element association (NumPy)")
        pixel_elts_np, pixel_nodes_np = build_pixel_to_element_mapping_numpy(
            pixel_coords, nodes_coord_np, elements_np
        )

        valid_mask = pixel_elts_np >= 0
        if not np.any(valid_mask):
            raise RuntimeError("No ROI pixel found after element lookup.")
        pixel_coords = pixel_coords[valid_mask]
        pixel_elts_np = pixel_elts_np[valid_mask]
        pixel_nodes_np = pixel_nodes_np[valid_mask]
        print(f"      - Pixels retained within ROI: {pixel_coords.shape[0]}")

        roi_mask = np.zeros((H, W), dtype=bool)
        roi_cols = np.clip(np.floor(pixel_coords[:, 0] - 0.5).astype(int), 0, W - 1)
        roi_rows = np.clip(np.floor(pixel_coords[:, 1] - 0.5).astype(int), 0, H - 1)
        roi_mask[roi_rows, roi_cols] = True
        self._roi_mask = roi_mask
        self._roi_flat_indices = np.ravel_multi_index((roi_rows, roi_cols), (H, W))

        pixel_coords_jax = jnp.asarray(pixel_coords)
        pixel_nodes_jax = jnp.asarray(pixel_nodes_np)
        pixel_elts = jnp.asarray(pixel_elts_np)
        print("      - Pixel→element mapping done")

        print("   [precompute] Step 3/5: Shape-function inversion (JAX)")
        pixel_N_jax, xi_eta = compute_pixel_shape_functions_jax(
            pixel_coords_jax,
            pixel_nodes_jax,
            nodes_coord_jax
        )
        pixel_N_jax = jnp.asarray(pixel_N_jax)
        xi_eta = jnp.asarray(xi_eta)
        print("      - Shape functions computed")

        # Cache everything needed by the solver
        self.pixel_coords_ref = pixel_coords_jax      # (Np,2)
        self.pixel_elts = pixel_elts                  # (Np,)
        self.pixel_nodes = pixel_nodes_jax            # (Np,4)
        self.pixel_shapeN = pixel_N_jax               # (Np,4)
        self.pixel_xi_eta = xi_eta                    # optionnel
    
        # Build CSR-style node → pixel data structures
        n_nodes = self.node_coordinates_binned.shape[0]
        print("   [precompute] Step 4/5: Building node→pixel structures (JAX)")
        node_pixel_index, node_N_weight, node_degree = build_node_pixel_dense(
            pixel_nodes_np,
            self.pixel_shapeN,
            n_nodes,
        )
        self.node_pixel_index = jnp.asarray(node_pixel_index)   # (Nnodes, max_deg)
        self.node_N_weight    = jnp.asarray(node_N_weight)      # (Nnodes, max_deg)
        self.node_degree      = jnp.asarray(node_degree)        # (Nnodes,)
        print("      - Node→pixel CSR cached")

        # Build neighbor info for the spring regularization
        print("   [precompute] Step 5/5: Building node neighbor tables (JAX)")
        node_neighbor_index, node_neighbor_degree, node_neighbor_weight = build_node_neighbor_dense(
            elements_np,
            nodes_coord_np,
            n_nodes,
        )
        self.node_neighbor_index  = jnp.asarray(node_neighbor_index)
        self.node_neighbor_degree = jnp.asarray(node_neighbor_degree)
        self.node_neighbor_weight = jnp.asarray(node_neighbor_weight)
        print("      - Neighbor tables ready")


    # Cached mesh properties
    @property
    def node_coordinates(self):
        """Return nodal coordinates as a ``jax.numpy`` array."""
        return jnp.array(self.mesh.points) 
    
    @property
    def node_coordinates_binned(self):
        """Return nodal coordinates divided by the binning factor."""
        return jnp.array(self.mesh.points) / self.binning
    
    @property
    def element_conectivity(self):
        """Return quadrilateral connectivity as ``jax.numpy`` indices."""
        quad_blocks = []
        # meshio >=5 stores quads inside ``cells``; gather and concatenate them.
        for cell_block in getattr(self.mesh, "cells", []):
            cell_type = getattr(cell_block, "type", "")
            if cell_type.startswith("quad"):
                quad_blocks.append(cell_block.data)
        if quad_blocks:
            conn = np.concatenate(quad_blocks, axis=0)
            return jnp.array(conn)
        # Backward compatibility for older meshio versions.
        cells_dict = getattr(self.mesh, "cells_dict", {})
        if isinstance(cells_dict, dict) and "quad" in cells_dict:
            return jnp.array(cells_dict["quad"])
        raise ValueError("No recognizable quadrilateral elements were found in the mesh.")

    def _compute_element_centers_binned(self):
        """Return element centers in binned pixel coordinates (average of four nodes)."""
        nodes = np.asarray(self.node_coordinates_binned[:, :2])
        elements = np.asarray(self.element_conectivity, dtype=int)
        elem_nodes = nodes[elements]          # shape: (Nelements, 4, 2)
        centers = elem_nodes.mean(axis=1)     # shape: (Nelements, 2)
        return centers

    # API_CANDIDATE
    def get_mesh_as_polyCollection(self, displacement=0.0, **kwargs):
        """Return the quad mesh as a Matplotlib ``PolyCollection`` for plotting."""
        node_coordinates = self.node_coordinates[:, :2] + displacement
        vertices = node_coordinates[self.element_conectivity]
        return PolyCollection(vertices, **kwargs)
    
    # API_CANDIDATE
    def get_mesh_as_triangulation(self, displacement=0.0):
        """Convert the quadrilateral mesh into a Matplotlib triangulation."""
        # Split each quad into two triangles for Matplotlib's 2D triangulation.
        from matplotlib.tri import Triangulation
        x, y = (self.node_coordinates[:,:2]+displacement).T
        conn_tri = []
        for quad in self.element_conectivity:
            conn_tri.append([quad[0], quad[1], quad[2]])
            conn_tri.append([quad[0], quad[2], quad[3]])
        conn_tri = jnp.array(conn_tri)
        triangulation = Triangulation(x, y, conn_tri)

        return triangulation

    def _read_mesh(self):
        """Read the mesh file through ``meshio``."""
        self.mesh = meshio.read(self.mesh_path)

    def compute_green_lagrange_strain_nodes(
        self,
        displacement,
        k_ring=1,
        gauge_length=0.0,
    ):
        """Estimate nodal deformation gradients and Green–Lagrange strains via local LSQ fits.

        ``k_ring`` controls the neighborhood size; ``gauge_length`` uses the same units as ``node_coordinates_binned``.
        Returns ``(F_all, E_all)`` each shaped ``(Nnodes, 2, 2)``.
        """
        nodes_np = np.asarray(self.node_coordinates_binned[:, :2])
        disp_binned = np.asarray(displacement) / float(self.binning)

        elements_np = np.asarray(self.element_conectivity)
        n_nodes = nodes_np.shape[0]

        nb1_idx, nb1_deg, _ = build_node_neighbor_dense(
            elements_np, nodes_np, n_nodes
        )

        if k_ring <= 1:
            nb_idx, nb_deg = nb1_idx, nb1_deg
        else:
            nb_idx, nb_deg = build_k_ring_neighbors(nb1_idx, nb1_deg, k=k_ring)

        F_all, E_all = compute_green_lagrange_strain_nodes_lsq(
            disp_binned,
            nodes_np,
            nb_idx,
            nb_deg,
            gauge_length=gauge_length,
        )
        return F_all, E_all

    def compute_feature_disp_guess_big_motion(
        self,
        im_ref,
        im_def,
        n_patches=16,
        patch_win=41,
        patch_search=24,
        ransac_model="similarity",
        refine=False,
        win=31,
        search=3,
        search_dilation=0.0,
        use_element_centers=False,
        element_stride=1,
        initial_match_mode: str = "robust",
    ):
        """Build a nodal displacement guess from sparse ZNCC matches before full DIC.

        ``initial_match_mode`` can be ``"robust"`` (default rotation/scale-insensitive search +
        RANSAC) or ``"translation_zncc"`` which samples fixed element centers and performs
        pure translation searches via ZNCC.
        Samples textured patches (or element centers), filters them with local RANSAC,
        interpolates inlier displacements with weighted RBFs, and optionally runs subpixel NCC.
        Returns the nodal guess and a diagnostics dict (matches, scores, model, interpolator).
        """
        nodes = np.asarray(self.node_coordinates_binned[:, :2])
        elements = np.asarray(self.element_conectivity)
        centers_used = None
        stride = int(element_stride)
        if stride < 1:
            raise ValueError("element_stride must be a positive integer.")

        match_mode = str(initial_match_mode).lower()
        used_elem_centers_flag = bool(use_element_centers)

        if match_mode in {"translation", "translation_zncc"}:
            centers = self._compute_element_centers_binned()
            centers = centers[::stride]
            centers_used = centers
            used_elem_centers_flag = True
            pts_ref_raw, pts_def_raw, scores_raw = big_motion_sparse_matches(
                im_ref,
                im_def,
                win=patch_win,
                search=patch_search,
                centers=centers,
                mode="translation_zncc",
            )


        else:
            if use_element_centers:
                centers = self._compute_element_centers_binned()
                centers = centers[::stride]
                centers_used = centers
                pts_ref_raw, pts_def_raw, scores_raw = big_motion_sparse_matches(
                    im_ref,
                    im_def,
                    K=centers.shape[0],
                    win=patch_win,
                    search=patch_search,
                    interp_method="rbf",
                    centers=centers,
                    mode="robust",
                )
            else:
                pts_ref_raw, pts_def_raw, scores_raw = big_motion_sparse_matches(
                    im_ref,
                    im_def,
                    K=n_patches,
                    win=patch_win,
                    search=patch_search,
                    mode="robust",
                )

        if pts_ref_raw.shape[0] == 0:
            raise ValueError("No sparse matches could be extracted from the images.")

        mask = _points_in_mesh_area(
            pts_ref_raw,
            mesh_nodes=nodes,
            mesh_elements=elements,
            dilation=search_dilation,
        )
        pts_ref = pts_ref_raw[mask]
        pts_def = pts_def_raw[mask]
        scores = scores_raw[mask]

        Model = AffineTransform if ransac_model == "affine" else SimilarityTransform
        min_samples = 3 if ransac_model == "affine" else 2

        if pts_ref.shape[0] < min_samples:
            raise ValueError(
                f"Not enough sparse matches fall inside the mesh footprint, "
                f"{nodes.shape[0]} nodes available, {min_samples} required."
            )

        inliers = local_ransac_outlier_filter(
            pts_ref,
            pts_def,
            model=ransac_model,
            k_neighbors=min(30, pts_ref.shape[0]),
            residual_threshold=2.5,
            max_trials=500,
        )
        pts_ref_in = pts_ref[inliers]
        pts_def_in = pts_def[inliers]
        scores_in = scores[inliers]

        if pts_ref_in.shape[0] == 0:
            raise ValueError(
                "Local RANSAC rejected all sparse matches; cannot build an initial field."
            )

        # Fit the requested global transform on all inliers (plain least squares).
        model_robust = Model()
        model_robust.estimate(pts_ref_in, pts_def_in)
        print(f"pts_ref_in shape: {pts_ref_in.shape}")
        if pts_ref_in.shape[0] < 3:
            # Not enough inliers: keep a constant displacement equal to the mean.
            mean_disp = (pts_def_in - pts_ref_in).mean(axis=0)

            def interp_fn(xy):
                q = np.atleast_2d(xy)
                return np.repeat(mean_disp[None, :], q.shape[0], axis=0)

        else:
            # Otherwise interpolate the inlier field with a weighted RBF.
            disp_in = pts_def_in - pts_ref_in
            try:
                interp_rbf = build_weighted_rbf_interpolator(
                    pts_ref_in,
                    disp_in,
                    w=scores_in,
                    smoothing=1e-3,
                )

                def interp_fn(xy):
                    q = np.atleast_2d(xy)
                    return np.asarray(interp_rbf(q))

            except Exception:
                # If the RBF solve fails, revert to the mean displacement.
                mean_disp = disp_in.mean(axis=0)
                print(f"RBF interpolation failed; using mean displacement: {mean_disp}")

                def interp_fn(xy):
                    q = np.atleast_2d(xy)
                    return np.repeat(mean_disp[None, :], q.shape[0], axis=0)

        disp_guess = interp_fn(nodes)
        print(f"disp_guess shape: {disp_guess.shape}")
        extras = {
            "model": model_robust,
            "pts_ref": pts_ref_in,
            "pts_def": pts_def_in,
            "scores": scores_in,
            "raw_pts_ref": pts_ref_raw,
            "raw_pts_def": pts_def_raw,
            "raw_scores": scores_raw,
            "rbf_interpolator": interp_fn,
            "used_element_centers": used_elem_centers_flag,
            "element_stride": stride,
            "match_mode": match_mode,
        }
        if centers_used is not None:
            extras["element_centers"] = centers_used

        # if refine and pts_ref_in.shape[0] > 0:
        #     pts_def_pred = model_robust(pts_ref_in)
        #     subpix = refine_matches_ncc(im_ref, im_def, pts_ref_in, pts_def_pred, win=win, search=search)
        #     extras["subpixel_offsets"] = subpix
        #     if subpix.size:
        #         offset = np.median(subpix, axis=0)
        #         disp_guess = disp_guess + offset
        
        return jnp.asarray(disp_guess), extras

    # API_CANDIDATE
    def run_dic(self, im1, im2,
                disp_guess=None,
                max_iter=50,
                tol=1e-3,
                reg_type="spring",
                alpha_reg=0.1,
                save_history=False):
        """Global pixelwise DIC solve on the mesh with optional spring regularization.

        Accepts an optional nodal ``disp_guess`` and runs the jitted CG loop with the
        provided ``max_iter``/``tol``; when ``save_history`` is True the (J, ||grad||)
        history is returned alongside the optimized displacement.
        """
        nodes_coord = jnp.asarray(self.node_coordinates_binned[:, :2])

        if disp_guess is None:
            displacement = jnp.zeros_like(nodes_coord)
        else:
            # TODO: revisit this scaling when binning differs from 1.
            displacement = jnp.asarray(disp_guess) / self.binning

        # Follow the global JAX default dtype (controlled via jax_enable_x64).
        im1 = jnp.asarray(im1)
        im2 = jnp.asarray(im2)
        im1_T = jnp.transpose(im1, (1, 0))
        im2_T = jnp.transpose(im2, (1, 0))

        if reg_type != "spring":
            raise ValueError(f"Unsupported reg_type '{reg_type}' (supported: 'spring').")

        max_iter = int(max_iter)
        disp_sol, history_arr, iterations, _, _ = self._solve_jit(
            displacement,
            im1_T,
            im2_T,
            self.pixel_coords_ref,
            self.pixel_nodes,
            self.pixel_shapeN,
            self.node_neighbor_index,
            self.node_neighbor_degree,
            self.node_neighbor_weight,
            alpha_reg,
            max_iter,
            tol,
            save_history,
        )

        disp_sol = disp_sol * self.binning
        iterations = int(iterations)
        history = None
        if save_history:
            history_np = np.asarray(history_arr[:iterations])
            history = [(float(val[0]), float(val[1])) for val in history_np]
        return disp_sol, history

    def run_dic_nodal(self, im1, im2,
                      disp_init=None,
                      n_sweeps=5,
        lam=1e-3,
        reg_type="spring_jacobi",
        alpha_reg=0.0,
        max_step=0.2,
        omega_local=0.5):
        """Local Gauss–Seidel nodal sweeps with optional spring regularization.

        Starts from ``disp_init`` (zeros if missing), damps the pixel Hessian with ``lam``,
        limits each update by ``max_step``, and relaxes with ``omega_local`` while using
        the spring neighborhood when ``reg_type == 'spring_jacobi'``.
        """
        nodes_coord = jnp.asarray(self.node_coordinates_binned[:, :2])
        if disp_init is None:
            displacement = jnp.zeros_like(nodes_coord)
        else:
            displacement = jnp.asarray(disp_init) / self.binning

        im1 = jnp.asarray(im1)
        im2 = jnp.asarray(im2)
        im1_T = jnp.transpose(im1, (1, 0))
        im2_T = jnp.transpose(im2, (1, 0))

        # Precompute image-2 gradients once (NumPy).
        gx2_np, gy2_np = compute_image_gradient(np.asarray(im2))
        gx2_T = jnp.asarray(gx2_np.T)
        gy2_T = jnp.asarray(gy2_np.T)

        if reg_type != "spring_jacobi":
            raise ValueError(f"Unsupported reg_type '{reg_type}' (supported: 'spring_jacobi').")

        for s in range(n_sweeps):
            r, _x_def, gx_def, gy_def = compute_pixel_state(
                displacement,
                im1_T, im2_T,
                self.pixel_coords_ref,
                self.pixel_nodes,
                self.pixel_shapeN,
                gx2_T, gy2_T,
            )

            displacement = jacobi_nodal_step_spring(
                displacement,
                r,
                gx_def, gy_def,
                self.node_pixel_index,
                self.node_N_weight,
                self.node_degree,
                self.node_neighbor_index,
                self.node_neighbor_degree,
                self.node_neighbor_weight,
                lam=lam,
                max_step=max_step,
                alpha_reg=alpha_reg,
                omega=omega_local,
            )

            jax.debug.print(
                "[Nodal-{}] sweep {}/{}, J={:.4e}",
                reg_type,
                s + 1,
                n_sweeps,
                0.5 * jnp.mean(r ** 2),
            )

        displacement = displacement * self.binning
        return displacement

    def compute_q1_pixel_displacement_field(self, nodal_displacement):
        """Evaluate the Q1 nodal field at every ROI pixel and return ``(Ux_map, Uy_map)``.

        Requires ``precompute_pixel_data``; outputs match the image shape with NaNs outside
        the ROI, and projected coordinates follow the deformed configuration for plotting.
        """
        if self._image_shape is None or self._roi_mask is None:
            raise RuntimeError(
                "precompute_pixel_data must be called before evaluating the Q1 field."
            )
        if not hasattr(self, "pixel_nodes") or not hasattr(self, "pixel_shapeN"):
            raise RuntimeError(
                "Pixel/element projection information is not available."
            )

        disp = jnp.asarray(nodal_displacement)
        n_nodes_expected = self.node_coordinates.shape[0]
        if disp.shape != (n_nodes_expected, 2):
            raise ValueError(
                f"nodal_displacement must have shape ({n_nodes_expected}, 2)."
            )

        # Interpolation Q1 : u_p = sum_i N_i(x_p) * u_i
        node_values = disp[self.pixel_nodes]                         # (Np, 4, 2)
        pixel_disp = jnp.sum(self.pixel_shapeN[..., None] * node_values, axis=1)
        pixel_disp_np = np.asarray(pixel_disp)

        # Project pixel centers into the deformed configuration for plotting.
        pixel_coords_ref = np.asarray(self.pixel_coords_ref)
        pixel_coords_def = pixel_coords_ref + pixel_disp_np

        def _scatter(values):
            values = np.asarray(values)
            H, W = self._image_shape
            grid = np.full((H, W), np.nan, dtype=values.dtype)
            if values.size == 0:
                return grid

            x_cont = pixel_coords_def[:, 0] - 0.5
            y_cont = pixel_coords_def[:, 1] - 0.5
            j0 = np.floor(x_cont).astype(int)
            i0 = np.floor(y_cont).astype(int)
            fx = x_cont - j0
            fy = y_cont - i0

            accum = np.zeros((H, W), dtype=values.dtype)
            weights = np.zeros((H, W), dtype=np.float64)

            def add(i_idx, j_idx, w):
                mask = (i_idx >= 0) & (i_idx < H) & (j_idx >= 0) & (j_idx < W) & (w > 0)
                if not np.any(mask):
                    return
                np.add.at(accum, (i_idx[mask], j_idx[mask]), values[mask] * w[mask])
                np.add.at(weights, (i_idx[mask], j_idx[mask]), w[mask])

            add(i0, j0, (1.0 - fx) * (1.0 - fy))
            add(i0, j0 + 1, fx * (1.0 - fy))
            add(i0 + 1, j0, (1.0 - fx) * fy)
            add(i0 + 1, j0 + 1, fx * fy)

            valid = weights > 0
            grid[valid] = accum[valid] / weights[valid]
            return grid

        ux_map = _scatter(pixel_disp_np[:, 0])
        uy_map = _scatter(pixel_disp_np[:, 1])
        return ux_map, uy_map
