"""Digital Image Correlation utilities built on top of D2IC."""

import jax.numpy as jnp
from jax import jit, value_and_grad
import numpy as np
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
    """
    Build a ZNCC-weighted thin-plate spline RBF interpolator for 2D displacement.

    Parameters
    ----------
    x : array-like, shape (N, 2)
        Reference positions.
    u : array-like, shape (N, 2)
        Displacements associated with ``x``.
    w : array-like, shape (N,)
        Non-negative weights (e.g., ZNCC scores).
    smoothing : float, optional
        Tikhonov-like diagonal damping added to the normal equations.

    Returns
    -------
    callable
        Function ``interp(q)`` returning interpolated displacements at query
        positions ``q``.
    """
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


class Dic():
    """Pixelwise DIC solver built on a quadrilateral mesh."""

    # API_CANDIDATE
    def __init__(self, mesh_path=""):
        """
        Initialize the solver by reading the mesh and basic geometry caches.

        Parameters
        ----------
        mesh_path : str
            Path to the Gmsh ``.msh`` file describing the ROI mesh.
        """
        self.mesh = None
        self.mesh_path = mesh_path
        self._read_mesh()
        self.binning = 1.
        self._image_shape = None
        self._roi_mask = None
        self._roi_flat_indices = None
    
    def __repr__(self):
        """String representation of the Dic object."""
        return f"Dic(mesh_path={self.mesh_path})"

    def precompute_pixel_data(self, im):
        """
        Precompute pixel-to-element mappings and shape functions for one ROI.

        Parameters
        ----------
        im : array-like
            Reference image (H, W) after optional binning.

        Notes
        -----
        Builds pixel coordinates, filters them inside the mesh footprint,
        assigns each pixel to an element, and precomputes Q4 shape functions
        ``N_i(x_p)`` so pixelwise residuals can be evaluated efficiently.
        Also constructs CSR-like node→pixel structures and geometric
        neighborhoods for mechanical regularization.
        """
        H, W = im.shape[:2]
        self._image_shape = (H, W)

        # 1) Build pixel coordinates over the ROI
        jj, ii = np.meshgrid(np.arange(W), np.arange(H))
        pixel_coords = np.stack([jj.ravel() + 0.5,
                                 ii.ravel() + 0.5], axis=1)  # (Np_all,2)

        # Restrict to pixels lying in or near the mesh footprint
        mask = _points_in_mesh_area(
            pixel_coords,
            mesh_nodes=np.asarray(self.node_coordinates_binned[:, :2]),
            mesh_elements=np.asarray(self.element_conectivity)
        )
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        self._roi_mask = mask.reshape(H, W)
        self._roi_flat_indices = np.flatnonzero(mask)
        pixel_coords = pixel_coords[mask]

        # 2) Mesh data (NumPy + JAX views)
        nodes_coord_np = np.asarray(self.node_coordinates_binned[:, :2])
        elements_np = np.asarray(self.element_conectivity)
        nodes_coord_jax = jnp.asarray(nodes_coord_np)

        pixel_elts_np, pixel_nodes_np = build_pixel_to_element_mapping_numpy(
            pixel_coords, nodes_coord_np, elements_np
        )
        pixel_coords_jax = jnp.asarray(pixel_coords)
        pixel_nodes_jax = jnp.asarray(pixel_nodes_np, dtype=jnp.int32)
        pixel_elts = jnp.asarray(pixel_elts_np, dtype=jnp.int32)

        pixel_N_jax, xi_eta = compute_pixel_shape_functions_jax(
            pixel_coords_jax,
            pixel_nodes_jax,
            nodes_coord_jax
        )

        # 4) Store
        self.pixel_coords_ref = pixel_coords_jax      # (Np,2)
        self.pixel_elts = jnp.asarray(pixel_elts)     # (Np,)
        self.pixel_nodes = pixel_nodes_jax            # (Np,4)
        self.pixel_shapeN = pixel_N_jax               # (Np,4)
        self.pixel_xi_eta = xi_eta                    # optionnel

        # CSR node → pixels
        n_nodes = self.node_coordinates_binned.shape[0]
        node_pixel_index, node_N_weight, node_degree = build_node_pixel_dense(
            np.asarray(self.pixel_nodes),
            np.asarray(self.pixel_shapeN),
            n_nodes,
        )
        self.node_pixel_index = jnp.asarray(node_pixel_index, dtype=jnp.int32)   # (Nnodes, max_deg)
        self.node_N_weight    = jnp.asarray(node_N_weight,    dtype=jnp.float32) # (Nnodes, max_deg)
        self.node_degree      = jnp.asarray(node_degree,      dtype=jnp.int32)   # (Nnodes,)

        # Mechanical neighbors (mesh geometry)
        node_neighbor_index, node_neighbor_degree, node_neighbor_weight = build_node_neighbor_dense(
            elements_np,
            nodes_coord_np,
            n_nodes,
        )
        self.node_neighbor_index  = jnp.asarray(node_neighbor_index,  dtype=jnp.int32)
        self.node_neighbor_degree = jnp.asarray(node_neighbor_degree, dtype=jnp.int32)
        self.node_neighbor_weight = jnp.asarray(node_neighbor_weight, dtype=jnp.float32)


    # properties definition
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
        # meshio >=5 stores the blocks in ``cells``; concatenate all quad blocks.
        for cell_block in getattr(self.mesh, "cells", []):
            cell_type = getattr(cell_block, "type", "")
            if cell_type.startswith("quad"):
                quad_blocks.append(cell_block.data)
        if quad_blocks:
            conn = np.concatenate(quad_blocks, axis=0)
            return jnp.array(conn)
        # fallback for older meshio versions where quads were not concatenated
        cells_dict = getattr(self.mesh, "cells_dict", {})
        if isinstance(cells_dict, dict) and "quad" in cells_dict:
            return jnp.array(cells_dict["quad"])
        raise ValueError("No recognizable quadrilateral elements were found in the mesh.")

    def _compute_element_centers_binned(self):
        """
        Return the (x, y) center of each finite element in binned pixel coordinates.
        The center is computed as the mean of the four node coordinates.
        """
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
        # creat a triangulation of the mesh
        # each quad element is divided into two triangles
        # work only for 2d meshes
        from matplotlib.tri import Triangulation
        x, y = (self.node_coordinates[:,:2]+displacement).T
        # creat connectivity for the triangulation by splitting each quad element into two triangles
        conn_tri = []
        for quad in self.element_conectivity:
            conn_tri.append([quad[0], quad[1], quad[2]])
            conn_tri.append([quad[0], quad[2], quad[3]])
        conn_tri = jnp.array(conn_tri)
        # create the triangulation
        triangulation = Triangulation(x, y, conn_tri)
 
        return triangulation

    def _read_mesh(self):
        """Charge le maillage depuis le disque via ``meshio``."""
        self.mesh = meshio.read(self.mesh_path)

    def compute_green_lagrange_strain_nodes(
        self,
        displacement,
        k_ring=1,
        gauge_length=0.0,
    ):
        """Compute nodal Green–Lagrange strains via local LSQ interpolation.

        Parameters
        ----------
        displacement : (Nnodes, 2)
            Nodal displacement field (pixels).
        k_ring : int, optional
            Number of neighbor rings in the stencil (1 = immediate neighbors).
        gauge_length : float, optional
            Gauge length in same unit as ``node_coordinates_binned``.

        Returns
        -------
        tuple of (F_all, E_all)
            Nodal deformation gradient and Green–Lagrange tensors, shape (Nnodes, 2, 2).

        Notes
        -----
        Builds a k-ring neighborhood, solves a least-squares fit of the displacement
        gradient over the stencil, then forms ``F = I + grad u`` and ``E = 0.5*(F.T@F - I)``.
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
    ):
        """High-level initializer for large motions using sparse matches + smooth interpolation.

        Recommended entry point to obtain a displacement guess for challenging,
        non-affine motion before running the full DIC solver. Outliers are rejected
        via local, neighborhood-based RANSAC; the global model is then estimated on
        the retained inliers (no global RANSAC).

        This routine extracts sparse patch correspondences robust to large displacements,
        filters them inside the FE mesh footprint, rejects outliers with local RANSAC,
        then builds a smooth displacement field by interpolating the inlier matches (RBF).
        The interpolated field is evaluated at mesh nodes to provide a nodal initial
        guess suitable for subsequent DIC refinement.

        Parameters
        ----------
        im_ref, im_def : array-like
            Reference and deformed images.
        n_patches : int, optional
            Target number of patch centers sampled across the image.
        patch_win : int, optional
            Patch size (odd) used for local ZNCC matching.
        patch_search : int, optional
            Half-width of the search window around the initial prediction.
        ransac_model : {'affine', 'similarity'}, optional
            Geometric model used only for outlier rejection/diagnostics.
        refine : bool, optional
            If True, perform a sub-pixel NCC refinement and shift the nodal guess
            by the median offset of the refined inlier matches.
        win : int, optional
            Window size used for sub-pixel refinement.
        search : int, optional
            Half-width of the search area during sub-pixel refinement.
        search_dilation : float, optional
            Dilation (in pixels) applied to the mesh footprint when filtering patches.
        use_element_centers : bool, optional
            If True, perform one ZNCC match per (subsampled) finite element center
            instead of automatically sampling textured patches.
        element_stride : int, optional
            Subsampling step applied to element centers (e.g., 2 keeps one center
            every two elements).

        Returns
        -------
        jnp.ndarray
            Nodal displacement guess of shape ``(Nnodes, 2)``.
        dict
            Diagnostics including raw matches, inliers, scores, robust model, and
            interpolator used to build the nodal guess.

        Examples
        --------
        >>> disp_guess, extras = dic.compute_feature_disp_guess_big_motion(
        ...     im_ref, im_def,
        ...     n_patches=64,
        ...     refine=True,
        ...     patch_win=21,
        ...     patch_search=51,
        ...     search_dilation=1.0,
        ... )
        >>> u_dic, history = dic.run_dic(im_ref, im_def, disp_guess=disp_guess)
        """
        nodes = np.asarray(self.node_coordinates_binned[:, :2])
        elements = np.asarray(self.element_conectivity)
        centers_used = None
        stride = int(element_stride)
        if stride < 1:
            raise ValueError("element_stride must be a positive integer.")

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
            )
        else:
            pts_ref_raw, pts_def_raw, scores_raw = big_motion_sparse_matches(
                im_ref,
                im_def,
                K=n_patches,
                win=patch_win,
                search=patch_search,
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

        # Estimate a global transform on the filtered inliers (deterministic LS fit).
        model_robust = Model()
        model_robust.estimate(pts_ref_in, pts_def_in)

        if pts_ref_in.shape[0] < 3:
            # Too few inliers: fall back to a constant displacement equal to the mean.
            mean_disp = (pts_def_in - pts_ref_in).mean(axis=0)

            def interp_fn(xy):
                q = np.atleast_2d(xy)
                return np.repeat(mean_disp[None, :], q.shape[0], axis=0)

        else:
            # Smooth, possibly non-affine displacement field by weighted RBF interpolation.
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
                # Numerical failure in RBF → conservative constant model.
                mean_disp = disp_in.mean(axis=0)

                def interp_fn(xy):
                    q = np.atleast_2d(xy)
                    return np.repeat(mean_disp[None, :], q.shape[0], axis=0)

        disp_guess = interp_fn(nodes)
        extras = {
            "model": model_robust,
            "pts_ref": pts_ref_in,
            "pts_def": pts_def_in,
            "scores": scores_in,
            "raw_pts_ref": pts_ref_raw,
            "raw_pts_def": pts_def_raw,
            "raw_scores": scores_raw,
            "rbf_interpolator": interp_fn,
            "used_element_centers": use_element_centers,
            "element_stride": stride,
        }
        if centers_used is not None:
            extras["element_centers"] = centers_used

        if refine and pts_ref_in.shape[0] > 0:
            pts_def_pred = model_robust(pts_ref_in)
            subpix = refine_matches_ncc(im_ref, im_def, pts_ref_in, pts_def_pred, win=win, search=search)
            extras["subpixel_offsets"] = subpix
            if subpix.size:
                offset = np.median(subpix, axis=0)
                disp_guess = disp_guess + offset

        return jnp.asarray(disp_guess), extras

    # API_CANDIDATE
    def run_dic(self, im1, im2,
                disp_guess=None,
                max_iter=50,
                tol=1e-3,
                reg_type="spring",
                alpha_reg=0.1):
        """
        Run global pixelwise DIC with optional Tikhonov regularization.

        Parameters
        ----------
        im1, im2 : array-like
            Reference and deformed images.
        disp_guess : array-like, optional
            Initial nodal displacement guess ``(Nnodes, 2)``.
        max_iter : int
            Maximum Conjugate Gradient iterations on the Gauss–Newton quadratic.
        tol : float
            Convergence tolerance on gradient norm.
        reg_type : {'none', 'laplace', 'spring'}
            Global regularization energy added to the image mismatch.
        alpha_reg : float
            Weight of the regularization term.

        Returns
        -------
        displacement : np.ndarray
            Optimized nodal displacement field.
        history : list of (J, ||grad||)
            Per-iteration objective and gradient norms.

        Notes
        -----
        Minimizes ``0.5 * ||I2(x+u) - I1(x)||^2`` with optional smoothness
        penalties (Laplace or spring-based) using a preconditioned CG loop on
        the normal equations.
        """
        nodes_coord = jnp.asarray(self.node_coordinates_binned[:, :2])

        if disp_guess is None:
            displacement = jnp.zeros_like(nodes_coord)
        else:
            # ⚠ if binning != 1 we might need to revisit this division by self.binning
            displacement = jnp.asarray(disp_guess) / self.binning

        im1 = jnp.asarray(im1)
        im2 = jnp.asarray(im2)

        if reg_type != "spring":
            raise ValueError(f"Unsupported reg_type '{reg_type}' (supported: 'spring').")

        def J_wrapper(disp):
            J_img = J_pixelwise_core(
                disp,
                im1, im2,
                self.pixel_coords_ref,
                self.pixel_nodes,
                self.pixel_shapeN,
            )
            if alpha_reg == 0.0:
                return J_img
            J_reg = reg_energy_spring_global(
                disp,
                self.node_neighbor_index,
                self.node_neighbor_degree,
                self.node_neighbor_weight,
            )
            return J_img + alpha_reg * J_reg

        J_wrapper_jit = jit(J_wrapper)
        J_val_and_grad = jit(value_and_grad(J_wrapper))


        # ---- CG initialization ----
        d = jnp.zeros_like(displacement)
        g_prev = jnp.zeros_like(displacement)
        first = True

        history = []

        for k in range(max_iter):
            J_val, g = J_val_and_grad(displacement)
            grad_norm = float(jnp.linalg.norm(g))
            J_val_float = float(J_val)
            history.append((J_val_float, grad_norm))

            print(f"[CG] iter {k:3d}  J={J_val_float:.4e}  ||g||={grad_norm:.3e}")

            # stopping criterion
            if grad_norm < tol:
                print(f"[CG] converged at iteration {k}, ||grad|| = {grad_norm:.3e}")
                break

            # ---- direction computation ----
            if first:
                d = -g
                first = False
            else:
                num = jnp.sum(g * (g - g_prev))
                den = jnp.sum(g_prev * g_prev) + 1e-16
                beta = jnp.maximum(num / den, 0.0)
                d = -g + beta * d

            # gradient·direction dot product
            gd = jnp.sum(g * d)

            # ⚠ restart when the direction is not descending
            if gd >= 0:
                d = -g
                gd = -jnp.sum(g * g)

            gd_float = float(gd)

            # ---- Armijo backtracking ----
            alpha = 1.0
            c = 1e-4

            for _ in range(10):
                new_disp = displacement + alpha * d
                J_new = float(J_wrapper_jit(new_disp))
                if J_new <= J_val_float + c * alpha * gd_float:
                    break
                alpha *= 0.5

            displacement = displacement + alpha * d
            g_prev = g

            print(f"        alpha={alpha:.3e}, J_new={J_new:.4e}")

        displacement = displacement * self.binning
        return displacement, history

    def run_dic_nodal(self, im1, im2,
                      disp_init=None,
                      n_sweeps=5,
        lam=1e-3,
        reg_type="spring_jacobi",
        alpha_reg=0.0,
        max_step=0.2,
        omega_local=0.5):
        """
        Local nodal refinement with Gauss–Seidel sweeps and optional regularization.

        Parameters
        ----------
        im1, im2 : array-like
            Reference and deformed images.
        disp_init : array-like, optional
            Initial nodal displacement.
        n_sweeps : int
            Number of Gauss–Seidel passes.
        lam : float
            Damping factor on the pixelwise Hessian.
        reg_type : {'none', 'laplace', 'spring'}
            Regularization flavour added to the pixel residual.
        alpha_reg : float
            Regularization weight (0 disables regularization).
        max_step : float
            Maximal step per node (pixels) for stability.
        omega_local : float
            Relaxation factor in the Gauss–Seidel update.

        Returns
        -------
        np.ndarray
            Refined nodal displacement.

        Notes
        -----
        Performs in-place Gauss–Seidel updates on nodes using the pixelwise
        residual gradient/Hessian (optionally damped by ``lam``), with either
        pure image data, Laplacian, or spring-weighted smoothness terms.
        """
        nodes_coord = jnp.asarray(self.node_coordinates_binned[:, :2])
        if disp_init is None:
            displacement = jnp.zeros_like(nodes_coord)
        else:
            displacement = jnp.asarray(disp_init) / self.binning

        im1 = jnp.asarray(im1)
        im2 = jnp.asarray(im2)

        # Precompute image-2 gradients once (NumPy)
        gx2_np, gy2_np = compute_image_gradient(np.asarray(im2))
        gx2 = jnp.asarray(gx2_np)
        gy2 = jnp.asarray(gy2_np)

        if reg_type != "spring_jacobi":
            raise ValueError(f"Unsupported reg_type '{reg_type}' (supported: 'spring_jacobi').")

        for s in range(n_sweeps):
            r, _x_def, gx_def, gy_def = compute_pixel_state(
                displacement,
                im1, im2,
                self.pixel_coords_ref,
                self.pixel_nodes,
                self.pixel_shapeN,
                gx2, gy2,
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

            print(f"[Nodal-{reg_type}] sweep {s+1}/{n_sweeps}, J={float(0.5*jnp.mean(r**2)):.4e}")

        displacement = displacement * self.binning
        return displacement

    def compute_q1_pixel_displacement_field(self, nodal_displacement):
        """
        Evaluate the Q1 displacement field pixel by pixel inside the ROI.

        Parameters
        ----------
        nodal_displacement : array-like, shape (Nnodes, 2)
            Nodal field (Ux, Uy) used to interpolate the displacement within the
            pixels of the reference image (after optional binning).

        Returns
        -------
        tuple of ndarray
            Two matrices ``(Ux_map, Uy_map)`` with the size of the processed
            image, filled with ``NaN`` outside the ROI. Values are projected to
            pixel positions in the deformed configuration to ease visualization
            over the deformed image.
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

        # Project pixel centers to the deformed configuration for visualization.
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
