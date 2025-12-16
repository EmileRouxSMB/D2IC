import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax import jit
import numpy as np
from matplotlib.path import Path


from  dm_pix import flat_nd_linear_interpolate


def build_node_neighbor_dense(elements, nodes_coord, n_nodes):
    """
    elements    : (Ne,4) connectivité quad (numpy array int)
    nodes_coord : (Nnodes,2) coordonnées nodales (numpy)
    n_nodes     : nombre total de noeuds

    Retourne :
      node_neighbor_index  : (n_nodes, max_deg) int32   -- indices des voisins
      node_neighbor_degree : (n_nodes,) int32           -- nb de voisins
      node_neighbor_weight : (n_nodes, max_deg) float32 -- poids 'mécaniques' w_ij
    """
    # 1) construire les ensembles de voisins
    neigh_sets = [set() for _ in range(n_nodes)]
    for elt in elements:
        elt = np.asarray(elt, dtype=int)
        for a in range(4):
            i = int(elt[a])
            for b in range(4):
                if b == a:
                    continue
                j = int(elt[b])
                neigh_sets[i].add(j)

    degrees = [len(s) for s in neigh_sets]
    max_deg = max(degrees) if degrees else 0

    node_neighbor_index  = -np.ones((n_nodes, max_deg), dtype=np.int32)
    node_neighbor_degree = np.asarray(degrees, dtype=np.int32)
    node_neighbor_weight = np.zeros((n_nodes, max_deg), dtype=np.float32)

    # 2) remplir les arrays denses + poids w_ij ~ 1 / ||x_i - x_j||
    for i in range(n_nodes):
        if degrees[i] == 0:
            continue
        neigh_i = np.asarray(sorted(neigh_sets[i]), dtype=np.int32)
        node_neighbor_index[i, :degrees[i]] = neigh_i

        xi = nodes_coord[i, :2]  # (2,)
        xj = nodes_coord[neigh_i, :2]  # (deg_i,2)
        dist = np.linalg.norm(xj - xi[None, :], axis=1)
        # petit epsilon pour éviter division par 0
        w = 1.0 / (dist + 1e-6)
        node_neighbor_weight[i, :degrees[i]] = w.astype(np.float32)

    return node_neighbor_index, node_neighbor_degree, node_neighbor_weight


def build_k_ring_neighbors(node_neighbor_index, node_neighbor_degree, k=2):
    """
    Construit un voisinage 'k-anneaux' à partir du premier anneau dense.

    Parameters
    ----------
    node_neighbor_index : (Nnodes, max_deg) int32
        Indices du 1er anneau (issu de build_node_neighbor_dense).
    node_neighbor_degree : (Nnodes,) int32
        Nombre de voisins du 1er anneau par nœud.
    k : int, optional
        Nombre d'anneaux souhaité (k=1 : inchangé, k=2 : voisins des voisins, ...).

    Returns
    -------
    neigh_index_k : (Nnodes, max_deg_k) int32
        Indices des voisins jusqu'au kᵉ anneau (triés, sans doublons).
    neigh_degree_k : (Nnodes,) int32
        Degré effectif pour chaque nœud.
    """
    node_neighbor_index = np.asarray(node_neighbor_index)
    node_neighbor_degree = np.asarray(node_neighbor_degree)
    Nnodes, _ = node_neighbor_index.shape

    # Ensembles uniques pour éviter les doublons.
    neigh_sets = [set() for _ in range(Nnodes)]
    for i in range(Nnodes):
        deg_i = int(node_neighbor_degree[i])
        for d in range(deg_i):
            j = int(node_neighbor_index[i, d])
            neigh_sets[i].add(j)

    # Propagation sur k-1 anneaux supplémentaires.
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

    neigh_index_k = -np.ones((Nnodes, max_deg_k), dtype=np.int32)
    neigh_degree_k = np.asarray(degrees, dtype=np.int32)

    for i in range(Nnodes):
        if degrees[i] == 0:
            continue
        neigh_i = np.asarray(sorted(neigh_sets[i]), dtype=np.int32)
        neigh_index_k[i, :degrees[i]] = neigh_i

    return neigh_index_k, neigh_degree_k


def build_pixel_to_element_mapping_numpy(pixel_coords, nodes_coord, elements):
    """
    pixel_coords : (Np,2) float64 ou float32
    nodes_coord  : (Nn,2)
    elements     : (Ne,4) indices des noeuds (numpy int)

    Retourne:
      pixel_elts  : (Np,) index d'élément
      pixel_nodes : (Np,4) indices des 4 noeuds
    """
    Np = pixel_coords.shape[0]
    pixel_elts = -np.ones(Np, dtype=int)
    pixel_nodes = np.zeros((Np, 4), dtype=int)

    # Prépare Path + bounding box par élément
    elt_paths = []
    elt_bboxes = []
    for e in elements:
        Xe = nodes_coord[e]  # (4,2)
        path = Path(Xe)
        elt_paths.append(path)
        xmin, ymin = Xe.min(axis=0)
        xmax, ymax = Xe.max(axis=0)
        elt_bboxes.append([xmin, xmax, ymin, ymax])
    elt_bboxes = np.asarray(elt_bboxes)

    # Vectorisé : itère par élément (rare) plutôt que par pixel (souvent)
    for e, path in enumerate(elt_paths):
        # Pixels encore non affectés + dans la bbox de l'élément
        mask_bbox = (
            (pixel_elts < 0)
            & (pixel_coords[:, 0] >= elt_bboxes[e, 0])
            & (pixel_coords[:, 0] <= elt_bboxes[e, 1])
            & (pixel_coords[:, 1] >= elt_bboxes[e, 2])
            & (pixel_coords[:, 1] <= elt_bboxes[e, 3])
        )
        if not np.any(mask_bbox):
            continue

        inside = path.contains_points(pixel_coords[mask_bbox])
        if not np.any(inside):
            continue

        idx = np.nonzero(mask_bbox)[0][inside]
        pixel_elts[idx] = e
        pixel_nodes[idx] = elements[e]

        # early exit si tout est rempli
        if np.all(pixel_elts >= 0):
            break

    return pixel_elts, pixel_nodes


# ------------------------
# Fonctions de forme Q4
# ------------------------

def shape_functions_jax(xi, eta):
    """Fonctions de forme Q4 bilinéaires sur [0,1]^2."""
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
    """
    Xi_eta : (2,) = (xi, eta)
    Xe     : (4,2) coord. des noeuds de l'élément
    """
    xi, eta = xi_eta
    N = shape_functions_jax(xi, eta)  # (4,)
    return N @ Xe  # (2,)


def jacobian_quad_jax(xi_eta, Xe):
    """
    Jacobienne 2x2 du mapping Q4 (∂x/∂xi, ∂x/∂eta).
    """
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
    """
    xp : (2,) coord pixel
    Xe : (4,2) coord noeuds element
    Newton 2D pour inverser le mapping Q4.
    Pas de test de tolérance pour garder le corps JIT-compatible :
    on fait `max_iter` itérations fixes.
    """
    def body_fun(i, xi_eta):
        x_map = map_quad_jax(xi_eta, Xe)
        r = xp - x_map
        J = jacobian_quad_jax(xi_eta, Xe)
        delta = jnp.linalg.solve(J, r)
        return xi_eta + delta

    xi_eta0 = jnp.array([0.5, 0.5])
    xi_eta = lax.fori_loop(0, max_iter, body_fun, xi_eta0)
    xi_eta = jnp.clip(xi_eta, 0.0, 1.0)  # sécurité
    return xi_eta  # (2,)


# Vectorization Newton sur tous les pixels
vmap_newton = jax.jit(
    vmap(_newton_one_pixel, in_axes=(0, 0)),
    static_argnames=()  # max_iter est codé en dur ici
)


@jax.jit
def compute_pixel_shape_functions_jax(pixel_coords,
                                      pixel_nodes,
                                      nodes_coord):
    """
    pixel_coords : (Np,2) float32/float64 jax
    pixel_nodes  : (Np,4) int32 jax
    nodes_coord  : (Nn,2) jax

    Retourne:
      pixel_N     : (Np,4) valeurs Ni(ξp,ηp)
      xi_eta_all  : (Np,2) coordonnées locales (xi,eta)
    """
    # Coord des 4 noeuds pour chaque pixel (Np,4,2)
    Xe_all = nodes_coord[pixel_nodes]  # indexation avancée JAX OK

    # Inversion (vectorisée) : (Np,2)
    xi_eta_all = vmap_newton(pixel_coords, Xe_all)

    xi = xi_eta_all[:, 0]
    eta = xi_eta_all[:, 1]

    # Fonctions de forme (vectorisé sur les pixels)
    def _shape_from_xieta(xi, eta):
        return shape_functions_jax(xi, eta)

    pixel_N = vmap(_shape_from_xieta)(xi, eta)  # (Np,4)

    return pixel_N, xi_eta_all


@jit
def residuals_pixelwise_core(displacement,
                             im1, im2,
                             pixel_coords,
                             pixel_nodes,
                             pixel_shapeN):
    """
    displacement : (Nnodes,2)
    im1, im2     : (H,W)
    pixel_coords : (Np,2)
    pixel_nodes  : (Np,4)
    pixel_shapeN : (Np,4)
    """
    disp = jnp.asarray(displacement)
    im1 = jnp.asarray(im1)
    im2 = jnp.asarray(im2)

    # (Np,4,1)
    shapeN = pixel_shapeN[..., None]

    # déplacements nodaux des 4 nœuds par pixel
    disp_local = disp[pixel_nodes]          # (Np,4,2)

    # u(x_p) = Σ_i N_i * u_i
    u_pix = (shapeN * disp_local).sum(axis=1)  # (Np,2)

    x_ref = pixel_coords                    # (Np,2)
    x_def = x_ref + u_pix                   # (Np,2)

    # interpolation images (note : im.T comme dans ton code existant)
    I1 = flat_nd_linear_interpolate(im1.T, x_ref.T)  # (Np,)
    I2 = flat_nd_linear_interpolate(im2.T, x_def.T)  # (Np,)

    return I2 - I1                           # (Np,)


@jit
def J_pixelwise_core(displacement,
                     im1, im2,
                     pixel_coords,
                     pixel_nodes,
                     pixel_shapeN):
    r = residuals_pixelwise_core(displacement,
                                 im1, im2,
                                 pixel_coords,
                                 pixel_nodes,
                                 pixel_shapeN)
    return 0.5 * jnp.mean(r**2)


def build_node_pixel_dense(pixel_nodes, pixel_shapeN, n_nodes):
    """
    pixel_nodes : (Np,4) int  -- indices des 4 noeuds de chaque pixel
    pixel_shapeN : (Np,4) float -- Ni(p) pour les 4 noeuds
    n_nodes : nombre total de noeuds

    Retourne :
      node_pixel_index : (n_nodes, max_deg) int32   -- indices de pixels
      node_N_weight    : (n_nodes, max_deg) float32 -- Ni(p)
      node_degree      : (n_nodes,) int32           -- nb de pixels utiles
    """
    Np = pixel_nodes.shape[0]

    # 1) construire des listes par nœud
    per_node_pixels = [[] for _ in range(n_nodes)]
    per_node_weights = [[] for _ in range(n_nodes)]

    for p in range(Np):
        for loc in range(4):
            i = int(pixel_nodes[p, loc])
            Ni = float(pixel_shapeN[p, loc])
            per_node_pixels[i].append(p)
            per_node_weights[i].append(Ni)

    # 2) trouver le max de voisins
    degrees = [len(lst) for lst in per_node_pixels]
    max_deg = max(degrees) if degrees else 0

    # 3) remplir les tableaux denses
    node_pixel_index = -np.ones((n_nodes, max_deg), dtype=np.int32)
    node_N_weight = np.zeros((n_nodes, max_deg), dtype=np.float32)
    node_degree = np.asarray(degrees, dtype=np.int32)

    for i in range(n_nodes):
        deg_i = degrees[i]
        if deg_i == 0:
            continue
        node_pixel_index[i, :deg_i] = np.asarray(per_node_pixels[i], dtype=np.int32)
        node_N_weight[i, :deg_i] = np.asarray(per_node_weights[i], dtype=np.float32)

    return node_pixel_index, node_N_weight, node_degree


@jax.jit
def compute_pixel_state(displacement,
                        im1, im2,
                        pixel_coords,
                        pixel_nodes,
                        pixel_shapeN,
                        gradx2, grady2):
    """
    displacement : (Nnodes,2)
    im1, im2     : (H,W)
    pixel_coords : (Np,2)
    pixel_nodes  : (Np,4)
    pixel_shapeN : (Np,4)
    gradx2, grady2 : (H,W) gradients de I2 sur la grille pixels

    Retourne :
      r      : (Np,) résidus
      x_def  : (Np,2)
      gx_def : (Np,) dI2/dx en x_def
      gy_def : (Np,) dI2/dy en x_def
    """
    shapeN = pixel_shapeN[..., None]            # (Np,4,1)
    disp_local = displacement[pixel_nodes]      # (Np,4,2)
    u_pix = (shapeN * disp_local).sum(axis=1)   # (Np,2)

    x_ref = pixel_coords
    x_def = x_ref + u_pix

    I1 = flat_nd_linear_interpolate(im1.T, x_ref.T)
    I2 = flat_nd_linear_interpolate(im2.T, x_def.T)

    # gradients d'image 2 interpolés en x_def
    gx_def = flat_nd_linear_interpolate(gradx2.T, x_def.T)
    gy_def = flat_nd_linear_interpolate(grady2.T, x_def.T)

    r = I2 - I1

    return r, x_def, gx_def, gy_def

def compute_image_gradient(im):
    im = np.asarray(im, dtype=np.float32)
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
    """
    Raffinement nodal LOCAL de type Jacobi relaxé avec régularisation ressorts.

    - On calcule pour chaque noeud i un incrément δu_i à partir de l'état courant
      'displacement' (NON mis à jour dans la boucle).
    - À la fin, on applique : u_new = u_old + omega * δu.
    - Ça évite l'oscillation pair/impair typique du Gauss–Seidel.

    displacement         : (Nnodes,2)
    r                    : (Np,)
    gx_def, gy_def       : (Np,)
    node_pixel_index     : (Nnodes, max_deg_pix)
    node_N_weight        : (Nnodes, max_deg_pix)
    node_degree          : (Nnodes,)
    node_neighbor_index  : (Nnodes, max_deg_neigh)
    node_neighbor_degree : (Nnodes,)
    node_neighbor_weight : (Nnodes, max_deg_neigh)
    lam                  : damping LM (image)
    max_step             : norme max de δu_i
    alpha_reg            : poids régularisation ressorts
    omega                : facteur de sous-relaxation (0<omega<=1)
    """
    disp0 = displacement
    Nnodes, max_deg_pix   = node_pixel_index.shape
    _,      max_deg_neigh = node_neighbor_index.shape

    def body_fun(i, delta_acc):
        deg_pix   = node_degree[i]
        deg_neigh = node_neighbor_degree[i]

        # ---------- cas deg_pix == 0 : pas d'info image ----------
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

                # limite de pas
                norm_delta = jnp.linalg.norm(delta_i)
                factor = jnp.minimum(1.0, max_step / (norm_delta + 1e-12))
                delta_i = delta_i * factor

                return delta_acc_inner.at[i].set(delta_i)

            cond_reg = jnp.logical_and(alpha_reg != 0.0, deg_neigh > 0)
            return lax.cond(cond_reg, apply_reg_only, only_return_delta, operand=None)

        # ---------- cas deg_pix > 0 : image + ressorts ----------
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

            # ressorts
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

    # mise à jour simultanée + sous-relaxation
    displacement_new = displacement + omega * delta_all
    return displacement_new

@jax.jit
def reg_energy_spring_global(displacement,
                             node_neighbor_index,
                             node_neighbor_degree,
                             node_neighbor_weight):
    """
    Énergie 'ressorts' globale :

      J_reg = 0.5 * Σ_{(i,j)} w_ij ||u_i - u_j||^2

    où w_ij vient de build_node_neighbor_dense (∼1/||x_i-x_j||).
    """
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

    # même raisonnement, mais pondéré
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
    """
    Champ de Green-Lagrange nodal via un ajustement affine local (LSQ pondéré).

    Parameters
    ----------
    displacement : (Nnodes, 2)
        Champ de déplacement nodal (config de référence).
    nodes_coord : (Nnodes, 2)
        Coordonnées nodales dans la configuration de référence.
    node_neighbor_index : (Nnodes, max_deg)
        Indices des voisins pour chaque noeud (1er anneau ou k-anneaux).
    node_neighbor_degree : (Nnodes,)
        Nombre de voisins effectifs pour chaque noeud.
    gauge_length : float, optional
        Longueur de jauge. Si > 0, w_j = exp(-(||dX_j|| / L)^2), sinon poids uniformes.
    eps : float, optional
        Régularisation de la matrice normale.

    Returns
    -------
    F_all : (Nnodes, 2, 2)
        Gradient de déformation F = I + ∇u.
    E_all : (Nnodes, 2, 2)
        Tenseur de Green-Lagrange E = 0.5 (FᵀF - I).
    """
    disp = jnp.asarray(displacement)
    X = jnp.asarray(nodes_coord)
    node_neighbor_index = jnp.asarray(node_neighbor_index)
    node_neighbor_degree = jnp.asarray(node_neighbor_degree)

    Nnodes, max_deg = node_neighbor_index.shape
    L = jnp.squeeze(jnp.asarray(gauge_length, dtype=jnp.float32))

    def one_node(i):
        xi = X[i]        # (2,)
        ui = disp[i]     # (2,)

        idx_all = node_neighbor_index[i]   # (max_deg,)
        deg = node_neighbor_degree[i]      # scalaire int
        idx_range = jnp.arange(max_deg)
        mask = idx_range < deg             # (max_deg,) bool

        # Voisins effectifs (indices) - 0 utilisé comme remplissage masqué.
        neigh_ids = jnp.where(mask, idx_all, 0)
        xj = X[neigh_ids]                  # (max_deg, 2)
        uj = disp[neigh_ids]               # (max_deg, 2)

        dX = xj - xi[None, :]              # (max_deg, 2)
        du = uj - ui[None, :]              # (max_deg, 2)

        # Distances et poids
        r = jnp.linalg.norm(dX, axis=1)    # (max_deg,)
        base_w = mask.astype(jnp.float32)  # 0 pour les positions non actives
        w_exp = base_w * jnp.exp(-(r / (L + 1e-12)) ** 2)
        w = jnp.where(L > 0.0, w_exp, base_w)  # (max_deg,)

        w_col = w[:, None]                 # (max_deg, 1)
        Xw = dX * w_col                    # (max_deg, 2)

        # Matrice normale A et seconds membres b0/b1
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
