"""Objective 2D kinematics and fast isotropic plane-stress histories."""
from functools import partial
import jax
import jax.numpy as jnp
from ..constitutive.isotropic_plane_stress import isotropic_plane_stress_increment
from .quadrature import q4_gauss_rule, q4_shape_gradients


def inverse_2x2(A):
    """Inverse of nonsingular (...,2,2) arrays, using the explicit determinant."""
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    adj = jnp.stack((jnp.stack((A[..., 1, 1], -A[..., 0, 1]), -1),
                     jnp.stack((-A[..., 1, 0], A[..., 0, 0]), -1)), -2)
    return adj / det[..., None, None]


def rotation_matrix(angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack((jnp.stack((c, -s), -1), jnp.stack((s, c), -1)), -2)


def symmetric_log(U):
    """SPD matrix logarithm, analytic and differentiable at repeated eigenvalues."""
    m = .5 * jnp.trace(U, axis1=-2, axis2=-1)
    D = U - m[..., None, None] * jnp.eye(2)
    v2 = .5 * jnp.sum(D * D, axis=(-2, -1))
    t = v2 / m**2
    safe_t = jnp.where(t > 1e-8, t, .25)
    v = jnp.sqrt(safe_t)
    ratio = jnp.where(t > 1e-8, jnp.arctanh(v) / v, 1 + t/3 + t*t/5 + t*t*t/7) / m
    det = U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]
    return .5 * jnp.log(det)[..., None, None] * jnp.eye(2) + ratio[..., None, None] * D


def symmetric_exp(L):
    """Symmetric matrix exponential with a regular isotropic limit."""
    m = .5 * jnp.trace(L, axis1=-2, axis2=-1)
    D = L - m[..., None, None] * jnp.eye(2)
    v2 = .5 * jnp.sum(D * D, axis=(-2, -1))
    safe_v = jnp.sqrt(jnp.where(v2 > 1e-8, v2, 1.))
    ratio = jnp.where(v2 > 1e-8, jnp.sinh(safe_v)/safe_v, 1 + v2/6 + v2*v2/120)
    cosh = jnp.where(v2 > 1e-8, jnp.cosh(safe_v), 1 + v2/2 + v2*v2/24)
    return jnp.exp(m)[..., None, None] * (cosh[..., None, None]*jnp.eye(2) + ratio[..., None, None]*D)


def polar_log(delta):
    """Return rotation angle and log stretch for orientation-preserving 2D F."""
    angle = jnp.arctan2(delta[..., 1, 0] - delta[..., 0, 1],
                       delta[..., 0, 0] + delta[..., 1, 1])
    R = rotation_matrix(angle)
    U = jnp.swapaxes(R, -1, -2) @ delta
    U = .5 * (U + jnp.swapaxes(U, -1, -2))
    return angle, symmetric_log(U)


@partial(jax.jit, static_argnames=('substeps',))
def prepare_history(F, substeps=1):
    """Prepare material strain increments and orientations (T,S,Ne,Ng,...).

    Interpolate each relative gradient as R(alpha) exp(alpha log U). This
    preserves pure rotations. Intermediate history is an assumed path, not
    additional measured data. Reference F[0] is the identity.
    """
    delta = F[1:] @ inverse_2x2(F[:-1])
    angle, logU = polar_log(delta)
    alpha = jnp.arange(1, substeps + 1, dtype=F.dtype) / substeps
    R = rotation_matrix(angle[:, None] * alpha[None, :, None, None])
    U = symmetric_exp(logU[:, None] * alpha[None, :, None, None, None, None])
    intermediate = R @ U @ F[:-1, None]
    refined = intermediate.reshape((-1,) + F.shape[1:])

    def step(carry, new_F):
        old_F, orientation = carry
        angle, stretch = polar_log(new_F @ inverse_2x2(old_F))
        strain = jnp.swapaxes(orientation, -1, -2) @ stretch @ orientation
        de = jnp.stack((strain[..., 0, 0], strain[..., 1, 1], 2 * strain[..., 0, 1]), -1)
        new_orientation = rotation_matrix(angle) @ orientation
        return (new_F, new_orientation), (de, new_orientation)

    shape = F.shape[1:3]
    _, (de, rotations) = jax.lax.scan(step, (F[0], jnp.broadcast_to(jnp.eye(2), shape + (2, 2))), refined)
    de = de.reshape((F.shape[0]-1, substeps) + shape + (3,))
    rotations = rotations.reshape((F.shape[0]-1, substeps) + shape + (2, 2))
    return de, rotations


@jax.jit
def integrate_prepared(de, rotations, theta, initial_state=None, start=0):
    """Integrate prepared kinematics; return frame states (including initial), fields.

    An initial cached state permits exact suffix integration at fixed theta.
    Material state is (stress4, accumulated plastic strain, log thickness).
    """
    shape = de.shape[2:-1]
    # Backward-compatible three-parameter input denotes Voce. Identification
    # passes five kernel values for either Voce or Swift.
    if theta.shape[0] == 3:
        theta = jnp.concatenate((theta, jnp.asarray([1., 0.], dtype=theta.dtype)))
    if initial_state is None:
        initial_state = (jnp.zeros(shape+(4,), de.dtype), jnp.zeros(shape, de.dtype), jnp.zeros(shape, de.dtype))

    def frame(state, inputs):
        increments, orientations, index = inputs
        def step(state, increment):
            state, (ok, residual) = isotropic_plane_stress_increment(increment, state, theta)
            return state, (ok, residual)
        def advance(state):
            return jax.lax.scan(step, state, increments)
        def skip(state):
            return state, (jnp.ones(increments.shape[:-1], dtype=bool),
                           jnp.zeros(increments.shape[:-1], dtype=de.dtype))
        state, (ok, residual) = jax.lax.cond(index >= start, advance, skip, state)
        stress, p, thickness = state
        local = jnp.stack((jnp.stack((stress[..., 0], stress[..., 3]), -1),
                           jnp.stack((stress[..., 3], stress[..., 1]), -1)), -2)
        Q = orientations[-1]
        fields = dict(cauchy_stress=Q @ local @ jnp.swapaxes(Q,-1,-2),
                      equivalent_plastic_strain=p, thickness_ratio=jnp.exp(thickness),
                      converged=jnp.all(ok,axis=0), residual=jnp.max(residual,axis=0))
        return state, (state, fields)

    _, (states, fields) = jax.lax.scan(frame, initial_state, (de, rotations, jnp.arange(de.shape[0])) )
    states = jax.tree_util.tree_map(lambda first, rest: jnp.concatenate((first[None], rest)), initial_state, states)
    return states, fields


@partial(jax.jit, static_argnames=('substeps',))
def isotropic_history(F, theta, substeps=1):
    """Return measured-frame fields from (T+1,Ne,Ng,2,2) gradients."""
    de, rotations = prepare_history(F, substeps)
    _, fields = integrate_prepared(de, rotations, theta)
    valid = jnp.all(jnp.isfinite(F)) & jnp.all(jnp.linalg.det(F)>0)
    fields['converged'] = fields['converged'] & valid
    return fields


@jax.jit
def q4_operators(nodes, elements):
    """Precompute (Ne,4,4,2) reference gradients and (Ne,4) measures."""
    gauss, weights = q4_gauss_rule(nodes.dtype)
    dN = jax.vmap(q4_shape_gradients)(gauss)
    jac = jnp.einsum('eai,gaj->egij',nodes[elements],dN)
    gradients = jnp.einsum('gai,egij->egaj',dN,inverse_2x2(jac))
    return gradients, jnp.linalg.det(jac), gauss, weights


@jax.jit
def gradients_from_displacement(u, elements, gradients):
    """Affine FE operator: (T,Nn,2) -> (T,Ne,4,2,2)."""
    return jnp.eye(2) + jnp.einsum('teai,egaj->tegij',u[:,elements],gradients)


@jax.jit
def forces_from_fields(F, stress, thickness, elements, gradients, measures, node_template):
    """Assemble forces using fixed reference operators and mm-scaled measures."""
    J = jnp.linalg.det(F)*thickness
    P = J[...,None,None]*stress @ jnp.swapaxes(inverse_2x2(F),-1,-2)
    elemental = jnp.einsum('tegij,egaj,eg->teai',P,gradients,measures)
    return jax.vmap(lambda values:jnp.zeros_like(node_template).at[elements].add(values))(elemental)
