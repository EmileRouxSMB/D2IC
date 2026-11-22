"""Batched J2 isotropic return mapping with plane stress and implicit derivatives.

Stress order is [11,22,33,12]; in-plane strain uses [e11,e22,gamma12].
The kernel accepts [sigma0,K,eps0,n,model], with model=0 for Voce and
model=1 for the standard Swift law K*(eps0+p)**n.
"""
import jax
import jax.numpy as jnp


def _trial(de, stress, e33, young, poisson):
    """Elastic trial invariants for prescribed in-plane strain and unknown e33."""
    shear = young / (2 * (1 + poisson))
    bulk = young / (3 * (1 - 2 * poisson))
    lam = bulk - 2 * shear / 3
    trace = de[..., 0] + de[..., 1] + e33
    # Scalar components avoid an XLA GPU tiling failure on concatenate(3,1)
    # and keep the four-component stress arithmetic readily fusible.
    s11 = stress[..., 0] + 2 * shear * de[..., 0] + lam * trace
    s22 = stress[..., 1] + 2 * shear * de[..., 1] + lam * trace
    s33 = stress[..., 2] + 2 * shear * e33 + lam * trace
    s12 = stress[..., 3] + shear * de[..., 2]
    mean = (s11 + s22 + s33) / 3
    d11, d22, d33 = s11 - mean, s22 - mean, s33 - mean
    deviator = jnp.stack((d11, d22, d33, s12), axis=-1)
    # A finite inactive elastic derivative at hydrostatic/zero trial stress.
    q = jnp.sqrt(jnp.maximum(1.5 * (d11*d11 + d22*d22 + d33*d33 + 2*s12*s12), 1e-30))
    return mean, deviator, q, shear, bulk


def _elastic_guess(de, stress, young, poisson):
    """Eliminate sigma33 analytically assuming an elastic increment."""
    shear = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    return -(stress[..., 2] + lam * (de[..., 0] + de[..., 1])) / (lam + 2 * shear)


def _system(z, de, stress, p, theta, young, poisson):
    """Dimensionless local residual (...,2) and analytic Jacobian (...,2,2)."""
    dp, e33 = z[..., 0], z[..., 1]
    mean, dev, q, shear, bulk = _trial(de, stress, e33, young, poisson)
    e_elastic = _elastic_guess(de, stress, young, poisson)
    q_elastic = _trial(de, stress, e_elastic, young, poisson)[2]
    if theta.shape[0] == 3:
        theta = jnp.concatenate((theta, jnp.asarray([1., 0.], dtype=theta.dtype)))
    sigma0, hardening, eps0, exponent, model = theta
    voce_old = sigma0 - hardening * jnp.expm1(-p / eps0)
    swift_old = hardening * (eps0 + p) ** exponent
    flow_old = jnp.where(model < 0.5, voce_old, swift_old)
    plastic = q_elastic > flow_old
    voce_flow = sigma0 - hardening * jnp.expm1(-(p + dp) / eps0)
    swift_flow = hardening * (eps0 + p + dp) ** exponent
    flow = jnp.where(model < 0.5, voce_flow, swift_flow)
    voce_slope = hardening / eps0 * jnp.exp(-(p + dp) / eps0)
    swift_slope = hardening * exponent * (eps0 + p + dp) ** (exponent - 1)
    slope = jnp.where(model < 0.5, voce_slope, swift_slope)
    factor = 1 - 3 * shear * dp / q
    sigma33 = mean + factor * dev[..., 2]
    r0 = jnp.where(plastic, (q - 3 * shear * dp - flow) / young, dp)
    r1 = sigma33 / young
    # dq/de33 = 3G*s33_trial/q_trial.
    dq = 3 * shear * dev[..., 2] / q
    j00 = jnp.where(plastic, (-3 * shear - slope) / young, 1.)
    j01 = jnp.where(plastic, dq / young, 0.)
    j10 = -3 * shear * dev[..., 2] / (q * young)
    j11 = (bulk + factor * 4 * shear / 3 +
           3 * shear * dp / q**2 * dq * dev[..., 2]) / young
    jac = jnp.stack((jnp.stack((j00, j01), -1), jnp.stack((j10, j11), -1)), -2)
    return jnp.stack((r0, r1), -1), jac


def solve_2x2(matrix, rhs):
    """Solve nonsingular (...,2,2) systems without a general LU kernel."""
    a, b = matrix[..., 0, 0], matrix[..., 0, 1]
    c, d = matrix[..., 1, 0], matrix[..., 1, 1]
    det = a * d - b * c
    return jnp.stack((d * rhs[..., 0] - b * rhs[..., 1],
                      a * rhs[..., 1] - c * rhs[..., 0]), -1) / det[..., None]


def _solve_root(de, stress, p, theta, young, poisson):
    """Batched damped Newton with a dimensionless norm and nonnegative flow."""
    e33 = _elastic_guess(de, stress, young, poisson)
    z = jnp.stack((jnp.zeros_like(p), e33), axis=-1)
    tolerance = 1e-13

    def error(z):
        return jnp.max(jnp.abs(_system(z, de, stress, p, theta, young, poisson)[0]), axis=-1)

    def condition(carry):
        iteration, _, residual = carry
        return (iteration < 30) & jnp.any(residual > tolerance)

    def iteration(carry):
        count, z, previous = carry
        residual, jacobian = _system(z, de, stress, p, theta, young, poisson)
        correction = solve_2x2(jacobian, residual)
        active = previous > tolerance

        def backtrack(k, best):
            candidate = z - (0.5**k) * correction
            admissible = candidate[..., 0] >= 0.
            candidate_error = error(candidate)
            improve = active & admissible & jnp.isfinite(candidate_error) & (candidate_error < best[1])
            return (jnp.where(improve[..., None], candidate, best[0]),
                    jnp.where(improve, candidate_error, best[1]))

        new_z, new_error = jax.lax.fori_loop(0, 6, backtrack, (z, previous))
        return count + 1, new_z, new_error

    count, root, _ = jax.lax.while_loop(condition, iteration, (jnp.asarray(0), z, error(z)))
    return root, count


@jax.custom_jvp
def plane_stress_root(de, stress, p, theta, young, poisson):
    """Solve (Delta p, Delta e33); derivatives use the implicit local equations."""
    return _solve_root(de, stress, p, theta, young, poisson)[0]


@plane_stress_root.defjvp
def _root_jvp(primals, tangents):
    root = plane_stress_root(*primals)
    jacobian = _system(root, *primals)[1]
    _, rhs = jax.jvp(lambda *inputs: _system(root, *inputs)[0], primals, tangents)
    return root, -solve_2x2(jacobian, rhs)


def isotropic_plane_stress_increment(de, state, theta, young=210000., poisson=.3):
    """Advance batched points; state=(stress (...,4), p (...), log_thickness).

    Returns updated state and dimensionless convergence residual. Positive
    plastic increments and sigma33=0 are checked before accepting the state.
    """
    stress, p, thickness = state
    z = plane_stress_root(de, stress, p, theta, young, poisson)
    dp, e33 = z[..., 0], z[..., 1]
    mean, dev, q, shear, _ = _trial(de, stress, e33, young, poisson)
    factor = 1 - 3 * shear * dp / q
    new_stress = factor[..., None] * dev
    new_stress = new_stress + jnp.stack((mean, mean, mean, jnp.zeros_like(mean)), -1)
    residual = jnp.max(jnp.abs(_system(z, de, stress, p, theta, young, poisson)[0]), axis=-1)
    converged = jnp.isfinite(residual) & (residual <= 1e-12) & (dp >= 0.)
    new_stress = new_stress.at[..., 2].set(0.)
    return (new_stress, p + dp, thickness + e33), (converged, residual)
