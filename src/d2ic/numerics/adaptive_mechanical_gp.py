"""JAX kernels for a GP-adaptive weak mechanical residual penalty."""

import jax
import jax.numpy as jnp


G_MIN = -20.0
G_MAX = 20.0
KERNEL_JITTER = 1.0e-8


@jax.jit
def assemble_linear_elastic_residual_q4(
    displacement, elements, grad, area_weights, elasticity_tensor, active_dofs
):
    """Assemble masked Q4 weak residual for small-strain linear elasticity.

    Parameters use displacement `(Nn,2)` [length], connectivity `(Ne,4)`,
    reference gradients `(Ne,Ng,4,2)` [1/length], and integration weights
    `(Ne,Ng)` [length²]. `elasticity_tensor` is the explicit plane constitutive
    tensor `(2,2,2,2)` [stress]; pass plane-stress or plane-strain constants
    from the caller. The returned residual `(Nn,2)` is force per unit thickness.
    `active_dofs` `(Nn,2)` excludes boundary/reaction components from inference.
    """
    du = displacement[elements]
    displacement_gradient = jnp.einsum("eai,egaj->egij", du, grad)
    strain = 0.5 * (
        displacement_gradient + jnp.swapaxes(displacement_gradient, -1, -2)
    )
    stress = jnp.einsum("ijkl,egkl->egij", elasticity_tensor, strain)
    element_force = jnp.einsum("egij,egaj,eg->eai", stress, grad, area_weights)
    residual = jnp.zeros_like(displacement).at[elements].add(element_force)
    return jnp.where(active_dofs, residual, 0.0)


@jax.jit
def rbf_covariance(x, y, length_scale, amplitude=1.0):
    """Squared-exponential covariance; coordinates and length scale share units."""
    delta = (x[:, None, :] - y[None, :, :]) / length_scale
    return amplitude**2 * jnp.exp(-0.5 * jnp.sum(delta * delta, axis=-1))


@jax.jit
def gp_log_variance(x, inducing_points, inducing_values, mean, length_scale,
                    amplitude=1.0):
    """Interpolate latent `g=log(s²)` from inducing values (all `(N,2)`/`(M,)`)."""
    k_xz = rbf_covariance(x, inducing_points, length_scale, amplitude)
    k_zz = rbf_covariance(inducing_points, inducing_points, length_scale, amplitude)
    k_zz = k_zz + KERNEL_JITTER * jnp.eye(inducing_points.shape[0], dtype=x.dtype)
    # ``inducing_values`` are values of g, not zero-mean GP coefficients.
    weights = jnp.linalg.solve(k_zz, inducing_values - mean)
    return jnp.clip(mean + k_xz @ weights, G_MIN, G_MAX)


@jax.jit
def mechanical_residual_nll(residual, log_variance, active_dofs, force_scale):
    """Normalized Gaussian NLL over active force components, including log variance.

    `residual` and `active_dofs` have shape `(Nn,2)`, `log_variance` `(Nn,)`;
    `force_scale` has residual force units and is strictly positive.
    """
    d = jnp.sum(active_dofs, axis=-1)
    squared = jnp.sum(jnp.where(active_dofs, (residual / force_scale) ** 2, 0.0), axis=-1)
    return 0.5 * jnp.sum(jnp.exp(-log_variance) * squared + d * log_variance)


@jax.jit
def gp_inducing_prior(inducing_values, inducing_points, mean, length_scale,
                      amplitude=1.0):
    """Half negative log GP prior for inducing values around constant `mean`."""
    k_zz = rbf_covariance(inducing_points, inducing_points, length_scale, amplitude)
    k_zz = k_zz + KERNEL_JITTER * jnp.eye(inducing_points.shape[0], dtype=inducing_points.dtype)
    centered = inducing_values - mean
    return 0.5 * centered @ jnp.linalg.solve(k_zz, centered)


@jax.jit
def mechanical_confidence(log_variance):
    """Return `lambda=1/s²`; clipping keeps weights in [exp(-20), exp(20)]."""
    return jnp.exp(-jnp.clip(log_variance, G_MIN, G_MAX))


@jax.jit
def laplace_log_variance_uncertainty(
    residual, active_dofs, force_scale, x, inducing_points, inducing_values,
    mean, length_scale, amplitude=1.0, prior_weight=1.0, objective_scale=1.0,
):
    """Approximate marginal variance of `g(x)` by a conditional GP Laplace model.

    The small `(M,M)` Hessian is differentiated from NLL plus GP prior. The
    returned `(Nn,)` variance adds conditional GP variance to the projected
    inverse-Hessian term and is floored at zero for roundoff. It is conditional
    on the optimized displacement; ``objective_scale`` applies the same
    likelihood tempering as the coupled DIC objective.
    """
    def objective(values):
        g = gp_log_variance(x, inducing_points, values, mean, length_scale, amplitude)
        posterior = mechanical_residual_nll(residual, g, active_dofs, force_scale) + prior_weight * gp_inducing_prior(
            values, inducing_points, mean, length_scale, amplitude
        )
        return objective_scale * posterior

    hessian = jax.hessian(objective)(inducing_values)
    hessian = hessian + KERNEL_JITTER * jnp.eye(inducing_points.shape[0], dtype=x.dtype)
    covariance_inducing = jnp.linalg.solve(hessian, jnp.eye(inducing_points.shape[0], dtype=x.dtype))
    k_xz = rbf_covariance(x, inducing_points, length_scale, amplitude)
    k_zz = rbf_covariance(inducing_points, inducing_points, length_scale, amplitude)
    k_zz = k_zz + KERNEL_JITTER * jnp.eye(inducing_points.shape[0], dtype=x.dtype)
    projection = jnp.linalg.solve(k_zz, k_xz.T).T
    conditional = amplitude**2 - jnp.sum(projection * k_xz, axis=1)
    projected = jnp.einsum("ni,ij,nj->n", projection, covariance_inducing, projection)
    return jnp.maximum(conditional + projected, 0.0)
