"""Physics and implicit-sensitivity checks for the optimized isotropic kernels."""
import jax
import jax.numpy as jnp
import numpy as np
from d2ic.constitutive.isotropic_plane_stress import _system, isotropic_plane_stress_increment, _solve_root
from d2ic.constitutive.plane_stress import integrate_plane_stress_increment
from d2ic.constitutive.state import Hill48Material, initial_material_state
from d2ic.mechanics.isotropic_history import (isotropic_history, polar_log, prepare_history,
    integrate_prepared, q4_operators, gradients_from_displacement, forces_from_fields)
from d2ic.mechanics import deformation_gradient_history_q4
from d2ic.mechanics.equilibrium import assemble_internal_force_history_q4

jax.config.update('jax_enable_x64', True)
THETA=jnp.asarray([300.,300.,.2])


def test_analytic_local_jacobian():
    args=(jnp.asarray([.02,-.005,.01]),jnp.zeros(4),jnp.asarray(.01),THETA,210000.,.3)
    z=jnp.asarray([.015,-.012])
    automatic=jax.jacfwd(lambda z:_system(z,*args)[0])(z)
    np.testing.assert_allclose(_system(z,*args)[1],automatic,rtol=1e-12,atol=1e-12)


def test_specialized_matches_general_loading_unloading_shear():
    material=Hill48Material(jnp.asarray(210000.),jnp.asarray(.3),
                            jnp.asarray([.5,.5,.5,1.5,1.5,1.5]),jnp.r_[THETA,1.],'voce')
    general=initial_material_state()
    state=(jnp.zeros(4),jnp.asarray(0.),jnp.asarray(0.))
    for de in ([1e-5,-3e-6,0.],[.008,-.003,.002],[-.004,.001,0.],[0.,0.,.02]):
        de=jnp.asarray(de)
        general,_,ok,_=integrate_plane_stress_increment(de,general,material)
        state,(fast_ok,res)=isotropic_plane_stress_increment(de,state,THETA)
        assert bool(ok & fast_ok)
        np.testing.assert_allclose(state[0],general.stress[jnp.array([0,1,2,3])],rtol=1e-8,atol=1e-7)
        np.testing.assert_allclose(state[1],general.equivalent_plastic_strain,atol=1e-10)
        np.testing.assert_allclose(state[2],general.log_thickness_strain,atol=1e-10)
    root,count=_solve_root(de,jnp.zeros(4),jnp.asarray(0.),THETA,210000.,.3)
    assert int(count)<10


def test_rigid_rotations_stay_stress_free_and_derivatives_finite():
    for angle in (.5,1.57,2.4):
        R=jnp.asarray([[jnp.cos(angle),-jnp.sin(angle)],[jnp.sin(angle),jnp.cos(angle)]])
        F=jnp.stack((jnp.eye(2),R,jnp.eye(2)))[:,None,None]
        for substeps in (1,2,4):
            fields=isotropic_history(F,THETA,substeps)
            assert np.all(fields['converged'])
            np.testing.assert_allclose(fields['cauchy_stress'],0.,atol=1e-7)
            np.testing.assert_allclose(fields['equivalent_plastic_strain'],0.,atol=1e-12)
    jac=jax.jacfwd(lambda F:polar_log(F)[1])(jnp.eye(2))
    assert np.all(np.isfinite(jac))
    direction=jnp.asarray([[.1,.2],[.3,-.1]])
    np.testing.assert_allclose(jnp.einsum('ijkl,kl->ij',jac,direction),.5*(direction+direction.T),atol=1e-12)


def test_implicit_derivatives_match_central_differences():
    F=jnp.asarray([np.eye(2),np.diag([1.02,.994]),np.diag([1.01,.997])])[:,None,None]
    def cost(theta):
        return jnp.sum(isotropic_history(F,theta,2)['cauchy_stress']**2)
    exact=jax.jit(jax.grad(cost))(THETA)
    forward=jax.jit(jax.jacfwd(cost))(THETA)
    fd=[]
    for i in range(3):
        step=1e-5*float(THETA[i]);delta=jnp.eye(3)[i]*step
        fd.append((cost(THETA+delta)-cost(THETA-delta))/(2*step))
    np.testing.assert_allclose(exact,fd,rtol=2e-5,atol=1e-5)
    np.testing.assert_allclose(exact,forward,rtol=1e-10)
    def deformation_cost(x):
        trial=F.at[1,0,0,0,0].set(x)
        return jnp.sum(isotropic_history(trial,THETA,2)['cauchy_stress']**2)
    derivative=jax.jit(jax.grad(deformation_cost))(1.02)
    fd=(deformation_cost(1.0200001)-deformation_cost(1.0199999))/2e-7
    np.testing.assert_allclose(derivative,fd,rtol=1e-5)


def test_cached_suffix_is_exact_and_geometry_matches_existing_operators():
    nodes=jnp.asarray([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    elements=jnp.asarray([[0,1,2,3]])
    U=jnp.stack((jnp.zeros_like(nodes),nodes*jnp.asarray([.02,-.006]),nodes*jnp.asarray([.01,-.003])))
    gradients,detJ,_,weights=q4_operators(nodes,elements)
    F=gradients_from_displacement(U,elements,gradients)
    original=deformation_gradient_history_q4(nodes,elements,U)[0]
    np.testing.assert_allclose(F,original,atol=1e-14)
    de,Q=prepare_history(F,2)
    states,full=integrate_prepared(de,Q,THETA)
    _,suffix=integrate_prepared(de[1:],Q[1:],THETA,jax.tree.map(lambda x:x[1],states))
    for key in full:
        np.testing.assert_allclose(suffix[key],full[key][1:],atol=1e-12)
    fast=forces_from_fields(F[1:],full['cauchy_stress'],full['thickness_ratio'],elements,
                           gradients,detJ*weights*.1,jnp.zeros_like(nodes))
    old=assemble_internal_force_history_q4(nodes,elements,F[1:],full['cauchy_stress'],full['thickness_ratio'],1.,.1)
    np.testing.assert_allclose(fast,old,rtol=1e-12,atol=1e-12)


def test_elastic_identity_gradient_and_changed_suffix():
    F=jnp.broadcast_to(jnp.eye(2),(4,1,1,2,2))
    def stress_sum(x):
        trial=F.at[1,0,0,0,0].set(x)
        return jnp.sum(isotropic_history(trial,THETA,2)['cauchy_stress'][0])
    derivative=jax.jit(jax.grad(stress_sum))(1.)
    assert np.isfinite(derivative)
    fd=(stress_sum(1.+1e-7)-stress_sum(1.-1e-7))/2e-7
    np.testing.assert_allclose(derivative,fd,rtol=1e-6)
    F=F.at[1:,0,0,0,0].set(jnp.asarray([1.01,1.02,1.015]))
    de,Q=prepare_history(F,2)
    states,_=integrate_prepared(de,Q,THETA)
    trial=F.at[2,0,0,0,1].add(.003)
    de_new,Q_new=prepare_history(trial,2)
    _,full=integrate_prepared(de_new,Q_new,THETA)
    _,cached=integrate_prepared(de_new,Q_new,THETA,jax.tree.map(lambda x:x[1],states),start=1)
    for key in full:
        np.testing.assert_allclose(cached[key][1:],full[key][1:],atol=1e-10)



def test_specialized_swift_matches_general_model():
    parameters = jnp.asarray([1000., .005, .2])
    kernel = jnp.asarray([0., *parameters, 1.])
    material = Hill48Material(
        jnp.asarray(210000.), jnp.asarray(.3),
        jnp.asarray([.5, .5, .5, 1.5, 1.5, 1.5]),
        jnp.asarray([0., *parameters]), "swift",
    )
    general = initial_material_state()
    state = (jnp.zeros(4), jnp.asarray(0.), jnp.asarray(0.))
    for increment in ([.004, -.001, 0.], [.01, -.003, .002], [-.003, .001, 0.]):
        increment = jnp.asarray(increment)
        general, _, general_ok, _ = integrate_plane_stress_increment(increment, general, material)
        state, (fast_ok, _) = isotropic_plane_stress_increment(increment, state, kernel)
        assert bool(general_ok & fast_ok)
        np.testing.assert_allclose(
            state[0], general.stress[jnp.array([0, 1, 2, 3])], rtol=1e-8, atol=1e-7,
        )
        np.testing.assert_allclose(state[1], general.equivalent_plastic_strain, atol=1e-10)
