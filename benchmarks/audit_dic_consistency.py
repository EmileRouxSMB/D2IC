"""Reproduce DIC consistency findings; observations, not acceptance tests."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from d2ic.numerics import global_cg as cg
from d2ic.numerics import local_gauss_newton as gn
jax.config.update('jax_enable_x64', True)

def run():
    image = jnp.arange(16.,dtype=jnp.float64).reshape(4,4)
    coords = jnp.array([[1.5,1.5]])
    result = {'bilinear_internal':float(cg._bilinear_sample(image,coords)[0])}
    if cg.flat_nd_linear_interpolate is not None:
        result['bilinear_dm_pix_current_call']=float(cg.flat_nd_linear_interpolate(image,coords.T)[0])
        result['bilinear_dm_pix_shifted_call']=float(cg.flat_nd_linear_interpolate(image,(coords-.5).T)[0])
    ref=jnp.zeros((4,4)); deformed=ref.at[2,1].set(10.)
    coords=jnp.array([[.5,1.5],[2.5,1.5]])
    ids=jnp.array([[0],[1]]); weights=jnp.ones((2,1)); u=jnp.zeros((2,2))
    f=lambda u:cg.J_pixelwise_core(u,ref,deformed,coords,ids,weights,interpolation='linear')
    result['cost_before_pixel_exit']=float(f(u))
    result['cost_after_bad_pixel_exit']=float(f(u.at[1,0].set(2.)))
    gx,gy=gn._compute_image_gradient_jax(deformed.T)
    r,_,_,_,valid=gn._pixel_state(u+10,ref,deformed,coords,ids,weights,gx.T,gy.T,interpolation='linear')
    result['local_valid_pixels_outside']=int(valid.sum())
    result['local_reported_rms_outside']=float(jnp.sqrt(jnp.sum(r*r)/jnp.maximum(valid.sum(),1)))
    # Pure regularization: compare local step to gradient of global energy.
    u=jnp.array([[0.,0.],[1.,0.]])
    idx=jnp.array([[1],[0]]); degree=jnp.ones(2,dtype=jnp.int32)
    w=jnp.ones((2,1)); nodal_w=jnp.array([1.,3.])
    grad=jax.grad(cg.reg_energy_spring_global)(u,idx,degree,w,nodal_w)
    local_g=nodal_w[:,None]*jnp.sum(w[...,None]*(u[:,None,:]-u[idx]),axis=1)
    result['global_regularization_gradient']=np.asarray(grad).tolist()
    result['local_regularization_gradient']=np.asarray(local_g).tolist()
    # NaN cannot be repaired by rejecting a line search if 0*NaN is applied.
    args=(jnp.zeros((1,2)),jnp.full((3,3),jnp.nan),jnp.ones((3,3)),
          jnp.array([[1.,1.]]),jnp.array([[0]]),jnp.ones((1,1)),
          jnp.zeros((1,0),dtype=jnp.int32),jnp.zeros(1,dtype=jnp.int32),
          jnp.zeros((1,0)),jnp.ones(1),0.,1,1e-6,'linear',False,False)
    out=cg._cg_solve(*args)[0]
    result['cg_output_finite_on_nonfinite_image']=bool(jnp.isfinite(out).all())
    return result

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
