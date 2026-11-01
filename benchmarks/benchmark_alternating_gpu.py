"""Reproducible before/after kernels, sensitivities and photo precision benchmark.

Run with XLA_PYTHON_CLIENT_PREALLOCATE=false in the science environment.
The general Hill48 solver is the reference; both kernels use the SAME objective
rotation/stretch substep path. Compilation is excluded from execution timings.
"""
from pathlib import Path
import argparse
import json
import platform
import sys
import time

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'IdentVirual'))
import alternating_identification as a
from d2ic.mechanics.isotropic_history import (
    inverse_2x2, polar_log, rotation_matrix, symmetric_exp, prepare_history,
    integrate_prepared, q4_operators, gradients_from_displacement, forces_from_fields,
)

jax.config.update('jax_enable_x64', True)


def measure(function, arguments, repeats):
    """Compile explicitly, warm up, then time synchronized execution."""
    start = time.perf_counter()
    compiled = jax.jit(function).lower(*arguments).compile()
    compile_seconds = time.perf_counter() - start
    value = jax.block_until_ready(compiled(*arguments))
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = jax.block_until_ready(compiled(*arguments))
        timings.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    result = dict(compile_seconds=compile_seconds, median_seconds=float(np.median(timings)),
                  min_seconds=min(timings), samples_seconds=timings)
    if memory is not None:
        result['compiled_memory_bytes'] = {name: getattr(memory, name) for name in
            ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
    return result, value


def objective_substeps(F, substeps):
    """Identical prescribed intermediate gradients for the two material solvers."""
    angle, stretch = polar_log(F[1:] @ inverse_2x2(F[:-1]))
    alpha = jnp.arange(1,substeps+1,dtype=F.dtype)/substeps
    intermediate = (rotation_matrix(angle[:,None]*alpha[None,:,None,None]) @
                    symmetric_exp(stretch[:,None]*alpha[None,:,None,None,None,None]) @ F[:-1,None])
    return jnp.concatenate((F[:1],intermediate.reshape((-1,)+F.shape[1:])))


def case(nx, ny, frames):
    x,y=np.meshgrid(np.linspace(0.,40.,nx+1),np.linspace(0.,80.,ny+1))
    nodes=np.column_stack((x.ravel(),y.ravel()))
    elements=[]
    for j in range(ny):
        for i in range(nx):
            k=j*(nx+1)+i
            elements.append([k,k+1,k+nx+2,k+nx+1])
    elements=np.asarray(elements,np.int32)
    U=np.zeros((frames+1,len(nodes),2))
    for t,strain in enumerate(np.linspace(0.,.025,frames+1)):
        U[t,:,0]=-.3*strain*nodes[:,0]
        U[t,:,1]=strain*nodes[:,1]
    return jnp.asarray(nodes),jnp.asarray(elements),jnp.asarray(U)


def run(args):
    if args.real_data:
        data=a.load_dic_history_npz(args.real_data,prepend_reference=True)
        nodes,elements,U=map(jnp.asarray,(data.nodes_xy,data.elements,data.displacement))
        args.frames=len(U)-1
    else:
        nodes,elements,U=case(args.nx,args.ny,args.frames)
    theta=jnp.asarray([300.,300.,.2])
    gradients,detJ,_,weights=q4_operators(nodes,elements)
    measures=detJ*weights*.1
    F=gradients_from_displacement(U,elements,gradients)
    de,Q=prepare_history(F,args.substeps)
    states,base_fields=integrate_prepared(de,Q,theta)
    base_forces=forces_from_fields(F[1:],base_fields['cauchy_stress'],base_fields['thickness_ratio'],
                                  elements,gradients,measures,nodes)
    frame=max(1,args.frames//2+1)
    prefix=jax.tree.map(lambda x:x[frame-1],states)

    def legacy(F,theta):
        refined=objective_substeps(F,args.substeps)
        _,fields=a.integrate_gauss_point_histories(refined,a.make_material(theta),jnp.asarray(0.))
        return fields['cauchy_stress'][args.substeps-1::args.substeps], jnp.all(fields['converged'])

    def optimized(F,theta):
        fields=a.constitutive_history(F,theta,args.substeps)
        return fields['cauchy_stress'], jnp.all(fields['converged'])

    def prepared(de,Q,theta):
        fields=integrate_prepared(de,Q,theta)[1]
        return fields['cauchy_stress'], jnp.all(fields['converged'])

    def nodal_cost(delta_u, theta, suffix=False):
        trial=U.at[frame].add(delta_u)
        trial_F=gradients_from_displacement(trial,elements,gradients)
        trial_de,trial_Q=prepare_history(trial_F,args.substeps)
        _,fields=integrate_prepared(trial_de,trial_Q,theta,
                                   prefix if suffix else None, start=frame-1 if suffix else 0)
        forces=forces_from_fields(trial_F[1:],fields['cauchy_stress'],fields['thickness_ratio'],
                                 elements,gradients,measures,nodes)
        if suffix:
            forces=jnp.where((jnp.arange(args.frames)>=frame-1)[:,None,None],forces,base_forces)
        return .5*jnp.mean((forces/100.)**2)

    zero=jnp.zeros_like(nodes)
    results={}
    results['general_hill48'],legacy_stress=measure(legacy,(F,theta),args.repeats)
    print('reference material',results['general_hill48']['median_seconds'],flush=True)
    results['isotropic_implicit'],fast_stress=measure(optimized,(F,theta),args.repeats)
    results['prepared_kinematics'],_=measure(prepared,(de,Q,theta),args.repeats)
    if not bool(legacy_stress[1]) or not bool(fast_stress[1]):
        raise RuntimeError('Benchmark requires converged reference and optimized solutions.')
    results['stress_max_abs_error_MPa']=float(jnp.max(jnp.abs(fast_stress[0]-legacy_stress[0])))
    np.testing.assert_allclose(fast_stress[0],legacy_stress[0],rtol=1e-7,atol=1e-6)
    results['material_speedup']=results['general_hill48']['median_seconds']/results['isotropic_implicit']['median_seconds']
    results['preparation_reuse_speedup']=results['isotropic_implicit']['median_seconds']/results['prepared_kinematics']['median_seconds']

    full=lambda delta,theta:nodal_cost(delta,theta,False)
    suffix=lambda delta,theta:nodal_cost(delta,theta,True)
    results['implicit_full_gradient'],full_g=measure(jax.value_and_grad(full),(zero,theta),args.repeats)
    results['implicit_suffix_gradient'],suffix_g=measure(jax.value_and_grad(suffix),(zero,theta),args.repeats)
    results['suffix_gradient_max_error']=float(jnp.max(jnp.abs(full_g[1]-suffix_g[1])))
    results['suffix_speedup']=results['implicit_full_gradient']['median_seconds']/results['implicit_suffix_gradient']['median_seconds']

    step=1e-3
    direction_count=min(18,nodes.size)
    directions=(jnp.eye(nodes.size,dtype=nodes.dtype)[:direction_count]*step).reshape(
        (direction_count,)+nodes.shape
    )
    def finite_difference(theta):
        def pair(delta):
            return (full(delta,theta)-full(-delta,theta))/(2*step)
        return jax.lax.map(pair,directions,batch_size=2)
    results['finite_difference_gradient'],fd=measure(finite_difference,(theta,),args.repeats)
    projected_gradient=jax.vmap(lambda direction:jnp.vdot(full_g[1],direction/step))(directions)
    results['gradient_relative_error']=float(jnp.linalg.norm(fd-projected_gradient)/jnp.maximum(jnp.linalg.norm(fd),1e-30))
    results['implicit_derivative_speedup']=results['finite_difference_gradient']['median_seconds']/results['implicit_full_gradient']['median_seconds']

    # Tangent tests for the actual three-parameter fitting path.
    response=lambda theta:prepared(de,Q,theta)[0].reshape(-1)
    def material_fd(theta):
        def one(i):
            h=1e-5*theta[i];delta=jnp.eye(3)[i]*h
            return (response(theta+delta)-response(theta-delta))/(2*h)
        return jax.vmap(one)(jnp.arange(3)).T
    results['material_parameter_fd'],param_fd=measure(material_fd,(theta,),args.repeats)
    results['material_parameter_jvp'],param_jvp=measure(jax.jacfwd(response),(theta,),args.repeats)
    results['material_parameter_relative_error']=float(jnp.linalg.norm(param_fd-param_jvp)/jnp.linalg.norm(param_fd))

    results['batches']={}
    for batch in (1,2,4,8):
        batched=jax.vmap(optimized,in_axes=(0,None))
        stats,_=measure(batched,(jnp.broadcast_to(F,(batch,)+F.shape),theta),args.repeats)
        stats['seconds_per_history']=stats['median_seconds']/batch
        results['batches'][str(batch)]=stats

    # Same coordinates and pixels; only bilinear weights/intensities change precision.
    rng=np.random.default_rng(123)
    image=rng.random((1920,1200)).astype(np.float32)
    coords=np.column_stack((rng.uniform(2,1197,65000),rng.uniform(2,1917,65000)))
    ids=jnp.zeros((len(coords),1),dtype=jnp.int32)
    shape=jnp.ones((len(coords),1))
    ref=jnp.asarray(rng.random(len(coords)))
    support=jnp.ones(len(coords),dtype=bool)
    def photo(u,img):
        return a.fixed_support_photo(u,img,jnp.asarray(coords),ids,shape,ref,support)[0]
    u=jnp.asarray([[.003,-.002]])
    results['photo_float64'],p64=measure(jax.value_and_grad(photo),(u,jnp.asarray(image,dtype=jnp.float64)),args.repeats)
    results['photo_mixed'],p32=measure(jax.value_and_grad(photo),(u,jnp.asarray(image)),args.repeats)
    results['photo_host_input'],_=measure(jax.value_and_grad(photo),(u,image),args.repeats)
    results['photo_residency_speedup']=results['photo_host_input']['median_seconds']/results['photo_mixed']['median_seconds']
    results['photo_cost_abs_error']=float(abs(p64[0]-p32[0]))
    results['photo_gradient_relative_error']=float(jnp.linalg.norm(p64[1]-p32[1])/jnp.linalg.norm(p64[1]))
    results['photo_speedup']=results['photo_float64']['median_seconds']/results['photo_mixed']['median_seconds']

    results['environment']=dict(jax=jax.__version__,jaxlib=jaxlib.__version__,python=platform.python_version(),
        devices=[str(device) for device in jax.devices()],device_kind=jax.devices()[0].device_kind,
        platform=jax.devices()[0].platform,x64=jax.config.x64_enabled)
    results['case']=dict(frames=args.frames,elements=len(elements),gauss_points=4,
        substeps=args.substeps,nodal_dofs=int(nodes.size),fd_directions=direction_count,
        suffix_first_frame=frame,repeats=args.repeats,
        baseline='general Hill48 on identical objective substep path',
        timing='separate compilation, warmup, synchronized median',
        memory='XLA compiled buffer estimates; not whole-process peak VRAM',
        gradient_objective='0.5 * mean((nodal_force / 100 N)**2), all nodal DOFs',
        parameter_jacobian='all Cauchy stress components versus 3 material parameters',
        photo_case='65000 fixed random samples, reproducible random 1920x1200 image')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    for key in ('material_speedup','preparation_reuse_speedup','implicit_derivative_speedup',
                'suffix_speedup','stress_max_abs_error_MPa','gradient_relative_error','photo_speedup'):
        print(key,results[key],flush=True)
    print('saved',args.output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-data',type=Path,default=None)
    parser.add_argument('--nx',type=int,default=4)
    parser.add_argument('--ny',type=int,default=8)
    parser.add_argument('--frames',type=int,default=4)
    parser.add_argument('--substeps',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('alternating_gpu_results.json'))
    run(parser.parse_args())
