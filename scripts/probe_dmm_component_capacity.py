"""Bypass image matching and normal roots: frozen-weight 3D latent controls."""
import json
import traceback
import importlib.util
from dataclasses import replace
import numpy as np
import torch
from dmm_stall_common import *
scene=scene_at('initial')
spec=importlib.util.spec_from_file_location('original_dmm_surface_probe',PROJECT/'.runtime/dmm_r4_best_val_20261015/package/third_party/DMM/dmm/surface.py')
original=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=original
spec.loader.exec_module(original)
precision={}
for arch,state in scene.arches.items():
    event(stage='original_vs_repaired_mean_root',arch=arch)
    detail={}
    try:
        with torch.no_grad():
            chart=original.extract_chart(state,SURFACE)
            original.attach_chart(state,chart,SURFACE)
        detail['original_status']='PASSED'
    except Exception as error:
        detail['original_status']='FAILED';detail['original_error']=str(error)
    with torch.no_grad():
        chart=surface.extract_chart(state,SURFACE)
        box=torch.as_tensor(chart.sampling_domain,device=chart.anchors.device,dtype=chart.anchors.dtype)
        roots,offset=root_module.solve_normal_roots(state,chart.anchors,chart.normals,SURFACE,surface._field_gradient,box)
        double=state.query_model(roots)['sdf'];single=state.query_model(roots.float())['sdf']
    detail.update(repaired_float64_max_abs=float(double.abs().max()),float32_at_same_roots_max_abs=float(single.abs().max()),
        float32_at_same_roots_over_1e6=int((single.abs()>1e-6).sum()),max_offset_model=float(offset.abs().max()),vertices=len(roots))
    precision[arch]=detail;write('root_precision_control.json',precision);event(root_precision=detail,arch=arch)
poses=json.loads((OUT/'pose_vs_shape_3d.json').read_text('utf-8'))['oracle_initial_poses']
target=np.load(OUT/'oracle_capacity_targets.npz')
results=[]
for arch,chosen in (('upper',(11,27)),('lower',(31,47))):
    state=scene.arches[arch]
    pose=np.array(poses[arch])
    for label in chosen:
        event(stage='direct_3d_component_fit',arch=arch,label=label)
        q=state.q[str(label)]
        with torch.no_grad():
            q.zero_();state.base_pose.copy_(state.base_pose.new_tensor(pose));state.pose_delta.zero_()
        points=(target[str(label)+'_fit']-pose[:3,3])@pose[:3,:3]/50
        heldout=(target[str(label)+'_heldout']-pose[:3,3])@pose[:3,:3]/50
        fit_points=torch.tensor(points,device=q.device,dtype=torch.float64)
        heldout_points=torch.tensor(heldout,device=q.device,dtype=torch.float64)
        def field(x):return state.precise_decoder.component(x,state.codes()[label].double(),label)['sdf']
        def first_order(x):
            x=x.detach().requires_grad_(True)
            value=field(x)
            gradient=torch.autograd.grad(value.sum(),x,retain_graph=True)[0]
            return value/gradient.detach().norm(dim=-1).clamp_min(1e-3)
        def snapshot(phase):
            # Detached MC supplies a geometry check independent of normal roots.
            component=recovery._ComponentField(state,label)
            chart=surface.extract_chart(component,replace(SURFACE,resolution=64))
            v=chart.anchors.detach().cpu().numpy()*50@pose[:3,:3].T+pose[:3,3]
            np.savez_compressed(OUT/f'capacity_{label}_{phase}.npz',vertices=v,faces=chart.faces.detach().cpu().numpy())
        row=dict(arch=arch,label=label,scope='DIAGNOSTIC ORACLE: fixed centroid-Kabsch 3D pose, component field only, original frozen weights; no image objective, no normal roots')
        # Independent decoder finite difference along one latent coordinate.
        value=field(heldout_points[:32]).mean()
        derivative=torch.autograd.grad(value,q)[0][0].item()
        step=1e-3
        with torch.no_grad():
            q[0]=step;positive=float(field(heldout_points[:32]).mean());q[0]=-step;negative=float(field(heldout_points[:32]).mean());q.zero_()
        row['latent_derivative_check']=dict(autograd=derivative,central_difference=(positive-negative)/(2*step))
        try:
            try:snapshot('before')
            except Exception as error:row['mesh_before_error']=str(error)
            row['before_heldout_first_order_abs_mm']=float(first_order(heldout_points).detach().abs().mean()*50)
            optimizer=torch.optim.Adam([q],lr=.05)
            history=[]
            for iteration in range(80):
                optimizer.zero_grad(set_to_none=True)
                distance=first_order(fit_points)
                loss=distance.square().mean()+1e-5*q.square().mean()
                loss.backward();optimizer.step()
                with torch.no_grad():
                    rms=q.square().mean().sqrt()
                    if rms>4:q.mul_(4/rms)
                if iteration%10==0 or iteration==79:
                    history.append(dict(iteration=iteration,loss=float(loss.detach()),q_rms=float(q.square().mean().sqrt())))
            row['after_heldout_first_order_abs_mm']=float(first_order(heldout_points).detach().abs().mean()*50)
            row['q']=q.detach().cpu().tolist();row['history']=history
            try:snapshot('after')
            except Exception as error:row['mesh_after_error']=str(error)
            row['status']='COMPLETE'
        except Exception:
            row['status']='FAILED_RETAINED';row['traceback']=traceback.format_exc()
        results.append(row);write('component_capacity.json',dict(results=results,quality_accepted=False))
        event(completed={k:v for k,v in row.items() if k not in ('history','q','scope')})
        with torch.no_grad():q.zero_()
