"""Optimize shared pose with the perturbed DMM meshes held completely fixed."""
from __future__ import annotations
import json
import numpy as np
from scipy.optimize import least_squares
from joint_common import pose_matrix, right_jacobian, skew
from shape_only_common import project_and_jacobian, make_pairs
from run_pose_only import render_masks, mean_iou, save_render_comparison, log


class FixedShapePoseObjective:
    def __init__(self,pairs,cameras,initial_pose,pose_scale):
        self.points=np.concatenate([p[0] for p in pairs.values()])
        self.targets=np.concatenate([p[1] for p in pairs.values()])
        self.indices=np.concatenate([p[2] for p in pairs.values()])
        self.cameras,self.initial_pose,self.pose_scale=cameras,initial_pose,pose_scale

    def residual(self,u):
        pixels,_=project_and_jacobian(self.points,self.cameras,self.indices,pose_matrix(self.initial_pose+u*self.pose_scale))
        return (pixels-self.targets).ravel()

    def jacobian(self,u):
        pose=self.initial_pose+u*self.pose_scale
        matrix=pose_matrix(pose)
        _,J=project_and_jacobian(self.points,self.cameras,self.indices,matrix)
        rotation=-np.einsum('nij,njk->nik',J,skew(self.points))@right_jacobian(pose[:3])
        translation=J@matrix[:3,:3].T
        return (np.concatenate([rotation,translation],axis=2)*self.pose_scale).reshape(-1,6)


def check_gradient(objective,u):
    J=objective.jacobian(u)
    records=[]
    for axis in range(6):
        for step in [1e-3,1e-4]:
            d=np.eye(6)[axis]*step
            finite=(objective.residual(u+d)-objective.residual(u-d))/(2*step)
            analytic=J[:,axis]
            error=float(np.linalg.norm(finite-analytic)/max(np.linalg.norm(finite),np.linalg.norm(analytic),1e-10))
            records.append({'axis':axis,'epsilon_normalized':step,'relative_error':error})
    return {'max_relative_error':max(r['relative_error'] for r in records),'directions':records,
        'scope':'fixed perturbed mesh; pose axes; frozen visibility and contour pairs'}


def optimize_pose(root,meshes,cameras,targets,labels,initial_pose,pose_scale,u,bound,local_step,config):
    opts=config['pose_warmup']
    masks,maps=render_masks(meshes,cameras,initial_pose+u*pose_scale,True)
    start_iou=mean_iou(masks,targets,labels)[0]
    best_u=u.copy(); best_iou=start_iou; best_iteration=0
    history=[]; stop_reason='outer_iteration_limit'
    shape_copies={k:(v[0].copy(),v[1].copy()) for k,v in meshes.items()}
    for outer in range(opts['outer_iterations']):
        pose=initial_pose+u*pose_scale
        pairs=make_pairs(masks,maps,targets,cameras,labels,pose_matrix(pose),config['boundary_samples_per_direction'])
        objective=FixedShapePoseObjective(pairs,cameras,initial_pose,pose_scale)
        if outer==0:
            gradient=check_gradient(objective,u)
            gradient['status']='POSE_WARMUP_GRADIENT_PASS' if gradient['max_relative_error']<=config['acceptance']['max_gradient_relative_error'] else 'POSE_WARMUP_GRADIENT_FAIL'
            (root/'warmup_gradient_check.json').write_text(json.dumps(gradient,indent=2))
            if gradient['status']!='POSE_WARMUP_GRADIENT_PASS':
                raise ValueError('warm-up pose derivative failed')
            log(root,f'{gradient["status"]} max_relative_error={gradient["max_relative_error"]:.3g}')
        solution=least_squares(objective.residual,u,jac=objective.jacobian,
            bounds=(np.maximum(-bound,u-local_step),np.minimum(bound,u+local_step)),
            loss='soft_l1',f_scale=1,max_nfev=opts['inner_max_nfev'],ftol=1e-7,xtol=1e-7,gtol=1e-7)
        change=float(np.linalg.norm(solution.x-u)); u=solution.x
        pose=initial_pose+u*pose_scale
        masks,maps=render_masks(meshes,cameras,pose,True)
        iou,per_view,micro=mean_iou(masks,targets,labels)
        record={'outer_iteration':outer+1,'pose':pose.tolist(),'active_mean_tooth_iou':iou,'per_view':per_view,
            'solver_success':bool(solution.success),'solver_nfev':solution.nfev,'cost':float(solution.cost),
            'normalized_pose_update_norm':change,'fixed_shape':True}
        history.append(record)
        log(root,f'POSE_WARMUP_ITER {outer+1} active_iou={iou:.6f} pose_update={change:.3g}',record)
        if iou>best_iou+opts['best_iou_epsilon']:
            best_iou=iou; best_u=u.copy(); best_iteration=outer+1
        np.savez_compressed(root/'warmup_latest_parameters.npz',pose_parameters=best_u)
        if outer+1>=opts['minimum_outer_iterations'] and outer+1-best_iteration>=opts['stop_after_no_best_iterations']:
            stop_reason='fitting_iou_plateau'; break
    unchanged=all(np.array_equal(shape_copies[k][0],v[0]) and np.array_equal(shape_copies[k][1],v[1]) for k,v in meshes.items())
    if not unchanged:
        raise ValueError('warm-up modified fixed shape')
    pose=initial_pose+best_u*pose_scale
    masks,maps=render_masks(meshes,cameras,pose,True)
    save_render_comparison(root,'warmup',masks,targets,cameras,labels)
    result={'status':'POSE_WARMUP_COMPLETED','initial_iou':start_iou,'active_mean_tooth_iou':best_iou,
        'pose':pose.tolist(),'best_iteration':best_iteration,'iterations':history,'stop_reason':stop_reason,
        'fixed_perturbed_meshes_unchanged':unchanged,'active_codes_optimized':False,
        'selection':'best fitting-view IoU only; no truth or heldout access'}
    (root/'warmup_report.json').write_text(json.dumps(result,indent=2))
    (root/'warmup_pose.json').write_text(json.dumps({'pose':pose.tolist(),'arch_to_world':pose_matrix(pose).tolist()},indent=2))
    return best_u,masks,maps,result
