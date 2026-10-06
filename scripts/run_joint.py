"""Minimal Joint fit: no active truth codes, true pose, source meshes or depth."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from run_forward_check import PROJECT_ROOT, sha256, save_ply
from run_pose_only import log, render_masks, mean_iou, save_render_comparison, target_features, centroid_residual
from shape_only_common import load_model, decode, make_pairs
from joint_common import JointImageObjective, pose_matrix, gradient_check


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/joint_minimal.json')
    parser.add_argument('--fit-input',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    fit=args.fit_input.resolve(strict=True)
    root=PROJECT_ROOT/'runs'/(dt.datetime.now(dt.timezone.utc).strftime('joint_%Y%m%dT%H%M%SZ_')+sha256(args.config)[:8])
    root.mkdir(exist_ok=False,parents=True)
    started=time.perf_counter()
    try:
        log(root,'START minimal Joint')
        manifest=json.loads((fit/'manifest.json').read_text())
        if manifest['status']!='JOINT_SYNTHETIC_INPUT_READY' or manifest['frozen_labels']:
            raise ValueError('expected active-only Joint input')
        if any(k in manifest for k in ['fixed_arch_to_world','ground_truth_arch_to_world']):
            raise ValueError('Joint fit input must not contain true/fixed pose')
        for name,digest in manifest['extra_input_sha256'].items():
            if sha256(fit/name)!=digest:
                raise ValueError(f'input changed: {name}')
        if sha256(fit/manifest['camera_file'])!=manifest['camera_sha256']:
            raise ValueError('camera changed')
        cameras=json.loads((fit/manifest['camera_file']).read_text())
        targets={}
        for item in manifest['masks']:
            if sha256(fit/item['path'])!=item['sha256']:
                raise ValueError('mask changed')
            targets[item['camera']]=np.asarray(Image.open(fit/item['path']))
        initial=dict(np.load(fit/manifest['initial_codes']))
        scales=dict(np.load(fit/manifest['training_scales']))
        boxes=json.loads((fit/manifest['sampling_boxes']).read_text())
        pose_data=json.loads((fit/manifest['initial_pose']).read_text())
        initial_pose=np.r_[pose_data['rotation_vector_radians'],pose_data['translation_dmm']]
        labels=manifest['active_labels']
        if labels!=config['active_labels'] or set(initial)!={f'label_{k}' for k in labels} or initial_pose.shape!=(6,) or not np.isfinite(initial_pose).all():
            raise ValueError('invalid Joint parameter input')
        pose_scale=np.r_[np.repeat(np.deg2rad(config['pose_rotation_scale_degrees']),3),np.repeat(config['pose_translation_scale_dmm'],3)]
        dimensions=6+sum(len(initial[f'label_{k}']) for k in labels)
        x=np.zeros(dimensions)
        bound=np.r_[np.repeat(config['pose_rotation_bound_degrees']/config['pose_rotation_scale_degrees'],3),
            np.repeat(config['pose_translation_bound_dmm']/config['pose_translation_scale_dmm'],3),
            np.repeat(config['global_bound_training_std'],dimensions-6)]
        local_step=np.r_[np.repeat(config['local_pose_rotation_step_degrees']/config['pose_rotation_scale_degrees'],3),
            np.repeat(config['local_pose_translation_step_dmm']/config['pose_translation_scale_dmm'],3),
            np.repeat(config['local_step_training_std'],dimensions-6)]
        code_names=['run_joint.py','joint_common.py','shape_only_common.py','run_forward_check.py','run_pose_only.py','prepare_joint.py']
        snapshot=root/'code_snapshot'; snapshot.mkdir()
        for name in code_names:
            shutil.copyfile(PROJECT_ROOT/'scripts'/name,snapshot/name)
        dmm=Path(config['dmm_root'])
        dmm_sources=list((dmm/'networks').glob('*.py'))+list((dmm/'utils').glob('*.py'))
        provenance={'run_id':root.name,'fit_input':str(fit),'fit_manifest_sha256':sha256(fit/'manifest.json'),
            'config_sha256':sha256(args.config),'model_sha256':manifest['model_sha256'],'checkpoint_epoch':manifest['checkpoint_epoch'],
            'active_truth_codes_read':False,'ground_truth_pose_read':False,'source_meshes_read':False,'depth_maps_read':False,
            'objective_inputs_only':['packaged masks, known cameras, perturbed initial codes/pose, training scales, sampling boxes','frozen DMM weights'],
            'diagnostic_oracle_conditions':manifest['diagnostic_oracle_conditions'],'active_labels':labels,'frozen_labels':[],
            'scene':manifest['scene'],'coarse_alignment':'image centroid fit using perturbed initial meshes, then full coupled solve',
            'code_sha256':{name:sha256(PROJECT_ROOT/'scripts'/name) for name in code_names},
            'dmm_source_sha256':{str(p):sha256(p) for p in dmm_sources},'python':sys.executable,'device':'cpu','sdf_dtype':'float64'}
        (root/'provenance.json').write_text(json.dumps(provenance,indent=2))
        (root/'resolved_config.json').write_text(json.dumps(config,indent=2))
        np.savez_compressed(root/'initial_codes.npz',**initial)
        (root/'initial_pose.json').write_text(json.dumps({'pose':initial_pose.tolist(),'arch_to_world':pose_matrix(initial_pose).tolist()},indent=2))
        net=load_model(config,manifest)
        meshes={}
        (root/'meshes/initial').mkdir(parents=True)
        for label in labels:
            meshes[label]=decode(net,label,initial[f'label_{label}'],boxes[str(label)])
            save_ply(root/'meshes/initial'/f'tooth{label}.ply',*meshes[label])
        masks,_=render_masks(meshes,cameras,initial_pose)
        iou,per_view,micro=mean_iou(masks,targets,labels)
        save_render_comparison(root,'initial',masks,targets,cameras,labels)
        initial_metrics={'pose':initial_pose.tolist(),'active_mean_tooth_iou':iou,'per_view':per_view,'micro_tooth_iou':micro}
        log(root,f'INITIAL_JOINT_IOU {iou:.6f}')
        features=target_features(targets,cameras,labels)
        centers={k:v[0].mean(axis=0).astype(float) for k,v in meshes.items()}
        def coarse_residual(u):
            pose=initial_pose+u*pose_scale
            return np.r_[centroid_residual(pose,cameras,labels,centers,features),config['centroid_initial_pose_damping']*u]
        coarse=least_squares(coarse_residual,x[:6],bounds=(-bound[:6],bound[:6]),max_nfev=100,loss='soft_l1',f_scale=5)
        x[:6]=coarse.x
        pose=initial_pose+x[:6]*pose_scale
        masks,maps=render_masks(meshes,cameras,pose,True)
        coarse_iou,per_view,micro=mean_iou(masks,targets,labels)
        save_render_comparison(root,'centroid',masks,targets,cameras,labels)
        coarse_metrics={'pose':pose.tolist(),'active_mean_tooth_iou':coarse_iou,'solver_nfev':coarse.nfev,'solver_success':bool(coarse.success)}
        log(root,f'CENTROID_IOU {coarse_iou:.6f}')
        pairs=make_pairs(masks,maps,targets,cameras,labels,pose_matrix(pose),config['boundary_samples_per_direction'])
        objective=JointImageObjective(net,labels,initial,scales,pairs,cameras,initial_pose,pose_scale,x,config['initial_code_damping'])
        check=gradient_check(objective,x)
        check['status']='JOINT_IMAGE_GRADIENT_PASS' if check['max_relative_error']<=config['acceptance']['max_gradient_relative_error'] else 'JOINT_IMAGE_GRADIENT_FAIL'
        (root/'gradient_check.json').write_text(json.dumps(check,indent=2))
        log(root,f'{check["status"]} max_relative_error={check["max_relative_error"]:.3g}')
        if check['status']!='JOINT_IMAGE_GRADIENT_PASS':
            raise ValueError('Joint derivative failed')
        best_x=x.copy() if coarse_iou>iou else np.zeros(dimensions)
        best_iou=max(iou,coarse_iou); best_iteration=0
        history=[]
        for outer in range(config['outer_iterations']):
            if outer:
                pose=initial_pose+x[:6]*pose_scale
                pairs=make_pairs(masks,maps,targets,cameras,labels,pose_matrix(pose),config['boundary_samples_per_direction'])
                objective=JointImageObjective(net,labels,initial,scales,pairs,cameras,initial_pose,pose_scale,x,config['initial_code_damping'])
            step=local_step.copy(); rejected=[]
            for retry in range(config['local_trust_retries']+1):
                try:
                    solution=least_squares(objective.residual,x,jac=objective.jacobian,
                        bounds=(np.maximum(-bound,x-step),np.minimum(bound,x+step)),loss='soft_l1',f_scale=1,
                        max_nfev=config['inner_max_nfev'],ftol=1e-5,xtol=1e-5,gtol=1e-5)
                    break
                except ValueError as error:
                    if str(error) not in ['local surface root failed to converge','singular implicit surface derivative'] or retry>=config['local_trust_retries']:
                        raise
                    rejected.append({'reason':str(error),'step':step.tolist()}); step*=.5
                    log(root,f'JOINT_LOCAL_STEP_RETRY {retry+1} reason={error}')
            x=solution.x
            for label in labels:
                key=f'label_{label}'
                meshes[label]=decode(net,label,initial[key]+x[objective.slices[label]]*scales[key],boxes[str(label)])
            pose=initial_pose+x[:6]*pose_scale
            masks,maps=render_masks(meshes,cameras,pose,True)
            iou,per_view,micro=mean_iou(masks,targets,labels)
            record={'outer_iteration':outer+1,'active_mean_tooth_iou':iou,'per_view':per_view,'pose':pose.tolist(),
                'solver_nfev':solution.nfev,'solver_success':bool(solution.success),'cost':solution.cost,
                'max_code_change_std':float(np.abs(x[6:]).max()),'rejected_trial_solves':rejected}
            history.append(record)
            log(root,f'JOINT_ITER {outer+1} active_iou={iou:.6f}',record)
            if iou>best_iou:
                best_x=x.copy(); best_iou=iou; best_iteration=outer+1
            np.savez_compressed(root/'latest_parameters.npz',parameters=best_x)
            if outer+1-best_iteration>=config['stop_after_no_best_iterations']:
                log(root,'STOP no new fitting-view IoU best'); break
        final_codes={f'label_{k}':initial[f'label_{k}']+best_x[objective.slices[k]]*scales[f'label_{k}'] for k in labels}
        np.savez_compressed(root/'final_codes.npz',**final_codes)
        final_pose=initial_pose+best_x[:6]*pose_scale
        (root/'final_pose.json').write_text(json.dumps({'pose':final_pose.tolist(),'arch_to_world':pose_matrix(final_pose).tolist()},indent=2))
        (root/'meshes/final').mkdir(parents=True)
        for label in labels:
            meshes[label]=decode(net,label,final_codes[f'label_{label}'],boxes[str(label)])
            save_ply(root/'meshes/final'/f'tooth{label}.ply',*meshes[label])
        masks,_=render_masks(meshes,cameras,final_pose)
        final_iou,per_view,micro=mean_iou(masks,targets,labels)
        save_render_comparison(root,'final',masks,targets,cameras,labels)
        report={'status':'JOINT_FIT_COMPLETED_EVAL_REQUIRED','active_labels':labels,'active_latent_dimensions':dimensions-6,
            'pose_dimensions':6,'total_dimensions':dimensions,'initial':initial_metrics,'centroid':coarse_metrics,
            'iterations':history,'best_iteration':best_iteration,'final':{'pose':final_pose.tolist(),
                'active_mean_tooth_iou':final_iou,'per_view':per_view,'micro_tooth_iou':micro},
            'selection':'best fitting-view IoU only; no geometry truth or heldout view','elapsed_seconds':time.perf_counter()-started}
        (root/'fit_report.json').write_text(json.dumps(report,indent=2))
        log(root,f'{report["status"]} active_iou={final_iou:.6f}')
        print(json.dumps({'status':report['status'],'run_root':str(root),'final_iou':final_iou}))
    except Exception as error:
        (root/'failure.json').write_text(json.dumps({'status':'JOINT_FIT_FAILED','error_type':type(error).__name__,'error':str(error)},indent=2))
        log(root,f'JOINT_FIT_FAILED {type(error).__name__}: {error}')
        raise


if __name__=='__main__':
    main()
