"""Alternating latent/pose and boundary-band experiments; no geometry truth."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from run_forward_check import PROJECT_ROOT,sha256,save_ply
from run_pose_only import render_masks,mean_iou,save_render_comparison,log
from shape_only_common import load_model,decode,make_pairs
from joint_common import pose_matrix,gradient_check
from pose_warmup import FixedShapePoseObjective
from joint_boundary_band import fixed_band_residual,fixed_band_jacobian
from joint_population_prior import PopulationJointObjective
from scipy.spatial import cKDTree
from run_pose_only import boundary_pixels

def selection_score(masks,targets,labels,x,initial,scales,population,radius,transition):
    image=0.
    for name in targets:
        for label in labels:
            a=boundary_pixels(masks[name],label); b=boundary_pixels(targets[name],label)
            if not len(a) or not len(b): return float('inf')
            for points,reference in [(a,b),(b,a)]:
                distance=cKDTree(reference).query(points)[0]
                d=np.maximum(distance-radius,0)
                if radius>0: d=np.where(d<transition,d*d/(2*transition),d-transition/2)
                image+=100*np.mean(2*(np.sqrt(1+d*d)-1))
    offset=6; prior=0.
    for label in labels:
        key=f'label_{label}'; n=len(initial[key]); code=initial[key]+x[offset:offset+n]*scales[key]; offset+=n
        residual=population[key+'_whitener']@(code-population[key+'_mean'])
        prior+=np.sum(2*(np.sqrt(1+residual**2)-1))
    return float(image+prior)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); args=parser.parse_args()
    config=read(args.config); fit=Path(config['fit_input']).resolve(strict=True)
    root=args.config.resolve().parent/'fit_attempt_01'
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('run must be under project runs')
    root.mkdir(exist_ok=False)
    started=time.perf_counter()
    try:
        log(root,'START alternating pose/latent and contour boundary band')
        manifest=read(fit/'manifest.json')
        if manifest['status']!='JOINT_SYNTHETIC_INPUT_READY' or manifest['frozen_labels'] or 'fixed_arch_to_world' in manifest:
            raise ValueError('invalid Joint input')
        for n,d in manifest['extra_input_sha256'].items():
            if sha256(fit/n)!=d:
                raise ValueError('input changed')
        cameras=read(fit/manifest['camera_file'])
        if sha256(fit/manifest['camera_file'])!=manifest['camera_sha256']:
            raise ValueError('cameras changed')
        targets={}
        for m in manifest['masks']:
            if sha256(fit/m['path'])!=m['sha256']:
                raise ValueError('mask changed')
            targets[m['camera']]=np.asarray(Image.open(fit/m['path']))
        population=dict(np.load(fit/'population_prior.npz')); initial=dict(np.load(fit/manifest['initial_codes'])); scales=dict(np.load(fit/manifest['training_scales']))
        boxes=read(fit/manifest['sampling_boxes']); labels=manifest['active_labels']
        source=Path(config['warm_start_run']).resolve(strict=True); sp=read(source/'provenance.json')
        if manifest['warm_start_manifest_sha256']!=sp['fit_manifest_sha256']:
            raise ValueError('warm-start observations differ')
        old_manifest=read(Path(sp['fit_input'])/'manifest.json')
        for field in ['active_labels','camera_file','camera_sha256','masks','initial_codes','training_scales','sampling_boxes','initial_pose','model_sha256','specs_sha256','checkpoint_epoch']:
            if manifest[field]!=old_manifest[field]:
                raise ValueError('warm-start input changed: '+field)
        for name,digest in old_manifest['extra_input_sha256'].items():
            if manifest['extra_input_sha256'].get(name)!=digest:
                raise ValueError('warm-start numerical input changed')
        old_initial=dict(np.load(source/'initial_codes.npz'))
        if any(not np.array_equal(initial[k],old_initial[k]) for k in initial):
            raise ValueError('warm-start shapes differ')
        if any(sp[k] for k in ['active_truth_codes_read','ground_truth_pose_read','source_meshes_read','depth_maps_read']):
            raise ValueError('unsafe warm-start provenance')
        old_config=read(source/'resolved_config.json')
        if config['acceptance']!=old_config['acceptance']:
            raise ValueError('acceptance changed')
        pd=read(fit/manifest['initial_pose']); initial_pose=np.r_[pd['rotation_vector_radians'],pd['translation_dmm']]
        scale=np.r_[np.repeat(np.deg2rad(config['pose_rotation_scale_degrees']),3),np.repeat(config['pose_translation_scale_dmm'],3)]
        dimensions=6+sum(len(initial[f'label_{k}']) for k in labels); x=np.zeros(dimensions)
        warm=read(source/'warmup_pose.json'); x[:6]=(np.array(warm['pose'])-initial_pose)/scale
        bound=np.r_[np.repeat(config['pose_rotation_bound_degrees']/config['pose_rotation_scale_degrees'],3),np.repeat(config['pose_translation_bound_dmm']/config['pose_translation_scale_dmm'],3),np.repeat(config['global_bound_training_std'],dimensions-6)]
        step=np.r_[np.repeat(config['local_pose_rotation_step_degrees']/config['pose_rotation_scale_degrees'],3),np.repeat(config['local_pose_translation_step_dmm']/config['pose_translation_scale_dmm'],3),np.repeat(config['local_step_training_std'],dimensions-6)]
        names=['run_joint_population.py','joint_population_prior.py','joint_boundary_band.py','joint_common.py','pose_warmup.py','shape_only_common.py','run_forward_check.py','run_pose_only.py']
        (root/'code_snapshot').mkdir()
        for n in names:
            shutil.copyfile(PROJECT_ROOT/'scripts'/n,root/'code_snapshot'/n)
        dmm=Path(config['dmm_root']); dmm_sources=list((dmm/'networks').glob('*.py'))+list((dmm/'utils').glob('*.py'))
        provenance={'run_id':root.name,'fit_input':str(fit),'fit_manifest_sha256':sha256(fit/'manifest.json'),
            'config_sha256':sha256(args.config),'model_sha256':manifest['model_sha256'],'checkpoint_epoch':manifest['checkpoint_epoch'],
            'active_truth_codes_read':False,'ground_truth_pose_read':False,'source_meshes_read':False,'depth_maps_read':False,
            'warm_start_run':str(source),'warm_start_pose_sha256':sha256(source/'warmup_pose.json'),
            'warm_start_initial_codes_sha256':sha256(source/'initial_codes.npz'),'warm_start_used_only_image_fitted_pose':True,
            'code_sha256':{n:sha256(PROJECT_ROOT/'scripts'/n) for n in names},'dmm_source_sha256':{str(p):sha256(p) for p in dmm_sources},
            'python':sys.executable,'stabilization':config['stabilization'],'device':'cpu','sdf_dtype':'float64'}
        (root/'provenance.json').write_text(json.dumps(provenance,indent=2)); (root/'resolved_config.json').write_text(json.dumps(config,indent=2))
        np.savez_compressed(root/'initial_codes.npz',**initial)
        (root/'initial_pose.json').write_text(json.dumps({'pose':initial_pose.tolist(),'arch_to_world':pose_matrix(initial_pose).tolist()},indent=2))
        shutil.copyfile(source/'warmup_pose.json',root/'warmup_pose.json')
        net=load_model(config,manifest); meshes={}
        (root/'meshes/initial').mkdir(parents=True)
        for label in labels:
            meshes[label]=decode(net,label,initial[f'label_{label}'],boxes[str(label)]); save_ply(root/'meshes/initial'/f'tooth{label}.ply',*meshes[label])
        initial_masks,_=render_masks(meshes,cameras,initial_pose)
        initial_iou,per,micro=mean_iou(initial_masks,targets,labels); save_render_comparison(root,'initial',initial_masks,targets,cameras,labels)
        initial_metrics={'pose':initial_pose.tolist(),'active_mean_tooth_iou':initial_iou,'per_view':per,'micro_tooth_iou':micro}
        masks,maps=render_masks(meshes,cameras,initial_pose+x[:6]*scale,True)
        warm_iou=mean_iou(masks,targets,labels)[0]; save_render_comparison(root,'warmup',masks,targets,cameras,labels)
        radius=config['stabilization']['band_radius_pixels']; transition=config['stabilization']['band_transition_pixels']
        best=x.copy(); best_iou=warm_iou; best_iteration=0; history=[]
        best_score=selection_score(masks,targets,labels,x,initial,scales,population,radius,transition)
        for outer in range(config['outer_iterations']):
            pose=initial_pose+x[:6]*scale
            pairs=make_pairs(masks,maps,targets,cameras,labels,pose_matrix(pose),config['boundary_samples_per_direction'])
            obj=PopulationJointObjective(net,labels,initial,scales,pairs,cameras,initial_pose,scale,x,config['initial_code_damping'],band_radius=radius,band_transition=transition,population=population)
            if outer==0:
                check=gradient_check(obj,x); check['status']='JOINT_IMAGE_GRADIENT_PASS' if check['max_relative_error']<=config['acceptance']['max_gradient_relative_error'] else 'JOINT_IMAGE_GRADIENT_FAIL'
                (root/'gradient_check.json').write_text(json.dumps(check,indent=2))
                log(root,f'{check["status"]} max_relative_error={check["max_relative_error"]:.3g}')
                if check['status']!='JOINT_IMAGE_GRADIENT_PASS':
                    raise ValueError('stabilized derivative failed')
            base=x.copy(); trial_step=step[6:].copy(); rejected=[]
            def embed(q):
                return np.r_[base[:6],q]
            for retry in range(config['local_trust_retries']+1):
                try:
                    shape_solution=least_squares(lambda q:obj.residual(embed(q)),x[6:],jac=lambda q:obj.jacobian(embed(q))[:,6:],
                        bounds=(np.maximum(-bound[6:],x[6:]-trial_step),np.minimum(bound[6:],x[6:]+trial_step)),
                        loss='soft_l1',f_scale=1,max_nfev=config['inner_max_nfev'],ftol=1e-5,xtol=1e-5,gtol=1e-5)
                    break
                except ValueError as error:
                    if str(error) not in ['local surface root failed to converge','singular implicit surface derivative'] or retry>=config['local_trust_retries']:
                        raise
                    rejected.append({'reason':str(error),'code_step_std':trial_step.tolist()}); trial_step*=.5
                    log(root,f'STABLE_LOCAL_STEP_RETRY {retry+1} {error}')
            x[6:]=shape_solution.x
            for label in labels:
                key=f'label_{label}'; meshes[label]=decode(net,label,initial[key]+x[obj.slices[label]]*scales[key],boxes[str(label)])
            masks,maps=render_masks(meshes,cameras,pose,True)
            pairs=make_pairs(masks,maps,targets,cameras,labels,pose_matrix(pose),config['boundary_samples_per_direction'])
            po=FixedShapePoseObjective(pairs,cameras,initial_pose,scale)
            pose_solution=least_squares(lambda u:fixed_band_residual(po,u,radius,transition),x[:6],jac=lambda u:fixed_band_jacobian(po,u,radius,transition),
                bounds=(np.maximum(-bound[:6],x[:6]-step[:6]),np.minimum(bound[:6],x[:6]+step[:6])),
                loss='soft_l1',f_scale=1,max_nfev=config['inner_max_nfev'],ftol=1e-5,xtol=1e-5,gtol=1e-5)
            x[:6]=pose_solution.x; pose=initial_pose+x[:6]*scale
            masks,maps=render_masks(meshes,cameras,pose,True); iou,per,micro=mean_iou(masks,targets,labels)
            record={'outer_iteration':outer+1,'active_mean_tooth_iou':iou,'pose':pose.tolist(),'per_view':per,
                'shape_nfev':shape_solution.nfev,'pose_nfev':pose_solution.nfev,'shape_cost':float(shape_solution.cost),'pose_cost':float(pose_solution.cost),
                'rejected_trial_solves':rejected,'max_code_change_std':float(np.abs(x[6:]).max())}
            history.append(record); log(root,f'STABLE_ITER {outer+1} active_iou={iou:.6f}',record)
            score=selection_score(masks,targets,labels,x,initial,scales,population,radius,transition)
            record['selection_score']=score
            if score<best_score:
                best_score=score
                best=x.copy(); best_iou=iou; best_iteration=outer+1
            np.savez_compressed(root/'latest_parameters.npz',parameters=best)
            if outer+1-best_iteration>=config['stop_after_no_best_iterations']:
                log(root,'STOP no new boundary-band plus population score best'); break
        final_codes={f'label_{k}':initial[f'label_{k}']+best[obj.slices[k]]*scales[f'label_{k}'] for k in labels}
        np.savez_compressed(root/'final_codes.npz',**final_codes)
        final_pose=initial_pose+best[:6]*scale
        (root/'final_pose.json').write_text(json.dumps({'pose':final_pose.tolist(),'arch_to_world':pose_matrix(final_pose).tolist()},indent=2))
        (root/'meshes/final').mkdir(parents=True)
        for label in labels:
            meshes[label]=decode(net,label,final_codes[f'label_{label}'],boxes[str(label)]); save_ply(root/'meshes/final'/f'tooth{label}.ply',*meshes[label])
        masks,_=render_masks(meshes,cameras,final_pose); final_iou,per,micro=mean_iou(masks,targets,labels)
        save_render_comparison(root,'final',masks,targets,cameras,labels)
        report={'status':'JOINT_FIT_COMPLETED_EVAL_REQUIRED','active_labels':labels,'active_latent_dimensions':dimensions-6,'total_dimensions':dimensions,
            'pose_dimensions':6,'initial':initial_metrics,'warmup':{'pose':warm['pose'],'active_mean_tooth_iou':warm_iou,'reused':True},
            'iterations':history,'best_iteration':best_iteration,'final':{'pose':final_pose.tolist(),'active_mean_tooth_iou':final_iou,'per_view':per,'micro_tooth_iou':micro},
            'selection':'minimum bidirectional full-contour boundary-band soft-L1 score plus population prior; 100 times per-direction mean; no truth/clean/heldout selection','selection_score':best_score,'elapsed_seconds':time.perf_counter()-started,
            'schedule':'shape then pose, fresh rendering/correspondences between blocks; up to 24 nfev per block'}
        (root/'fit_report.json').write_text(json.dumps(report,indent=2)); log(root,f'{report["status"]} iou={final_iou:.6f}')
        print(json.dumps({'status':report['status'],'run_root':str(root)}))
    except Exception as error:
        (root/'failure.json').write_text(json.dumps({'status':'STABILIZED_FIT_FAILED','error_type':type(error).__name__,'error':str(error)},indent=2))
        raise


if __name__=='__main__':
    main()
