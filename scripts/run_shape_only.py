"""Fit configured tooth latents with fixed cameras, arch pose and other teeth."""

from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from run_forward_check import PROJECT_ROOT, sha256, save_ply
from run_pose_only import log, save_render_comparison, mean_iou
from shape_only_common import (load_model, decode, fixed_render_cache, render_scene, make_pairs,
                               SurfaceImageObjective, gradient_check)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/shape_only.json')
    parser.add_argument('--fit-input', type=Path, required=True)
    parser.add_argument('--run-root', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    fit = args.fit_input.resolve(strict=True)
    root = args.run_root.resolve() if args.run_root else PROJECT_ROOT/'runs'/(dt.datetime.now(dt.timezone.utc).strftime('shape_only_%Y%m%dT%H%M%SZ_')+sha256(args.config)[:8])
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('run root must be inside project runs')
    root.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    try:
        log(root,'START shape-only fit')
        manifest = json.loads((fit/'manifest.json').read_text())
        if manifest['status'] != 'SHAPE_ONLY_SYNTHETIC_INPUT_READY':
            raise ValueError('shape input not ready')
        for name,digest in manifest['extra_input_sha256'].items():
            if sha256(fit/name) != digest:
                raise ValueError(f'input hash mismatch: {name}')
        if sha256(fit/manifest['camera_file']) != manifest['camera_sha256']:
            raise ValueError('camera hash mismatch')
        cameras = json.loads((fit/manifest['camera_file']).read_text())
        targets = {}
        for item in manifest['masks']:
            if sha256(fit/item['path']) != item['sha256']:
                raise ValueError('mask hash mismatch')
            targets[item['camera']] = np.asarray(Image.open(fit/item['path']))
        initial = dict(np.load(fit/manifest['initial_codes']))
        scales = dict(np.load(fit/manifest['training_scales']))
        boxes = json.loads((fit/manifest['sampling_boxes']).read_text())
        active = list(manifest['active_labels'])
        frozen = list(manifest['frozen_labels'])
        labels = sorted(active+frozen)
        if active != config['active_labels'] or set(targets)!={c['name'] for c in cameras}:
            raise ValueError('active labels or camera names disagree')
        pose = np.asarray(manifest['fixed_arch_to_world'])
        if pose.shape!=(4,4) or not np.allclose(pose[:3,:3]@pose[:3,:3].T,np.eye(3),atol=1e-8):
            raise ValueError('fixed pose invalid')
        net = load_model(config,manifest)
        provenance = {'run_id':root.name,'fit_input':str(fit),'fit_manifest_sha256':sha256(fit/'manifest.json'),
                      'config_sha256':sha256(args.config),'model_sha256':manifest['model_sha256'],
                      'objective_inputs_only':['packaged masks/cameras/initial codes/scales/sampling boxes/fixed pose','frozen DMM weights'],
                      'diagnostic_oracle_conditions':manifest['diagnostic_oracle_conditions'],
                      'active_truth_codes_read':False,'active_labels':active,'frozen_labels':frozen,
                      'fixed_arch_to_world':pose.tolist(),'checkpoint_epoch':manifest['checkpoint_epoch'],
                      'code_sha256':{name:sha256(PROJECT_ROOT/'scripts'/name) for name in
                                    ['run_shape_only.py','shape_only_common.py','run_forward_check.py']},
                      'python':sys.executable,'device':'cpu','sdf_dtype':'float64'}
        (root/'provenance.json').write_text(json.dumps(provenance,indent=2))
        (root/'resolved_config.json').write_text(json.dumps(config,indent=2))
        np.savez_compressed(root/'initial_codes.npz',**initial)
        meshes = {}
        initial_folder = root/'meshes/initial'
        initial_folder.mkdir(parents=True)
        for label in labels:
            meshes[label] = decode(net,label,initial[f'label_{label}'],boxes[str(label)])
            save_ply(initial_folder/f'tooth{label}.ply',*meshes[label])
        log(root,f'DMM_INITIAL_DECODED active={active} frozen={len(frozen)} pose_and_cameras_fixed')
        frozen_meshes = {k:meshes[k] for k in frozen}
        cache = fixed_render_cache(frozen_meshes,cameras,pose)
        active_meshes = {k:meshes[k] for k in active}
        masks,maps = render_scene(active_meshes,cache,cameras,pose,True)
        initial_iou,initial_per_view,initial_micro = mean_iou(masks,targets,active)
        save_render_comparison(root,'initial',masks,targets,cameras,active)
        initial_metrics = {'active_mean_tooth_iou':initial_iou,'active_micro_tooth_iou':initial_micro,
                           'per_view':initial_per_view}
        log(root,f'INITIAL_ACTIVE_IOU {initial_iou:.6f}')
        pairs = make_pairs(masks,maps,targets,cameras,active,pose,config['boundary_samples_per_direction'])
        q = {label:np.zeros_like(initial[f'label_{label}']) for label in active}
        objectives = {label:SurfaceImageObjective(net,label,initial[f'label_{label}'],scales[f'label_{label}'],
                       q[label],pairs[label],cameras,pose,config['initial_code_damping']) for label in active}
        check = gradient_check(objectives)
        check['status'] = 'IMAGE_TO_LATENT_GRADIENT_PASS' if check['max_relative_error'] <= config['acceptance']['max_gradient_relative_error'] else 'GRADIENT_FAIL'
        (root/'gradient_check.json').write_text(json.dumps(check,indent=2))
        log(root,f"{check['status']} max_relative_error={check['max_relative_error']:.3g}")
        if check['status'] != 'IMAGE_TO_LATENT_GRADIENT_PASS':
            raise ValueError('image-to-latent derivative failed finite difference check')
        best_q = {k:v.copy() for k,v in q.items()}
        best_iou = initial_iou
        history = []
        best_iteration = 0
        for outer in range(config['outer_iterations']):
            if outer:
                pairs = make_pairs(masks,maps,targets,cameras,active,pose,config['boundary_samples_per_direction'])
                objectives = {label:SurfaceImageObjective(net,label,initial[f'label_{label}'],scales[f'label_{label}'],
                    q[label],pairs[label],cameras,pose,config['initial_code_damping']) for label in active}
            solver_records = {}
            for label in active:
                obj = objectives[label]
                bound = config['global_bound_training_std']
                step = config['local_step_training_std']
                rejected_trials=[]
                for retry in range(int(config.get('local_trust_retries',0))+1):
                    try:
                        solution = least_squares(obj.residual,q[label],jac=obj.jacobian,
                            bounds=(np.maximum(-bound,q[label]-step),np.minimum(bound,q[label]+step)),
                            loss='soft_l1',f_scale=1.0,max_nfev=config['inner_max_nfev'],
                            ftol=1e-5,xtol=1e-5,gtol=1e-5)
                        break
                    except ValueError as error:
                        if str(error) not in ['local surface root failed to converge','singular implicit surface derivative'] or retry>=int(config.get('local_trust_retries',0)):
                            raise
                        rejected_trials.append({'step_training_std':step,'reason':str(error)})
                        step*=0.5
                        log(root,f'LOCAL_STEP_RETRY tooth={label} step_std={step:.4g} reason={error}')
                q[label] = solution.x
                code = initial[f'label_{label}']+q[label]*scales[f'label_{label}']
                active_meshes[label] = decode(net,label,code,boxes[str(label)])
                solver_records[str(label)] = {'nfev':solution.nfev,'success':bool(solution.success),
                    'cost':float(solution.cost),'max_initial_relative_change_std':float(np.abs(q[label]).max()),
                    'local_step_training_std':step,'rejected_trial_solves':rejected_trials}
            masks,maps = render_scene(active_meshes,cache,cameras,pose,True)
            iou,per_view,micro = mean_iou(masks,targets,active)
            item = {'outer_iteration':outer+1,'active_mean_tooth_iou':iou,'active_micro_tooth_iou':micro,
                    'per_view':per_view,'solvers':solver_records}
            history.append(item)
            log(root,f'SHAPE_ITER {outer+1} active_iou={iou:.6f}',item)
            if iou>best_iou:
                best_iou=iou
                best_q = {k:v.copy() for k,v in q.items()}
                best_iteration = outer+1
            np.savez_compressed(root/'latest_codes.npz',**{f'label_{k}':initial[f'label_{k}']+
                best_q[k]*scales[f'label_{k}'] if k in active else initial[f'label_{k}'] for k in labels})
            if outer>=4 and max(h['active_mean_tooth_iou'] for h in history[-3:])-min(h['active_mean_tooth_iou'] for h in history[-3:])<2e-4:
                break
            if config.get('stop_after_no_best_iterations') and outer+1-best_iteration>=config['stop_after_no_best_iterations']:
                log(root,'STOP no further independently rasterized IoU improvement')
                break
        final_codes = {f'label_{k}':initial[f'label_{k}']+best_q[k]*scales[f'label_{k}']
                       if k in active else initial[f'label_{k}'].copy() for k in labels}
        if any(not np.array_equal(initial[f'label_{k}'],final_codes[f'label_{k}']) for k in frozen):
            raise ValueError('frozen code drifted')
        np.savez_compressed(root/'final_codes.npz',**final_codes)
        final_folder=root/'meshes/final'
        final_folder.mkdir(parents=True)
        for label in labels:
            mesh=decode(net,label,final_codes[f'label_{label}'],boxes[str(label)]) if label in active else frozen_meshes[label]
            save_ply(final_folder/f'tooth{label}.ply',*mesh)
            if label in active:
                active_meshes[label]=mesh
        masks,_=render_scene(active_meshes,cache,cameras,pose)
        final_iou,final_per_view,final_micro=mean_iou(masks,targets,active)
        save_render_comparison(root,'final',masks,targets,cameras,active)
        report={'status':'SHAPE_ONLY_FIT_COMPLETED_EVAL_REQUIRED','run_id':root.name,
                'active_latent_dimensions':sum(len(initial[f'label_{k}']) for k in active),
                'active_labels':active,'frozen_codes_unchanged':True,'pose_and_cameras_unchanged':True,
                'initial':initial_metrics,'iterations':history,'best_iteration':best_iteration,
                'final':{'active_mean_tooth_iou':final_iou,'active_micro_tooth_iou':final_micro,'per_view':final_per_view},
                'elapsed_seconds':round(time.perf_counter()-started,2)}
        (root/'fit_report.json').write_text(json.dumps(report,indent=2))
        log(root,f"{report['status']} active_iou={final_iou:.6f}")
        print(json.dumps({'status':report['status'],'run_root':str(root),'initial_iou':initial_iou,'final_iou':final_iou}))
    except Exception as exc:
        failure={'status':'SHAPE_ONLY_FIT_FAILED','error_type':type(exc).__name__,'error':str(exc)}
        (root/'failure.json').write_text(json.dumps(failure,indent=2))
        log(root,f"SHAPE_ONLY_FIT_FAILED {type(exc).__name__}: {exc}")
        raise


if __name__=='__main__':
    main()
