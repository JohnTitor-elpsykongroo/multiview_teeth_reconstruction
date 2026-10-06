"""Frozen-plan, resumable controlled robustness jobs; failures never stop siblings."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,binary_dilation,distance_transform_edt
from run_forward_check import PROJECT_ROOT,sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def prepare(root,job,control):
    directory=root/'jobs'/job['id']; directory.mkdir(parents=True)
    provenance=read(control/'provenance.json')
    source=Path(provenance['fit_input'])
    case=directory/'case'; shutil.copytree(source.parent,case)
    fit,truth=case/'fit_input',case/'truth'
    manifest=read(fit/'manifest.json')
    if sha256(source/'manifest.json')!=provenance['fit_manifest_sha256']:
        raise ValueError('control fit inputs changed')
    clean={'camera_file':'clean_fit/cameras.json','camera_sha256':manifest['camera_sha256'],'masks':[]}
    (truth/'clean_fit/masks').mkdir(parents=True)
    shutil.copyfile(source/manifest['camera_file'],truth/clean['camera_file'])
    for item in manifest['masks']:
        path='clean_fit/'+item['path']
        shutil.copyfile(source/item['path'],truth/path)
        clean['masks'].append(dict(item,path=path))
    write(truth/'clean_fit_manifest.json',clean)
    config=read(control/'resolved_config.json')
    labels=manifest['active_labels']
    evidence={'source_control':str(control),'axis':job,'unchanged_input_files':{},'alterations':[]}
    if job['kind']=='initialization':
        gt=dict(np.load(truth/'latents.npz'))
        scales=dict(np.load(fit/manifest['training_scales']))
        initial={}
        for label in labels:
            key=f'label_{label}'
            rng=np.random.default_rng(np.random.SeedSequence([job['seed'],label]))
            initial[key]=gt[key].astype(np.float64)+rng.normal(size=len(scales[key]))*scales[key]*config['perturbation_training_std']
        np.savez_compressed(fit/manifest['initial_codes'],**initial)
        config['perturbation_seed']=job['seed']
        evidence['alterations'].append('initial codes only; same sigma and pose')
    elif job['kind']=='mask_boundary':
        if job['radius_pixels']!=1:
            raise ValueError('pilot only specifies one pixel Chebyshev morphology')
        kernel=np.ones((3,3),dtype=bool)
        for item in manifest['masks']:
            ids=np.asarray(Image.open(fit/item['path']))
            if job['operation']=='erode':
                changed=np.zeros_like(ids)
                for label in labels:
                    changed[binary_erosion(ids==label,structure=kernel)]=label
            elif job['operation']=='dilate':
                candidates=np.stack([binary_dilation(ids==k,structure=kernel) for k in labels])
                distances=np.stack([distance_transform_edt(ids!=k) for k in labels])
                distances[~candidates]=np.inf
                nearest=np.argmin(distances,axis=0)
                changed=np.where(candidates.any(axis=0),np.array(labels,dtype=ids.dtype)[nearest],0).astype(ids.dtype)
                changed[ids>0]=ids[ids>0]
            else:
                raise ValueError('unknown boundary operation')
            if any(np.count_nonzero(changed==k)<100 for k in labels):
                raise ValueError('morphology removed a tooth observation')
            evidence['alterations'].append({'camera':item['camera'],'changed_pixels':int(np.count_nonzero(ids!=changed)),
                'tooth_pixel_counts':{str(k):int(np.count_nonzero(changed==k)) for k in labels}})
            Image.fromarray(changed).save(fit/item['path'])
            item['sha256']=sha256(fit/item['path'])
    elif job['kind']=='view_drop':
        cameras=read(fit/manifest['camera_file'])
        retained=[c for c in cameras if c['name']!=job['drop_camera']]
        if len(retained)!=len(cameras)-1 or len(retained)<2:
            raise ValueError('unexpected view-drop request')
        write(fit/manifest['camera_file'],retained)
        manifest['camera_sha256']=sha256(fit/manifest['camera_file'])
        manifest['masks']=[m for m in manifest['masks'] if m['camera']!=job['drop_camera']]
        evidence['alterations'].append({'dropped_camera':job['drop_camera'],'retained_cameras':[c['name'] for c in retained]})
    else:
        raise ValueError('unknown perturbation kind')
    for name in manifest['extra_input_sha256']:
        manifest['extra_input_sha256'][name]=sha256(fit/name)
    manifest['robustness_axis']=job
    manifest['observation_type']='hard FDI labels; controlled robustness perturbation; two teeth only'
    write(fit/'manifest.json',manifest)
    # Geometry-generation files never enter the fit payload.
    for name in [manifest['training_scales'],manifest['sampling_boxes'],manifest['initial_pose']]:
        evidence['unchanged_input_files'][name]=sha256(fit/name)==sha256(source/name)
    if job['kind']!='initialization':
        evidence['unchanged_input_files']['initial_codes.npz']=sha256(fit/'initial_codes.npz')==sha256(source/'initial_codes.npz')
    if job['kind']!='view_drop':
        evidence['unchanged_input_files']['cameras.json']=sha256(fit/'cameras.json')==sha256(source/'cameras.json')
    if job['kind']!='mask_boundary':
        evidence['unchanged_input_files']['retained_masks']=all(sha256(fit/m['path'])==sha256(source/m['path']) for m in manifest['masks'])
    if not all(evidence['unchanged_input_files'].values()):
        raise ValueError('unexpected input change')
    write(fit/'robustness_contract.json',{'prepared_manifest_sha256':sha256(fit/'manifest.json'),
        'control_fit_manifest_sha256':provenance['fit_manifest_sha256'],'evidence':evidence})
    config['robustness']={'control_run':str(control),'axis':job,'clean_evaluation':'all three clean original cameras plus 65 degree heldout; post-fit only',
        'prior_weight_note':'original unnormalized residual sum is preserved; losing a view also changes relative damping contribution'}
    write(directory/'config.json',config)
    return directory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,default=PROJECT_ROOT/'configs/joint_robustness.json')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.resume:
        root=args.resume.resolve(strict=True); plan=read(root/'plan.json')
        state=read(root/'status.json')
        if any(j['phase'] in ['fit_running','eval_running'] for j in state['jobs']):
            raise ValueError('running/interrupted child recorded; inspect PID before resuming, never duplicate')
    else:
        plan=read(args.plan)
        root=PROJECT_ROOT/'runs'/(dt.datetime.now(dt.timezone.utc).strftime('joint_robustness_%Y%m%dT%H%M%SZ_')+sha256(args.plan)[:8])
        root.mkdir(parents=True,exist_ok=False)
        write(root/'plan.json',plan)
        control=Path(plan['control_run']).resolve(strict=True)
        state={'status':'ROBUSTNESS_RUNNING','run_root':str(root),'control_status':read(control/'evaluation.json')['status'],'jobs':[]}
        for job in plan['jobs']:
            directory=prepare(root,job,control)
            state['jobs'].append({'id':job['id'],'axis':job,'directory':str(directory),'phase':'prepared'})
        (root/'code_snapshot').mkdir()
        names=['run_joint_robustness.py','run_joint_robust.py','evaluate_joint_robust.py','pose_warmup.py','joint_common.py',
            'shape_only_common.py','run_forward_check.py','run_pose_only.py','evaluate_shape_only.py','surface_metric_utils.py']
        hashes={}
        for name in names:
            shutil.copyfile(PROJECT_ROOT/'scripts'/name,root/'code_snapshot'/name)
            hashes[name]=sha256(PROJECT_ROOT/'scripts'/name)
        write(root/'code_sha256.json',hashes)
        write(root/'status.json',state)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('batch must be under project runs')
    hashes=read(root/'code_sha256.json')
    if not all(sha256(PROJECT_ROOT/'scripts'/n)==sha256(root/'code_snapshot'/n)==d for n,d in hashes.items()):
        raise ValueError('batch code changed')
    print(json.dumps({'run_root':str(root),'jobs':len(state['jobs']),'max_workers':plan['max_workers']}),flush=True)
    active={}
    while True:
        for index,item in enumerate(state['jobs']):
            if len(active)>=plan['max_workers']:
                break
            if item['phase']!='prepared':
                continue
            directory=Path(item['directory'])
            stream=(directory/'fit_process.log').open('w',encoding='utf-8')
            command=[sys.executable,'-u',str(PROJECT_ROOT/'scripts/run_joint_robust.py'),'--config',str(directory/'config.json'),
                '--fit-input',str(directory/'case/fit_input')]
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=PROJECT_ROOT)
            active[index]=(process,stream,'fit'); item.update(phase='fit_running',pid=process.pid)
            print(f'START {item["id"]} pid={process.pid}',flush=True)
        for index,(process,stream,phase) in list(active.items()):
            code=process.poll()
            if code is None:
                continue
            stream.close(); item=state['jobs'][index]; directory=Path(item['directory'])
            if code!=0:
                item.update(phase=phase+'_failed',returncode=code); del active[index]
                print(f'FAIL {item["id"]} {phase} returncode={code}',flush=True)
            elif phase=='fit':
                stream=(directory/'eval_process.log').open('w',encoding='utf-8')
                process=subprocess.Popen([sys.executable,'-u',str(PROJECT_ROOT/'scripts/evaluate_joint_robust.py'),
                    str(directory/'fit_attempt_01')],stdout=stream,stderr=subprocess.STDOUT,cwd=PROJECT_ROOT)
                active[index]=(process,stream,'eval'); item.update(phase='eval_running',pid=process.pid)
                print(f'EVALUATE {item["id"]}',flush=True)
            else:
                ev=read(directory/'fit_attempt_01/evaluation.json')
                item.update(phase='evaluated',evaluation_status=ev['status'],failed_checks=ev['failed_checks'])
                del active[index]
                print(f'DONE {item["id"]} {ev["status"]} failures={ev["failed_checks"]}',flush=True)
        write(root/'status.json',state)
        if not active and all(item['phase']!='prepared' for item in state['jobs']):
            break
        time.sleep(2)
    state['status']='ROBUSTNESS_EXPERIMENTS_COMPLETED_REVIEW_REQUIRED'
    write(root/'status.json',state)
    print(json.dumps({'status':state['status'],'run_root':str(root),'jobs':state['jobs']}),flush=True)


if __name__=='__main__':
    main()
