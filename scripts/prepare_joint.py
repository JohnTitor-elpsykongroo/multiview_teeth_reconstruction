"""Generate the two-tooth scene, with geometry and pose truth isolated."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from run_forward_check import PROJECT_ROOT, sha256, load_ply, rasterize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/joint_minimal.json')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config['scene']!='active_teeth_only':
        raise ValueError('minimal joint requires an active-only scene')
    source=Path(config['source_run']).resolve(strict=True)
    provenance=json.loads((source/'provenance.json').read_text())
    original_truth=json.loads((source/'truth/manifest.json').read_text())
    gt_file=source/'truth'/original_truth['latent_file']
    if sha256(gt_file)!=original_truth['latent_sha256']:
        raise ValueError('generator latent changed')
    if original_truth['ground_truth_arch_pose']!='identity in the saved DMM world frame':
        raise ValueError('unsupported generator pose')
    experiment=Path(config['experiment'])
    model_file=experiment/'ModelParameters'/f'dmm_{config["checkpoint"]}.pth'
    latent_file=experiment/'LatentCodes'/f'latent_vecs_{config["checkpoint"]}.pth'
    for path in [model_file,latent_file]:
        if sha256(path)!=provenance['input_sha256'][str(path)]:
            raise ValueError(f'checkpoint changed: {path}')
    checkpoint=torch.load(latent_file,map_location='cpu',weights_only=True)
    if checkpoint['epoch']!=original_truth['checkpoint_epoch']:
        raise ValueError('checkpoint epoch mismatch')
    active=list(config['active_labels'])
    if active!=sorted(set(active)) or any(k==0 or k not in provenance['labels'] for k in active):
        raise ValueError('invalid active labels')
    gt_all=dict(np.load(gt_file))
    gt,initial,scales={},{},{}
    for label in active:
        key=f'label_{label}'
        gt[key]=gt_all[key]
        matrix=checkpoint['latent_codes'][f'{label}.module.weight'].numpy()
        scales[key]=matrix.std(axis=0,ddof=1).astype(np.float64)
        if np.any(scales[key]<=0):
            raise ValueError('invalid training scales')
        rng=np.random.default_rng(np.random.SeedSequence([config['perturbation_seed'],label]))
        initial[key]=gt[key].astype(np.float64)+rng.normal(size=len(scales[key]))*scales[key]*config['perturbation_training_std']
    root=PROJECT_ROOT/'runs'/(dt.datetime.now(dt.timezone.utc).strftime('joint_case_%Y%m%dT%H%M%SZ_')+sha256(args.config)[:8])
    root.mkdir(exist_ok=False,parents=True)
    fit,truth=root/'fit_input',root/'truth'
    (fit/'masks').mkdir(parents=True)
    (truth/'meshes').mkdir(parents=True)
    camera_source=source/'fit_input/cameras.json'
    source_manifest=json.loads((source/'fit_input/manifest.json').read_text())
    if sha256(camera_source)!=source_manifest['camera_sha256']:
        raise ValueError('source cameras changed')
    shutil.copyfile(camera_source,fit/'cameras.json')
    cameras=json.loads((fit/'cameras.json').read_text())
    meshes={}
    mesh_hashes={}
    for label in active:
        path=source/'meshes/training_case'/f'tooth{label}.ply'
        meshes[label]=load_ply(path)
        shutil.copyfile(path,truth/'meshes'/path.name)
        mesh_hashes[path.name]=sha256(truth/'meshes'/path.name)
    masks=[]
    for camera in cameras:
        path=fit/'masks'/f'{camera["name"]}.png'
        ids=rasterize(meshes,camera,include_gum=False,shade=False)[0]
        Image.fromarray(ids).save(path)
        masks.append({'camera':camera['name'],'path':str(path.relative_to(fit)).replace('\\','/'),'sha256':sha256(path)})
    boxes_all=json.loads((source/'mesh_metrics.json').read_text())['zero']
    boxes={}
    for label in active:
        box={k:boxes_all[str(label)][k] for k in ['grid_origin','grid_top','grid_n']}
        origin,top=np.array(box['grid_origin']),np.array(box['grid_top'])
        step=(top-origin)/(box['grid_n']-1)
        padding=config['sampling_padding_cells']
        box.update(grid_origin=(origin-padding*step).tolist(),grid_top=(top+padding*step).tolist(),grid_n=box['grid_n']+2*padding)
        boxes[str(label)]=box
    np.savez_compressed(fit/'initial_codes.npz',**initial)
    np.savez_compressed(fit/'training_scales.npz',**scales)
    (fit/'sampling_boxes.json').write_text(json.dumps(boxes,indent=2))
    pose=np.r_[np.deg2rad(config['initial_rotation_degrees_xyz']),config['initial_translation_dmm']]
    (fit/'initial_pose.json').write_text(json.dumps({'rotation_vector_radians':pose[:3].tolist(),'translation_dmm':pose[3:].tolist()},indent=2))
    manifest={'status':'JOINT_SYNTHETIC_INPUT_READY','active_labels':active,'frozen_labels':[],
        'visible_fdi_ids':active,'scene':'active_teeth_only','source_run_id':source.name,
        'camera_file':'cameras.json','camera_sha256':sha256(fit/'cameras.json'),'masks':masks,
        'initial_codes':'initial_codes.npz','training_scales':'training_scales.npz','sampling_boxes':'sampling_boxes.json',
        'initial_pose':'initial_pose.json','extra_input_sha256':{name:sha256(fit/name) for name in
            ['initial_codes.npz','training_scales.npz','sampling_boxes.json','initial_pose.json']},
        'model_sha256':sha256(model_file),'specs_sha256':sha256(experiment/'specs.json'),
        'checkpoint_epoch':checkpoint['epoch'],'diagnostic_oracle_conditions':['known cameras','initialization perturbed from generator codes'],
        'ground_truth_pose_available_to_fitter':False,'active_truth_codes_available_to_fitter':False,
        'observation_type':'hard FDI labels; two teeth only, no gum, no other teeth, no lip occluder'}
    (fit/'manifest.json').write_text(json.dumps(manifest,indent=2))
    np.savez_compressed(truth/'latents.npz',**gt)
    (truth/'pose_gt.json').write_text(json.dumps({'arch_to_world':np.eye(4).tolist()},indent=2))
    heldout=json.loads((source/'review_camera.json').read_text())
    (truth/'heldout_camera.json').write_text(json.dumps(heldout,indent=2))
    Image.fromarray(rasterize(meshes,heldout,include_gum=False,shade=False)[0]).save(truth/'heldout_labels.png')
    truth_manifest={'status':'JOINT_GENERATOR_TRUTH_ISOLATED','source_case':original_truth['source_case'],
        'latent_file':'latents.npz','latent_sha256':sha256(truth/'latents.npz'),'pose_file':'pose_gt.json',
        'pose_sha256':sha256(truth/'pose_gt.json'),'mesh_sha256':mesh_hashes,'heldout_camera_file':'heldout_camera.json',
        'heldout_camera_sha256':sha256(truth/'heldout_camera.json'),'heldout_mask_file':'heldout_labels.png',
        'heldout_mask_sha256':sha256(truth/'heldout_labels.png'),'preparation_config_sha256':sha256(args.config),
        'warning':'truth is for post-fit evaluation only'}
    (truth/'manifest.json').write_text(json.dumps(truth_manifest,indent=2))
    (root/'resolved_config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps({'status':manifest['status'],'prepared_root':str(root),'fit_input':str(fit)}))


if __name__=='__main__':
    main()
