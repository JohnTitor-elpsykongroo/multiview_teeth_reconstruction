"""Image-only coarse and fixed perturbed-shape pose warm-up for new scenes."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from run_forward_check import PROJECT_ROOT,sha256
from run_pose_only import target_features,centroid_residual
from shape_only_common import load_model,decode
from pose_warmup import optimize_pose

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); args=parser.parse_args()
    config=json.loads(args.config.read_text()); fit=Path(config['fit_input']); root=args.config.parent/'fit_attempt_01'
    root.mkdir(exist_ok=False); m=json.loads((fit/'manifest.json').read_text())
    if m['frozen_labels'] or 'fixed_arch_to_world' in m: raise ValueError('invalid joint input')
    for n,h in m['extra_input_sha256'].items():
        if sha256(fit/n)!=h: raise ValueError('input changed')
    if sha256(fit/m['camera_file'])!=m['camera_sha256']: raise ValueError('camera changed')
    cameras=json.loads((fit/m['camera_file']).read_text()); targets={}
    for item in m['masks']:
        if sha256(fit/item['path'])!=item['sha256']: raise ValueError('mask changed')
        targets[item['camera']]=np.asarray(Image.open(fit/item['path']))
    codes=dict(np.load(fit/m['initial_codes'])); boxes=json.loads((fit/m['sampling_boxes']).read_text()); labels=m['active_labels']
    pd=json.loads((fit/m['initial_pose']).read_text()); initial=np.r_[pd['rotation_vector_radians'],pd['translation_dmm']]
    scale=np.r_[np.repeat(np.deg2rad(config['pose_rotation_scale_degrees']),3),np.repeat(config['pose_translation_scale_dmm'],3)]
    bound=np.r_[np.repeat(config['pose_rotation_bound_degrees']/config['pose_rotation_scale_degrees'],3),np.repeat(config['pose_translation_bound_dmm']/config['pose_translation_scale_dmm'],3)]
    step=np.r_[np.repeat(config['local_pose_rotation_step_degrees']/config['pose_rotation_scale_degrees'],3),np.repeat(config['local_pose_translation_step_dmm']/config['pose_translation_scale_dmm'],3)]
    names=['run_pose_warmup_only.py','pose_warmup.py','joint_common.py','shape_only_common.py','run_forward_check.py','run_pose_only.py']
    (root/'code_snapshot').mkdir()
    for n in names: shutil.copyfile(PROJECT_ROOT/'scripts'/n,root/'code_snapshot'/n)
    provenance={'fit_input':str(fit),'fit_manifest_sha256':sha256(fit/'manifest.json'),'model_sha256':m['model_sha256'],
        'active_truth_codes_read':False,'ground_truth_pose_read':False,'source_meshes_read':False,'depth_maps_read':False,
        'code_sha256':{n:sha256(PROJECT_ROOT/'scripts'/n) for n in names},'scope':'image-only pose initialization; no joint acceptance claim'}
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)); (root/'resolved_config.json').write_text(json.dumps(config,indent=2)); np.savez_compressed(root/'initial_codes.npz',**codes)
    net=load_model(config,m); meshes={k:decode(net,k,codes[f'label_{k}'],boxes[str(k)]) for k in labels}
    centers={k:v[0].mean(0).astype(float) for k,v in meshes.items()}; features=target_features(targets,cameras,labels)
    def coarse(u): return np.r_[centroid_residual(initial+u*scale,cameras,labels,centers,features),config['centroid_initial_pose_damping']*u]
    solution=least_squares(coarse,np.zeros(6),bounds=(-bound,bound),max_nfev=100,loss='soft_l1',f_scale=5)
    u,_,_,metrics=optimize_pose(root,meshes,cameras,targets,labels,initial,scale,solution.x,bound,step,config)
    (root/'fit_report.json').write_text(json.dumps({'status':'IMAGE_ONLY_POSE_WARMUP_COMPLETED','warmup':metrics},indent=2))
    print(str(root))
if __name__=='__main__': main()
