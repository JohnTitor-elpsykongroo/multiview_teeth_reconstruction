"""Post-fit local Jacobian diagnosis; reads fit inputs/results, no generator truth."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from run_forward_check import PROJECT_ROOT, load_ply
from run_pose_only import render_masks
from shape_only_common import load_model, make_pairs
from joint_common import JointImageObjective, pose_matrix
from PIL import Image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_root',type=Path)
    parser.add_argument('--output',default='observability.json')
    args=parser.parse_args()
    root=args.run_root.resolve(strict=True)
    if Path(args.output).name!=args.output or not args.output.endswith('.json'):
        raise ValueError('output must be a JSON filename')
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()) or (root/args.output).exists():
        raise ValueError('invalid root or diagnosis already exists')
    report=json.loads((root/'fit_report.json').read_text())
    if report['status']!='JOINT_FIT_COMPLETED_EVAL_REQUIRED':
        raise ValueError('fit has not completed')
    config=json.loads((root/'resolved_config.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    fit=Path(provenance['fit_input'])
    manifest=json.loads((fit/'manifest.json').read_text())
    labels=manifest['active_labels']
    initial=dict(np.load(fit/manifest['initial_codes']))
    scales=dict(np.load(fit/manifest['training_scales']))
    cameras=json.loads((fit/manifest['camera_file']).read_text())
    targets={v['camera']:np.asarray(Image.open(fit/v['path'])) for v in manifest['masks']}
    initial_pose=np.array(json.loads((root/'initial_pose.json').read_text())['pose'])
    final_pose=np.array(json.loads((root/'final_pose.json').read_text())['pose'])
    x=np.load(root/'latest_parameters.npz')['parameters']
    pose_scale=np.r_[np.repeat(np.deg2rad(config['pose_rotation_scale_degrees']),3),np.repeat(config['pose_translation_scale_dmm'],3)]
    if not np.allclose(initial_pose+x[:6]*pose_scale,final_pose,atol=1e-12):
        raise ValueError('saved best parameters differ from final pose')
    meshes={k:load_ply(root/'meshes/final'/f'tooth{k}.ply') for k in labels}
    masks,maps=render_masks(meshes,cameras,final_pose,True)
    pairs=make_pairs(masks,maps,targets,cameras,labels,pose_matrix(final_pose),config['boundary_samples_per_direction'])
    net=load_model(config,manifest)
    obj=JointImageObjective(net,labels,initial,scales,pairs,cameras,initial_pose,pose_scale,x,config['initial_code_damping'])
    image_J=obj.jacobian(x)[:obj.image_rows]
    pose_J,code_J=image_J[:,:6],image_J[:,6:]
    U,S,_=np.linalg.svd(code_J,full_matrices=False)
    rank=int(np.count_nonzero(S>S[0]*max(code_J.shape)*np.finfo(float).eps))
    basis=U[:,:rank]
    remaining=pose_J-basis@(basis.T@pose_J)
    records=[]
    for index,name in enumerate(['rotation_x','rotation_y','rotation_z','translation_x','translation_y','translation_z']):
        relative=float(np.linalg.norm(remaining[:,index])/np.linalg.norm(pose_J[:,index]))
        records.append({'axis':name,'residual_norm_fraction_after_code_projection':relative,
            'pose_image_energy_explained_by_latent_span':1-relative**2})
    values=np.linalg.svd(image_J,compute_uv=False)
    pose_basis,_=np.linalg.qr(pose_J)
    cosines=np.clip(np.linalg.svd(basis.T@pose_basis,compute_uv=False),0,1)
    result={'scope':'local final-fit image Jacobian, fixed visibility/correspondences; no generator truth read',
        'parameter_scaling':'configured pose scales and per-dimension training code std',
        'code_image_rank':rank,'joint_image_singular_values':values.tolist(),
        'joint_image_condition_number':float(values[0]/values[-1]),'pose_latent_overlap':records,
        'pose_latent_principal_angle_cosines':cosines.tolist(),
        'pose_latent_principal_angles_degrees':np.rad2deg(np.arccos(cosines)).tolist(),
        'note':'linearized span overlap diagnoses compensation directions; not a proof of an exact global gauge'}
    (root/args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'condition':result['joint_image_condition_number'],'principal_angles_degrees':result['pose_latent_principal_angles_degrees'],'pose_latent_overlap':records}))


if __name__=='__main__':
    main()
