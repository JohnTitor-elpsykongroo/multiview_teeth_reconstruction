"""Robustness: retain corrupted-target fit scores, judge clean observations after fit."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from run_forward_check import PROJECT_ROOT, sha256, load_ply
from run_pose_only import mean_iou, render_masks, save_render_comparison
from shape_only_common import world_meshes
from evaluate_shape_only import surface_metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_root',type=Path)
    args=parser.parse_args()
    root=args.run_root.resolve(strict=True)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('evaluation root must be in project runs')
    if (root/'evaluation.json').exists():
        raise FileExistsError('evaluation already exists')
    report=json.loads((root/'fit_report.json').read_text())
    if report['status']!='JOINT_FIT_COMPLETED_EVAL_REQUIRED':
        raise ValueError('Joint fit has not completed')
    provenance=json.loads((root/'provenance.json').read_text())
    if any(provenance[k] for k in ['active_truth_codes_read','ground_truth_pose_read','source_meshes_read','depth_maps_read']):
        raise ValueError('fitter read geometric truth')
    config=json.loads((root/'resolved_config.json').read_text())
    for name,digest in provenance['code_sha256'].items():
        if sha256(root/'code_snapshot'/name)!=digest or sha256(PROJECT_ROOT/'scripts'/name)!=digest:
            raise ValueError(f'fit code changed: {name}')
    for name,digest in provenance['dmm_source_sha256'].items():
        if sha256(Path(name))!=digest:
            raise ValueError('DMM source changed')
    fit=Path(provenance['fit_input'])
    if sha256(fit/'manifest.json')!=provenance['fit_manifest_sha256']:
        raise ValueError('fit manifest changed')
    manifest=json.loads((fit/'manifest.json').read_text())
    for name,digest in manifest['extra_input_sha256'].items():
        if sha256(fit/name)!=digest:
            raise ValueError('fit input changed')
    if sha256(fit/manifest['camera_file'])!=manifest['camera_sha256']:
        raise ValueError('camera changed')
    for item in manifest['masks']:
        if sha256(fit/item['path'])!=item['sha256']:
            raise ValueError('mask changed')
    model=Path(config['experiment'])/'ModelParameters'/f'dmm_{config["checkpoint"]}.pth'
    if sha256(model)!=manifest['model_sha256']:
        raise ValueError('model changed')
    labels=manifest['active_labels']
    initial_codes=dict(np.load(root/'initial_codes.npz'))
    final_codes=dict(np.load(root/'final_codes.npz'))
    input_codes=dict(np.load(fit/manifest['initial_codes']))
    if any(not np.array_equal(initial_codes[k],input_codes[k]) for k in initial_codes):
        raise ValueError('saved initial codes differ from input')
    initial_pose_data=json.loads((root/'initial_pose.json').read_text())
    final_pose_data=json.loads((root/'final_pose.json').read_text())
    input_pose_data=json.loads((fit/manifest['initial_pose']).read_text())
    if not np.array_equal(np.r_[input_pose_data['rotation_vector_radians'],input_pose_data['translation_dmm']],initial_pose_data['pose']):
        raise ValueError('initial pose differs from input')
    gradient=json.loads((root/'gradient_check.json').read_text())

    # The evaluator first opens geometry truth here, after completion and integrity checks.
    truth_root=fit.parent/'truth'
    truth_path=truth_root/'manifest.json'
    truth=json.loads(truth_path.read_text())
    for name,digest in [(truth['latent_file'],truth['latent_sha256']),(truth['pose_file'],truth['pose_sha256']),
                        (truth['heldout_camera_file'],truth['heldout_camera_sha256']),(truth['heldout_mask_file'],truth['heldout_mask_sha256'])]:
        if sha256(truth_root/name)!=digest:
            raise ValueError('truth changed')
    for name,digest in truth['mesh_sha256'].items():
        if sha256(truth_root/'meshes'/name)!=digest:
            raise ValueError('truth mesh changed')
    gt_matrix=np.array(json.loads((truth_root/truth['pose_file']).read_text())['arch_to_world'])
    gt_codes=dict(np.load(truth_root/truth['latent_file']))
    gt_meshes={k:load_ply(truth_root/'meshes'/f'tooth{k}.ply') for k in labels}
    gt_world=world_meshes(gt_meshes,gt_matrix)
    scales=dict(np.load(fit/manifest['training_scales']))
    meshes_by_stage={stage:{k:load_ply(root/'meshes'/stage/f'tooth{k}.ply') for k in labels} for stage in ['initial','final']}
    poses={'initial':initial_pose_data,'final':final_pose_data}
    surface,geometry,latent={},{},{}
    for label in labels:
        surface[str(label)]={}
        for stage in ['initial','final']:
            mesh=meshes_by_stage[stage][label]
            world=world_meshes({label:mesh},poses[stage]['arch_to_world'])[label]
            surface[str(label)][stage]={'canonical':surface_metrics(mesh,gt_meshes[label]),'world':surface_metrics(world,gt_world[label])}
        values=surface[str(label)]
        values['canonical_mean_improvement_fraction']=1-values['final']['canonical']['symmetric_sampled_surface_mean_dmm']/values['initial']['canonical']['symmetric_sampled_surface_mean_dmm']
        values['world_mean_improvement_fraction']=1-values['final']['world']['symmetric_sampled_surface_mean_dmm']/values['initial']['world']['symmetric_sampled_surface_mean_dmm']
        mesh=trimesh.Trimesh(*meshes_by_stage['final'][label],process=False)
        geometry[str(label)]={'watertight':bool(mesh.is_watertight),'connected_components':len(mesh.split(only_watertight=False))}
        key=f'label_{label}'
        latent[str(label)]={f'{stage}_rms_error_training_std':float(np.sqrt(np.mean(((codes[key]-gt_codes[key])/scales[key])**2)))
            for stage,codes in [('initial',initial_codes),('final',final_codes)]}
    pose_errors={}
    for stage in ['initial','final']:
        matrix=np.array(poses[stage]['arch_to_world'])
        pose_errors[stage]={'rotation_degrees':float(np.rad2deg(Rotation.from_matrix(matrix[:3,:3]@gt_matrix[:3,:3].T).magnitude())),
            'translation_dmm':float(np.linalg.norm(matrix[:3,3]-gt_matrix[:3,3]))}
    camera=json.loads((truth_root/truth['heldout_camera_file']).read_text())
    name=camera['name']; targets={name:np.asarray(Image.open(truth_root/truth['heldout_mask_file']))}
    heldout={}
    for stage in ['initial','final']:
        predicted,_=render_masks(meshes_by_stage[stage],[camera],np.array(poses[stage]['pose']))
        iou,per_view,micro=mean_iou(predicted,targets,labels)
        save_render_comparison(root,f'heldout_{stage}',predicted,targets,[camera],labels)
        heldout[stage]={'active_mean_tooth_iou':iou,'per_view':per_view,'micro_tooth_iou':micro}
    aggregate={}
    for frame in ['canonical','world']:
        values={stage:float(np.mean([v[stage][frame]['symmetric_sampled_surface_mean_dmm'] for v in surface.values()])) for stage in ['initial','final']}
        values['mean_improvement_fraction']=1-values['final']/values['initial']
        aggregate[frame]=values
    limits=config['acceptance']
    noisy_scores={'initial':report['initial']['active_mean_tooth_iou'],'final':report['final']['active_mean_tooth_iou']}
    clean_manifest=json.loads((truth_root/'clean_fit_manifest.json').read_text())
    clean_cameras=json.loads((truth_root/clean_manifest['camera_file']).read_text())
    if sha256(truth_root/clean_manifest['camera_file'])!=clean_manifest['camera_sha256']:
        raise ValueError('clean evaluation cameras changed')
    clean_targets={}
    for item in clean_manifest['masks']:
        if sha256(truth_root/item['path'])!=item['sha256']:
            raise ValueError('clean evaluation masks changed')
        clean_targets[item['camera']]=np.asarray(Image.open(truth_root/item['path']))
    clean_fit={}
    for stage in ['initial','final']:
        predicted,_=render_masks(meshes_by_stage[stage],clean_cameras,np.array(poses[stage]['pose']))
        iou,per_view,micro=mean_iou(predicted,clean_targets,labels)
        save_render_comparison(root,f'clean_{stage}',predicted,clean_targets,clean_cameras,labels)
        clean_fit[stage]={'active_mean_tooth_iou':iou,'per_view':per_view,'micro_tooth_iou':micro}
    # Absolute original image gates are applied to all three clean cameras.
    # Noisy/two-view target scores remain separately recorded, never compared as clean IoU.
    for stage in ['initial','final']:
        report[stage]=dict(report[stage],active_mean_tooth_iou=clean_fit[stage]['active_mean_tooth_iou'])
    fit_gain=report['final']['active_mean_tooth_iou']-report['initial']['active_mean_tooth_iou']
    heldout_gain=heldout['final']['active_mean_tooth_iou']-heldout['initial']['active_mean_tooth_iou']
    checks={'joint_gradient':gradient['max_relative_error']<=limits['max_gradient_relative_error'],
        'rotation_error':pose_errors['final']['rotation_degrees']<=limits['max_rotation_error_degrees'],
        'translation_error':pose_errors['final']['translation_dmm']<=limits['max_translation_error_dmm'],
        'fitting_iou':report['final']['active_mean_tooth_iou']>=limits['min_active_mean_tooth_iou'],
        'fitting_iou_gain':fit_gain>=limits['min_active_iou_gain'],'heldout_iou_gain':heldout_gain>=limits['min_heldout_iou_gain'],
        'canonical_shape_improvement':aggregate['canonical']['mean_improvement_fraction']>=limits['min_surface_mean_improvement_fraction'],
        'world_geometry_improvement':aggregate['world']['mean_improvement_fraction']>=limits['min_world_surface_improvement_fraction'],
        'each_tooth_canonical_nondegradation':all(v['canonical_mean_improvement_fraction']>=-limits['max_individual_surface_degradation_fraction'] for v in surface.values()),
        'each_tooth_canonical_p95_nondegradation':all(v['final']['canonical']['symmetric_sampled_surface_p95_dmm']<=v['initial']['canonical']['symmetric_sampled_surface_p95_dmm']*(1+limits['max_individual_surface_p95_degradation_fraction']) for v in surface.values()),
        'closed_connected_meshes':all(v['watertight'] and v['connected_components']==1 for v in geometry.values()),
        'pose_and_codes_both_changed':not np.array_equal(initial_pose_data['pose'],final_pose_data['pose']) and all(not np.array_equal(initial_codes[k],final_codes[k]) for k in initial_codes)}
    checks={k:bool(v) for k,v in checks.items()}
    evaluation={'status':'JOINT_MINIMAL_PASS' if all(checks.values()) else 'JOINT_MINIMAL_FAIL',
        'evaluated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'truth_read_after_fit':True,
        'truth_manifest_sha256':sha256(truth_path),'active_labels':labels,'active_latent_dimensions':report['active_latent_dimensions'],
        'total_dimensions':report['total_dimensions'],'pose_errors':pose_errors,'surface_aggregate':aggregate,'surface_per_tooth':surface,
        'heldout':heldout,'initial_active_mean_tooth_iou':report['initial']['active_mean_tooth_iou'],
        'final_active_mean_tooth_iou':report['final']['active_mean_tooth_iou'],'fitting_iou_gain':fit_gain,'heldout_iou_gain':heldout_gain,
        'gradient_max_relative_error':gradient['max_relative_error'],'geometry':geometry,'latent_error_diagnostic_only':latent,
        'acceptance_limits':limits,'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],
        'scope':'two-teeth-only robustness; known cameras; perturbed observations/initialization; absolute image gates on all three clean post-fit views',
        'observed_target_iou':noisy_scores,'clean_fit_views':clean_fit,'robustness':config['robustness'],
        'clean_evaluation_truth_read_after_fit':True}
    (root/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
    print(json.dumps({'status':evaluation['status'],'run_root':str(root),'pose_errors':pose_errors,
        'surface_aggregate':aggregate,'heldout_final_iou':heldout['final']['active_mean_tooth_iou'],'failed_checks':evaluation['failed_checks']}))


if __name__=='__main__':
    main()
