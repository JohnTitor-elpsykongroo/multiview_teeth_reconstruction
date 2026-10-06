"""Post-fit truth and held-out-view evaluation for the limited Shape-only case."""

from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from run_forward_check import load_ply, rasterize, sha256
from run_pose_only import mean_iou, save_render_comparison
from shape_only_common import world_meshes
from surface_metric_utils import finite_nearest_triangle_distances


def sampled_points(mesh, count, seed):
    vertices,faces = mesh
    triangles = vertices[faces].astype(float)
    areas = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles),size=count,p=areas/areas.sum())]
    u = np.sqrt(rng.uniform(size=count))
    v = rng.uniform(size=count)
    return (1-u[:,None])*selected[:,0] + (u*(1-v))[:,None]*selected[:,1] + (u*v)[:,None]*selected[:,2]


def nearest_triangle_distances(points, mesh):
    # Candidate triangles from centroids; dense regular marching-cubes meshes
    # keep candidates local. This is a sampled surface metric, not Hausdorff.
    return finite_nearest_triangle_distances(points, mesh)


def surface_metrics(first, second):
    diagnostics=[{},{}]
    distances=np.r_[finite_nearest_triangle_distances(sampled_points(first,6000,410),second,diagnostics[0]),
                    finite_nearest_triangle_distances(sampled_points(second,6000,411),first,diagnostics[1])]
    return {'symmetric_sampled_surface_mean_dmm':float(distances.mean()),
            'symmetric_sampled_surface_p95_dmm':float(np.percentile(distances,95)),
            'samples_per_direction':6000,'nearest_triangle_candidates':32,
            'distance_diagnostics':diagnostics}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_root',type=Path)
    args=parser.parse_args()
    root=args.run_root.resolve(strict=True)
    if (root/'evaluation.json').exists():
        raise FileExistsError('evaluation already exists; refusing to overwrite')
    report=json.loads((root/'fit_report.json').read_text())
    if report['status']!='SHAPE_ONLY_FIT_COMPLETED_EVAL_REQUIRED':
        raise ValueError('fit has not completed')
    provenance=json.loads((root/'provenance.json').read_text())
    config=json.loads((root/'resolved_config.json').read_text())
    fit=Path(provenance['fit_input'])
    if sha256(fit/'manifest.json')!=provenance['fit_manifest_sha256']:
        raise ValueError('fit manifest changed')
    manifest=json.loads((fit/'manifest.json').read_text())
    if provenance['active_truth_codes_read']:
        raise ValueError('fitter used active truth')
    for name,digest in manifest['extra_input_sha256'].items():
        if sha256(fit/name)!=digest:
            raise ValueError('fit input changed')
    initial_codes=dict(np.load(root/'initial_codes.npz'))
    final_codes=dict(np.load(root/'final_codes.npz'))
    original_initial=dict(np.load(fit/manifest['initial_codes']))
    if any(not np.array_equal(initial_codes[k],v) for k,v in original_initial.items()):
        raise ValueError('saved initial codes differ from fit input')
    frozen_unchanged=all(np.array_equal(initial_codes[f'label_{k}'],final_codes[f'label_{k}'])
                         for k in manifest['frozen_labels'])
    active=manifest['active_labels']
    labels=manifest['visible_fdi_ids']
    gradient=json.loads((root/'gradient_check.json').read_text())

    # Generator truth and held-out images are first opened after fit completion.
    truth_root=fit.parent/'truth'
    truth_path=truth_root/'manifest.json'
    truth=json.loads(truth_path.read_text())
    if truth['ground_truth_arch_pose']!='identity in the saved DMM world frame':
        raise ValueError('unsupported ground truth pose')
    latent_path=truth_root/truth['latent_file']
    if sha256(latent_path)!=truth['latent_sha256']:
        raise ValueError('truth latent hash mismatch')
    gt_codes=dict(np.load(latent_path))
    source=Path(truth['source_run'])
    gt_meshes={k:load_ply(source/'meshes/training_case'/f'tooth{k}.ply') for k in active}
    initial_meshes={k:load_ply(root/'meshes/initial'/f'tooth{k}.ply') for k in labels}
    final_meshes={k:load_ply(root/'meshes/final'/f'tooth{k}.ply') for k in labels}
    surface,geometry,latent={},{},{}
    scales=dict(np.load(fit/manifest['training_scales']))
    for label in active:
        key=f'label_{label}'
        a=surface_metrics(initial_meshes[label],gt_meshes[label])
        b=surface_metrics(final_meshes[label],gt_meshes[label])
        surface[str(label)]={'initial':a,'final':b,'mean_improvement_fraction':
            1-b['symmetric_sampled_surface_mean_dmm']/a['symmetric_sampled_surface_mean_dmm']}
        mesh=trimesh.Trimesh(*final_meshes[label],process=False)
        geometry[str(label)]={'watertight':bool(mesh.is_watertight),
                              'connected_components':len(mesh.split(only_watertight=False))}
        latent[str(label)]={'initial_rms_error_training_std':float(np.sqrt(np.mean(((initial_codes[key]-gt_codes[key])/scales[key])**2))),
                           'final_rms_error_training_std':float(np.sqrt(np.mean(((final_codes[key]-gt_codes[key])/scales[key])**2)))}
    camera_path=truth_root/truth['heldout_camera_file']
    mask_path=truth_root/truth['heldout_mask_file']
    if sha256(camera_path)!=truth['heldout_camera_sha256'] or sha256(mask_path)!=truth['heldout_mask_sha256']:
        raise ValueError('heldout input hash mismatch')
    camera=json.loads(camera_path.read_text())
    name=camera['name']
    target={name:np.asarray(Image.open(mask_path))}
    heldout={}
    for stage,meshes in [('initial',initial_meshes),('final',final_meshes)]:
        masks={name:rasterize(world_meshes(meshes,manifest['fixed_arch_to_world']),camera,include_gum=False,shade=False)[0]}
        iou,per_view,micro=mean_iou(masks,target,active)
        heldout[stage]={'active_mean_tooth_iou':iou,'active_micro_tooth_iou':micro,'per_view':per_view}
        comparison_root=root/'renders'/f'heldout_{stage}'
        if comparison_root.exists():
            saved_mask=np.asarray(Image.open(comparison_root/f'{name}_predicted_labels.png'))
            if not np.array_equal(saved_mask,masks[name]):
                raise ValueError('previous partial evaluation render differs')
        else:
            save_render_comparison(root,f'heldout_{stage}',masks,target,[camera],active)
    surface_initial=np.mean([v['initial']['symmetric_sampled_surface_mean_dmm'] for v in surface.values()])
    surface_final=np.mean([v['final']['symmetric_sampled_surface_mean_dmm'] for v in surface.values()])
    surface_gain=1-surface_final/surface_initial
    fit_gain=report['final']['active_mean_tooth_iou']-report['initial']['active_mean_tooth_iou']
    heldout_gain=heldout['final']['active_mean_tooth_iou']-heldout['initial']['active_mean_tooth_iou']
    fit_error_reduction=fit_gain/max(1-report['initial']['active_mean_tooth_iou'],1e-8)
    heldout_error_reduction=heldout_gain/max(1-heldout['initial']['active_mean_tooth_iou'],1e-8)
    limits=config['acceptance']
    checks={
        'image_to_latent_gradient':gradient['max_relative_error']<=limits['max_gradient_relative_error'],
        'active_mean_tooth_iou':report['final']['active_mean_tooth_iou']>=limits['min_active_mean_tooth_iou'],
        'active_iou_gain':fit_gain>=limits['min_active_iou_gain'],
        'surface_improvement':surface_gain>=limits['min_surface_mean_improvement_fraction'],
        'heldout_iou_improvement':heldout_gain>=limits['min_heldout_active_iou_gain'],
        'frozen_codes_unchanged':frozen_unchanged,
        'closed_connected_active_meshes':all(v['watertight'] and v['connected_components']==1 for v in geometry.values()),
    }
    checks={key:bool(value) for key,value in checks.items()}
    if 'min_active_iou_error_reduction_fraction' in limits:
        checks['active_iou_error_reduction']=bool(fit_error_reduction>=limits['min_active_iou_error_reduction_fraction'])
        checks['heldout_iou_error_reduction']=bool(heldout_error_reduction>=limits['min_heldout_iou_error_reduction_fraction'])
        checks['each_tooth_surface_nondegradation']=all(v['mean_improvement_fraction']>=-limits['max_individual_surface_degradation_fraction'] for v in surface.values())
        checks['each_tooth_surface_p95_nondegradation']=all(v['final']['symmetric_sampled_surface_p95_dmm']<=v['initial']['symmetric_sampled_surface_p95_dmm']*(1+limits['max_individual_surface_p95_degradation_fraction']) for v in surface.values())
    evaluation={
        'status':'SHAPE_ONLY_SUBSET_PASS' if all(checks.values()) else 'SHAPE_ONLY_SUBSET_FAIL',
        'run_id':root.name,'evaluated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'active_labels':active,'active_latent_dimensions':report['active_latent_dimensions'],
        'truth_read_after_fit':True,'truth_manifest_sha256':sha256(truth_path),
        'initial_active_mean_tooth_iou':report['initial']['active_mean_tooth_iou'],
        'final_active_mean_tooth_iou':report['final']['active_mean_tooth_iou'],
        'active_iou_gain':fit_gain,'active_iou_error_reduction_fraction':fit_error_reduction,
        'heldout_iou_error_reduction_fraction':heldout_error_reduction,
        'gradient_max_relative_error':gradient['max_relative_error'],
        'initial_surface_mean_dmm':float(surface_initial),'final_surface_mean_dmm':float(surface_final),
        'surface_mean_improvement_fraction':float(surface_gain),'surface_per_tooth':surface,
        'heldout':heldout,'heldout_iou_gain':heldout_gain,'latent_error_diagnostic_only':latent,
        'geometry':geometry,'acceptance_limits':limits,'checks':checks,
        'surface_metric_revision':'v1: same samples and nearest-32 candidates; nonfinite distances use exact segment/plane fallback',
        'metric_code_sha256':{name:sha256(Path(__file__).parent/name) for name in ['evaluate_shape_only.py','surface_metric_utils.py']},
        'scope':f'{len(active)} active teeth, same-model noiseless synthetic masks; known pose/cameras; {len(manifest["frozen_labels"])} inactive teeth fixed at true codes',
    }
    (root/'evaluation.json').write_text(json.dumps(evaluation,indent=2,allow_nan=False))
    print(json.dumps({'status':evaluation['status'],'run_root':str(root),
                     'surface_mean_improvement_fraction':surface_gain,'heldout_iou_gain':heldout_gain,
                     'checks':checks}))


if __name__=='__main__':
    main()
