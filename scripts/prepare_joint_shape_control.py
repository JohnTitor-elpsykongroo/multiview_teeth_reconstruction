"""Package a matched Shape-only oracle-pose diagnostic for the Joint scene."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
import numpy as np
from run_forward_check import PROJECT_ROOT, sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('joint_case_root',type=Path)
    args=parser.parse_args()
    source=args.joint_case_root.resolve(strict=True)
    if not source.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('source must be a project run')
    original=json.loads((source/'fit_input/manifest.json').read_text())
    if original['status']!='JOINT_SYNTHETIC_INPUT_READY' or original['frozen_labels']:
        raise ValueError('expected active-only Joint generator package')
    joint_config=json.loads((source/'resolved_config.json').read_text())
    config=json.loads((PROJECT_ROOT/'configs/shape_only.json').read_text())
    for key in ['active_labels','perturbation_seed','perturbation_training_std','cpu_threads','outer_iterations','inner_max_nfev',
                'boundary_samples_per_direction','initial_code_damping','global_bound_training_std','local_step_training_std',
                'local_trust_retries','stop_after_no_best_iterations','sampling_padding_cells']:
        config[key]=joint_config[key]
    config['acceptance']=json.loads((PROJECT_ROOT/'configs/shape_expansion.json').read_text())['acceptance']
    root=PROJECT_ROOT/'runs'/(dt.datetime.now(dt.timezone.utc).strftime('joint_shape_control_case_%Y%m%dT%H%M%SZ_')+sha256(source/'fit_input/manifest.json')[:8])
    shutil.copytree(source,root)
    fit=root/'fit_input'
    manifest=json.loads((fit/'manifest.json').read_text())
    manifest.update(status='SHAPE_ONLY_SYNTHETIC_INPUT_READY',fixed_arch_to_world=np.eye(4).tolist(),
        diagnostic_oracle_conditions=['known cameras','true upper arch pose for matched Shape-only diagnostic','same initial codes/masks as Joint'],
        ground_truth_pose_available_to_fitter=True)
    manifest.pop('initial_pose')
    manifest['extra_input_sha256'].pop('initial_pose.json')
    # Keep the old perturbed-pose file as provenance; Shape-only does not consume it.
    manifest['matched_joint_fit_manifest_sha256']=sha256(source/'fit_input/manifest.json')
    (fit/'manifest.json').write_text(json.dumps(manifest,indent=2))
    truth_path=root/'truth/manifest.json'
    truth=json.loads(truth_path.read_text())
    truth.update(ground_truth_arch_pose='identity in the saved DMM world frame',source_run=joint_config['source_run'])
    truth_path.write_text(json.dumps(truth,indent=2))
    (root/'control_config.json').write_text(json.dumps(config,indent=2))
    (root/'matched_control_provenance.json').write_text(json.dumps({'source_joint_case':str(source),
        'only_fit_condition_changed':'pose fixed to generator identity instead of optimized',
        'codes_sha256_equal':sha256(fit/'initial_codes.npz')==sha256(source/'fit_input/initial_codes.npz'),
        'cameras_sha256_equal':sha256(fit/'cameras.json')==sha256(source/'fit_input/cameras.json'),
        'mask_sha256_equal':all(sha256(fit/m['path'])==m['sha256'] for m in original['masks'])},indent=2))
    print(json.dumps({'prepared_root':str(root),'config':str(root/'control_config.json'),'fit_input':str(fit)}))


if __name__=='__main__':
    main()
