"""Frozen four-tooth single-factor checks; clean replay gates execution, not accuracy.

The historical engine is a private project with its own scripts/ and runs/.
Every altered observation/initialization gets its own image-only pose warmup.
Truth codes are used only by input preparation for prescribed synthetic seeds;
geometry truth is opened only by the independent postfit evaluator.
"""
from __future__ import annotations
import argparse
import copy
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
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from run_forward_check import PROJECT_ROOT, sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def fingerprint_tree(path):
    return {str(p.resolve()): sha256(p) for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def verify(root):
    guard = read(root/'integrity_guard.json')
    for category, files in guard.items():
        for name, digest in files.items():
            if sha256(Path(name)) != digest:
                raise ValueError(f'Frozen input/code changed: {category}: {name}')


def prepare(root, job, control):
    directory = root/'engine/runs/jobs'/job['id']
    directory.mkdir(parents=True)
    cp = read(control/'provenance.json')
    source = Path(cp['fit_input'])
    if sha256(source/'manifest.json') != cp['fit_manifest_sha256']:
        raise ValueError('Historical control input changed')
    shutil.copytree(source.parent, directory/'case')
    fit = directory/'case/fit_input'
    manifest = read(fit/'manifest.json')
    config = read(control/'resolved_config.json')
    warm_source = Path(config['warm_start_run'])
    warm_config = read(warm_source/'resolved_config.json')
    labels = manifest['active_labels']
    allowed = set()
    evidence = {'axis': job, 'source_control': str(control), 'alterations': [],
                'preparation_truth_codes_read': job['kind'] == 'initialization'}
    if job['kind'] == 'initialization':
        # This is synthetic input generation, never solver initialization from truth.
        gt = dict(np.load(directory/'case/truth/latents.npz'))
        scales = dict(np.load(fit/manifest['training_scales']))
        codes = {}
        for label in labels:
            key = f'label_{label}'
            rng = np.random.default_rng(np.random.SeedSequence([job['seed'], label]))
            codes[key] = gt[key].astype(np.float64) + rng.normal(size=len(scales[key])) * scales[key] * config['perturbation_training_std']
        np.savez_compressed(fit/manifest['initial_codes'], **codes)
        config['perturbation_seed'] = warm_config['perturbation_seed'] = job['seed']
        allowed.add(manifest['initial_codes'])
        evidence['alterations'].append('Only prescribed code perturbation direction; unchanged 0.25 training-std scale and pose')
    elif job['kind'] == 'pose_initialization':
        pose = read(fit/manifest['initial_pose'])
        for field in ['rotation_vector_radians', 'translation_dmm']:
            original = np.asarray(pose[field])
            pose[field] = (-original).tolist()
            assert np.linalg.norm(original) == np.linalg.norm(pose[field])
        write(fit/manifest['initial_pose'], pose)
        for cfg in [config, warm_config]:
            cfg['initial_rotation_degrees_xyz'] = np.rad2deg(pose['rotation_vector_radians']).tolist()
            cfg['initial_translation_dmm'] = pose['translation_dmm']
        allowed.add(manifest['initial_pose'])
        evidence['alterations'].append('Negated rotation vector and translation, preserving their norms; shape unchanged')
    elif job['kind'] == 'mask_boundary':
        if job['radius_pixels'] != 1:
            raise ValueError('Only predeclared 1 pixel radius allowed')
        kernel = np.ones((3, 3), dtype=bool)
        for item in manifest['masks']:
            ids = np.asarray(Image.open(fit/item['path']))
            if job['operation'] == 'erode':
                changed = np.zeros_like(ids)
                for label in labels:
                    changed[binary_erosion(ids == label, structure=kernel)] = label
            elif job['operation'] == 'dilate':
                candidates = np.stack([binary_dilation(ids == k, structure=kernel) for k in labels])
                distances = np.stack([distance_transform_edt(ids != k) for k in labels])
                distances[~candidates] = np.inf
                changed = np.where(candidates.any(axis=0), np.array(labels, dtype=ids.dtype)[np.argmin(distances, axis=0)], 0).astype(ids.dtype)
                changed[ids > 0] = ids[ids > 0]
            else:
                raise ValueError('Unknown operation')
            counts = {str(k): int(np.count_nonzero(changed == k)) for k in labels}
            if min(counts.values()) == 0:
                raise ValueError('Perturbation erased an observed tooth; retain failure')
            Image.fromarray(changed).save(fit/item['path'])
            item['sha256'] = sha256(fit/item['path'])
            allowed.add(item['path'])
            evidence['alterations'].append({'camera': item['camera'], 'changed_pixels': int(np.count_nonzero(ids != changed)), 'tooth_pixel_counts': counts})
    elif job['kind'] == 'view_drop':
        cameras = read(fit/manifest['camera_file'])
        retained = [c for c in cameras if c['name'] != job['drop_camera']]
        if len(retained) != len(cameras)-1 or len(retained) != 2:
            raise ValueError('Invalid predeclared drop')
        write(fit/manifest['camera_file'], retained)
        manifest['camera_sha256'] = sha256(fit/manifest['camera_file'])
        dropped = next(m for m in manifest['masks'] if m['camera'] == job['drop_camera'])
        (fit/dropped['path']).unlink()
        manifest['masks'] = [m for m in manifest['masks'] if m['camera'] != job['drop_camera']]
        allowed.update([manifest['camera_file'], dropped['path']])
        evidence['alterations'].append({'dropped_camera': job['drop_camera'], 'retained': [c['name'] for c in retained]})
    elif job['kind'] != 'clean_replay':
        raise ValueError('Unknown axis')
    # Enforce the single-factor payload difference before updating metadata.
    unchanged = {}
    for src in source.rglob('*'):
        if src.is_file():
            name = src.relative_to(source).as_posix()
            if name != 'manifest.json' and name not in allowed:
                unchanged[name] = sha256(fit/name) == sha256(src)
    if not all(unchanged.values()):
        raise ValueError('Unexpected payload change')
    evidence['unchanged_payload_files'] = unchanged
    evidence['allowed_changed_payload_files'] = sorted(allowed)
    for name in manifest['extra_input_sha256']:
        manifest['extra_input_sha256'][name] = sha256(fit/name)
    config['fit_input'] = str(fit)
    config['robustness'] = {'axis': job, 'control_run': str(control), 'clean_evaluation': 'all three original views and 65 deg holdout; postfit only'}
    if job['kind'] != 'clean_replay':
        manifest.pop('warm_start_manifest_sha256', None)
        manifest['robustness_axis'] = job
        manifest['observation_type'] = 'hard FDI labels; four active teeth; controlled single-factor input'
        warm = directory/'warmup'
        shutil.copytree(fit, warm/'fit_input')
        write(warm/'fit_input/manifest.json', manifest)
        warm_config['fit_input'] = str(warm/'fit_input')
        warm_config['robustness'] = config['robustness']
        write(warm/'config.json', warm_config)
        config['warm_start_run'] = str(warm/'fit_attempt_01')
    write(fit/'manifest.json', manifest)
    write(directory/'config.json', config)
    write(directory/'input_contract.json', {'evidence': evidence,
          'prepared_manifest_sha256': sha256(fit/'manifest.json'),
          'truth_copy_unchanged': fingerprint_tree(directory/'case/truth')})
    return directory


def finish_warmup(directory):
    fit = directory/'case/fit_input'
    manifest = read(fit/'manifest.json')
    manifest['warm_start_manifest_sha256'] = sha256(directory/'warmup/fit_input/manifest.json')
    write(fit/'manifest.json', manifest)
    contract = read(directory/'input_contract.json')
    contract['final_manifest_sha256'] = sha256(fit/'manifest.json')
    write(directory/'input_contract.json', contract)


def compare_replay(root, control, directory, tolerances):
    a = read(control/'evaluation.json'); b = read(directory/'fit_attempt_01/evaluation.json')
    delta = {'clean_iou': abs(a['final_active_mean_tooth_iou'] - b['final_active_mean_tooth_iou']),
             'heldout_iou': abs(a['heldout']['final']['active_mean_tooth_iou'] - b['heldout']['final']['active_mean_tooth_iou']),
             'rotation_degrees': abs(a['pose_errors']['final']['rotation_degrees'] - b['pose_errors']['final']['rotation_degrees']),
             'translation_dmm': abs(a['pose_errors']['final']['translation_dmm'] - b['pose_errors']['final']['translation_dmm']),
             'canonical_mean': abs(a['surface_aggregate']['canonical']['final'] - b['surface_aggregate']['canonical']['final'])}
    ca = dict(np.load(control/'final_codes.npz')); cb = dict(np.load(directory/'fit_attempt_01/final_codes.npz'))
    scales = dict(np.load(directory/'case/fit_input/training_scales.npz'))
    delta['normalized_code_max_abs'] = max(float(np.max(np.abs((ca[k]-cb[k])/scales[k]))) for k in ca)
    checks = {k: value <= tolerances[k] for k, value in delta.items()}
    checks['same_failed_gates'] = a['failed_checks'] == b['failed_checks']
    write(root/'clean_replay_comparison.json', {'status': 'REPLAY_MATCH' if all(checks.values()) else 'REPLAY_MISMATCH',
          'absolute_deltas': delta, 'tolerances': tolerances, 'checks': checks})
    return all(checks.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=PROJECT_ROOT/'configs/joint_robustness_four.json')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.resume:
        root = args.resume.resolve(strict=True)
        plan = read(root/'plan.json'); state = read(root/'status.json')
        if any(j['phase'].endswith('_running') for j in state['jobs']):
            raise ValueError('Recorded child may still be active: inspect recorded PID before recovery; never duplicate')
    else:
        plan = read(args.plan)
        root = PROJECT_ROOT/'runs'/(dt.datetime.now(dt.timezone.utc).strftime('joint_robustness_four_%Y%m%dT%H%M%SZ_')+sha256(args.plan)[:8])
        root.mkdir(parents=True, exist_ok=False)
        write(root/'plan.json', plan)
        engine = root/'engine/scripts'; engine.mkdir(parents=True)
        for source in Path(plan['engine_snapshot']).glob('*.py'):
            shutil.copyfile(source, engine/source.name)
        control = Path(plan['control_run'])
        cp = read(control/'provenance.json'); cfg = read(control/'resolved_config.json')
        for name, digest in cp['code_sha256'].items():
            if sha256(engine/name) != digest:
                raise ValueError('Historical engine does not match baseline')
        warm = Path(cfg['warm_start_run']); wp = read(warm/'provenance.json')
        for name, digest in wp['code_sha256'].items():
            if sha256(engine/name) != digest:
                raise ValueError('Warmup engine does not match baseline')
        shutil.copyfile(Path(__file__), root/'controller_snapshot.py')
        model = Path(cfg['experiment'])/'ModelParameters'/f'dmm_{cfg["checkpoint"]}.pth'
        write(root/'integrity_guard.json', {'engine': fingerprint_tree(engine), 'controller': {str(root/'controller_snapshot.py'): sha256(Path(__file__))},
              'control_case': fingerprint_tree(Path(cp['fit_input']).parent), 'control_fit': fingerprint_tree(control),
              'control_warmup': fingerprint_tree(warm), 'dmm_sources': cp['dmm_source_sha256'],
              'model': {str(model): cp['model_sha256']}})
        state = {'status': 'ROBUSTNESS_PREPARED', 'run_root': str(root), 'jobs': [], 'control_status': read(control/'evaluation.json')['status']}
        jobs = [{'id': 'clean_replay', 'kind': 'clean_replay'}] + plan['jobs']
        for job in jobs:
            item = {'id': job['id'], 'axis': job, 'phase': 'prepared'}
            try:
                item['directory'] = str(prepare(root, job, control))
            except Exception as error:
                item.update(phase='preparation_failed', error=repr(error))
            state['jobs'].append(item)
        write(root/'status.json', state)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('Batch outside workspace runs')
    verify(root)
    print(json.dumps({'run_root': str(root), 'jobs': state['jobs']}), flush=True)
    if args.prepare_only:
        return
    if (root/'clean_replay_comparison.json').exists() and read(root/'clean_replay_comparison.json')['status'] != 'REPLAY_MATCH':
        raise ValueError('Clean replay mismatch: perturbation campaign blocked')
    state['status'] = 'ROBUSTNESS_RUNNING'
    engine = root/'engine'; active = {}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    while True:
        replay_ready = (root/'clean_replay_comparison.json').exists()
        for index, item in enumerate(state['jobs']):
            if len(active) >= plan['max_workers']:
                break
            if item['phase'] not in ['prepared', 'warm_completed', 'fit_completed']:
                continue
            if item['id'] != 'clean_replay' and not replay_ready:
                continue
            verify(root)
            directory = Path(item['directory'])
            if item['phase'] == 'fit_completed':
                phase = 'eval'; command = ['evaluate_joint_robust.py', str(directory/'fit_attempt_01')]
            elif item['id'] == 'clean_replay' or item['phase'] == 'warm_completed':
                phase = 'fit'; command = ['run_joint_population.py', '--config', str(directory/'config.json')]
            else:
                phase = 'warm'; command = ['run_pose_warmup_only.py', '--config', str(directory/'warmup/config.json')]
            stream = (directory/f'{phase}_process.log').open('x', encoding='utf-8')
            command[0] = str(engine/'scripts'/command[0])
            process = subprocess.Popen([sys.executable, '-u'] + command, stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=engine, env=env, creationflags=flags)
            active[index] = (process, stream, phase)
            item.update(phase=phase+'_running', pid=process.pid)
            write(root/'status.json', state)
            print(f'START {item["id"]} {phase} pid={process.pid}', flush=True)
        for index, (process, stream, phase) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close(); item = state['jobs'][index]; directory = Path(item['directory']); del active[index]
            if code:
                item.update(phase=phase+'_failed', returncode=code)
                print(f'FAIL {item["id"]} {phase}: {code}', flush=True)
            elif phase == 'warm':
                finish_warmup(directory); item['phase'] = 'warm_completed'
            elif phase == 'fit':
                item['phase'] = 'fit_completed'
            else:
                ev = read(directory/'fit_attempt_01/evaluation.json')
                item.update(phase='evaluated', evaluation_status=ev['status'], failed_checks=ev['failed_checks'])
                print(f'DONE {item["id"]}: {ev["status"]} {ev["failed_checks"]}', flush=True)
                if item['id'] == 'clean_replay':
                    if not compare_replay(root, Path(plan['control_run']), directory, plan['replay_tolerances']):
                        state['status'] = 'CLEAN_REPLAY_MISMATCH'; write(root/'status.json', state); return
        write(root/'status.json', state)
        clean = state['jobs'][0]
        if clean['phase'].endswith('_failed'):
            state['status'] = 'CLEAN_REPLAY_EXECUTION_FAILED'; write(root/'status.json', state); return
        if not active and all(j['phase'] not in ['prepared', 'warm_completed', 'fit_completed'] for j in state['jobs']):
            break
        time.sleep(2)
    verify(root)
    state['status'] = 'ROBUSTNESS_EXPERIMENTS_COMPLETED_REVIEW_REQUIRED'
    write(root/'status.json', state)
    print(json.dumps({'status': state['status'], 'run_root': str(root)}), flush=True)


if __name__ == '__main__':
    main()
