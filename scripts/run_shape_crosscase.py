"""Predeclared multi-case Shape-only campaign; no replacement of failed cases."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import trimesh
from run_forward_check import PROJECT_ROOT, sha256, load_ply
from run_shape_expansion import write_json


CORE = ['run_shape_crosscase.py', 'run_forward_check.py', 'run_pose_only.py',
        'render_review_views.py', 'package_forward_observations.py',
        'prepare_shape_only.py', 'shape_only_common.py', 'run_shape_only.py',
        'evaluate_shape_only.py', 'run_shape_expansion.py', 'surface_metric_utils.py']


def child_running(pid):
    if not pid:
        return False
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def recorded_batch(log_path):
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text().splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get('batch_root'):
            return value['batch_root']
    return None


def select_cases(definition, forward):
    dataset = Path(forward['dataset'])
    split_path = dataset/'splits/train_split.json'
    split = set(json.loads(split_path.read_text()))
    names = sorted(p.name for p in (dataset/'SdfSamples').glob('*.npz') if p.name in split)
    latent_path = Path(forward['experiment'])/'LatentCodes'/f'latent_vecs_{forward["checkpoint"]}.pth'
    saved = torch.load(latent_path, map_location='cpu', weights_only=True)
    matrices = [saved['latent_codes'][f'{label}.module.weight'].numpy().astype(float)
                for label in definition['active_labels']]
    if any(len(m) != len(names) for m in matrices):
        raise ValueError('latent row count differs from resolved split')
    matrix = np.concatenate(matrices, axis=1)
    scale = matrix.std(axis=0, ddof=1)
    if np.any(scale <= 0):
        raise ValueError('zero latent training standard deviation')
    scores = np.sqrt(np.mean(((matrix-matrix.mean(axis=0))/scale)**2, axis=1))
    eligible, excluded = [], []
    required = set(definition['active_labels']) | {0}
    for row, name in enumerate(names):
        if '__mirror' in name or name in definition['exclude_cases']:
            continue
        pkl = dataset/'SdfSamples'/Path(name).with_suffix('.pkl')
        try:
            with pkl.open('rb') as handle:
                present = set(map(int, pickle.load(handle))) | {0}
            if present != required:
                raise ValueError('incomplete or unexpected tooth labels')
        except Exception as error:
            excluded.append({'case': name, 'reason': str(error)})
            continue
        eligible.append({'case': name, 'case_row': row, 'latent_standardized_rms_from_training_mean': float(scores[row])})
    ranked = sorted(eligible, key=lambda r: (r['latent_standardized_rms_from_training_mean'], r['case']))
    selected = []
    for quantile in definition['selection_quantiles_in_execution_order']:
        index = int(np.floor(quantile*(len(ranked)-1)+0.5))
        item = dict(ranked[index], selection_quantile=quantile, rank=index)
        if any(s['case'] == item['case'] for s in selected):
            raise ValueError('quantile selection duplicates a case')
        selected.append(item)
    return {'method': 'full-label nonmirror cases; nearest ordered rank at predeclared quantiles of 280D standardized latent RMS',
            'scope': 'training cases only; latent distance is not an anatomical diversity metric',
            'training_rows': len(names), 'checkpoint_epoch': saved['epoch'], 'eligible_count': len(ranked),
            'selected': selected, 'eligible_ranked': ranked, 'eligibility_excluded': excluded,
            'split_sha256': sha256(split_path), 'latent_checkpoint_sha256': sha256(latent_path)}


def execute(root, state, case_id, phase, arguments):
    plan = json.loads((root/'plan.json').read_text())
    for name, digest in plan['code_sha256'].items():
        if sha256(PROJECT_ROOT/'scripts'/name) != digest:
            raise RuntimeError(f'campaign code changed: {name}; preserve this campaign and create a new one')
    state.update({'status': 'CROSSCASE_RUNNING', 'current_case': case_id, 'phase': phase,
                  'updated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    case_root = root/'cases'/case_id
    log_path = case_root/f'{phase}.log'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    with log_path.open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-u', *arguments], cwd=PROJECT_ROOT,
                                 stdout=log, stderr=subprocess.STDOUT, env=env,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        state.update({'child_pid': child.pid, 'log': str(log_path)})
        write_json(root/'status.json', state)
        code = child.wait()
    state['child_pid'] = None
    write_json(root/'status.json', state)
    return code


def persist_config(path, config):
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f'configuration changed on resume: {path}')
    else:
        write_json(path, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/shape_crosscase.json')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.resume:
        root = args.resume.resolve(strict=True)
        plan = json.loads((root/'plan.json').read_text())
    else:
        definition = json.loads(args.config.read_text())
        forward = json.loads((PROJECT_ROOT/definition['forward_config']).read_text())
        shape = json.loads((PROJECT_ROOT/definition['shape_config']).read_text())
        expansion = json.loads((PROJECT_ROOT/definition['expansion_config']).read_text())
        selection = select_cases(definition, forward)
        root = PROJECT_ROOT/'runs'/dt.datetime.now(dt.timezone.utc).strftime('shape_crosscase_%Y%m%dT%H%M%SZ_')
        root = root.with_name(root.name+sha256(args.config)[:8])
        root.mkdir(parents=True, exist_ok=False)
        plan = {'definition': definition, 'forward_config': forward, 'shape_config': shape,
                'expansion_config': expansion, 'selection': selection,
                'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'case_gate': 'all planned seeds pass; every preselected case retained; independent cases continue after failures',
                'code_sha256': {name: sha256(PROJECT_ROOT/'scripts'/name) for name in CORE}}
        snapshot = root/'code_snapshot'
        snapshot.mkdir()
        for name in CORE:
            shutil.copyfile(PROJECT_ROOT/'scripts'/name, snapshot/name)
        source = Path(forward['dmm_root'])
        plan['dmm_source_sha256'] = {str(p.relative_to(source)): sha256(p)
            for folder in ['networks', 'utils'] for p in (source/folder).glob('*.py')}
        write_json(root/'plan.json', plan)
        write_json(root/'selection.json', selection)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('campaign must stay in project runs')
    state = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'cases': {}}
    if child_running(state.get('child_pid')):
        raise RuntimeError('a recorded child is still running; inspect this campaign instead of launching a duplicate')
    if args.prepare_only:
        state.update({'status': 'PLAN_READY', 'current_case': None, 'phase': 'idle', 'child_pid': None})
        write_json(root/'status.json', state)
        print(json.dumps({'campaign_root': str(root), 'selected': plan['selection']['selected']}))
        return
    print(json.dumps({'campaign_root': str(root), 'status': 'CROSSCASE_STARTED'}), flush=True)
    for selected in plan['selection']['selected']:
        case_id = Path(selected['case']).stem
        if state['cases'].get(case_id, {}).get('terminal'):
            continue
        case_root = root/'cases'/case_id
        case_root.mkdir(parents=True, exist_ok=True)
        forward_config = dict(plan['forward_config'], case=selected['case'])
        forward_config_path = case_root/'forward_config.json'
        persist_config(forward_config_path, forward_config)
        attempts = sorted(case_root.glob('source_attempt_*'))
        source = next((p for p in reversed(attempts) if (p/'report.json').exists()), None)
        if source is None:
            source = case_root/f'source_attempt_{len(attempts)+1:02d}'
            code = execute(root, state, case_id, 'forward', ['scripts/run_forward_check.py', '--config',
                           str(forward_config_path), '--run-root', str(source)])
            if code:
                state['cases'][case_id] = {'status': 'GENERATOR_FAILED', 'terminal': True, 'source_run': str(source)}
                write_json(root/'status.json', state)
                continue
        if not (source/'review_view_metrics.json').exists():
            if execute(root, state, case_id, 'source_review', ['scripts/render_review_views.py', str(source)]):
                raise RuntimeError('source review generation failed; inspect log')
        provenance = json.loads((source/'provenance.json').read_text())
        mesh_metrics = json.loads((source/'mesh_metrics.json').read_text())
        geometry = {}
        for label in plan['definition']['active_labels']:
            mesh = trimesh.Trimesh(*load_ply(source/'meshes/training_case'/f'tooth{label}.ply'), process=False)
            geometry[str(label)] = {'watertight': bool(mesh.is_watertight),
                'components': len(mesh.split(only_watertight=False)),
                'boundary_nonpositive_count': mesh_metrics['training_case'][str(label)]['boundary_nonpositive_count']}
        geometry_pass = all(v['watertight'] and v['components']==1 and v['boundary_nonpositive_count']==0 for v in geometry.values())
        write_json(case_root/'source_geometry.json', {'teeth_only': True, 'geometry': geometry, 'pass': geometry_pass})
        if not geometry_pass:
            state['cases'][case_id] = {'status': 'SOURCE_GEOMETRY_FAILED', 'terminal': True, 'source_run': str(source)}
            write_json(root/'status.json', state)
            continue
        if not (source/'fit_input/manifest.json').exists():
            if execute(root, state, case_id, 'package', ['scripts/package_forward_observations.py', str(source)]):
                raise RuntimeError('packaging failed; inspect log before resume')
        shape = dict(plan['shape_config'], source_run=str(source))
        shape_path = case_root/'shape_config.json'
        persist_config(shape_path, shape)
        definition = dict(plan['expansion_config'])
        definition['base_config'] = str(shape_path)
        definition['stages'] = [{'id': 'full_14', 'active_labels': plan['definition']['active_labels'],
                                'sigma': plan['definition']['sigma'], 'seeds': plan['definition']['seeds']}]
        expansion_path = case_root/'expansion_config.json'
        persist_config(expansion_path, definition)
        expansion_root = case_root/'expansion_root.json'
        if not expansion_root.exists():
            recovered = recorded_batch(case_root/'shape.log')
            if recovered:
                write_json(expansion_root, {'batch_root': recovered})
        if not expansion_root.exists():
            code = execute(root, state, case_id, 'shape', ['scripts/run_shape_expansion.py', '--plan', str(expansion_path)])
            started = recorded_batch(case_root/'shape.log')
            if not started:
                raise RuntimeError('shape controller did not record a batch; inspect log')
            write_json(expansion_root, {'batch_root': started})
        else:
            batch = Path(json.loads(expansion_root.read_text())['batch_root'])
            if json.loads((batch/'status.json').read_text()).get('status') != 'EXPANSION_COMPLETED':
                code = execute(root, state, case_id, 'shape_resume', ['scripts/run_shape_expansion.py', '--resume', str(batch)])
            else:
                code = 0
        if code:
            raise RuntimeError('shape controller failed; inspect logs and resume same campaign')
        batch = Path(json.loads(expansion_root.read_text())['batch_root'])
        batch_state = json.loads((batch/'status.json').read_text())
        state['cases'][case_id] = {'status': 'PASS' if batch_state['stages']['full_14']=='PASS' else 'SHAPE_FAIL',
            'terminal': True, 'source_run': str(source), 'shape_batch': str(batch),
            'case_row': provenance['case_row'], 'selection_quantile': selected['selection_quantile'], 'jobs': batch_state['jobs']}
        write_json(root/'status.json', state)
        print(json.dumps({'case': case_id, 'status': state['cases'][case_id]['status']}), flush=True)
    state.update({'status': 'CROSSCASE_COMPLETED', 'current_case': None, 'phase': 'idle', 'child_pid': None})
    write_json(root/'status.json', state)
    print(json.dumps({'campaign_root': str(root), 'cases': {k: v['status'] for k,v in state['cases'].items()}}))


if __name__ == '__main__':
    main()
