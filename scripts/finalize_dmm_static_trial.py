"""Finish plots, post-fit geometry evaluation and integrity audit after fitting."""
import concurrent.futures
import difflib
import hashlib
import html
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUN = Path((PROJECT / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
NAME = 'fit_pilot_008_resume_persistent'
FIT = RUN / NAME
PYTHON = Path(r'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe')
PACKAGE = PROJECT / '.runtime/dmm_r4_best_val_20261015/package'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


state_path = RUN / 'finalization_state.json'
write(state_path, dict(status='WAITING_FOR_NATIVE_FIT', fit=NAME))
deadline = time.monotonic()+6*3600
while True:
    control = RUN / (NAME + '_state.json')
    if control.exists():
        try:
            state = json.loads(control.read_text('utf-8'))
            if state['status'] != 'RESUME_RUNNING':
                assert state['status'] == 'RESUME_FINISHED' and state['returncode'] == 0, state
                assert (FIT / 'result.json').is_file()
                break
        except json.JSONDecodeError:
            pass
    assert time.monotonic() < deadline, 'Finalization wait exceeded six hours; fit artifacts retained'
    time.sleep(10)
write(state_path, dict(status='BUILDING_REVIEW_AND_GEOMETRY_EVALUATION', fit=NAME))


def execute(script, log_name):
    with (RUN / log_name).open('w', encoding='utf-8') as log:
        result = subprocess.run([str(PYTHON), '-B', str(PROJECT / 'scripts' / script), NAME],
                                cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, log_name


try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(execute, 'review_dmm_static_trial.py', 'review_generation.log'),
                pool.submit(execute, 'evaluate_dmm_static_geometry.py', 'geometry_evaluation.log')]
        for job in jobs:
            job.result()
    review = RUN / ('review_' + NAME)
    evaluation = RUN / ('geometry_evaluation_' + NAME)
    metrics = json.loads((evaluation / 'geometry_metrics.json').read_text('utf-8'))
    summary = json.loads((review / 'summary.json').read_text('utf-8'))
    summary['geometry_evaluation'] = dict(path=str(evaluation), phase_metrics=metrics['phase_metrics'],
        paired_teeth=metrics['paired_teeth'], paired_macro_rms_mm=metrics['paired_macro_rms_mm'],
        scope=metrics['scope'], partition=metrics['partition'])
    summary['execution_records'] = ['persistent_cache_execution_record.json', 'component_memoization_gradient_check_008.json',
                                    'interruption_audit_004.json']
    write(review / 'summary.json', summary)
    count = len(metrics['paired_teeth'])
    first, final = (metrics['paired_macro_rms_mm'][key] for key in ('initial', 'final'))
    extra = f'\n三维事后评价：同一世界坐标，不做ICP/Sim3或缩放对齐。\n共同有可评价预测面的牙位 {count}/28。\n'
    if first is not None:
        extra += f'共同牙位的逐牙对称表面采样RMS均值：{first:.6f} -> {final:.6f} mm（沿用源扫描假定mm尺度）。\n'
    extra += '每方向每牙1000个固定种子面积样本；点到三角面的距离精确，但不是完整连续表面的精确Hausdorff。\n预测按混合表面软语义分区，不能等同于人工牙齿解剖分割。\n'
    with (review / 'RESULT_ZH.txt').open('a', encoding='utf-8') as output:
        output.write(extra)
    image = '../' + evaluation.name + '/reference_initial_final_geometry.png'
    page = (review / 'index.html').read_text('utf-8')
    page += '<h2>Post-fit 3D reference comparison</h2><pre>'+html.escape(extra)+'</pre>'
    page += f'<a href="{image}"><img src="{image}"></a><p><a href="../{evaluation.name}/per_tooth_surface_error.csv">Per-tooth 3D errors CSV</a></p>'
    (review / 'index.html').write_text(page, encoding='utf-8')

    # Audit original data and the pristine deployment, separately from the repaired runtime.
    source = Path(json.loads((RUN / 'plan.json').read_text('utf-8'))['source'])
    checks = json.loads((source / 'checksums.json').read_text('utf-8'))
    changed = [name for name, expected in checks.items() if digest(source / name) != expected]
    package_checks = json.loads((PACKAGE / 'SHA256SUMS.json').read_text('utf-8'))
    package_changed = [name for name, expected in package_checks.items() if digest(PACKAGE / name) != expected]
    changes = []
    runtime = RUN / 'runtime_source/DMM'
    original = PACKAGE / 'third_party/DMM'
    for path in sorted(runtime.rglob('*.py')):
        relative = path.relative_to(runtime)
        donor = original / relative
        before = donor.read_text('utf-8-sig') if donor.exists() else ''
        after = path.read_text('utf-8-sig')
        if before != after:
            changes.append(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                fromfile='original/'+relative.as_posix(), tofile='trial/'+relative.as_posix())))
    (RUN / 'runtime_changes.diff').write_text(''.join(changes), encoding='utf-8')
    snapshots = RUN / 'source_snapshot/helpers'
    snapshots.mkdir(parents=True)
    helper_registry = {}
    for path in sorted((PROJECT / 'scripts').glob('*dmm*trial*.py')):
        shutil.copy2(path, snapshots / path.name)
        helper_registry[path.name] = dict(original_path=str(path), sha256=digest(path))
    for path in sorted((PROJECT / 'scripts').glob('*dmm_static*.py')):
        if path.name not in helper_registry:
            shutil.copy2(path, snapshots / path.name)
            helper_registry[path.name] = dict(original_path=str(path), sha256=digest(path))
    write(RUN / 'source_snapshot/helper_registry.json', helper_registry)
    audit = dict(original_source_file_count=len(checks), original_source_changed=changed,
                 pristine_deployment_file_count=len(package_checks), pristine_deployment_changed=package_changed,
                 fitting_source_sha256=json.loads((FIT / 'settings.json').read_text('utf-8'))['source_sha256'],
                 execution_wrapper_sha256=digest(RUN / 'persistent_cache_execution.py'),
                 input_and_original_models_preserved=not changed and not package_changed,
                 finished_utc=datetime.now(timezone.utc).isoformat())
    write(RUN / 'final_integrity_audit.json', audit)
    assert audit['input_and_original_models_preserved'], audit
    write(state_path, dict(status='COMPLETE', fit_status=summary['status'], review=str(review), geometry=str(evaluation),
                          summary=str(review / 'summary.json'), finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(json.loads(state_path.read_text('utf-8')), indent=2), flush=True)
except Exception:
    import traceback
    write(state_path, dict(status='FINALIZATION_FAILED_RETAINED', error=traceback.format_exc()))
    raise
