"""One declared synthetic scene, immutable inputs, exported R4 runtime."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEPLOY = PROJECT / '.runtime/dmm_r4_best_val_20261015'
PACKAGE = DEPLOY / 'package'
SOURCE = Path(r'D:\WorkSpace\BlenderProjects\IntraoralDataCreater\multiview_runs\dmm_static_20261015T113553Z_01f2be')
CASE = 'L428SD7J'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    run = PROJECT / 'runs' / ('dmm_static_trial_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6])
    run.mkdir()
    print(str(run), flush=True)
    (PROJECT / 'tmp/dmm_static_trial_current.txt').write_text(str(run), encoding='utf-8')
    shutil.copy2(__file__, run / 'runner_snapshot.py')
    configs = run / 'configs'
    configs.mkdir()
    for name in ('fit_static_v1.json', 'surface_static_v1.json', 'semanticxy_static_v1.json'):
        shutil.copy2(PACKAGE / 'configs' / name, configs / name)
    shutil.copy2(SOURCE / 'configs/render_512.json', configs / 'render_512.json')
    manifest = SOURCE / 'scenes' / CASE / 'fit_input/manifest.json'
    plan = dict(run=str(run), source=str(SOURCE), case=CASE,
        selection='First case in delivery_summary.json, selected before observing fit results',
        requested_trial='One dual-arch fit, all 15 supplied 512x512 views',
        manifest=str(manifest), manifest_sha256=digest(manifest), initialize_mean=True,
        fitting_access='Packaged scene RGB/masks/valid/cameras/presence/initialization/model_bundle only',
        truth_policy='No target 3D mesh or source pose used for fitting/initialization; post-fit evaluation only',
        python=sys.executable, package=str(PACKAGE),
        configs={p.name: digest(p) for p in configs.iterdir()})
    write(run / 'plan.json', plan)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               TEMP=str(PROJECT / 'tmp'), TMP=str(PROJECT / 'tmp'))
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    cli = [sys.executable, '-u', '-B', str(PACKAGE / 'third_party/DMM/dmm_cli.py')]
    state = dict(status='INPUT_HASH_VERIFICATION', case=CASE, started_utc=datetime.now(timezone.utc).isoformat())
    write(run / 'state.json', state)
    try:
        expected = json.loads((SOURCE / 'checksums.json').read_text('utf-8'))
        mismatches = []
        for relative, wanted in expected.items():
            if digest(SOURCE / relative) != wanted:
                mismatches.append(relative)
        write(run / 'input_hash_verification.json', dict(file_count=len(expected), mismatches=mismatches))
        assert not mismatches, mismatches
        for arch in ('upper', 'lower'):
            for name in ('model.json', 'weights.pth', 'latent_statistics.npz'):
                assert digest(SOURCE / 'scenes' / CASE / 'model_bundle' / arch / name) == digest(PACKAGE / 'model_bundle' / arch / name)
        validate = cli + ['validate-scene', str(manifest), '--device', 'cpu']
        write(run / 'validate_command.json', validate)
        with (run / 'validate_scene.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(validate, cwd=PACKAGE, env=env, stdout=log, stderr=subprocess.STDOUT)
        assert completed.returncode == 0, 'validate-scene failed; inspect validate_scene.log'
        command = cli + ['fit-scene', str(manifest), '--device', 'cuda:0',
            '--fit-config', str(configs / 'fit_static_v1.json'),
            '--surface-config', str(configs / 'surface_static_v1.json'),
            '--render-config', str(configs / 'render_512.json'),
            '--matching-config', str(configs / 'semanticxy_static_v1.json'),
            '--initialize-mean', '--output', str(run / 'fit_attempt_001')]
        write(run / 'fit_command.json', command)
        state.update(status='FIT_RUNNING', fit_started_utc=datetime.now(timezone.utc).isoformat())
        write(run / 'state.json', state)
        started = time.monotonic()
        with (run / 'fit_attempt_001.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen(command, cwd=PACKAGE, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = proc.pid
            write(run / 'state.json', state)
            returncode = proc.wait()
        state.update(returncode=returncode, seconds=time.monotonic() - started)
        result = run / 'fit_attempt_001/result.json'
        if result.exists():
            state['result'] = json.loads(result.read_text('utf-8'))['status']
        state['status'] = 'FIT_FINISHED' if returncode == 0 else 'FIT_FAILED_RETAINED'
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write(run / 'state.json', state)
        print(json.dumps(state), flush=True)
        return returncode
    except BaseException:
        state.update(status='FAILED_RETAINED', error=traceback.format_exc())
        write(run / 'state.json', state)
        print(state['error'], flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
