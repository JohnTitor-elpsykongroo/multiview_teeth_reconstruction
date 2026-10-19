"""Retry only the declared trial, using the isolated execution repair."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

project = Path(__file__).resolve().parents[1]
run = Path((project / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
check = json.loads((run / 'safeguarded_surface_diagnostic.json').read_text('utf-8'))
assert all(row['passed'] for row in check['arches'].values()), 'Surface gates must pass before retry'
command = json.loads((run / 'fit_command.json').read_text('utf-8'))
command[3] = str(run / 'runtime_source/DMM/dmm_cli.py')
command[-1] = str(run / 'fit_attempt_002')
assert not Path(command[-1]).exists()
(run / 'fit_attempt_002_command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
           TEMP=str(project / 'tmp'), TMP=str(project / 'tmp'))
env.pop('PYTHONPATH', None)
env.pop('PYTHONHOME', None)
state = dict(status='FIT_RETRY_RUNNING', attempt=2, started_utc=datetime.now(timezone.utc).isoformat(),
             scope='Safeguarded float64 surface roots; unchanged model values, trust/residual/image gates; float32 raster')
(run / 'attempt_002_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
start = time.monotonic()
with (run / 'fit_attempt_002.log').open('w', encoding='utf-8') as log:
    proc = subprocess.Popen(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
    state['child_pid'] = proc.pid
    (run / 'attempt_002_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    state['returncode'] = proc.wait()
state.update(status='FIT_FINISHED' if state['returncode'] == 0 else 'FIT_FAILED_RETAINED', seconds=time.monotonic()-start)
(run / 'attempt_002_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
print(json.dumps(state), flush=True)
sys.exit(state['returncode'])
