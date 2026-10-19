"""Preserve every failed attempt and run a new declared attempt directory."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

project = Path(__file__).resolve().parents[1]
run = Path((project / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
tag = sys.argv[1]
assert tag.isdigit()
target = run / ('fit_pilot_' + tag)
assert not target.exists()
command = json.loads((run / 'pilot_command.json').read_text('utf-8'))
command[-1] = str(target)
(run / (target.name + '_command.json')).write_text(json.dumps(command, indent=2), encoding='utf-8')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TEMP=str(project / 'tmp'), TMP=str(project / 'tmp'))
env.pop('PYTHONPATH', None)
env.pop('PYTHONHOME', None)
state = dict(status='PILOT_RUNNING', started_utc=datetime.now(timezone.utc).isoformat(), attempt=target.name)
start = time.monotonic()
with (run / (target.name + '.log')).open('w', encoding='utf-8') as log:
    proc = subprocess.Popen(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
    state['child_pid'] = proc.pid
    (run / (target.name + '_state.json')).write_text(json.dumps(state, indent=2), encoding='utf-8')
    status = proc.wait()
state.update(status='PILOT_FINISHED' if status == 0 else 'PILOT_FAILED_RETAINED', returncode=status, seconds=time.monotonic()-start)
(run / (target.name + '_state.json')).write_text(json.dumps(state, indent=2), encoding='utf-8')
print(json.dumps(state), flush=True)
sys.exit(status)
