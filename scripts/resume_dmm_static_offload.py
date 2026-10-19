"""Resume losslessly with an explicitly recorded execution-only memory hook."""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

project = Path(__file__).resolve().parents[1]
run = Path((project / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
checkpoint = run / 'fit_pilot_005_resume/checkpoint.json'
assert json.loads(checkpoint.read_text('utf-8'))['status'] == 'RUNNING'
command = json.loads((run / 'pilot_command.json').read_text('utf-8'))
target = run / 'fit_pilot_006_resume_offload'
assert not target.exists()
command[3] = str(run / 'offload_execution.py')
command[-1] = str(target)
command += ['--resume', str(checkpoint)]
(run / 'fit_pilot_006_resume_offload_command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TEMP=str(project / 'tmp'), TMP=str(project / 'tmp'))
env.pop('PYTHONPATH', None)
env.pop('PYTHONHOME', None)
state = dict(status='RESUME_RUNNING', checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    started_utc=datetime.now(timezone.utc).isoformat(), purpose='Exact CPU autograd saved-tensor offload; native source/settings/checkpoint unchanged')
start = time.monotonic()
with (run / 'fit_pilot_006_resume_offload.log').open('w', encoding='utf-8') as log:
    proc = subprocess.Popen(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
    state['child_pid'] = proc.pid
    (run / 'fit_pilot_006_resume_offload_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    code = proc.wait()
state.update(status='RESUME_FINISHED' if code == 0 else 'RESUME_FAILED_RETAINED', returncode=code, seconds=time.monotonic()-start)
(run / 'fit_pilot_006_resume_offload_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
print(json.dumps(state), flush=True)
sys.exit(code)
