"""Strict native resume from the retained interrupted checkpoint."""
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
checkpoint = run / 'fit_pilot_004/checkpoint.json'
assert json.loads(checkpoint.read_text('utf-8'))['status'] == 'RUNNING'
audit = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
    status='PROCESS_AND_TERMINAL_SESSION_ABSENT_NO_EXCEPTION_LOG',
    checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    cause='Unknown; no completion or failure record; original files retained')
(run / 'interruption_audit_004.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
command = json.loads((run / 'pilot_command.json').read_text('utf-8'))
target = run / 'fit_pilot_005_resume'
assert not target.exists()
command[-1] = str(target)
command += ['--resume', str(checkpoint)]
(run / 'fit_pilot_005_resume_command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TEMP=str(project / 'tmp'), TMP=str(project / 'tmp'))
env.pop('PYTHONPATH', None)
env.pop('PYTHONHOME', None)
state = dict(status='RESUME_RUNNING', checkpoint_sha256=audit['checkpoint_sha256'], started_utc=datetime.now(timezone.utc).isoformat())
start = time.monotonic()
with (run / 'fit_pilot_005_resume.log').open('w', encoding='utf-8') as log:
    proc = subprocess.Popen(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
    state['child_pid'] = proc.pid
    (run / 'fit_pilot_005_resume_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    code = proc.wait()
state.update(status='RESUME_FINISHED' if code == 0 else 'RESUME_FAILED_RETAINED', returncode=code, seconds=time.monotonic()-start)
(run / 'fit_pilot_005_resume_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
print(json.dumps(state), flush=True)
sys.exit(code)
