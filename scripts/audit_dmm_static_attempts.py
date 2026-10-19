"""Explain retained execution attempts without treating resumes as new cases."""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUN = Path((PROJECT / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
records = [
    ('fit_attempt_001', 'numerical_initialization_failure',
     'Original deployment entrypoint; all upper mean-initialization candidates rejected by implicit normal-root trust/convergence gates; no optimization completed.'),
    ('fit_attempt_002', 'intentional_stop_before_optimization',
     'Isolated safeguarded float64 surface runtime, original 512-pixel views and cube search; stopped to use a bounded 8 GB pilot.'),
    ('fit_pilot_003', 'execution_adapter_dtype_error',
     '256-pixel pilot; mixed float32/float64 recovery adapter error before optimization, repaired only in the isolated runtime.'),
    ('fit_pilot_004', 'unexplained_process_absence_after_pose_checkpoint',
     'Ten pose steps completed. Later process and terminal were absent without an exception or completion record; cause unknown. Native checkpoint retained; original initial_render is the comparison baseline.'),
    ('fit_pilot_005_resume', 'intentional_execution_adjustment',
     'Strict native resume from the same ten-pose-step checkpoint; stopped before another completed step to add lossless saved-tensor CPU offload.'),
    ('fit_pilot_006_resume_offload', 'intentional_execution_adjustment',
     'Same native checkpoint, objective and schedule; stopped before another completed step to add exact repeated-component evaluation memoization.'),
    ('fit_pilot_007_resume_cached', 'intentional_execution_adjustment',
     'Same checkpoint; stopped before another completed step to add persistent exact-input root memoization. Forward/gradient comparison retained.'),
    ('fit_pilot_008_resume_persistent', 'final_native_resume',
     'Continues the same single-case pilot with exact-input caches, live numerical gates and fresh implicit gradient graphs. Completion/quality status comes from this folder result.json, not the attempt number.'),
]
audit = dict(case='L428SD7J', independent_cases=1,
    interpretation='Execution attempts and strict resumes of one development-case fit; not eight independent reconstruction trials.',
    numerical_gate_policy='Original residual 1e-6, normal-offset trust 0.1 and minimum image IoU 0.85 retained.',
    invalidated_preflight='surface preflight 004 was explicitly invalidated because its source changed while running; excluded from acceptance evidence.',
    records=[dict(folder=name, classification=kind, explanation=explanation) for name, kind, explanation in records])
(RUN / 'attempt_outcomes.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(RUN / 'attempt_outcomes.json')
