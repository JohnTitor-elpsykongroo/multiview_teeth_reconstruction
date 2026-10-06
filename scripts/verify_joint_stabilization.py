"""Verify immutable fit provenance, warm-start identity and original gates."""
import json
from pathlib import Path
import sys
import numpy as np
from run_forward_check import sha256
root=Path(sys.argv[1]).resolve(strict=True)
checks={}
for job in (root/'jobs').iterdir():
    fitroot=job/'fit_attempt_01'; path=fitroot/'provenance.json'
    if not path.exists(): continue
    p=json.loads(path.read_text()); fit=Path(p['fit_input']); m=json.loads((fit/'manifest.json').read_text())
    for n,h in p['code_sha256'].items():
        checks[f'{job.name}:fit_snapshot:{n}']=sha256(fitroot/'code_snapshot'/n)==h
        checks[f'{job.name}:current_fit_code:{n}']=sha256(Path(__file__).parent/n)==h
    checks[f'{job.name}:manifest']=sha256(fit/'manifest.json')==p['fit_manifest_sha256']
    for n,h in m['extra_input_sha256'].items(): checks[f'{job.name}:input:{n}']=sha256(fit/n)==h
    checks[f'{job.name}:cameras']=sha256(fit/m['camera_file'])==m['camera_sha256']
    for item in m['masks']: checks[f'{job.name}:mask:{item["camera"]}']=sha256(fit/item['path'])==item['sha256']
    for n,h in p['dmm_source_sha256'].items(): checks[f'{job.name}:DMM:{n}']=sha256(Path(n))==h
    warm=Path(p['warm_start_run'])
    checks[f'{job.name}:warm_pose']=sha256(warm/'warmup_pose.json')==p['warm_start_pose_sha256']
    checks[f'{job.name}:warm_codes']=sha256(warm/'initial_codes.npz')==p['warm_start_initial_codes_sha256']
    c=json.loads((fitroot/'resolved_config.json').read_text()); old=json.loads((warm/'resolved_config.json').read_text())
    checks[f'{job.name}:acceptance_unchanged']=c['acceptance']==old['acceptance']
    checks[f'{job.name}:no_truth_in_fitter']=not any(p[n] for n in ['active_truth_codes_read','ground_truth_pose_read','source_meshes_read','depth_maps_read'])
    if (fit/'population_prior_provenance.json').exists():
        prior=json.loads((fit/'population_prior_provenance.json').read_text())
        checks[f'{job.name}:patient_and_mirror_excluded']=all(v['excluded_rows']==[0,1] and v['rows']==1050 for v in prior['details'].values())
result={'status':'INTEGRITY_PASS' if all(checks.values()) else 'INTEGRITY_FAIL','checks':checks,'failed':[k for k,v in checks.items() if not v]}
out=root/'review'; out.mkdir(exist_ok=True); target=out/'integrity.json'
if target.exists(): raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2)); print(json.dumps({'status':result['status'],'checks':len(checks),'failed':result['failed']}))
