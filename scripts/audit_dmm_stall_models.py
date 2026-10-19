"""Audit weight/statistics bindings, validation scope and actual case exclusion."""
import hashlib
import json
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
OUT=Path((PROJECT/'tmp/dmm_diagnostic_current.txt').read_text('utf-8-sig'))
PACKAGE=PROJECT/'.runtime/dmm_r4_best_val_20261015/package'
DATA=Path(r'D:\WorkSpace\Dental\data\Teeth3DS_DualArch_v1')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
excluded=json.loads((DATA/'reports/excluded_cases.json').read_text('utf-8'))
result=dict(models={},case_exclusions=[r for r in excluded if r.get('patient_id')=='L428SD7J'],
    validation_metric_scope='One-way observed-to-generated surface sample mean, after direct 3D latent fitting; not comparable to image-fit symmetric triangle-distance RMS')
for arch in ('upper','lower'):
    folder=PACKAGE/'model_bundle'/arch
    model=json.loads((folder/'model.json').read_text('utf-8'))
    manifest=DATA/'training'/arch/'manifest.json'
    local=json.loads(manifest.read_text('utf-8'))
    local_train={r['case_id'] for r in local['cases'] if r['split']=='train'}
    with np.load(folder/'latent_statistics.npz') as data:
        stats={k:dict(shape=list(data[k].shape),minimum=float(data[k].min()),maximum=float(data[k].max())) for k in data.files if data[k].dtype.kind in 'fiu'}
    val=model['selection']['validation']
    result['models'][arch]=dict(model_id=model['model_id'],epoch=model['training_epoch'],
        weights_sha256=digest(folder/'weights.pth'),weights_hash_ok=digest(folder/'weights.pth')==model['weights']['sha256'],
        statistics_hash_ok=digest(folder/'latent_statistics.npz')==model['latent_statistics']['sha256'],
        local_training_manifest_hash_ok=digest(manifest)==model['training_manifest_sha256'],
        local_train_case_set_matches=local_train==set(model['training_case_ids']),
        contains_case=any('L428SD7J' in name for name in model['training_case_ids']),
        validation_status=val['status'],validation_score_mm=val['score_mm'],validation_cases=len(val['cases']),
        validation_failures=val['failures'],statistics=stats)
(OUT/'model_and_population_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:{a:b for a,b in v.items() if a!='statistics'} for k,v in result['models'].items()},indent=2))
