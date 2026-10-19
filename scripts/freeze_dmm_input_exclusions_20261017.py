"""Version training input selection; never mutate source or historical manifests."""
from pathlib import Path
import json, hashlib, collections
ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'runs/dmm_raw_risk_census_20261017_01'
OUT=ROOT/'runs/dmm_training_inputs_filtered_20261017_01'
DATA=Path('D:/WorkSpace/Dental/data/Teeth3DS_DualArch_v1/training')
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def absolutize(x,base):
    if isinstance(x,dict):
        return {k:str((base/v).resolve()) if k=='path' and isinstance(v,str) else absolutize(v,base) for k,v in x.items()}
    if isinstance(x,list): return [absolutize(v,base) for v in x]
    return x
assert not (OUT/'summary.json').exists(), 'Completed selection is immutable'
OUT.mkdir(exist_ok=True)
excluded=read(AUDIT/'approximately_reference_0_4_percent_cases.json')
ids={x['case_id'] for x in excluded}
assert len(ids)==7
for x in excluded:
    assert sha(Path(x['obj']))==x['source_sha256']
    assert sha(Path(x['annotation']))==x['annotation_sha256']
save(OUT/'exclusion_policy.json',dict(status='USER_APPROVED_EXCLUSION',scope='Exact seven arch scans and augmentation descendants; partner arches retained',excluded_case_ids=sorted(ids),source_audit=str(AUDIT),original_files_retained=True))
save(OUT/'excluded_sources_archive_index.json',excluded)
allrows=read(AUDIT/'all_case_metrics.json')
remaining=[x for x in allrows if x['status']=='SCREENED' and x['case_id'] not in ids]
assert len(remaining)==1793
save(OUT/'preprocessing_candidates.json',dict(status='AWAITING_PREPROCESSING_DESIGN_AND_QC_NOT_TRAINING_READY',cases=remaining))
save(OUT/'missing_annotation_holdout.json',[x for x in allrows if x['status']!='SCREENED'])
summary={'excluded_source_scans':7,'preprocessing_candidates':len(remaining),'missing_annotation_holdout':100,'arches':{}}
for arch in ['upper','lower']:
    source=DATA/arch/'manifest.json';original_sha=sha(source);m=read(source)
    removed=set(ids)
    while True:
        nxt=removed|{r['case_id'] for r in m['cases'] if r.get('augmentation_parent_id') in removed}
        if nxt==removed: break
        removed=nxt
    dropped=[r for r in m['cases'] if r['case_id'] in removed]
    kept=[r for r in m['cases'] if r['case_id'] not in removed]
    for i,r in enumerate(kept): r['embedding_row']=i
    m['cases']=kept;m=absolutize(m,source.parent)
    dest=OUT/arch/'manifest.json';save(dest,m)
    save(OUT/arch/'excluded_records.json',dropped)
    for r in m['cases']:
        assert r['case_id'] not in removed and r.get('augmentation_parent_id') not in removed
        for key in ['samples','centers','source_geometry','source_annotation']:
            assert Path(r[key]['path']).is_file()
    assert sha(source)==original_sha
    config_path=ROOT/'configs/training_handoff_v1'/f'{arch}_pilot.json'
    cfg=absolutize(read(config_path),config_path.parent)
    cfg['manifest']={'path':str(dest),'sha256':sha(dest)}
    if 'training_case_ids' in cfg: cfg['training_case_ids']=[i for i in cfg['training_case_ids'] if i not in removed]
    save(OUT/arch/'legacy_filtered_pilot.json',cfg)
    model=read(ROOT/f'.runtime/dmm_r4_best_val_20261015/package/model_bundle/{arch}/model.json')
    old_ids=model['training_case_ids'];new_ids=[i for i in old_ids if i not in removed]
    save(OUT/arch/'r4_training_case_ids_filtered.json',new_ids)
    summary['arches'][arch]=dict(historical_manifest=str(source),historical_sha256=original_sha,removed_records=len(dropped),remaining_records=len(kept),remaining_by_split=dict(collections.Counter(r['split'] for r in kept)),r4_original=len(old_ids),r4_remaining=len(new_ids),r4_removed=len(old_ids)-len(new_ids))
save(OUT/'summary.json',summary)
save(OUT/'delivery_manifest.json',[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in OUT.rglob('*') if p.is_file()])
print(json.dumps(summary,indent=2))
