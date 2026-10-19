"""Verify the diagnostic delivery, not anatomical acceptance."""
import sys,json,re,hashlib,platform,shutil
from pathlib import Path
from PIL import Image
import numpy,pyvista,vtk,scipy
from diagnose_preprocess12 import read,write,sha,now
root=Path(sys.argv[1]).resolve()
assert root.is_relative_to((Path(__file__).resolve().parents[1]/'runs').resolve())
summary=read(root/'summary.json');ms=summary['cases'];cohort=read(root/'protocol/cohort.json')['cases']
assert len(ms)==len(cohort)==12 and len({r['patient_id'] for r in cohort})==12
assert sum(r['arch']=='upper' for r in cohort)==6
assert all(r['assigned_split']=='train' and not r['old_profile_test_patient'] for r in cohort)
assert all(not m['invalid_fdi'] and m['zero_area_faces']==0 and m['duplicate_face_excess']==0 and m['topology']['inconsistent_edges']==0 for m in ms)
assert all(all(len(v)==1 for v in m['instance_to_labels'].values()) for m in ms)
assert all(all(m['source_hashes_match_prior'].values()) for m in ms)
content=(root/'index.html').read_text(encoding='utf-8')
refs=re.findall(r'(?:href|src)="([^"]+)"',content)
missing=[s for s in refs if not s.startswith('#') and not (root/s).is_file()]
assert not missing,missing
manifests=read(root/'artifact_manifest.json')
assert all(sha(root/x['path'])==x['sha256'] for x in manifests)
main=[p for s in ['geometry','semantics','normals'] for p in (root/'cases').glob(f'*/{s}.png')]
assert len(main)==36
for p in root.rglob('*.png'):
    with Image.open(p) as im:im.verify()
for r,m in zip(cohort,ms):
    assert r['case_id']==m['case_id']
    assert sha(r['source_obj'])==m['source_hashes_before']['source_obj']
    assert sha(r['source_annotation'])==m['source_hashes_before']['source_annotation']
    d=numpy.load(root/'cases'/r['case_id']/'raw_index_preserved.npz')
    assert d['vertices'].shape==(m['vertices'],3) and d['faces'].shape==(m['faces'],3)
    assert len(d['vertex_labels'])==m['vertices']
    assert numpy.max(numpy.abs(numpy.linalg.norm(d['face_normals'],axis=1)-1))<1e-10
    assert max(m['display_basis_spans_mm'])<110, 'Full arch may exceed orthographic frame'
report=dict(verified_utc=now(),case_count=12,patient_count=12,upper=6,lower=6,main_four_view_figures=36,
    local_figures=len(list((root/'cases').glob('*/local_fdi_*.png'))),
    nonmanifold_cases=sum(m['topology']['nonmanifold_edges']>0 for m in ms),
    local_page_references_verified=len(refs),artifact_hashes_verified=len(manifests),
    source_files_rehashed=24,source_unchanged=True,source_hashes_match_previous_metadata=True,
    structural_and_artifact_checks='PASS',human_visual_review='PENDING',accepted_for_training=False,
    assistant_visual_review_scope='All three 12-case occlusal contact sheets and representative four-view/local figures; not an independent clinical or anatomical acceptance',
    runtime=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,pyvista=pyvista.__version__,vtk=vtk.vtkVersion.GetVTKVersion()),
    preserved_drafts='First selection attempt made empty directories only; previous contact aspect and flat-normal display drafts are retained and excluded from review links.')
write(root/'qa_report.json',report)
shutil.copyfile(__file__,root/'protocol/verify_preprocess12_delivery.py')
write(root/'delivery_manifest.json',[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()])
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
