"""Independent exhaustive delivery checks, plus sampled geometry replay per case."""
from pathlib import Path
import argparse
import json
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pipeline as p


def require(value,message):
    if not value:raise AssertionError(message)


def verify_part(config,expected=None,part_index=0,part_count=1):
    out=Path(config['output_root']);rows=p.read(out/'case_results.json');start=time.time()
    inputs=p.read(out/'inputs.json');excluded=set(inputs['excluded_ids']);failures=[];checks=Counter();patients={}
    require(p.sha(config['candidate_manifest'])==inputs['candidate_manifest_sha256'],'candidate manifest changed during batch')
    require(p.sha(config['exclusion_policy'])==inputs['exclusion_policy_sha256'],'exclusion policy changed during batch')
    declared={r['case_id']:r for r in p.read(config['candidate_manifest'])['cases']}
    used_config=p.read(out/'run_config.json')
    for key in ['geometry_implementation_sha256','processing_implementation_sha256']:
        require(config[key]==used_config[key],'numeric implementation drift during batch')
    if expected is not None:require(len(rows)==expected,'case count')
    require(set(r['case_id'] for r in rows)==set(inputs['selected_ids']),'input coverage')
    rows=rows[part_index::part_count]
    for i,row in enumerate(rows):
        cid=row['case_id'];arch=row['arch']
        try:
            require(cid not in excluded,'excluded source included')
            for key,hk in [('obj','source_sha256'),('annotation','annotation_sha256')]:
                require(p.sha(row[key])==row[hk],'source changed '+key);checks['source_hashes']+=1
            split=row['assigned_split'];patient=row['patient_id']
            require(split==declared[cid]['assigned_split'] and patient==declared[cid]['patient_id'],'input split or patient identity changed')
            require(patient not in patients or patients[patient]==split,'patient split leakage');patients[patient]=split
            if not row.get('geometry_path'):continue
            require(p.sha(row['geometry_path'])==row['geometry_sha256'],'geometry hash')
            with np.load(row['geometry_path']) as z:geometry={k:z[k] for k in z.files}
            v,f=geometry['vertices_mm'],geometry['faces']
            require(len(v)==row['source_vertices'] and len(f)==row['retained_faces'],'geometry dimensions')
            require(np.isfinite(v).all() and f.min()>=0 and f.max()<len(v),'geometry finite/index')
            checks['geometry_caches']+=1
            if not row.get('entries'):continue
            provenance=out/'cache'/cid/'sampling.npz'
            with np.load(provenance) as z:blueprint={k:z[k] for k in z.files}
            require(np.isfinite(blueprint['barycentric']).all() and np.min(blueprint['barycentric'])>=-1e-7,
                    'barycentric validity')
            require(np.max(np.abs(blueprint['barycentric'].sum(1)-1))<2e-7,'barycentric sum')
            orig=geometry['original_face_id'];lookup={int(k):i for i,k in enumerate(orig)}
            tri=v[f];edge_cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(edge_cross,axis=1)
            triplets=geometry['vertex_labels'][f];strict=(triplets[:,0]==triplets[:,1])&(triplets[:,1]==triplets[:,2])
            actual_centers={}
            for k in p.TEETH[arch]:
                take=strict&(triplets[:,0]==k)
                if take.any():actual_centers[k]=np.average(tri[take].mean(1),weights=area[take],axis=0)
            for k,value in actual_centers.items():np.testing.assert_allclose(value,row['centers_raw_mm'][str(k)],atol=1e-10)
            checks['area_weighted_centroid_cases']+=1
            for entry in row['entries']:
                eid=entry['case_id'];require(eid.split('__mirror')[0] not in excluded,'excluded augmentation')
                require(entry['split']==split,'derived split drift')
                require(p.sha(entry['samples_path'])==entry['samples_sha256'],'NPZ hash')
                require(p.sha(entry['centers_path'])==entry['centers_sha256'],'PKL hash')
                with np.load(entry['samples_path']) as z:
                    require(set(z.files)=={'surf','normals','pos','neg'},'NPZ keys')
                    surf,n,pos,neg=(z[k] for k in ['surf','normals','pos','neg'])
                    for a in [surf,n,pos,neg]:require(a.dtype==np.float32 and np.isfinite(a).all(),'float32 finite')
                    require(surf.shape[1]==5 and n.shape==(len(surf),3),'surface shape')
                    require(pos.shape==neg.shape==(config['offsurface_points']//2,5),'space count')
                    require(np.max(np.abs(np.linalg.norm(n,axis=1)-1))<2e-6,'unit normals')
                    require(set(np.unique(surf[:,4]).astype(int))<=set(p.TEETH[arch])|{0},'FDI set')
                    counts=Counter(surf[:,4].astype(int))
                    require(counts[0]==config['surface_gum'],'gum quota')
                    require(all(counts[k]==config['surface_per_tooth'] for k in counts if k),'tooth quota')
                    require(np.all(surf[:,3]==0) and np.all(pos[:,3]==0) and np.all(neg[:,3]==0),'placeholder')
                    require(np.all(pos[:,4]==-1) and np.all(neg[:,4]==-1),'space labels')
                    require(np.max(np.abs(pos[:,:3]))<=1 and np.max(np.abs(neg[:,:3]))<=1,'space domain')
                    for off in [pos,neg]:
                        require(np.max(np.abs(off[:,:3].mean(0)))<.01,'uniform-space first moment')
                        require(np.max(np.abs(off[:,:3].var(0)-1/3))<.01,'uniform-space variance')
                    with open(entry['centers_path'],'rb') as stream:centers=pickle.load(stream)
                    require(all(type(k) is int and value.shape==(3,) and value.dtype==np.float32 for k,value in centers.items()),'PKL contract')
                    for k,value in centers.items():np.testing.assert_allclose(value,entry['centers'][str(k)],atol=1e-7)
                    ids=np.linspace(0,len(surf)-1,min(512,len(surf))).astype(int)
                    face_id=np.array([lookup[int(k)] for k in blueprint['original_face_id'][ids]])
                    triangles=v[f[face_id]];b=blueprint['barycentric'][ids].astype(float)
                    raw=np.einsum('ni,nij->nj',b,triangles)
                    matrix=np.array(entry['source_mm_to_model']);expected_points=raw@matrix[:3,:3].T+matrix[:3,3]
                    np.testing.assert_allclose(surf[ids,:3],expected_points,atol=2e-7)
                    orig_norm=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);orig_norm/=np.linalg.norm(orig_norm,axis=1,keepdims=True)
                    expected_norm=orig_norm@np.linalg.inv(matrix[:3,:3]);expected_norm/=np.linalg.norm(expected_norm,axis=1,keepdims=True)
                    np.testing.assert_allclose(n[ids],expected_norm,atol=2e-6)
                    inverse=np.array(entry['model_to_source_mm']);recovered=surf[ids,:3]@inverse[:3,:3].T+inverse[:3,3]
                    np.testing.assert_allclose(recovered,raw,atol=2e-5)
                    expected_labels=blueprint['labels'][ids]
                    source_labels=geometry['vertex_labels'][f[face_id]];a,b,c=source_labels.T
                    source_majority=np.where(a==b,a,np.where(a==c,a,np.where(b==c,b,-1)))
                    require(np.array_equal(source_majority,expected_labels),'original face semantic replay')
                    for k,value in actual_centers.items():
                        output_label=int(entry['fdi_map'][str(k)]) if entry['parent_id'] else k
                        np.testing.assert_allclose(centers[output_label],value@matrix[:3,:3].T+matrix[:3,3],atol=2e-7)
                    if entry['parent_id']:
                        mapping=entry['fdi_map'];expected_labels=np.array([0 if k==0 else int(mapping[str(k)]) for k in expected_labels])
                        require(np.linalg.det(matrix[:3,:3])<0,'mirror reflection')
                    else:require(np.linalg.det(matrix[:3,:3])>0,'proper original')
                    require(np.array_equal(surf[ids,4],expected_labels),'blueprint/FDI replay')
                    if row['qc_status']=='PASS':require(np.max(np.abs(surf[:,:3]))<=1+1e-7,'PASS surface domain')
                checks['NPZ_PKL_pairs']+=1;checks['surface_replay_points']+=len(ids)
        except Exception as e:failures.append(dict(case_id=cid,error=f'{type(e).__name__}: {e}'))
        if (i+1)%25==0 or i+1==len(rows):print(json.dumps(dict(stage='verify',part=part_index,done=i+1,total=len(rows),failures=len(failures),seconds=round(time.time()-start,1))),flush=True)
    for arch in p.TEETH:
        folder=out/'data'/arch;entries=p.read(folder/'manifest.json')['cases'];avg=np.loadtxt(folder/'avg_centroids.txt')
        require(avg.shape==(50,3),'average shape')
        for k in p.TEETH[arch]:
            means=[e['centers'][str(k)] for e in entries if e['split']=='train' and e['qc_status']=='PASS' and str(k) in e['centers']]
            np.testing.assert_allclose(avg[k],np.mean(means,axis=0),atol=1e-9)
        for split in ['train','val','test']:
            actual=p.read(folder/'splits'/f'{split}_computational_pass.json')
            require(set(actual)=={e['case_id']+'.npz' for e in entries if e['split']==split and e['qc_status']=='PASS'},'split membership')
        expected_files={e['case_id']+ext for e in entries for ext in ['.npz','.pkl']}
        require({x.name for x in (folder/'SdfSamples').iterdir() if x.is_file()}==expected_files,'stale or missing final files')
    # Excluded inputs are never opened or written by this portable project.
    report=dict(status='PASS' if not failures else 'FAIL',case_count=len(rows),case_ids=[r['case_id'] for r in rows],checks=dict(checks),failures=failures,
                all_source_files_unchanged=checks['source_hashes']==len(rows)*2,elapsed_seconds=time.time()-start,
                no_train_val_test_patient_overlap=True,human_visual_acceptance='PENDING',accepted_for_training=False,
                geometry_replay_scope='512 deterministic points per original/mirror; all arrays, hashes, dimensions, labels, quotas, space-domain and average-centroid checks exhaustive')
    if failures:print(json.dumps(dict(failures=failures),ensure_ascii=False),flush=True)
    require(not failures,'delivery verification failed')
    return report


def _part_job(args):
    return verify_part(*args)


def verify(config,expected=None):
    start=time.time();workers=min(config['workers'],4);reports=[];checks=Counter();out=Path(config['output_root'])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(_part_job,(config,expected,i,workers)) for i in range(workers)]
        for future in as_completed(pending):
            report=future.result();reports.append(report);checks.update(report['checks'])
    ids=[cid for r in reports for cid in r['case_ids']]
    rows=p.read(out/'case_results.json')
    require(len(ids)==len(set(ids))==len(rows) and set(ids)=={r['case_id'] for r in rows},'verification part coverage')
    patients={}
    for row in rows:
        patient,split=row['patient_id'],row['assigned_split']
        require(patient not in patients or patients[patient]==split,'global patient split leakage')
        patients[patient]=split
    failures=[f for r in reports for f in r['failures']]
    report=dict(status='PASS' if not failures else 'FAIL',case_count=len(ids),checks=dict(checks),failures=failures,
        all_source_files_unchanged=all(r['all_source_files_unchanged'] for r in reports),
        elapsed_seconds=time.time()-start,parallel_processes=workers,no_train_val_test_patient_overlap=True,
        human_visual_acceptance='PENDING',accepted_for_training=False,
        geometry_replay_scope=reports[0]['geometry_replay_scope'])
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    require(not failures,'delivery verification failed')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(p.ROOT/'config.json'));parser.add_argument('--expected',type=int)
    a=parser.parse_args();verify(p.load_config(a.config),a.expected)
