"""Full surface-domain audit and exact unsigned distance audit; no dataset writes."""
from pathlib import Path
import argparse, json, time, hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point
from audit_space_sampling_20261018 import DATA,OUT,read,write,quant,seed

def domain_job(job):
    row,e=job
    try:
        with np.load(e['samples_path']) as z:surf=z['surf']
        p=surf[:,:3];lab=surf[:,4].astype(int)
        distance=np.linalg.norm(np.maximum(np.abs(p)-1,0),axis=1)
        outside=np.any(np.abs(p)>1,axis=1);mm=1/np.linalg.norm(np.array(e['source_mm_to_model'])[0,:3])
        per=[]
        for k in sorted(set(lab)):
            ids=lab==k
            per.append(dict(fdi=int(k),n=int(ids.sum()),outside=int(np.sum(outside&ids)),more_than_1mm=int(np.sum((distance*mm>1)&ids)),max_mm=float(np.max(distance[ids])*mm)))
        return dict(case_id=e['case_id'],parent_id=e.get('parent_id'),arch=row['arch'],split=row['assigned_split'],status=row['qc_status'],n=len(p),outside=int(outside.sum()),outside_more_than_1mm=int(np.sum(distance*mm>1)),max_distance_mm=float(distance.max()*mm),tooth_outside=int(np.sum(outside&(lab!=0))),gum_outside=int(np.sum(outside&(lab==0))),per_label=per)
    except Exception as ex:return dict(case_id=e['case_id'],error=f'{type(ex).__name__}: {ex}')

def domain():
    rows=read(DATA/'case_results.json');jobs=[(r,e) for r in rows for e in r.get('entries',[])];results=[];start=time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(domain_job,j) for j in jobs])):
            results.append(f.result())
            if (i+1)%300==0:print(json.dumps(dict(phase='domain',done=i+1,total=len(jobs),seconds=round(time.time()-start,1))),flush=True)
    write(OUT/'domain_entries.json',sorted(results,key=lambda r:r['case_id']))
    summary={}
    for group,predicate in [('all',lambda r:True),('original',lambda r:not r['parent_id']),('mirror',lambda r:bool(r['parent_id'])),('train_original',lambda r:r['split']=='train' and not r['parent_id']),('PASS',lambda r:r['status']=='PASS'),('REVIEW',lambda r:r['status']=='REVIEW')]:
        selected=[r for r in results if 'error' not in r and predicate(r)]
        summary[group]=dict(entries=len(selected),affected_entries=sum(r['outside']>0 for r in selected),affected_tooth_entries=sum(r['tooth_outside']>0 for r in selected),outside_points=sum(r['outside'] for r in selected),outside_tooth_points=sum(r['tooth_outside'] for r in selected),outside_gum_points=sum(r['gum_outside'] for r in selected),more_than_1mm_points=sum(r['outside_more_than_1mm'] for r in selected),max_outside_distance_mm=max((r['max_distance_mm'] for r in selected),default=0))
    summary['errors']=[r for r in results if 'error' in r];summary['elapsed_seconds']=time.time()-start
    write(OUT/'domain_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


class ExactSurface:
    """Guaranteed triangle candidates using centroid/radius groups and a vertex-distance upper bound."""
    def __init__(self,v,f):
        self.tri=np.asarray(v[f],float);centers=self.tri.mean(1)
        radii=np.linalg.norm(self.tri-centers[:,None,:],axis=2).max(1)
        bucket=np.floor(np.log2(np.maximum(radii,1e-12))).astype(int)
        self.groups=[]
        for b in sorted(set(bucket)):
            ids=np.flatnonzero(bucket==b);self.groups.append((ids,cKDTree(centers[ids]),float(radii[ids].max())))
        self.vertex_tree=cKDTree(np.asarray(v)[np.unique(f)])
        self.lo=self.tri.min(1);self.hi=self.tri.max(1)

    def query(self,q):
        best,_=self.vertex_tree.query(q,workers=1);best=best**2
        for start in range(0,len(q),64):
            points=q[start:start+64];dist=best[start:start+64]
            for ids,tree,radius in self.groups:
                lists=tree.query_ball_point(points,np.sqrt(dist)+radius+1e-10,workers=1)
                counts=np.array([len(a) for a in lists]);total=int(counts.sum())
                if not total:continue
                qi=np.repeat(np.arange(len(points)),counts);fi=ids[np.concatenate(lists).astype(int)]
                delta=np.maximum(np.maximum(self.lo[fi]-points[qi],points[qi]-self.hi[fi]),0)
                keep=np.sum(delta**2,axis=1)<=dist[qi]+1e-12
                qi,fi=qi[keep],fi[keep]
                if not len(qi):continue
                cp=closest_point(self.tri[fi],points[qi]);ds=np.sum((cp-points[qi])**2,axis=1)
                np.minimum.at(dist,qi,ds)
        return np.sqrt(best)

def validate_exact():
    rng=np.random.default_rng(20261018);v=rng.normal(size=(60,3));f=np.arange(60).reshape(-1,3);q=rng.normal(size=(100,3))*3
    method=ExactSurface(v,f);got=method.query(q)
    cp=closest_point(np.tile(v[f],(len(q),1,1)),np.repeat(q,len(f),axis=0));brute=np.sqrt(np.sum((cp-np.repeat(q,len(f),axis=0))**2,axis=1).reshape(len(q),len(f)).min(1))
    np.testing.assert_allclose(got,brute,atol=1e-9)
    return float(np.max(np.abs(got-brute)))

def exact():
    check=validate_exact();rows=read(DATA/'case_results.json');byid={r['case_id']:r for r in rows}
    ids=['01328DDN_upper','01JNZKAX_lower','KX9HKC3D_upper','01FAXM1D_upper','38JDN4ZV_upper','XNNFHCFV_lower']
    selection=read(OUT/'deep_selection.json')
    for arch in ['upper','lower']:
        candidates=[byid[x['case_id']] for x in selection if x['arch']==arch]
        for key in ['outside_domain_fraction','relative_scale_to_reference']:
            r=max(candidates,key=lambda r:r['alignment'][key]);ids.append(r['case_id'])
        ids.append(min(candidates,key=lambda r:r['alignment']['relative_scale_to_reference'])['case_id'])
    results=[];start=time.time()
    for cid in dict.fromkeys(ids):
        if cid not in byid or not byid[cid].get('entries'):continue
        row=byid[cid];e=row['entries'][0]
        with np.load(row['geometry_path']) as z:v,f=z['vertices_mm'],z['faces']
        surface=ExactSurface(v,f)
        variants=[];legacy_evidence={}
        with np.load(e['samples_path']) as z:current=[z['pos'],z['neg']]
        inv=np.array(e['model_to_source_mm']);variants.append(('current_uniform',current,lambda q:q@inv[:3,:3].T+inv[:3,3]))
        oldroot=Path(r'D:\WorkSpace\Dental\data\Teeth3DS_DMM')
        old=oldroot/'05_dataset_profiles'/'upper_v1.3_strict_fixed_offsurface'/'SdfSamples'/(cid+'.npz')
        transform=oldroot/'03_alignment_profiles'/'strict_fixed'/'transforms'/(cid+'.json')
        if old.exists() and transform.exists() and row['arch']=='upper':
            t=read(transform);rot=np.array(t['rotation']);tr=np.array(t['translation']);s=t['scale']
            with np.load(old) as z:
                previous=[z['pos'],z['neg']];legacy_surface=z['surf'];check_ids=np.linspace(0,len(legacy_surface)-1,512).astype(int)
                mapped=(legacy_surface[check_ids,:3]-tr)@rot/s
            errors=surface.query(mapped)
            actual_sha=hashlib.sha256(old.read_bytes()).hexdigest();meta=read(old.with_name(old.name+'.meta.json'))
            assert actual_sha==meta['content_sha256']
            assert np.max(errors)<.05
            legacy_evidence=dict(npz_sha256=actual_sha,npz_meta_sha_matches=True,legacy_surface_mapping_distance_mm=quant(errors),legacy_surface_mapping_check_points=512)
            variants.append(('legacy_fixed_offsurface',previous,lambda q:(q-tr)@rot/s))
        for name,slots,to_raw in variants:
            rng=np.random.default_rng(seed(cid)+5)
            q=np.concatenate([a[rng.choice(len(a),2048,replace=False),:3] for a in slots]);q=to_raw(q)
            distances=surface.query(q)
            results.append(dict(case_id=cid,arch=row['arch'],status=row['qc_status'],variant=name,query_count=len(q),distance_unsigned_triangle_mm=quant(distances),counts={str(t):int(np.sum(distances<=t)) for t in [.25,.5,1,2,5]},near_2mm_fraction=float(np.mean(distances<=2)),expected_per_2048_space_draw=float(np.mean(distances<=2)*2048),legacy_evidence=legacy_evidence if name=='legacy_fixed_offsurface' else None))
            print(json.dumps(dict(phase='exact',case_id=cid,variant=name,near_2mm=float(np.mean(distances<=2)),seconds=round(time.time()-start,1))),flush=True)
            write(OUT/'exact_distance.json',dict(method='unsigned nearest retained observed triangle; centroid/radius candidate pruning with nearest vertex upper bound; no signed-distance or inside classification',brute_force_validation_max_error=check,results=results,elapsed_seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['domain','exact'],required=True);a=p.parse_args()
    if a.phase=='domain':domain()
    else:exact()
