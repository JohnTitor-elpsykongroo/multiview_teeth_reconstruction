"""Manifest-driven DMM preprocessing. One latest cache; no freeze lifecycle."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
TEETH = {'upper': tuple(range(11,18))+tuple(range(21,28)),
         'lower': tuple(range(31,38))+tuple(range(41,48))}
VERSION = '2.0.0'





def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def seed(cid, stage, base):
    return int(hashlib.sha256(f'{base}/{cid}/{stage}'.encode()).hexdigest()[:16],16)


def save_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)


def save_pickle(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:pickle.dump(value,f,protocol=4)
    tmp.replace(path)


def legacy(config):
    p=str(Path(config['legacy_root'])/'src')
    if p not in sys.path:sys.path.insert(0,p)
    from teeth3ds_dmm.io.obj_reader import read_obj
    from teeth3ds_dmm.alignment.canonical import build_canonical_frame
    from teeth3ds_dmm.alignment.similarity import solve_similarity
    return read_obj,build_canonical_frame,solve_similarity


def geometry_arrays(vertices,faces,labels):
    triangles=vertices[faces]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area2=np.linalg.norm(cross,axis=1)
    normals=cross/area2[:,None]
    a,b,c=labels[faces].T
    strict=(a==b)&(b==c)
    majority=np.where(a==b,a,np.where(a==c,a,np.where(b==c,b,-1))).astype(np.int16)
    return triangles,area2,normals,strict,majority


def topology(faces):
    e=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    lo,hi=np.min(e,1),np.max(e,1);direction=np.where(e[:,0]<e[:,1],1,-1)
    order=np.lexsort((hi,lo));lo,hi,direction=lo[order],hi[order],direction[order]
    starts=np.r_[0,np.flatnonzero((lo[1:]!=lo[:-1])|(hi[1:]!=hi[:-1]))+1]
    counts=np.diff(np.r_[starts,len(lo)]);winding=np.add.reduceat(direction,starts)
    return dict(boundary_edges=int((counts==1).sum()),nonmanifold_edges=int((counts>2).sum()),
                inconsistent_edges=int(((counts==2)&(winding!=0)).sum()))


def coverage(centers,arch):
    keys=set(map(int,centers));left,right=(2,1) if arch=='upper' else (3,4)
    return (len(keys)>=6 and any(k//10==left for k in keys) and any(k//10==right for k in keys)
            and any(k%10<=2 for k in keys) and any(k%10>=6 for k in keys))


def prepare(job):
    row,config=job;start=time.time();cid=row['case_id'];arch=row['arch']
    out=Path(config['output_root'])/'cache'/cid;meta=out/'geometry.json';npz=out/'geometry.npz'
    result={k:row[k] for k in ('case_id','patient_id','arch','obj','annotation','assigned_split','previous_status','previous_reason')}
    try:
        hs,ha=sha(row['obj']),sha(row['annotation'])
        if hs!=row['source_sha256'] or ha!=row['annotation_sha256']:raise ValueError('SOURCE_IDENTITY_CHANGED')
        signature=identity(['geometry_v2.1',hs,ha,config['geometry_implementation_sha256']])
        if meta.exists():
            prior=read(meta)
            if prior.get('signature')==signature and prior.get('geometry_sha256') and npz.exists() and sha(npz)==prior['geometry_sha256']:
                return dict(prior,cache_hit=True)
        result.update(signature=signature,source_sha256=hs,annotation_sha256=ha,issues=[],warnings=[])
        ann=read(row['annotation']);raw=legacy(config)[0](Path(row['obj']))
        v,f=raw.vertices,raw.faces
        if not len(v) or not len(f) or not np.isfinite(v).all():raise ValueError('EMPTY_OR_NONFINITE_GEOMETRY')
        if f.min()<0 or f.max()>=len(v):raise ValueError('FACE_INDEX_OUT_OF_RANGE')
        lab=np.asarray(ann['labels']);ins=np.asarray(ann['instances'])
        if lab.shape!=(len(v),) or ins.shape!=lab.shape:raise ValueError('VERTEX_ANNOTATION_LENGTH_MISMATCH')
        if not np.issubdtype(lab.dtype,np.integer) or not np.issubdtype(ins.dtype,np.integer):raise ValueError('NONINTEGER_ANNOTATION')
        if ann['id_patient']!=row['patient_id'] or ann['jaw']!=arch:raise ValueError('PATIENT_OR_ARCH_MISMATCH')
        allowed=set(TEETH[arch])|{0};wisdom={18,28} if arch=='upper' else {38,48}
        present=set(map(int,np.unique(lab)))
        if present-(allowed|wisdom):result['issues'].append('INVALID_FDI')
        if present&wisdom:result['issues'].append('WISDOM_TEETH_POLICY_HOLD')
        if np.any(ins<0) or not np.array_equal(lab==0,ins==0):result['issues'].append('LABEL_INSTANCE_MISMATCH')
        if any(len(np.unique(ins[lab==k]))!=1 for k in present):result['issues'].append('MULTIPLE_INSTANCES_PER_FDI')
        if any(len(np.unique(lab[ins==k]))!=1 for k in np.unique(ins)):result['issues'].append('MIXED_INSTANCE_LABELS')
        if row.get('previous_reason')=='PREVIOUS_NORMAL_REJECTION':result['issues'].append('HISTORICAL_MANUAL_NORMAL_HOLD')
        cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
        valid=np.isfinite(cross).all(1)&(np.linalg.norm(cross,axis=1)>1e-12)
        original=np.flatnonzero(valid);f=f[valid]
        if not len(f):raise ValueError('NO_VALID_TRIANGLES')
        _,unique=np.unique(np.sort(f,axis=1),axis=0,return_index=True);unique.sort()
        duplicates=len(f)-len(unique);f=f[unique];original=original[unique]
        tri,area2,norm,strict,majority=geometry_arrays(v,f,lab)
        centers={}
        for k in TEETH[arch]:
            mask=strict&(majority==k)
            if mask.any():centers[str(k)]=np.average(tri[mask].mean(1),weights=area2[mask],axis=0).tolist()
            elif k in present:result['issues'].append(f'NO_STRICT_SURFACE_{k}')
        if not coverage(centers,arch):result['issues'].append('INSUFFICIENT_CENTROID_COVERAGE')
        topo=topology(f)
        if topo['nonmanifold_edges']:result['issues'].append('NONMANIFOLD_SURFACE_REVIEW')
        if topo['inconsistent_edges']:result['issues'].append('INCONSISTENT_WINDING_REVIEW')
        missing=[k for k in TEETH[arch] if k not in present]
        semantic_bad=any(x in result['issues'] for x in ['INVALID_FDI','WISDOM_TEETH_POLICY_HOLD','LABEL_INSTANCE_MISMATCH','MULTIPLE_INSTANCES_PER_FDI','MIXED_INSTANCE_LABELS','INSUFFICIENT_CENTROID_COVERAGE'])
        if 0 not in present:result['issues'].append('NO_GUM_SURFACE');semantic_bad=True
        save_npz(npz,vertices_mm=v,faces=f.astype(np.int32),original_face_id=original.astype(np.int32),
                 vertex_labels=lab.astype(np.int16),vertex_instances=ins.astype(np.int32))
        result.update(status='GEOMETRY_READY',can_sample=not semantic_bad,centers_raw_mm=centers,
                      source_vertices=len(v),source_faces=len(raw.faces),retained_faces=len(f),
                      removed_degenerate=int((~valid).sum()),removed_duplicate_faces=duplicates,
                      ambiguous_faces=int((majority<0).sum()),boundary_majority_faces=int(((majority>=0)&~strict).sum()),
                      topology=topo,missing_teeth=missing,geometry_path=str(npz),geometry_sha256=sha(npz),
                      bounds_raw_mm=[v.min(0).tolist(),v.max(0).tolist()],cache_hit=False)
    except Exception as e:
        result.update(status='FAILED',can_sample=False,issues=[f'{type(e).__name__}: {e}'])
    result['seconds']=time.time()-start;write(meta,result);return result


def canonical(row,config):
    with np.load(row['geometry_path']) as d:v=d['vertices_mm']
    c={int(k):tuple(x) for k,x in row['centers_raw_mm'].items()};arch=row['arch']
    if len(c)!=14 or not row['can_sample'] or row['assigned_split']!='train':raise ValueError(f'INVALID_REFERENCE_{arch}')
    mapping={k:(k if arch=='upper' else (20+k%10 if k//10==3 else 10+k%10)) for k in c}
    frame=legacy(config)[1](reference_case_id=row['case_id'],centroids={mapping[k]:x for k,x in c.items()},
        mesh_vertices=v,right_group=tuple(range(11,18)),left_group=tuple(range(21,28)),
        anterior_group=(11,12,21,22),posterior_group=(16,17,26,27),
        reference_bbox_margin=config['reference_bbox_margin'],manual_orientation_confirmed=False)
    return dict(case_id=row['case_id'],arch=arch,source_sha256=row['source_sha256'],annotation_sha256=row['annotation_sha256'],
                transform=frame.transform.to_dict(),centers_model={str(k):list(frame.reference_centroids[mapping[k]]) for k in c},
                axes='+X patient-left; +Y anterior; +Z=X cross Y; upper crowns +Z, lower crowns -Z',
                bbox_half_extent_mm=frame.bbox_half_extent,reference_issues=row['issues'],
                reference_scope='training-side diagnostic reference; visual review recorded separately; no freeze requirement')


def homogeneous(t):
    h=np.eye(4);h[:3,:3]=t['scale']*np.asarray(t['rotation']);h[:3,3]=t['translation'];return h


def apply(points,t):
    return t['scale']*(np.asarray(points)@np.asarray(t['rotation']).T)+np.asarray(t['translation'])


def fit_centers(centers,reference,config):
    keys=sorted(set(centers)&set(reference['centers_model']),key=int)
    if not coverage({k:centers[k] for k in keys},reference['arch']):raise ValueError('INSUFFICIENT_CORRESPONDENCE')
    x=np.array([centers[k] for k in keys]);y=np.array([reference['centers_model'][k] for k in keys])
    fit=legacy(config)[2](x,y)
    if fit.singular_values[1]<=1e-10:raise ValueError('NEAR_COLLINEAR_CORRESPONDENCE')
    t=fit.to_dict();res=np.linalg.norm(fit.apply(x)-y,axis=1)
    return t,{k:float(r) for k,r in zip(keys,res)}


def alignment_qc(row,t,res,vertices,reference):
    transformed=apply(vertices,t);outside=np.any(np.abs(transformed)>1+1e-7,axis=1)
    h=homogeneous(t);inverse=np.linalg.inv(h)
    recovered=(transformed-h[:3,3])@np.linalg.inv(h[:3,:3]).T
    s=t['scale'];r=np.array(t['rotation'])
    if abs(np.linalg.det(r)-1)>1e-8 or np.max(np.abs(r.T@r-np.eye(3)))>1e-8:raise ValueError('IMPROPER_ROTATION')
    errors=np.array(list(res.values()))
    q=dict(transform=t,source_mm_to_model=h.tolist(),model_to_source_mm=inverse.tolist(),
           relative_scale_to_reference=s/reference['transform']['scale'],mm_per_model_unit=1/s,
           residuals_model=res,rmse_model=float(np.sqrt(np.mean(errors**2))),
           rmse_case_mm=float(np.sqrt(np.mean(errors**2))/s),max_residual_case_mm=float(errors.max()/s),
           inverse_roundtrip_max_mm=float(np.max(np.linalg.norm(recovered-vertices,axis=1))),
           outside_domain_vertices=int(outside.sum()),outside_domain_fraction=float(outside.mean()),
           model_bounds=[transformed.min(0).tolist(),transformed.max(0).tolist()])
    return q






def sample_surface(d,arch,config,cid):
    v,f,lab=d['vertices_mm'],d['faces'],d['vertex_labels']
    tri,area2,norm,strict,majority=geometry_arrays(v,f,lab)
    rng=np.random.default_rng(seed(cid,'surface',config['seed']))
    selected=[];barys=[];labs=[]
    for k in (0,*TEETH[arch]):
        pool=np.flatnonzero(majority==k)
        if not len(pool):
            if k in set(map(int,np.unique(lab))):raise ValueError(f'NO_SAMPLING_SUPPORT_{k}')
            continue
        n=config['surface_gum'] if k==0 else config['surface_per_tooth']
        cumulative=np.cumsum(area2[pool]);idx=pool[np.searchsorted(cumulative,rng.random(n)*cumulative[-1],side='right')]
        u,vv=rng.random(n),rng.random(n);root=np.sqrt(u)
        bary=np.c_[1-root,root*(1-vv),root*vv].astype(np.float32)
        bary[:,2]=1-bary[:,0]-bary[:,1]
        selected.append(idx);barys.append(bary);labs.append(np.full(n,k,np.int16))
    ids=np.concatenate(selected);bary=np.concatenate(barys);lab=np.concatenate(labs)
    points=np.einsum('ni,nij->nj',bary.astype(np.float64),tri[ids])
    return points,norm[ids],lab,ids,bary


def normal_qc(points,normals,labels,reference_surface,config):
    per={};issues=[]
    for k in sorted(set(map(int,labels))-{0}):
        ids=np.flatnonzero(labels==k);idx=ids[np.linspace(0,len(ids)-1,min(512,len(ids))).astype(int)]
        p,n=reference_surface[k]
        distance,nearest=cKDTree(p).query(points[idx],workers=1)
        valid=distance<config['normal_distance_gate_model']
        dots=np.einsum('ij,ij->i',normals[idx][valid],n[nearest[valid]])
        per[str(k)]=dict(matches=int(valid.sum()),median_dot=float(np.median(dots)) if len(dots) else None,
                        negative_fraction=float(np.mean(dots<0)) if len(dots) else None,
                        match_distance_p95=float(np.quantile(distance,.95)))
        if len(dots)<128:issues.append(f'NORMAL_REFERENCE_SUPPORT_{k}')
        elif np.median(dots)<0 or np.mean(dots<0)>.5:issues.append(f'NORMAL_DIRECTION_REVIEW_{k}')
    return per,issues


def export_entry(cid,points,normals,labels,centers,config,arch):
    folder=Path(config['output_root'])/'data'/arch/'SdfSamples'
    rng=np.random.default_rng(seed(cid,'offsurface',config['seed']))
    space=rng.uniform(-1,1,(config['offsurface_points'],3)).astype(np.float32)
    perm=rng.permutation(len(space));space=space[perm]
    off=np.zeros((len(space),5),np.float32);off[:,:3]=space;off[:,4]=-1
    surf=np.zeros((len(points),5),np.float32);surf[:,:3]=points;surf[:,4]=labels
    npz=folder/f'{cid}.npz';pkl=folder/f'{cid}.pkl'
    save_npz(npz,surf=surf,normals=normals.astype(np.float32),pos=off[:len(off)//2],neg=off[len(off)//2:])
    save_pickle(pkl,{int(k):np.asarray(v,dtype=np.float32) for k,v in centers.items()})
    return dict(case_id=cid,samples_path=str(npz),samples_sha256=sha(npz),centers_path=str(pkl),centers_sha256=sha(pkl),
                surface_points=len(points),offsurface_points=len(off),centers=centers,
                true_inside_outside=False,offsurface_semantics='uniform canonical cube; loader-only pos/neg; sdf unused placeholder')


def process(job):
    row,config,reference,refsurface=job;start=time.time();row=dict(row);cid=row['case_id'];arch=row['arch']
    out=Path(config['output_root']);meta=out/'cache'/cid/'processed.json'
    sig=identity(['process_v2.1',row.get('geometry_sha256'),reference,config['processing_recipe'],config['processing_implementation_sha256']])
    try:
        if meta.exists():
            prior=read(meta)
            files=prior.get('files',[])
            if prior.get('process_signature')==sig and files and all(Path(f['path']).is_file() and sha(f['path'])==f['sha256'] for f in files):
                return dict(prior,cache_hit=True)
        row.update(process_signature=sig,entries=[],files=[],cache_hit=False)
        if row['status']!='GEOMETRY_READY':return row
        with np.load(row['geometry_path']) as g:d={k:g[k] for k in g.files}
        if not coverage(row['centers_raw_mm'],arch):
            row['qc_status']='HOLD';write(meta,row);return row
        t,res=fit_centers(row['centers_raw_mm'],reference,config)
        aq=alignment_qc(row,t,res,d['vertices_mm'],reference);row['alignment']=aq
        if aq['outside_domain_vertices']:row['issues'].append('OUTSIDE_CANONICAL_CUBE')
        if aq['rmse_case_mm']>5:row['issues'].append('ALIGNMENT_RMSE_REVIEW')
        if not .7<aq['relative_scale_to_reference']<1.3:row['issues'].append('RELATIVE_SCALE_REVIEW')
        if row['can_sample']:
            p,n,l,ids,bary=sample_surface(d,arch,config,cid)
            points=apply(p,t);normals=n@np.asarray(t['rotation']).T;normals/=np.linalg.norm(normals,axis=1,keepdims=True)
            nq,ni=normal_qc(points,normals,l,refsurface,config);row['normal_qc']=nq;row['issues']+=ni
            provenance=out/'cache'/cid/'sampling.npz'
            save_npz(provenance,original_face_id=d['original_face_id'][ids],barycentric=bary,labels=l)
            row['files'].append(dict(path=str(provenance),sha256=sha(provenance)))
            centers={k:apply(v,t).tolist() for k,v in row['centers_raw_mm'].items()}
            entry=export_entry(cid,points,normals,l,centers,config,arch)
            entry.update(parent_id=None,source_mm_to_model=aq['source_mm_to_model'],model_to_source_mm=aq['model_to_source_mm'],
                         contains_reflection=False,split=row['assigned_split'])
            row['entries'].append(entry)
            if row['assigned_split']=='train' and config['mirror_train']:
                mapping={str(k):str(k+10 if k in TEETH[arch][:7] else k-10) for k in TEETH[arch]}
                reflected={mapping[k]:(np.array(v)*[-1,1,1]).tolist() for k,v in centers.items()}
                mt,mres=fit_centers(reflected,reference,config);mr=np.array(mt['rotation'])
                mp=apply(points*[-1,1,1],mt);mn=(normals*[-1,1,1])@mr.T;mn/=np.linalg.norm(mn,axis=1,keepdims=True)
                ml=np.array([0 if k==0 else int(mapping[str(k)]) for k in l],np.int16)
                mc={k:apply(v,mt).tolist() for k,v in reflected.items()}
                mirror=export_entry(cid+'__mirror',mp,mn,ml,mc,config,arch)
                F=np.diag([-1.,1,1,1]);combo=homogeneous(mt)@F@homogeneous(t)
                mirror.update(parent_id=cid,source_mm_to_model=combo.tolist(),model_to_source_mm=np.linalg.inv(combo).tolist(),
                              contains_reflection=True,reverse_mesh_winding_if_materialized=True,split='train',fdi_map=mapping,
                              mirror_realign_transform=mt,sampling_blueprint_parent=cid,
                              outside_domain_surface_points=int(np.any(np.abs(mp)>1+1e-7,axis=1).sum()))
                if mirror['outside_domain_surface_points']:row['issues'].append('MIRROR_OUTSIDE_CANONICAL_CUBE')
                row['entries'].append(mirror)
            row['sample_replay']=dict(normal_norm_max_error=float(np.max(np.abs(np.linalg.norm(normals.astype(np.float32),axis=1)-1))),
                surface_inverse_max_mm=float(np.max(np.linalg.norm((points-np.array(t['translation']))@np.array(t['rotation'])/t['scale']-p,axis=1))),
                barycentric_sum_max_error=float(np.max(np.abs(bary.sum(1)-1))))
            row['qc_status']='REVIEW' if row['issues'] else 'PASS'
        else:
            row['qc_status']='HOLD'
        for e in row['entries']:
            for pk,hk in [('samples_path','samples_sha256'),('centers_path','centers_sha256')]:row['files'].append(dict(path=e[pk],sha256=e[hk]))
        row['issues']=sorted(set(row['issues']));row['seconds']=time.time()-start;write(meta,row)
        return row
    except Exception as e:
        row.update(qc_status='FAILED',process_error=f'{type(e).__name__}: {e}',seconds=time.time()-start)
        write(meta,row);return row


def parallel(fn,jobs,config,stage):
    output=[];start=time.time();last=start;counts=Counter()
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        future={pool.submit(fn,j):j[0]['case_id'] for j in jobs}
        for f in as_completed(future):
            r=f.result();output.append(r);counts[r.get('qc_status',r['status'])]+=1
            now=time.time()
            if len(output)%20==0 or now-last>20 or len(output)==len(jobs):
                msg=dict(stage=stage,done=len(output),total=len(jobs),seconds=round(now-start,1),status=dict(counts),case_id=r['case_id'])
                print(json.dumps(msg),flush=True);last=now
    return sorted(output,key=lambda r:r['case_id'])


def build_reference_surfaces(rows,refs,config):
    result={}
    for arch,reference in refs.items():
        row=next(r for r in rows if r['case_id']==reference['case_id'])
        with np.load(row['geometry_path']) as g:d={k:g[k] for k in g.files}
        p,n,l,_,_=sample_surface(d,arch,config,row['case_id'])
        p=apply(p,reference['transform']);n=n@np.array(reference['transform']['rotation']).T
        result[arch]={}
        for k in TEETH[arch]:
            ids=np.flatnonzero(l==k);idx=ids[np.linspace(0,len(ids)-1,min(4096,len(ids))).astype(int)]
            result[arch][k]=(p[idx],n[idx])
    return result


def finalize(rows,config,refs,started):
    out=Path(config['output_root']);write(out/'case_results.json',rows)
    summary=dict(schema='dmm_preprocess_latest/2',version=VERSION,completed_utc=datetime.now(timezone.utc).isoformat(),
                 input_cases=len(rows),status_counts=dict(Counter(r.get('qc_status',r['status']) for r in rows)),
                 reason_counts=dict(Counter(x for r in rows for x in r.get('issues',[]))),
                 export_originals=sum(bool(r.get('entries')) for r in rows),export_entries=sum(len(r.get('entries',[])) for r in rows),
                 permanent_exclusions=7,source_files_modified=False,training_started=False,
                 human_visual_acceptance='PENDING',accepted_for_training=False,
                 elapsed_seconds=time.time()-started,output_root=str(out),arches={})
    for arch in TEETH:
        subset=[r for r in rows if r['arch']==arch];folder=out/'data'/arch;folder.mkdir(parents=True,exist_ok=True)
        entries=[dict(e,patient_id=r['patient_id'],qc_status=r.get('qc_status'),issues=r['issues']) for r in subset for e in r.get('entries',[])]
        write(folder/'manifest.json',dict(schema='dmm_fixed_similarity/2',arch=arch,reference=refs[arch],domain=[-1,1],
            fixed_model_unit_mm=None,per_case_inverse_required=True,accepted_for_training=False,cases=entries))
        split_counts={};avg=np.zeros((50,3));support={}
        for split in ['train','val','test']:
            allfiles=sorted(e['case_id']+'.npz' for e in entries if e['split']==split)
            passing=sorted(e['case_id']+'.npz' for e in entries if e['split']==split and e['qc_status']=='PASS')
            write(folder/'splits'/f'{split}_all_exported.json',allfiles)
            write(folder/'splits'/f'{split}_computational_pass.json',passing);split_counts[split]=dict(all_exported=len(allfiles),computational_pass=len(passing))
        for k in TEETH[arch]:
            values=[e['centers'][str(k)] for e in entries if e['split']=='train' and e['qc_status']=='PASS' and str(k) in e['centers']]
            if not values:raise ValueError(f'NO_PASS_TRAIN_SUPPORT_{arch}_{k}')
            avg[k]=np.mean(values,axis=0);support[str(k)]=len(values)
        np.savetxt(folder/'avg_centroids.txt',avg,fmt='%.10g')
        write(folder/'avg_centroids_support.json',dict(source='computational_pass train entries including realigned mirrors; missing teeth omitted',support=support))
        specs=read(Path(config['archive_root'])/'configs'/f'{arch}_specs.json');write(folder/'specs.json',specs)
        summary['arches'][arch]=dict(inputs=len(subset),status_counts=dict(Counter(r.get('qc_status',r['status']) for r in subset)),splits=split_counts)
    return summary




def load_config(path):
    config=read(path);out=Path(config['output_root']).resolve();project=Path(config['project_root']).resolve()
    if out!=project/'output'/'latest':raise ValueError('OUTPUT_MUST_BE_PROJECT_LATEST')
    if not project.is_dir():raise ValueError('PROJECT_MISSING')
    config['geometry_implementation_sha256']=identity([inspect.getsource(f) for f in [prepare,geometry_arrays,topology,coverage]])
    config['processing_implementation_sha256']=identity([inspect.getsource(f) for f in [canonical,fit_centers,alignment_qc,sample_surface,normal_qc,export_entry,process]])
    config['processing_recipe']={k:config[k] for k in ['seed','surface_per_tooth','surface_gum','offsurface_points','mirror_train','normal_distance_gate_model']}
    return config


def run(config,limit=None,stage='all'):
    start=time.time();out=Path(config['output_root']);out.mkdir(parents=True,exist_ok=True)
    lock=out/'running.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:
        source=read(config['candidate_manifest'])['cases'];excluded=set(read(config['exclusion_policy'])['excluded_case_ids'])
        if len(source)!=1793 or len({r['case_id'] for r in source})!=1793:raise ValueError('CANDIDATE_COUNT_OR_IDS_CHANGED')
        if any(r['case_id'] in excluded for r in source) or len(excluded)!=7:raise ValueError('EXCLUSION_SCOPE_MISMATCH')
        if any(r['assigned_split'] not in ['train','val','test'] for r in source):raise ValueError('MISSING_EXISTING_SPLIT')
        patients={}
        for r in source:
            if r['patient_id'] in patients and patients[r['patient_id']]!=r['assigned_split']:raise ValueError('PATIENT_SPLIT_LEAKAGE')
            patients[r['patient_id']]=r['assigned_split']
        selected=source if limit is None else [r for r in source if r['case_id'] in set(limit)|set(config['reference_cases'].values())]
        write(out/'run_config.json',config)
        inputs=dict(candidate_manifest=config['candidate_manifest'],candidate_manifest_sha256=sha(config['candidate_manifest']),
                    exclusion_policy_sha256=sha(config['exclusion_policy']),selected_ids=[r['case_id'] for r in selected],
                    preserved_patient_splits=True,excluded_ids=sorted(excluded),no_raw_glob=True)
        write(out/'inputs.json',inputs)
        geometry=parallel(prepare,[(r,config) for r in selected],config,'geometry')
        if stage=='geometry':return dict(geometry_cases=len(geometry))
        refs={a:canonical(next(r for r in geometry if r['case_id']==cid),config) for a,cid in config['reference_cases'].items()}
        write(out/'references.json',refs);refsurf=build_reference_surfaces(geometry,refs,config)
        ready=[r for r in geometry if r['status']=='GEOMETRY_READY'];failed=[dict(r,qc_status='FAILED') for r in geometry if r['status']!='GEOMETRY_READY']
        rows=parallel(process,[(r,config,refs[r['arch']],refsurf[r['arch']]) for r in ready],config,'align_sample_export')+failed
        rows=sorted(rows,key=lambda r:r['case_id']);summary=finalize(rows,config,refs,start)
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True);return summary
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=str(ROOT/'config.json'))
    parser.add_argument('--cases',nargs='*');parser.add_argument('--stage',choices=['all','geometry'],default='all');args=parser.parse_args()
    run(load_config(args.config),args.cases,args.stage)


if __name__=='__main__':main()
