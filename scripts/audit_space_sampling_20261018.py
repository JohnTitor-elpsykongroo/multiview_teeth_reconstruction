"""Read-only audit of the latest fixed-reference DMM space samples."""
from pathlib import Path
import argparse, hashlib, json, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import hypergeom

PROJECT = Path(r'D:\WorkSpace\Dental\Teeth3DS_DualArch_Preprocess')
DATA = PROJECT/'output'/'latest'
OUT = Path(r'D:\WorkSpace\Dental\multiview_teeth_reconstruction\runs\space_sampling_audit_20261018_01')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(cid):return int(hashlib.sha256(f'20261017/{cid}/offsurface'.encode()).hexdigest()[:16],16)
def quant(x):return dict(zip(['min','p05','p25','p50','p75','p95','max'],map(float,np.quantile(x,[0,.05,.25,.5,.75,.95,1]))))

def global_job(job):
    e, row = job
    cid=e['case_id']; start=time.time()
    try:
        stat=Path(e['samples_path']).stat()
        with np.load(e['samples_path'],allow_pickle=False) as z:pos,neg=z['pos'],z['neg']
        flags=[]
        for name,a in [('pos',pos),('neg',neg)]:
            if a.shape!=(250000,5) or a.dtype!=np.float32 or not np.isfinite(a).all():flags.append(name+'_shape_dtype_finite')
            if np.any(a[:,3]!=0) or np.any(a[:,4]!=-1):flags.append(name+'_semantics')
        x=np.concatenate((pos[:,:3],neg[:,:3]))
        if np.any(np.abs(x)>1):flags.append('outside_cube')
        rng=np.random.default_rng(seed(cid));expected=rng.uniform(-1,1,(500000,3)).astype(np.float32);expected=expected[rng.permutation(500000)]
        replay=np.array_equal(x,expected)
        if not replay:flags.append('seed_replay_mismatch')
        mean=x.mean(0,dtype=np.float64);var=x.var(0,dtype=np.float64)
        correlations=[float(np.mean((x[:,i].astype(float)-mean[i])*(x[:,j].astype(float)-mean[j]))/np.sqrt(var[i]*var[j])) for i,j in [(0,1),(0,2),(1,2)]]
        bin8=np.clip(((x+1)*4).astype(np.int16),0,7)
        hist=np.bincount((bin8[:,0]*64+bin8[:,1]*8+bin8[:,2]).astype(int),minlength=512)
        reduced_chi=float(np.sum((hist-500000/512)**2/(500000/512))/511)
        slot_means=[a[:,:3].mean(0,dtype=np.float64).tolist() for a in [pos,neg]]
        if np.max(np.abs(mean))>.01 or np.max(np.abs(var-1/3))>.01 or max(map(abs,correlations))>.01 or reduced_chi>2:flags.append('gross_nonuniformity')
        subset=np.concatenate((x[:2048],x[250000:252048]))
        subset_dups=len(subset)-len(np.unique(subset,axis=0))
        if subset_dups:flags.append('subset_duplicate_rows')
        scale=float(np.linalg.norm(np.array(e['source_mm_to_model'])[0,:3]))
        return dict(case_id=cid,parent_id=e.get('parent_id'),arch=row['arch'],split=row['assigned_split'],status=row['qc_status'],
            seed=seed(cid),n=len(x),mean=mean.tolist(),var=var.tolist(),axis_corr=correlations,slot_means=slot_means,
            min=x.min(0).tolist(),max=x.max(0).tolist(),joint_8cube_reduced_chi2=reduced_chi,
            slot_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in [pos,neg]],xyz_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            replay_exact=replay,subset_duplicate_rows=subset_dups,mm_per_model_unit=1/scale,
            pool_density_per_mm3=500000/(8/scale**3),expected_2048_step_density_per_mm3=2048/(8/scale**3),
            flags=flags,file_size=stat.st_size,file_mtime_ns=stat.st_mtime_ns,seconds=time.time()-start)
    except Exception as ex:return dict(case_id=cid,error=f'{type(ex).__name__}: {ex}')

def run_global():
    OUT.mkdir(exist_ok=False,parents=True)
    rows=read(DATA/'case_results.json');before=sha(DATA/'case_results.json');start=time.time()
    jobs=[(e,r) for r in rows for e in r.get('entries',[])]
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(global_job,j) for j in jobs]
        for f in as_completed(futures):
            results.append(f.result())
            if len(results)%100==0 or len(results)==len(jobs):print(json.dumps(dict(phase='global',done=len(results),total=len(jobs),seconds=round(time.time()-start,1))),flush=True)
    results.sort(key=lambda r:r['case_id']);write(OUT/'global_entries.json',results)
    good=[r for r in results if 'error' not in r];duplicate_groups=[]
    for key in ['xyz_sha256','slot_sha256']:
        groups=defaultdict(list)
        for r in good:
            values=[r[key]] if key=='xyz_sha256' else r[key]
            for i,h in enumerate(values):groups[h].append(r['case_id']+(f'/slot{i}' if key=='slot_sha256' else ''))
        duplicate_groups.extend(dict(key=key,entries=g) for g in groups.values() if len(g)>1)
    summary=dict(entry_count=len(results),point_count=sum(r['n'] for r in good),errors=[r for r in results if 'error' in r],flagged=[dict(case_id=r['case_id'],flags=r['flags']) for r in good if r['flags']],
        exact_replays=sum(r['replay_exact'] for r in good),unique_seeds=len({r['seed'] for r in good}),duplicate_groups=duplicate_groups,
        max_abs_mean=max(max(map(abs,r['mean'])) for r in good),max_abs_var_error=max(max(abs(v-1/3) for v in r['var']) for r in good),
        max_abs_correlation=max(max(map(abs,r['axis_corr'])) for r in good),reduced_chi2=quant([r['joint_8cube_reduced_chi2'] for r in good]),
        training_original_scale_mm=quant([r['mm_per_model_unit'] for r in good if r['split']=='train' and not r['parent_id']]),
        training_original_step_density_mm3=quant([r['expected_2048_step_density_per_mm3'] for r in good if r['split']=='train' and not r['parent_id']]),
        dataset_sha256_before=before,dataset_sha256_after=sha(DATA/'case_results.json'),elapsed_seconds=time.time()-start,
        scope='all exported entries, full 500k coordinate/semantics/replay; duplicate-row check 4096 deterministic rows per entry; exact entire-cloud/slot hash comparisons')
    write(OUT/'global_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def select_deep(rows):
    selected={}
    for arch in ['upper','lower']:
        group=[r for r in rows if r['arch']==arch and r['assigned_split']=='train' and r.get('entries')]
        for key in ['relative_scale_to_reference','rmse_case_mm','outside_domain_fraction']:
            order=sorted(group,key=lambda r:r['alignment'][key])
            for q in [0,.25,.5,.75,.95,1]:
                r=order[round((len(order)-1)*q)];selected[r['case_id']]=r
        for status in ['PASS','REVIEW']:
            order=[r for r in group if r['qc_status']==status]
            for r in order[:2]:selected[r['case_id']]=r
        for r in group:
            if r['case_id'] in ['KX9HKC3D_upper','01FAXM1D_upper','38JDN4ZV_upper','XNNFHCFV_lower','01328DDN_upper','01JNZKAX_lower']:selected[r['case_id']]=r
    return list(selected.values())

def deep_job(row):
    start=time.time();out=[]
    try:
        for e in row['entries']:
            if e.get('parent_id') and row['case_id'] not in ['01328DDN_upper','01JNZKAX_lower','KX9HKC3D_upper','XNNFHCFV_lower']:continue
            with np.load(e['samples_path']) as z:surf,pos,neg,normals=[z[k] for k in ['surf','pos','neg','normals']]
            scale=np.linalg.norm(np.array(e['source_mm_to_model'])[0,:3]);mm=1/scale
            points=surf[:,:3];labels=surf[:,4].astype(int);space=np.concatenate((pos[:,:3],neg[:,:3]))
            exact_duplicates=len(space)-len(np.unique(space,axis=0))
            outside=np.any(np.abs(points)>1,axis=1)
            outside_distance=np.linalg.norm(np.maximum(np.abs(points)-1,0),axis=1)*mm
            upper_tree=cKDTree(points);qids=np.random.default_rng(seed(e['case_id'])+1).choice(len(space),20000,replace=False)
            # Distance to dense observed surface samples is an upper bound, not exact triangle distance.
            distances,idx=upper_tree.query(space[qids],workers=1);distances_mm=distances*mm
            tooth_tree=cKDTree(points[labels!=0]);td,ti=tooth_tree.query(space[qids],workers=1)
            counts_near={str(t):int(np.sum(distances_mm<=t)) for t in [.25,.5,1,2,5]}
            tree_pos=cKDTree(pos[:,:3]);tree_neg=cKDTree(neg[:,:3])
            anchor_ids=np.concatenate([np.flatnonzero(labels==k)[np.linspace(0,np.sum(labels==k)-1,64).astype(int)] for k in sorted(set(labels))])
            anchor=points[anchor_ids]
            coverage={}
            for rad_mm in [.5,1,2]:
                kp=tree_pos.query_ball_point(anchor,rad_mm/mm,return_length=True,workers=1);kn=tree_neg.query_ball_point(anchor,rad_mm/mm,return_length=True,workers=1)
                zero=np.exp(hypergeom.logpmf(0,250000,kp,1024)+hypergeom.logpmf(0,250000,kn,1024))
                coverage[str(rad_mm)]=dict(pool_neighbor_counts=quant(kp+kn),mean_expected_points_per_2048_step=float(np.mean((kp+kn)*2048/500000)),mean_probability_no_local_point_per_2048_step=float(zero.mean()),pool_zero_anchors=int(np.sum(kp+kn==0)))
            per_label=[]
            for k in sorted(set(labels)):
                ids=np.flatnonzero(labels==k)
                near_count=int(np.sum((distances_mm<=2)&(labels[idx]==k)))
                per_label.append(dict(fdi=int(k),surface_pool=len(ids),outside_count=int(outside[ids].sum()),outside_fraction=float(outside[ids].mean()),outside_distance_max_mm=float(outside_distance[ids].max()),
                    nearest_component_2mm_query_count=near_count,expected_near_points_per_2048_step=near_count/20000*2048))
            record=dict(case_id=e['case_id'],parent_id=e.get('parent_id'),arch=row['arch'],status=row['qc_status'],split=row['assigned_split'],mm_per_model_unit=float(mm),
                exact_duplicate_rows=exact_duplicates,surface_count=len(surf),outside_surface_count=int(outside.sum()),outside_surface_fraction=float(outside.mean()),
                outside_to_cube_distance_mm=quant(outside_distance[outside]) if outside.any() else None,
                surface_bounds=[points.min(0).tolist(),points.max(0).tolist()],query_count=20000,
                distance_to_surface_pool_mm=quant(distances_mm),near_surface_upper_bound_counts=counts_near,
                near_tooth_pool_2mm_fraction=float(np.mean(td*mm<=2)),
                expected_near_points_per_2048_step={k:v/20000*2048 for k,v in counts_near.items()},local_anchor_neighborhood=coverage,per_label=per_label)
            # Actual official RAM-loader batches, not training, summarized by surface-tree queries below.
            import sys
            sys.path.insert(0,r'D:\WorkSpace\Dental\DMM')
            from data.data_with_labels import read_sdf_samples_into_ram,unpack_sdf_samples_from_ram
            data=read_sdf_samples_into_ram(e['samples_path']);np.random.seed(seed(e['case_id'])%2**32)
            batches={str(n):[] for n in [4096,8192]};all_draws=[]
            for n in [4096,8192]:
                for repeat in range(20):
                    batch,mask,_,_=unpack_sdf_samples_from_ram(data,n)
                    q=batch[mask[:,0]==-1,:3].numpy();d,_=upper_tree.query(q,workers=1)
                    batches[str(n)].append(dict(space_points=len(q),near_1mm=int(np.sum(d*mm<=1)),near_2mm=int(np.sum(d*mm<=2))))
                    if n==4096:all_draws.append(q)
            record['official_loader_batches']=batches
            record['elapsed_seconds']=time.time()-start;out.append(record)
        return out
    except Exception as ex:return [dict(case_id=row['case_id'],error=f'{type(ex).__name__}: {ex}')]

def run_deep():
    rows=read(DATA/'case_results.json');selected=select_deep(rows);write(OUT/'deep_selection.json',[dict(case_id=r['case_id'],arch=r['arch'],status=r['qc_status'],split=r['assigned_split']) for r in selected]);start=time.time();results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(deep_job,r) for r in selected]
        for i,f in enumerate(as_completed(futures)):
            results.extend(f.result());write(OUT/'deep_entries.json',results)
            print(json.dumps(dict(phase='deep',done=i+1,total=len(selected),seconds=round(time.time()-start,1))),flush=True)
    write(OUT/'deep_entries.json',sorted(results,key=lambda r:r['case_id']))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=['global','deep'],required=True);args=a.parse_args()
    if args.phase=='global':run_global()
    else:run_deep()
