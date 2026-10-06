"""Read-only audit of frozen alignment, scale, DMM deformation and code regularization."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
import datetime as dt
import json
import pickle
import re
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from run_forward_check import PROJECT_ROOT,sha256
from shape_only_common import load_model


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stats(values):
    x=np.asarray(values,dtype=float)
    x=x[np.isfinite(x)]
    return {'count':len(x),**{k:float(v) for k,v in zip(['min','p05','median','p95','max'],np.percentile(x,[0,5,50,95,100]))}} if len(x) else {'count':0}


def rigid_fit(x,y):
    mx,my=x.mean(0),y.mean(0)
    U,S,Vt=np.linalg.svd((x-mx).T@(y-my))
    D=np.eye(3);D[-1,-1]=np.linalg.det(Vt.T@U.T)
    R=Vt.T@D@U.T;t=my-R@mx
    residual=np.linalg.norm(x@R.T+t-y,axis=1)
    return R,t,residual


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=Path('D:/WorkSpace/Dental/data/Teeth3DS_DMM/05_dataset_profiles/upper_v1.3_strict_fixed_offsurface'))
    parser.add_argument('--experiment',type=Path,default=Path('D:/WorkSpace/Dental/experiments/DMM/20260824'))
    parser.add_argument('--dmm-root',type=Path,default=Path('D:/WorkSpace/Dental/DMM'))
    args=parser.parse_args()
    dataset,experiment,dmm=args.dataset.resolve(),args.experiment.resolve(),args.dmm_root.resolve()
    data_root=dataset.parents[1]; alignment=data_root/'03_alignment_profiles/strict_fixed'
    prep=Path('D:/WorkSpace/Dental/teeth3DS_to_DMM')
    root=PROJECT_ROOT/'runs'/dt.datetime.now(dt.timezone.utc).strftime('training_audit_%Y%m%dT%H%M%SZ')
    root.mkdir(exist_ok=False,parents=True)
    def save(name,value):
        (root/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    def log(message):
        line=f'{dt.datetime.now(dt.timezone.utc).isoformat()} {message}'
        print(line,flush=True)
        with (root/'progress.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
    log(f'START read-only training audit {root}')
    source_files=[dmm/'train_dmm.py',dmm/'data/data_with_labels.py',dmm/'networks/dmm_net.py',dmm/'networks/deform_net.py',dmm/'networks/loss.py',dmm/'networks/meta_modules.py',dmm/'utils/math.py',
        prep/'src/teeth3ds_dmm/alignment/similarity.py',prep/'src/teeth3ds_dmm/alignment/canonical.py',prep/'src/teeth3ds_dmm/alignment/stages.py',
        prep/'src/teeth3ds_dmm/dmm_export/stages.py',prep/'src/teeth3ds_dmm/dmm_export/mirror.py',
        experiment/'specs.json',experiment/'train.log',experiment/'ModelParameters/dmm_best.pth',experiment/'LatentCodes/latent_vecs_best.pth',
        dataset/'avg_centroids.txt',dataset/'freeze_manifest.json',dataset/'qc/validation_report.json',dataset/'splits/train_split.json',dataset/'splits/test_split.json',
        alignment/'freeze_manifest.json',alignment/'m5_freeze_manifest.json',data_root/'02_frozen_protocol/canonical_frame.json']
    digests={str(p):sha256(p) for p in source_files}
    save('input_sha256.json',digests)
    specs=read(experiment/'specs.json'); labels=specs['labels']
    train=sorted(read(dataset/'splits/train_split.json')); test=sorted(read(dataset/'splits/test_split.json'))
    originals=[n for n in train if '__mirror' not in n]
    qc=read(dataset/'qc/validation_report.json'); alignment_manifest=read(alignment/'freeze_manifest.json')
    frame=read(data_root/'02_frozen_protocol/canonical_frame.json')
    avg=np.loadtxt(dataset/'avg_centroids.txt')
    check_hashes=[];centroid_mismatches=[];all_centers={};alignment_rows=[];mirror_mismatches=[]
    per_label_original={k:[] for k in labels if k};per_label_augmented={k:[] for k in labels if k}
    for index,name in enumerate(train+test):
        case=Path(name).stem
        pkl=dataset/'SdfSamples'/f'{case}.pkl'
        digest=sha256(pkl); meta=read(Path(str(pkl)+'.meta.json'))
        check_hashes.append({'path':str(pkl),'sha256':digest,'matches_export_meta':digest==meta['content_sha256']})
        centers={int(k):np.asarray(v,dtype=float) for k,v in pickle.load(pkl.open('rb')).items()}
        all_centers[name]=centers
        if name in train:
            for label,point in centers.items():per_label_augmented[label].append(point)
        if '__mirror' in case:continue
        ap=alignment/'aligned_centroids'/f'{case}.json';tp=alignment/'transforms'/f'{case}.json'
        aj,tr=read(ap),read(tp)
        for path,relative in [(ap,f'aligned_centroids/{case}.json'),(tp,f'transforms/{case}.json')]:
            check_hashes.append({'path':str(path),'sha256':sha256(path),'matches_alignment_manifest':sha256(path)==alignment_manifest['files'][relative]})
        expected={int(k):np.asarray(v) for k,v in aj['centroids'].items()}
        mismatch=max(np.max(np.abs(centers[k]-expected[k])) for k in expected)
        if set(expected)!=set(centers) or mismatch>1e-6:centroid_mismatches.append({'case':case,'max_abs':float(mismatch)})
        if name in train:
            for k,p in expected.items():per_label_original[k].append(p)
        R=np.array(tr['rotation']); t=np.array(tr['translation']); common=tr['common_labels']
        X=np.stack([expected[k] for k in common]);Y=np.stack([frame['reference_centroids'][str(k)] for k in common])
        E=np.linalg.norm(X-Y,axis=1)
        # An additional rigid fit to the training mean is only a descriptor of
        # anatomy/reference bias; it is not a measured alignment error.
        Rm,tm,err=rigid_fit(X,avg[common])
        alignment_rows.append({'case':case,'split':'train' if name in train else 'test','status':tr['status'],
            'scale':tr['scale'],'scale_relative_reference':tr['scale']/frame['canonical_transform']['scale'],
            'det_error':float(abs(np.linalg.det(R)-1)),'orthogonality_error_recomputed':float(np.max(np.abs(R.T@R-np.eye(3)))),
            'rmse_reference':float(np.sqrt(np.mean(E**2))),'saved_rmse_difference':float(abs(np.sqrt(np.mean(E**2))-tr['rmse_after'])),
            'max_reference_distance':float(E.max()),'arch_orientation_angle_degrees':float(np.rad2deg(np.arccos(np.clip(tr['arch_orientation_dot'],-1,1)))),
            'canonical_relation_valid':tr['canonical_relation_valid'],'bbox_outside_fraction':tr['bbox_outside_fraction'],
            'present_count':len(centers),'centroid_pkl_max_abs_difference':float(mismatch),
            'extra_rigid_fit_to_mean_rotation_degrees':float(np.rad2deg(Rotation.from_matrix(Rm).magnitude())),
            'extra_rigid_fit_to_mean_translation_norm':float(np.linalg.norm(tm))})
        if index%150==0:log(f'CENTROIDS {index}/{len(train)+len(test)}')
    for name in originals:
        mirrored=name.replace('.npz','__mirror.npz')
        expected={(k+10 if 10<k<20 else k-10):v*np.array([-1,1,1]) for k,v in all_centers[name].items()}
        actual=all_centers[mirrored]
        if set(actual)!=set(expected) or any(not np.array_equal(actual[k],expected[k]) for k in expected):mirror_mismatches.append(name)
    mean_orig=np.zeros((50,3));mean_aug=np.zeros((50,3))
    for k in per_label_original:
        mean_orig[k]=np.mean(per_label_original[k],0);mean_aug[k]=np.mean(per_label_augmented[k],0)
    train_rows=[r for r in alignment_rows if r['split']=='train']
    split_summary={'train_files':len(train),'train_original_cases':len(originals),'mirror_files':len(train)-len(originals),'test_files':len(test),
        'base_case_overlap_train_test':sorted(set(originals)&set(test)),
        'note':'filename/base-case isolation checked; patient linkage not independently re-derived from source patient manifest'}
    summary={'splits':split_summary,'centroid_mismatches':centroid_mismatches,'mirror_centroid_mismatches':mirror_mismatches,
        'alignment_status_counts':{s:sum(r['status']==s for r in train_rows) for s in sorted({r['status'] for r in train_rows})},
        'train_alignment_statistics':{k:stats([r[k] for r in train_rows]) for k in ['scale','scale_relative_reference','det_error','orthogonality_error_recomputed','rmse_reference','saved_rmse_difference','arch_orientation_angle_degrees','bbox_outside_fraction','extra_rigid_fit_to_mean_rotation_degrees','extra_rigid_fit_to_mean_translation_norm']},
        'avg_centroid_max_abs_difference_original_mean':float(np.max(np.abs(avg-mean_orig))),
        'avg_centroid_max_abs_difference_augmented_mean':float(np.max(np.abs(avg-mean_aug))),
        'avg_centroid_per_tooth_augmented_mean_difference':{str(k):float(np.linalg.norm(avg[k]-mean_aug[k])) for k in per_label_original},
        'checks':{'all_pkl_hashes_match_export':all(r.get('matches_export_meta',True) for r in check_hashes),
            'all_alignment_hashes_match_manifest':all(r.get('matches_alignment_manifest',True) for r in check_hashes),
            'all_pkl_centroids_match_alignment':not centroid_mismatches,'all_mirror_centroids_consistent':not mirror_mismatches,
            'all_train_rotations_proper':all(r['det_error']<1e-8 and r['orthogonality_error_recomputed']<1e-8 for r in train_rows),
            'all_train_canonical_relations_valid':all(r['canonical_relation_valid'] for r in train_rows)}}
    save('alignment_cases.json',alignment_rows);save('data_integrity.json',check_hashes);save('dataset_summary.json',summary)
    log('ALL centroid/alignment metadata audited; select representative NPZ samples independently of model performance')
    # Fixed seeds plus alignment-scale/residual extremes; no selection by fit quality.
    selected={originals[0]}
    for key in ['scale','rmse_reference','bbox_outside_fraction']:
        rows=sorted(train_rows,key=lambda r:r[key])
        for fraction in [0,.5,1]:selected.add(rows[round(fraction*(len(rows)-1))]['case']+'.npz')
    selected.update(test[i] for i in [0,len(test)//2,len(test)-1])
    original_selected=sorted(selected)
    selected.update(n.replace('.npz','__mirror.npz') for n in original_selected if n in train)
    sampled_checks=[];surface_cache={}
    for i,name in enumerate(sorted(selected)):
        p=dataset/'SdfSamples'/name;meta=read(Path(str(p)+'.meta.json'))
        digest=sha256(p)
        with np.load(p) as z:
            a={k:z[k] for k in z.files}
        surf=a['surf']; normals=a['normals'];centers=all_centers[name]
        row={'case':name,'sha256_matches_export':digest==meta['content_sha256'],'array_shapes':{k:list(v.shape) for k,v in a.items()},
            'all_arrays_finite':all(np.isfinite(v).all() for v in a.values()),'surface_outside_cube_fraction':float(np.mean(np.any(abs(surf[:,:3])>1,axis=1))),
            'normal_max_unit_error':float(np.max(np.abs(np.linalg.norm(normals.astype(float),axis=1)-1))),
            'offsurface_sdf_all_zero':bool(np.all(a['pos'][:,3]==0) and np.all(a['neg'][:,3]==0)),
            'offsurface_labels_all_minus_one':bool(np.all(a['pos'][:,4]==-1) and np.all(a['neg'][:,4]==-1)),
            'surface_centroid_mean_distances':{str(k):float(np.linalg.norm(surf[surf[:,4]==k,:3].mean(0,dtype=float)-point)) for k,point in centers.items()}}
        if '__mirror' not in name and name in train:
            per_label={}
            for k in centers:
                points=surf[surf[:,4]==k,:3]
                per_label[k]=points[np.linspace(0,len(points)-1,128,dtype=int)].astype(float)
            surface_cache[name]=per_label
        if '__mirror' in name:
            with np.load(p.with_name(name.replace('__mirror',''))) as z:
                mismatches=[]
                for k in a:
                    expected=z[k].copy();expected[:,0]*=-1
                    if k=='surf':
                        old_labels=expected[:,4].copy()
                        for lab in labels:
                            if lab:expected[old_labels==lab,4]=lab+10 if lab<20 else lab-10
                    if not np.array_equal(expected,a[k]):mismatches.append(k)
                row['mirror_array_mismatches']=mismatches
        sampled_checks.append(row)
        log(f'NPZ sample {i+1}/{len(selected)} {name}')
    save('npz_sample_audit.json',sampled_checks)
    state=torch.load(experiment/'LatentCodes/latent_vecs_best.pth',map_location='cpu',weights_only=True)
    code_stats={};codes={};regularizer_values=[]
    for label in labels:
        X=state['latent_codes'][f'{label}.module.weight'].numpy().astype(float);codes[label]=X
        presence=np.array([label==0 or label in all_centers[n] for n in train])
        std=X.std(0,ddof=1);cov=np.cov(X.T);eig=np.linalg.eigvalsh(cov)
        rms=np.sqrt(np.mean(X**2,1));regularizer_values.append(float(1e6*np.mean(X**2)))
        code_stats[str(label)]={'shape':list(X.shape),'present_training_rows':int(presence.sum()),'abs_dimension_mean':stats(abs(X.mean(0))),
            'dimension_std':stats(std),'code_rms':stats(rms),'present_only_rms':stats(rms[presence]),'missing_only_rms':stats(rms[~presence]),
            'covariance_eigenvalues':eig.tolist(),'covariance_condition_number':float(eig[-1]/eig[0]) if eig[0]>0 else None,
            'global_mean_regularizer_1e6_mse':regularizer_values[-1]}
    log('LATENT statistics loaded; inspect trained spatial SE(3) fields')
    net=load_model({'dmm_root':str(dmm),'experiment':str(experiment),'checkpoint':'best','cpu_threads':4},
        {'model_sha256':sha256(experiment/'ModelParameters/dmm_best.pth'),'specs_sha256':sha256(experiment/'specs.json'),'checkpoint_epoch':state['epoch']})
    from utils.math import screw_axis_to_rt
    field_records=[]
    for name,points_by_label in surface_cache.items():
        row_index=train.index(name)
        for label,points in points_by_label.items():
            q=torch.tensor(points,dtype=torch.float64)
            z=torch.tensor(codes[label][row_index][None],dtype=torch.float64)
            with torch.no_grad():
                raw=net.deform_nets_dict[str(label)](q,z).reshape(-1,8)
                R,T=screw_axis_to_rt(raw[:,:6]);T=T.squeeze(-1)
                moved=(R@q[:,:,None]).squeeze(-1)+T
            if not torch.isfinite(moved).all():
                field_records.append({'case':name,'label':label,'finite':False});continue
            Rn,Tn,yn=R.numpy(),T.numpy(),moved.numpy()
            Rfit,tfit,res=rigid_fit(points,yn)
            Rcenter=Rn[0]
            angle=Rotation.from_matrix(Rn@Rcenter.T).magnitude()
            eig_screw=np.linalg.norm(raw[:,:3].numpy(),axis=1)
            record={'case':name,'label':label,'points':len(points),'finite':True,
                'raw_rotation_vector_norm':stats(eig_screw),'pointwise_rotation_difference_from_first_degrees':stats(np.rad2deg(angle)),
                'pointwise_translation_std_norm':float(np.linalg.norm(Tn.std(0))),
                'best_single_rigid_rotation_degrees':float(np.rad2deg(Rotation.from_matrix(Rfit).magnitude())),
                'best_single_rigid_translation_norm':float(np.linalg.norm(tfit)),
                'residual_to_best_single_rigid':stats(res),'sdf_correction_abs':stats(abs(raw[:,-2].numpy()))}
            # A fixed code direction, no fitting: directly measure how a 0.25
            # training-std change affects the same coordinate deformation field.
            rng=np.random.default_rng(np.random.SeedSequence([413,label]));d=rng.normal(size=len(z[0]));d/=np.sqrt(np.mean(d**2))
            z2=z+torch.tensor((.25*codes[label].std(0,ddof=1)*d)[None])
            with torch.no_grad():
                raw2=net.deform_nets_dict[str(label)](q,z2).reshape(-1,8);R2,T2=screw_axis_to_rt(raw2[:,:6]);y2=(R2@q[:,:,None]).squeeze(-1)+T2.squeeze(-1)
            Rd,td,rd=rigid_fit(yn,y2.numpy())
            record['latent_perturbation_deformation_change']={'sigma_rms':.25,'best_rigid_rotation_degrees':float(np.rad2deg(Rotation.from_matrix(Rd).magnitude())),
                'best_rigid_translation_norm':float(np.linalg.norm(td)),'nonrigid_residual_mean':float(rd.mean())}
            field_records.append(record)
        log(f'FIELD {name}')
    zeros=screw_axis_to_rt(torch.zeros((1,6),dtype=torch.float64))
    log_text=(experiment/'train.log').read_text(encoding='utf-8')
    matches=re.findall(r'Total loss:\s*([\d.]+); Sep loss:\s*([\d.]+); Embd loss:\s*([\d.]+); center loss:\s*([\d.]+); BCE loss:\s*([\d.]+)',log_text)
    logged=np.array(matches,dtype=float)
    latent_summary={'epoch':state['epoch'],'metric_name':state.get('metric_name'),'metric_value':state.get('metric_value'),
        'code_regularizer':'average over batch labels of 1e6 * mean(z_label**2); mean includes batch and latent dimensions',
        'estimated_full_row_full_label_regularizer':float(np.mean(regularizer_values)),
        'per_label':code_stats,'last_16_logged_batches_statistics':{k:stats(logged[-16:,i]) for i,k in enumerate(['total','separate','latent','center','BCE'])},
        'log_sampling_note':'last 16 logged batches only, not all-batch loss decomposition',
        'zero_screw_axis_finite':all(bool(torch.isfinite(t).all()) for t in zeros)}
    save('latent_regularization.json',latent_summary);save('rigid_field_probe.json',field_records)
    checks={'source_inputs_unchanged':all(sha256(Path(n))==d for n,d in digests.items()),
        'data_metadata_checks':all(summary['checks'].values()),'sampled_npz_hashes_and_finite':all(r['sha256_matches_export'] and r['all_arrays_finite'] for r in sampled_checks),
        'sampled_mirror_arrays_match':all(not r.get('mirror_array_mismatches',[]) for r in sampled_checks),
        'field_probes_finite':all(r['finite'] for r in field_records)}
    save('audit_status.json',{'status':'TRAINING_AUDIT_COMPLETED','checks':checks,'root':str(root),
        'scope':{'all_training_and_test_centroids':len(train)+len(test),'original_transforms':len(alignment_rows),'npz_sample_files':len(sampled_checks),'field_case_tooth_probes':len(field_records)},
        'model_training_launched':False,'model_or_dataset_modified':False})
    log(f'COMPLETE {json.dumps(checks)}')


if __name__=='__main__':main()
