"""Compare regenerated content with the lightweight source-machine baseline."""
import hashlib,json,pickle,platform,sys,time,zipfile
from pathlib import Path
import numpy as np
import scipy
import pipeline as p

ROOT=Path(__file__).resolve().parent

def runtime():
    return dict(python=platform.python_version(),system=platform.system(),numpy=np.__version__,scipy=scipy.__version__)

def array_sha(a):
    h=hashlib.sha256()
    h.update(json.dumps(dict(dtype=a.dtype.str,shape=list(a.shape)),sort_keys=True).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def zip_members(path):
    with zipfile.ZipFile(path) as z:
        return {i.filename:dict(crc32=f'{i.CRC:08x}',uncompressed_bytes=i.file_size) for i in z.infolist()}

def numeric_tree(actual,expected,atol=1e-12):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and set(actual)==set(expected) and all(numeric_tree(actual[k],v,atol) for k,v in expected.items())
    if isinstance(expected,list):
        return isinstance(actual,list) and len(actual)==len(expected) and all(numeric_tree(a,b,atol) for a,b in zip(actual,expected))
    if isinstance(expected,float):
        return isinstance(actual,(int,float)) and abs(actual-expected)<=atol
    return actual==expected

def compare(config,selected,full=False):
    start=time.time();out=Path(config['output_root']);rows=p.read(out/'case_results.json')
    expected=p.read(ROOT/'assets/baseline_cases.json');gold=p.read(ROOT/'assets/sample_arrays.json')['entries']
    failures=[];checks=dict(original_cases=len(rows),export_entries=0,npz_container_sha_equal=0,pkl_container_sha_equal=0,npz_member_crc_and_size_equal=0,pkl_value_identity=0,full_array_sha256_checks=0)
    probes={};tables=p.read(ROOT/'assets/tables.json')
    def require(ok,message):
        if not ok:failures.append(message)
    require({r['case_id'] for r in rows}==set(selected),'processed case membership differs')
    require(numeric_tree(p.read(out/'references.json'),p.read(ROOT/'assets/references.json')),'fixed reference differs (>1e-12 or structural mismatch)')
    inputs=p.read(out/'inputs.json');exclude=set(p.read(ROOT/'assets/exclusions.json')['excluded_case_ids'])
    require(set(inputs['excluded_ids'])==exclude and not (set(selected)&exclude),'permanent exclusion scope differs')
    per_arch={a:[] for a in p.TEETH}
    row_fields=('patient_id','arch','assigned_split','qc_status','source_sha256','annotation_sha256','source_vertices','source_faces','retained_faces','removed_degenerate','removed_duplicate_faces','ambiguous_faces','boundary_majority_faces','missing_teeth','topology')
    for row in rows:
        cid=row['case_id'];old=expected[cid]
        for key in row_fields:
            require(row.get(key)==old.get(key),f'{cid}: {key} differs')
        require(set(row.get('issues',[]))==set(old.get('issues',[])),f'{cid}: QC issues differ')
        for key in ('centers_raw_mm','alignment','normal_qc'):
            if key in old:require(numeric_tree(row.get(key),old[key]),f'{cid}: {key} differs (>1e-12 or structural mismatch)')
        require([e['case_id'] for e in row.get('entries',[])]==[e['case_id'] for e in old['entries']],f'{cid}: original/mirror export membership differs')
        old_entries={e['case_id']:e for e in old['entries']}
        for entry in row.get('entries',[]):
            eid=entry['case_id'];b=old_entries[eid];checks['export_entries']+=1
            per_arch[row['arch']].append(dict(entry,qc_status=row['qc_status']))
            for key in ('parent_id','split','contains_reflection','surface_points','offsurface_points','true_inside_outside','fdi_map'):
                require(entry.get(key)==b.get(key),f'{eid}: {key} differs')
            for key in ('source_mm_to_model','model_to_source_mm','centers','mirror_realign_transform'):
                if key in b:require(numeric_tree(entry.get(key),b[key]),f'{eid}: {key} differs (>1e-12)')
            npz=Path(entry['samples_path']);pkl=Path(entry['centers_path'])
            checks['npz_container_sha_equal']+=int(p.sha(npz)==b['samples_sha256'])
            checks['pkl_container_sha_equal']+=int(p.sha(pkl)==b['centers_sha256'])
            match=zip_members(npz)==b['array_members'];checks['npz_member_crc_and_size_equal']+=int(match)
            require(match,f'{eid}: NPY member CRC32/size differs (data content is not identical)')
            with pkl.open('rb') as f:centers=pickle.load(f)
            exact=set(centers)==set(map(int,b['centers'])) and all(np.array_equal(centers[int(k)],np.asarray(v,dtype=np.float32)) for k,v in b['centers'].items())
            checks['pkl_value_identity']+=int(exact);require(exact,f'{eid}: PKL center values differ')
            if eid in gold:
                probes[eid]={}
                with np.load(npz,allow_pickle=False) as z:
                    require(set(z.files)==set(gold[eid]['arrays']),f'{eid}: array keys differ')
                    for key,g in gold[eid]['arrays'].items():
                        a=z[key];exact=array_sha(a)==g['sha256'];checks['full_array_sha256_checks']+=1
                        require(exact,f'{eid}/{key}: full-array SHA256 differs')
                        delta=float(np.max(np.abs(a[g['probe_indices']].astype(np.float64)-np.asarray(g['probe_values'],dtype=np.float64))))
                        probes[eid][key]=dict(full_array_sha256_equal=exact,shape=list(a.shape),dtype=a.dtype.str,probe_max_abs_difference=delta)
    for arch,entries in per_arch.items():
        folder=out/'data'/arch
        require(p.read(folder/'specs.json')==p.read(ROOT/'assets/configs'/f'{arch}_specs.json'),f'{arch}: network specifications differ')
        for split in ('train','val','test'):
            for cohort in ('all_exported','computational_pass'):
                key=f'{split}_{cohort}'
                names=tables[arch]['splits'][key] if full else sorted(e['case_id']+'.npz' for e in entries if e['split']==split and (cohort=='all_exported' or e['qc_status']=='PASS'))
                require(p.read(folder/'splits'/(key+'.json'))==names,f'{arch}/{key}: split membership differs')
        if full:
            require(np.array_equal(np.loadtxt(folder/'avg_centroids.txt'),np.asarray(tables[arch]['avg_centroids'])),f'{arch}: full-batch average centers differ')
            require(p.read(folder/'avg_centroids_support.json')==tables[arch]['avg_centroids_support'],f'{arch}: average-center support differs')
    report=dict(status='PASS_REPRODUCED_DATA_CONTENT' if not failures else 'FAIL_REPRODUCTION',scope='full 1793 originals' if full else 'six selected originals; not a full-batch reproduction claim',
        selected_ids=sorted(selected),runtime=runtime(),baseline_runtime=p.read(ROOT/'assets/provenance.json')['runtime'],checks=checks,
        failures=failures,sample_arrays=probes,elapsed_seconds=time.time()-start,
        identity_methods=dict(all_exports='NPY member CRC32 and uncompressed size (not cryptographic); payload CRC checked by geometry verifier; exact float32 PKL center values',
            selected_exports='Full-array SHA256 including dtype and shape, plus 257 deterministic numeric probes per array',containers='Existing baseline NPZ/PKL SHA256 counted separately; container bytes can differ across compression runtimes'),
        full_training_started=False,raw_data_written=False,human_visual_acceptance='PENDING',accepted_for_training=False,
        sample_average_centroids_are_not_production=not full)
    return report
