"""Read the 2026-10 fixed-similarity NPZ/PKL exports without a 50 mm adapter.

Manifest-only discovery, train-only embedding rows, bounded per-process LRU.
pos/neg retain the upstream 50:50 offsurface sampling; neither is a sign label.
"""
from collections import OrderedDict, Counter
from pathlib import Path, PureWindowsPath
import hashlib
import os
import pickle
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from dmm import TEETH
from dmm.validation import read_json, require, sha256


def portable_path(value, owner=None):
    value = str(value)
    windows = PureWindowsPath(value)
    if os.name != 'nt' and windows.drive:
        require(re.fullmatch(r'[a-zA-Z]:', windows.drive) is not None, 'unsupported Windows data root')
        return Path('/mnt') / windows.drive[0].lower() / Path(*windows.parts[1:])
    path = Path(value)
    return path.resolve() if path.is_absolute() else ((Path(owner).parent if owner else Path.cwd()) / path).resolve()


class FixedSimilarityDataset(Dataset):
    def __init__(self, root, arch, split='train', cohort='computational_pass', samples_per_scene=4096,
                 seed=20261018, cache_cases=1, case_ids=None):
        require(arch in TEETH and split in ('train','val','test'), 'invalid arch/split')
        require(cohort in ('computational_pass','all_exported'), 'invalid cohort')
        require(type(samples_per_scene) is int and samples_per_scene >= 4, 'samples_per_scene must be >=4')
        require(type(cache_cases) is int and cache_cases >= 0, 'invalid cache_cases')
        self.root=portable_path(root);self.arch=arch;self.split=split;self.cohort=cohort
        self.samples_per_scene=samples_per_scene;self.seed=int(seed);self.epoch=0;self.cache_cases=cache_cases
        self.cache=OrderedDict();self.verified={}
        require(not (self.root.parent.parent/'running.lock').exists(), 'preprocessing is running; use a stable completed latest dataset')
        folder=self.root/'data'/arch;self.manifest_path=folder/'manifest.json'
        manifest=read_json(self.manifest_path)
        require(manifest.get('schema')=='dmm_fixed_similarity/2' and manifest.get('arch')==arch, 'expected fixed-similarity v2 manifest')
        require(manifest.get('fixed_model_unit_mm') is None and manifest.get('per_case_inverse_required') is True, 'per-case similarity units required')
        self.reference=manifest['reference'];self.domain=manifest['domain']
        require(self.domain==[-1,1], 'unsupported space domain')
        self.inputs=read_json(self.root/'inputs.json');excluded=set(self.inputs['excluded_ids'])
        candidates=set(self.inputs['selected_ids']);patients={};all_rows={}
        self.dataset_files={}
        # Cross-arch leakage check uses both small manifests, never globbed samples.
        for a in TEETH:
            mpath=self.root/'data'/a/'manifest.json'
            if not mpath.is_file():continue
            m=read_json(mpath);self.dataset_files[str(mpath)]=sha256(mpath)
            require(m.get('schema')=='dmm_fixed_similarity/2' and m['arch']==a, 'mixed dataset schema')
            ids=[e['case_id'] for e in m['cases']]
            require(len(ids)==len(set(ids)), 'duplicate exported case ID')
            lookup={e['case_id']:e for e in m['cases']}
            for e in m['cases']:
                cid=e['case_id'];parent=e.get('parent_id');original=parent or cid
                require(re.fullmatch(r'[A-Za-z0-9_]+',cid) is not None, 'invalid case filename')
                require(original in candidates and original not in excluded and cid not in excluded, 'excluded/unselected input in manifest')
                require(e['split'] in ('train','val','test') and e['qc_status'] in ('PASS','REVIEW'), 'invalid export status/split')
                patient=e['patient_id']
                require(patient not in patients or patients[patient]==e['split'], f'patient split leakage: {patient}')
                patients[patient]=e['split']
                if parent:
                    require(e['split']=='train' and parent in lookup and not lookup[parent].get('parent_id'), 'mirror parent/split invalid')
                    require(lookup[parent]['patient_id']==patient and lookup[parent]['split']==e['split'], 'mirror lineage mismatch')
                if a==arch:all_rows[cid]=e
        self.all_rows=all_rows
        self.split_files={}
        for part in ('train','val','test'):
            for group in ('all_exported','computational_pass'):
                path=folder/'splits'/f'{part}_{group}.json';names=read_json(path)
                expected={cid+'.npz' for cid,e in all_rows.items() if e['split']==part and (group=='all_exported' or e['qc_status']=='PASS')}
                require(len(names)==len(set(names)) and set(names)==expected, f'split manifest drift: {path.name}')
                self.split_files[(part,group)]=names;self.dataset_files[str(path)]=sha256(path)
        ids=[Path(n).stem for n in self.split_files[(split,cohort)]]
        if case_ids is not None:
            require(isinstance(case_ids,list) and bool(case_ids) and len(case_ids)==len(set(case_ids)), 'invalid explicit case subset')
            require(set(case_ids)<=set(ids), 'requested case is not in selected split/cohort')
            require(all(not all_rows[c].get('parent_id') or all_rows[c]['parent_id'] in case_ids for c in case_ids), 'subset must include mirror parent')
            ids=[cid for cid in ids if cid in case_ids]
        require(bool(ids), 'no cases in requested cohort/split')
        self.rows=[dict(all_rows[cid],embedding_row=i) for i,cid in enumerate(sorted(ids))]
        for e in self.rows:
            m=np.asarray(e['source_mm_to_model'],float);inv=np.asarray(e['model_to_source_mm'],float)
            require(m.shape==inv.shape==(4,4) and np.isfinite(m).all() and np.isfinite(inv).all(), 'invalid similarity matrices')
            require(np.allclose(m@inv,np.eye(4),atol=1e-7) and np.allclose(m[3],[0,0,0,1]), 'invalid inverse/homogeneous matrix')
            scale=np.linalg.norm(m[0,:3]);rotation=m[:3,:3]/scale
            require(scale>0 and np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7), 'transform is not a uniform similarity')
            require((np.linalg.det(rotation)<0)==bool(e.get('parent_id')), 'reflection/original mismatch')
            require(set(map(int,e['centers']))<=set(TEETH[arch]) and bool(e['centers']), 'centers FDI mismatch')
            require(e.get('true_inside_outside') is False, 'unexpected spatial label semantics')
            for suffix in ('.npz','.pkl'):require(self.path(e,suffix).is_file(), f'missing export: {e["case_id"]}{suffix}')
        avg_path=folder/'avg_centroids.txt';self.avg_centers=np.loadtxt(avg_path,dtype=np.float32)
        require(self.avg_centers.shape==(50,3) and np.isfinite(self.avg_centers).all(), 'invalid average centers')
        for path in (avg_path,folder/'specs.json',self.root/'inputs.json',self.root/'references.json'):
            self.dataset_files[str(path)]=sha256(path)
        self.specs=read_json(folder/'specs.json')
        require(self.specs['labels']==[0,*TEETH[arch]], 'network FDI must match dataset arch')
        # File paths are excluded from the identity to permit Windows/WSL moves.
        identities=sorted((Path(k).relative_to(self.root).as_posix(),v) for k,v in self.dataset_files.items())
        binding=dict(files=identities,arch=arch,split=split,cohort=cohort,rows=[(r['case_id'],r['embedding_row'],r['samples_sha256'],r['centers_sha256']) for r in self.rows])
        import json
        self.binding=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()

    def path(self,row,suffix):return self.root/'data'/self.arch/'SdfSamples'/(row['case_id']+suffix)

    def assert_current(self):
        require(not (self.root.parent.parent/'running.lock').exists(), 'preprocessing started during training')
        for path,digest in self.dataset_files.items():require(sha256(path)==digest, 'latest metadata changed; restart with new data: '+path)

    def _arrays(self,index):
        row=self.rows[index];cid=row['case_id'];path=self.path(row,'.npz');pkl=self.path(row,'.pkl')
        stamps=tuple((p.stat().st_size,p.stat().st_mtime_ns) for p in (path,pkl))
        if cid in self.cache and self.cache[cid][0]==stamps:
            self.cache.move_to_end(cid);return self.cache[cid][1]
        if self.verified.get(cid)!=stamps:
            require(sha256(path)==row['samples_sha256'] and sha256(pkl)==row['centers_sha256'], 'export hash changed: '+cid)
        with np.load(path,allow_pickle=False) as z:
            require(set(z.files)=={'surf','normals','pos','neg'}, 'unexpected sample keys')
            surf,normals,pos,neg=(z[k] for k in ('surf','normals','pos','neg'))
        require(surf.shape==(row['surface_points'],5) and normals.shape==(len(surf),3), 'surface/normal shape mismatch')
        require(pos.shape==neg.shape==(row['offsurface_points']//2,5), 'space shape mismatch')
        require(all(a.dtype==np.float32 and np.isfinite(a).all() for a in (surf,normals,pos,neg)), 'expected finite float32 arrays')
        require(np.allclose(np.linalg.norm(normals,axis=1),1,atol=2e-6), 'nonunit normals')
        require(set(np.unique(surf[:,4]))=={0,*map(int,row['centers'])}, 'surface FDI/centers mismatch')
        require(np.all(surf[:,3]==0), 'surface placeholder mismatch')
        for a in (pos,neg):
            require(np.all(a[:,3]==0) and np.all(a[:,4]==-1), 'pos/neg are non-surface slots, not signed targets')
            require(np.max(np.abs(a[:,:3]))<=1, 'space points outside cube')
        if row['qc_status']=='PASS':require(np.max(np.abs(surf[:,:3]))<=1+2e-7, 'PASS surface outside domain')
        # Local authenticated PKL, never an uploaded or unverified pickle.
        with pkl.open('rb') as stream:centers=pickle.load(stream)
        require(set(centers)==set(map(int,row['centers'])), 'PKL FDI mismatch')
        matrix=np.zeros((50,3),dtype=np.float32)
        for k,value in centers.items():
            value=np.asarray(value)
            require(value.shape==(3,) and value.dtype==np.float32 and np.isfinite(value).all(), 'invalid PKL centroid')
            require(np.allclose(value,row['centers'][str(k)],atol=2e-7), 'PKL/manifest centroid mismatch')
            matrix[k]=value
        arrays=(surf,normals,pos,neg,matrix);self.verified[cid]=stamps
        if self.cache_cases:
            self.cache[cid]=(stamps,arrays)
            while len(self.cache)>self.cache_cases:self.cache.popitem(last=False)
        return arrays

    def __len__(self):return len(self.rows)
    def set_epoch(self,epoch):self.epoch=int(epoch)

    def __getitem__(self,index):
        surf,normals,pos,neg,centers=self._arrays(index)
        cid=self.rows[index]['case_id'];case_seed=int(hashlib.sha256(cid.encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,case_seed]))
        ns=self.samples_per_scene//2;np_= (self.samples_per_scene-ns)//2;nn=self.samples_per_scene-ns-np_
        require(ns<=len(surf) and np_<=len(pos) and nn<=len(neg), 'batch quota exceeds source pool')
        si=rng.choice(len(surf),ns,replace=False);pi=rng.choice(len(pos),np_,replace=False);ni=rng.choice(len(neg),nn,replace=False)
        points=np.concatenate((surf[si],pos[pi],neg[ni]));mask=np.zeros((len(points),1),np.float32);mask[ns:]=-1
        normal=np.concatenate((normals[si],np.full((len(points)-ns,3),-1,dtype=np.float32)))
        return (torch.from_numpy(points),torch.from_numpy(mask),torch.from_numpy(normal),torch.from_numpy(centers.copy())),index

    def summary(self):
        return dict(arch=self.arch,split=self.split,cohort=self.cohort,cases=len(self),originals=sum(not r.get('parent_id') for r in self.rows),
                    mirrors=sum(bool(r.get('parent_id')) for r in self.rows),status_counts=dict(Counter(r['qc_status'] for r in self.rows)),
                    samples_per_scene=self.samples_per_scene,dataset_binding=self.binding,reference_case=self.reference['case_id'],
                    coordinate_contract='fixed_reference_similarity_v2; per-case inverse; no global mm/unit',
                    accepted_for_training=False,human_visual_acceptance='PENDING',cache_cases=self.cache_cases)
