"""Current NPZ/PKL and scale contract; independent temporary fixtures only."""
import copy
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import subprocess
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party'/'DMM'))
from dmm import TEETH
from dmm.validation import sha256,ContractError
from data.fixed_similarity_dataset import FixedSimilarityDataset,portable_path
from training.fixed_similarity import FORMAT,train_latest

def write(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def fixture(root):
    root.mkdir(exist_ok=True);rows_by_arch={};selected=[];rng=np.random.default_rng(6)
    for arch in TEETH:
        folder=root/'data'/arch;(folder/'SdfSamples').mkdir(parents=True);(folder/'splits').mkdir()
        gum=dict(latent_dim=2,model_type='sine',hyper_hidden_layers=0,hyper_hidden_features=4,mlp_input_dim=3,mlp_output_dim=5,mlp_num_hidden_layers=0,mlp_hidden_features=4)
        specs=dict(labels=[0,*TEETH[arch]],NetworkArchRef='mlp',NetworkSpecsRef=dict(init_dims=[4],output_dims=1,activation='sine'),GumDeformNetworkSpecs=gum,TeethDeformNetworkSpecs=dict(gum,mlp_output_dim=8))
        write(folder/'specs.json',specs);np.savetxt(folder/'avg_centroids.txt',np.zeros((50,3)))
        first=TEETH[arch][0];other=first+10
        rows=[]
        for name,split,status,parent in [('A','train','PASS',None),('A__mirror','train','PASS','A'),('B','train','REVIEW',None),('V','val','PASS',None),('T','test','PASS',None)]:
            cid=name.split('__')[0]+'_'+arch+('__mirror' if parent else '');base=cid.split('__')[0]
            if not parent:selected.append(base)
            centers={str(first):[-.2,.1,.3],str(other):[.2,.1,.3]}
            transform=np.diag([-.025 if parent else .025,.025,.025,1.]);inverse=np.linalg.inv(transform)
            labels=np.repeat([0,first,other],32).astype(np.float32)
            surf=np.zeros((len(labels),5),np.float32);surf[:,:3]=rng.uniform(-.5,.5,(len(labels),3));surf[:,4]=labels
            normals=np.tile(np.array([0,0,1],np.float32),(len(surf),1))
            off=np.zeros((128,5),np.float32);off[:,:3]=rng.uniform(-1,1,(128,3));off[:,4]=-1
            path=folder/'SdfSamples'/(cid+'.npz');np.savez(path,surf=surf,normals=normals,pos=off[:64],neg=off[64:])
            pkl=path.with_suffix('.pkl')
            with pkl.open('wb') as stream:pickle.dump({int(k):np.array(v,np.float32) for k,v in centers.items()},stream)
            rows.append(dict(case_id=cid,patient_id=name.split('__')[0],parent_id='A_'+arch if parent else None,split=split,qc_status=status,centers=centers,
                surface_points=len(surf),offsurface_points=len(off),samples_sha256=sha256(path),centers_sha256=sha256(pkl),
                source_mm_to_model=transform.tolist(),model_to_source_mm=inverse.tolist(),true_inside_outside=False))
        write(folder/'manifest.json',dict(schema='dmm_fixed_similarity/2',arch=arch,reference=dict(case_id='A_'+arch),domain=[-1,1],fixed_model_unit_mm=None,per_case_inverse_required=True,cases=rows))
        for split in ('train','val','test'):
            for cohort in ('all_exported','computational_pass'):
                names=[r['case_id']+'.npz' for r in rows if r['split']==split and (cohort=='all_exported' or r['qc_status']=='PASS')]
                write(folder/'splits'/f'{split}_{cohort}.json',names)
        rows_by_arch[arch]=rows
    write(root/'inputs.json',dict(excluded_ids=['EXCLUDED_upper'],selected_ids=selected));write(root/'references.json',dict(note='synthetic asymmetric reference'))
    config=root/'config.json';write(config,dict(schema=FORMAT,dataset_root='.',arch='upper',samples_per_scene=32,epochs=1,learning_rate=.0001,checkpoint_every=1,cache_cases=1))
    return config

class LatestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp');self.root=Path(self.tmp.name);self.config=fixture(self.root)
    def tearDown(self):self.tmp.cleanup()

    def test_native_schema_no_scale_and_slot_masks(self):
        ds=FixedSimilarityDataset(self.root,'upper',samples_per_scene=32)
        self.assertEqual([r['case_id'] for r in ds.rows],['A_upper','A_upper__mirror'])
        (points,mask,normals,centers),index=ds[0]
        self.assertEqual(int((mask==-1).sum()),16);self.assertTrue(torch.equal(points[16:,4],torch.full((16,),-1.)))
        self.assertLess(float(points[:16,:3].abs().max()),.51)
        with np.load(ds.path(ds.rows[0],'.npz')) as z:
            lookup={tuple(p):i for i,p in enumerate(z['surf'])}
            original_normals=z['normals']
        for p,n in zip(points[:16].numpy(),normals[:16].numpy()):
            self.assertIn(tuple(p),lookup)  # coordinates and FDI are passed through exactly
            np.testing.assert_array_equal(n,original_normals[lookup[tuple(p)]])
        self.assertTrue(torch.equal(centers[11],torch.tensor([-.2,.1,.3])))
        self.assertEqual(ds.summary()['coordinate_contract'],'fixed_reference_similarity_v2; per-case inverse; no global mm/unit')
        ds[1];self.assertEqual(len(ds.cache),1)
        ds.set_epoch(1);other,_=ds[0];self.assertFalse(torch.equal(points,other[0]))

    def test_review_is_explicit_cohort_not_permanent_exclusion(self):
        ds=FixedSimilarityDataset(self.root,'upper',cohort='all_exported',samples_per_scene=16)
        self.assertEqual(len(ds),3)
        with self.assertRaisesRegex(ContractError,'split/cohort'):
            FixedSimilarityDataset(self.root,'upper',case_ids=['V_upper'])
        with self.assertRaisesRegex(ContractError,'mirror parent'):
            FixedSimilarityDataset(self.root,'upper',case_ids=['A_upper__mirror'])

    def test_modified_samples_not_silently_reused(self):
        ds=FixedSimilarityDataset(self.root,'upper',samples_per_scene=16);ds[0]
        with ds.path(ds.rows[0],'.npz').open('ab') as stream:stream.write(b'modified')
        with self.assertRaisesRegex(ContractError,'hash changed'):ds[0]

    def test_cross_arch_patient_leakage_rejected(self):
        path=self.root/'data/lower/manifest.json';m=json.loads(path.read_text());m['cases'][-1]['patient_id']='A';write(path,m)
        with self.assertRaisesRegex(ContractError,'patient split leakage'):FixedSimilarityDataset(self.root,'upper')

    def test_resume_matches_uninterrupted_and_protects_embedding_identity(self):
        full=self.root/'full';train_latest(self.config,full,'cpu',epochs=2)
        first=self.root/'first';r=train_latest(self.config,first,'cpu',max_steps=1,epochs=2)
        self.assertEqual(r['optimizer_steps_this_run'],1);self.assertEqual(r['next_batch'],1)
        resumed=self.root/'resumed';train_latest(self.config,resumed,'cpu',resume=first/'latest.pth',epochs=2)
        a=torch.load(full/'final.pth',weights_only=True);b=torch.load(resumed/'final.pth',weights_only=True)
        for key in a['system_state']:torch.testing.assert_close(a['system_state'][key],b['system_state'][key],atol=0,rtol=0)
        self.assertIsNone(b['model_unit_mm']);self.assertEqual(b['case_ids'],['A_upper','A_upper__mirror'])
        self.assertGreater(sum(r['updated_parameter_tensors'].values()),0)
        with self.assertRaisesRegex(ContractError,'mismatch'):
            train_latest(self.config,self.root/'badresume','cpu',resume=first/'latest.pth',epochs=2,case_ids=['A_upper'])

    def test_train_dmm_entry_dispatches_current_schema(self):
        entry=Path(__file__).resolve().parents[1]/'third_party'/'DMM'/'train_dmm.py'
        out=self.root/'cli'
        result=subprocess.run([sys.executable,'-B',str(entry),'--config',str(self.config),'--output',str(out),'--device','cpu','--max-steps','1','--case-id','A_upper'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads((out/'report.json').read_text());self.assertEqual(report['optimizer_steps_this_run'],1)

if __name__=='__main__':unittest.main()
