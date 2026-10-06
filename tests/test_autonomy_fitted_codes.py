"""Export-equivalent coordinates, uncontaminated train support and code binding."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_autonomy_fitted_codes import export_statistics,component_prior,inspect
from inspect_autonomy_latents import sha
from test_dual_arch import training_fixture
from training.arch_training import train_arch,export_bundle


class FittedCodeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def fixture(self):
        folder=self.root/'evaluation';artifact=folder/'case_0000';artifact.mkdir(parents=True)
        rows=[dict(case_id=case,embedding_row=i,split=split,augmentation_parent_id=parent,presence={'11':present})
              for i,(case,split,parent,present) in enumerate([
                  ('a','train',None,True),('b','train',None,False),('c','train',None,True),
                  ('d','train',None,True),('mirror','train','a',True),('untrained','train',None,True),
                  ('v','val',None,True),('t','test',None,True)])]
        saved=dict(arch='upper',epoch=5,manifest_sha256='f'*64,canonical_reference_id='canonical',
                   training_case_ids=['a','b','c','d','mirror'],case_rows=rows,
                   system_state={'latents.0.weight':torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[100.,100.],[200.,200.],[300.,300.],[float('nan'),0.]]),
                                 'latents.11.weight':torch.tensor([[1.,0.],[999.,999.],[0.,1.],[.5,1.5],[100.,100.],[200.,200.],[300.,300.],[float('nan'),0.]])})
        mu,_,factor,_=component_prior(saved,11)
        code=mu+factor@np.asarray([1.,-2.])
        gum_mu=component_prior(saved,0)[0]
        np.savez(artifact/'codes.npz',model_unit_mm=50,code_0=gum_mu,code_11=code)
        (artifact/'field.json').write_text(json.dumps(dict(codes_sha256=sha(artifact/'codes.npz'))))
        checkpoint=self.root/'checkpoint.pth';torch.save(saved,checkpoint)
        (folder/'provenance.json').write_text(json.dumps(dict(split='val',checkpoint_sha256=sha(checkpoint),
                                                           manifest_sha256='f'*64,canonical_reference_id='canonical')))
        geometry=dict(score_mm=1.,per_tooth={'11':{'observed_to_mesh_sample_mean_mm':1.}})
        evaluation=dict(split='val',requested_case_count=1,successful_case_count=1,failed_case_count=0,failures=[],score_mm=1.,
                        cases=[dict(case_id='v',split='val',status='EVALUATED_NOT_QUALITY_ACCEPTED',code_mode='temporary_fitted',
                                    geometry=geometry,artifact_directory='case_0000')])
        (folder/'evaluation.json').write_text(json.dumps(evaluation))
        return folder,checkpoint,saved

    def test_statistics_match_actual_export_arrays_exactly(self):
        config,_=training_fixture(self.root,'upper')
        train_arch(config,self.root/'training')
        checkpoint=self.root/'training/final.pth'
        export_bundle(checkpoint,self.root/'bundle','unit')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        with np.load(self.root/'bundle/latent_statistics.npz',allow_pickle=False) as arrays:
            for label in saved['specs']['labels']:
                mu,covariance,factor,rows=component_prior(saved,label)
                for prefix,value in [('mu',mu),('cov',covariance),('L',factor)]:
                    np.testing.assert_array_equal(value,arrays[f'{prefix}_{label}'])
                self.assertEqual(len(rows),arrays[f'count_{label}'].item())

    def test_only_original_present_trained_support_and_known_q_recovered(self):
        folder,checkpoint,saved=self.fixture()
        self.assertEqual(component_prior(saved,11)[3],['a','c','d'])
        result=inspect(checkpoint,folder)
        self.assertEqual(result['failed'],0)
        codes=result['cases'][0]['components']
        np.testing.assert_allclose(codes['11']['standardized_q'],[1.,-2.],rtol=0,atol=1e-12)
        self.assertAlmostEqual(codes['11']['standardized_q_rms'],np.sqrt(2.5))
        self.assertTrue(codes['0']['bundle_gum_fixed'])
        self.assertAlmostEqual(codes['0']['standardized_q_rms'],0.)

    def test_test_rejected_before_checkpoint_or_metrics_access(self):
        folder=self.root/'test';folder.mkdir()
        (folder/'provenance.json').write_text(json.dumps(dict(split='test')))
        with self.assertRaisesRegex(ValueError,'excludes test'):
            inspect(self.root/'missing',folder)

    def test_missing_codes_and_tampered_codes_preserve_failed_case(self):
        folder,checkpoint,_=self.fixture();codes=folder/'case_0000/codes.npz'
        with codes.open('ab') as stream: stream.write(b'changed')
        result=inspect(checkpoint,folder)
        self.assertEqual(result['requested'],1);self.assertEqual(result['failed'],1)
        self.assertIn('SHA mismatch',result['cases'][0]['error'])
        codes.unlink()
        result=inspect(checkpoint,folder)
        self.assertIn('fresh evaluation required',result['cases'][0]['error'])

    def test_insufficient_support_and_zero_variance_rejected(self):
        with self.assertRaisesRegex(ValueError,'support'): export_statistics(np.zeros((3,0)))
        with self.assertRaisesRegex(ValueError,'support'): export_statistics([[1.,2.],[3.,4.]])
        with self.assertRaisesRegex(ValueError,'zero'): export_statistics([[1.,2.]]*3)


if __name__=='__main__': unittest.main()
