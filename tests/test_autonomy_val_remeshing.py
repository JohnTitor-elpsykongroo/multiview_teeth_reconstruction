"""Saved held-out code restoration, real geometry parity and failure isolation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from remesh_autonomy_val import remesh
from inspect_autonomy_latents import sha
import test_autonomy_remeshing as fixtures


class ValRemeshingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def fixture(self):
        folder,checkpoint,evaluation = fixtures.RemeshingTests.fixture(self)
        saved = torch.load(checkpoint,weights_only=True)
        saved['case_rows'].append(dict(case_id='v',patient_id='pv',embedding_row=1,split='val',
                                      augmentation_parent_id=None,presence={'11':True}))
        # Neither train nor val embedding row matches the saved post-fit code.
        saved['system_state']['latents.11.weight'] = torch.tensor([[7.,7.],[8.,8.]])
        saved['system_state']['latents.0.weight'] = torch.zeros(2,2)
        torch.save(saved,checkpoint)
        p = json.loads((folder/'provenance.json').read_text())
        p.update(split='val',checkpoint_sha256=sha(checkpoint))
        (folder/'provenance.json').write_text(json.dumps(p))
        evaluation['split']='val'
        evaluation['cases'][0].update(case_id='v',patient_id='pv',split='val',code_mode='temporary_fitted')
        (folder/'evaluation.json').write_text(json.dumps(evaluation))
        return folder,checkpoint,evaluation

    def test_train_and_test_rejected_before_checkpoint_or_metrics(self):
        folder = self.root/'heldout'
        folder.mkdir()
        for split in ('train','test'):
            (folder/'provenance.json').write_text(json.dumps(dict(split=split)))
            with patch('torch.load',side_effect=AssertionError('checkpoint accessed')):
                with self.assertRaisesRegex(ValueError,'only val'):
                    remesh(self.root/'missing',folder,'v',self.root/'output')
        self.assertFalse((self.root/'output').exists())

    def test_real_mesh_uses_exact_saved_fit_codes_without_optimizer(self):
        folder,checkpoint,evaluation = self.fixture()
        before = sha(checkpoint)
        with patch('networks.dmm_net.DMM',side_effect=lambda *a,**kw:fixtures.Sphere()), \
             patch('torch.optim.Adam',side_effect=AssertionError('refitting forbidden')):
            result = remesh(checkpoint,folder,'v',self.root/'output',resolutions=(8,12))
        self.assertEqual(result['grids'][0]['geometry'],evaluation['cases'][0]['geometry'])
        self.assertEqual(result['grids'][0]['parity'],'PASS_1e-5mm')
        with np.load(self.root/'output/grid_12/codes.npz') as codes:
            np.testing.assert_array_equal(codes['code_11'],np.array([.1,-.1],dtype=np.float32))
        self.assertEqual(before,sha(checkpoint))
        self.assertEqual(result['failed_grid_count'],0)

    def test_missing_or_tampered_codes_require_fresh_evaluation(self):
        folder,checkpoint,_ = self.fixture()
        codepath = folder/'case_0000/codes.npz'
        contents = codepath.read_bytes()
        codepath.unlink()
        with self.assertRaisesRegex(ValueError,'fresh evaluation required'):
            remesh(checkpoint,folder,'v',self.root/'missing',resolutions=(8,))
        codepath.write_bytes(contents+b'tampered')
        with self.assertRaisesRegex(ValueError,'field/code SHA'):
            remesh(checkpoint,folder,'v',self.root/'tampered',resolutions=(8,))

    def test_checkpoint_grid_and_original_presence_are_bound(self):
        folder,checkpoint,_ = self.fixture()
        with self.assertRaisesRegex(ValueError,'first grid'):
            remesh(checkpoint,folder,'v',self.root/'grid',resolutions=(12,))
        saved = torch.load(checkpoint,weights_only=True)
        saved['case_rows'][1]['presence']['11']=False
        torch.save(saved,checkpoint)
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            remesh(checkpoint,folder,'v',self.root/'swapped',resolutions=(8,))
        p = json.loads((folder/'provenance.json').read_text())
        p['checkpoint_sha256']=sha(checkpoint)
        (folder/'provenance.json').write_text(json.dumps(p))
        with self.assertRaisesRegex(ValueError,'presence mismatch'):
            remesh(checkpoint,folder,'v',self.root/'presence',resolutions=(8,))

    def test_failed_parity_preserves_codes_and_partial_and_stops(self):
        folder,checkpoint,evaluation = self.fixture()
        evaluation['cases'][0]['geometry']['per_tooth']['11']['observed_to_mesh_sample_p95_mm'] += 1
        (folder/'evaluation.json').write_text(json.dumps(evaluation))
        with patch('networks.dmm_net.DMM',side_effect=lambda *a,**kw:fixtures.Sphere()):
            with self.assertRaisesRegex(ValueError,'grid failed'):
                remesh(checkpoint,folder,'v',self.root/'failed',resolutions=(8,12))
        partial = json.loads((self.root/'failed/study_partial.json').read_text())
        self.assertIn('parity failed',partial['grids'][0]['error'])
        self.assertTrue((self.root/'failed/grid_8/codes.npz').exists())
        self.assertFalse((self.root/'failed/grid_12').exists())

    def test_fixed_gum_protocol_requires_exact_export_mean(self):
        folder,checkpoint,evaluation = self.fixture()
        protocol = dict(schema='dmm_code_fit_protocol_v1',arch='upper',fit_steps=20,gum_code_mode='fixed_train_mean')
        p = json.loads((folder/'provenance.json').read_text())
        p['fit_protocol']=protocol
        evaluation['fit_protocol']=protocol
        evaluation['budget']['gum_code_mode']='fixed_train_mean'
        evaluation['cases'][0]['code_mode']='temporary_teeth_fixed_train_mean_gum'
        (folder/'evaluation.json').write_text(json.dumps(evaluation))
        (folder/'provenance.json').write_text(json.dumps(p))
        with patch('networks.dmm_net.DMM',side_effect=lambda *a,**kw:fixtures.Sphere()):
            self.assertEqual(remesh(checkpoint,folder,'v',self.root/'fixed',resolutions=(8,))['failed_grid_count'],0)
        path = folder/'case_0000/codes.npz'
        np.savez_compressed(path,model_unit_mm=50,code_0=np.ones(2,dtype=np.float32),code_11=np.array([.1,-.1],dtype=np.float32))
        field = json.loads((folder/'case_0000/field.json').read_text())
        field['codes_sha256']=sha(path)
        (folder/'case_0000/field.json').write_text(json.dumps(field))
        with self.assertRaisesRegex(ValueError,'exported training mean'):
            remesh(checkpoint,folder,'v',self.root/'wrong_gum',resolutions=(8,))


if __name__=='__main__': unittest.main()
