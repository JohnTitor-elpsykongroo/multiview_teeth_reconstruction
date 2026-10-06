"""Reject statistics contamination and held-out probes before metric access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_autonomy_latents import latent_statistics, original_rows, fixed_train_probes, probe_binding, sha


class LatentDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(training_case_ids=['a','b','mirror'], specs={'labels':[0,11]},
            case_rows=[dict(case_id=case,embedding_row=i,split=split,augmentation_parent_id=parent,
                            presence={'11':present}) for i,(case,split,parent,present) in enumerate([
                ('a','train',None,True),('b','train',None,False),('mirror','train','a',True),
                ('untrained','train',None,True),('held','test',None,True)])],
            system_state={'latents.0.weight':torch.tensor([[1.,2.],[3.,4.],[100.,100.],[200.,200.],[float('nan'),0.]]),
                          'latents.11.weight':torch.tensor([[2.,4.],[300.,300.],[100.,100.],[200.,200.],[float('nan'),0.]])})

    def test_only_original_present_trained_rows_enter_statistics(self):
        stats,_ = latent_statistics(self.saved)
        self.assertEqual(stats['0']['case_ids'],['a','b'])
        self.assertEqual(stats['11']['case_ids'],['a'])
        self.assertEqual(stats['11']['mean_code_squared'],10.)
        self.assertEqual(stats['0']['mean'],[2.,3.])
        self.assertAlmostEqual(stats['0']['covariance_participation_ratio'],1.)
        self.assertEqual(stats['11']['covariance_participation_ratio'],0.)
        self.assertIsNone(stats['11']['covariance_condition_number'])

    def test_invalid_trained_subset_and_duplicate_rows_rejected(self):
        self.saved['training_case_ids'].append('held')
        with self.assertRaisesRegex(ValueError,'held-out'): original_rows(self.saved)
        self.saved['training_case_ids'].remove('held')
        self.saved['case_rows'][1]['embedding_row']=0
        with self.assertRaisesRegex(ValueError,'duplicate'): original_rows(self.saved)

    def test_nonfinite_original_present_code_rejected(self):
        self.saved['system_state']['latents.11.weight'][0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'): latent_statistics(self.saved)

    def test_held_out_probe_rejected_before_reading_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'provenance.json').write_text(json.dumps(dict(split='test')))
            # No evaluation.json exists: split check must happen first.
            with self.assertRaisesRegex(ValueError,'only learned-train'):
                fixed_train_probes(folder,['a'],16,original_rows(self.saved))

    def test_presence_and_original_trained_membership_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'case_0000').mkdir()
            (folder/'provenance.json').write_text(json.dumps(dict(split='train')))
            (folder/'evaluation.json').write_text(json.dumps(dict(split='train',cases=[dict(
                case_id='a',split='train',status='EVALUATED_NOT_QUALITY_ACCEPTED',artifact_directory='case_0000')])))
            path=folder/'case_0000'/'mesh.npz'
            np.savez(path,model_unit_mm=50,observed_points_model=np.zeros((4,3)),observed_labels=np.array([0,0,11,11]))
            _,probes,_=fixed_train_probes(folder,['a'],1,original_rows(self.saved))
            self.assertEqual(len(probes[0]['points']),2)
            with self.assertRaisesRegex(ValueError,'original trained'):
                fixed_train_probes(folder,['untrained'],1,original_rows(self.saved))
            np.savez(path,model_unit_mm=50,observed_points_model=np.zeros((4,3)),observed_labels=np.array([0,0,0,0]))
            with self.assertRaisesRegex(ValueError,'presence'):
                fixed_train_probes(folder,['a'],1,original_rows(self.saved))

    def test_older_provenance_requires_exact_config_and_frozen_references(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            manifest=folder/'manifest.json'
            manifest.write_text('{}')
            canonical=folder/'canonical.json'
            canonical.write_text(json.dumps(dict(reference_id='canonical')))
            config=folder/'config.json'
            config.write_text(json.dumps(dict(arch='upper',manifest=dict(path='manifest.json',sha256=sha(manifest)),
                canonical_reference=dict(path='canonical.json',sha256=sha(canonical)))))
            provenance=dict(config_sha256=sha(config))
            saved=dict(arch='upper',manifest_sha256=sha(manifest),canonical_reference_id='canonical')
            with self.assertRaisesRegex(ValueError,'exact --probe-config'): probe_binding(provenance,saved)
            hashes=probe_binding(provenance,saved,config)
            self.assertEqual(hashes[str(config)],sha(config))
            config.write_text('{}')
            with self.assertRaisesRegex(ValueError,'config hash mismatch'): probe_binding(provenance,saved,config)


if __name__=='__main__': unittest.main()
