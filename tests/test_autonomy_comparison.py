"""Protect the controlled factor, protocol, partitions and failure denominator."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_autonomy_evaluations import compare
from summarize_autonomy_run import sha


class ControlledComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.baseline,self.candidate=self.root/'baseline',self.root/'candidate'
        self.baseline.mkdir(); self.candidate.mkdir()
        self.base_config,self.candidate_config=self.root/'base.json',self.root/'candidate.json'
        self.fit=dict(version='masked_mean_v1',reduction='all_points',weights={'latent':1e6})
        self.config=dict(arch='upper',seed=42,training_case_ids=['t1','t2'],manifest={'sha256':'manifest'},recipe=self.fit)
        self.changed=copy.deepcopy(self.config)
        self.changed['recipe'].update(version='effective_count_v1',reduction='effective')
        self.base_config.write_text(json.dumps(self.config))
        self.candidate_config.write_text(json.dumps(self.changed))
        self.result=dict(split='val',requested_case_count=2,successful_case_count=2,failed_case_count=0,
            failures=[],score_mm=2.,fit_recipe=self.fit,budget=dict(fit_steps=20,seed=8),cases=[
            dict(case_id=cid,patient_id='p'+cid,split='val',status='EVALUATED_NOT_QUALITY_ACCEPTED',
                fit_index_sha256='fit'+cid,evaluation_index_sha256='eval'+cid,geometry=dict(score_mm=2.,
                metric='equal_tooth_observed_to_blended_mesh_samples_mm',resolution=48,samples_per_component=1024,
                per_tooth={'11':dict(observed_to_mesh_sample_mean_mm=1.,observed_to_mesh_sample_p95_mm=2.),
                           '12':dict(observed_to_mesh_sample_mean_mm=3.,observed_to_mesh_sample_p95_mm=4.)})) for cid in ('a','b')])
        self.candidate_result=copy.deepcopy(self.result)
        for case in self.candidate_result['cases']:
            case['geometry']['score_mm']=1.5
            for tooth in case['geometry']['per_tooth'].values(): tooth['observed_to_mesh_sample_mean_mm']-=.5
        self.candidate_result['score_mm']=1.5
        self.provenance=dict(split='val',manifest_sha256='manifest',canonical_reference_id='canonical',
            evaluation_source_sha256='source',runtime={'device_type':'cuda'},training_epoch=15,
            fit_recipe_sha256='common_fit',fit_recipe=self.fit,checkpoint_sha256='baseline')
        self.candidate_provenance=copy.deepcopy(self.provenance)
        self.candidate_provenance['checkpoint_sha256']='candidate'
        self.save()

    def save(self):
        for folder,result,provenance,config,path in (
                (self.baseline,self.result,self.provenance,self.config,self.base_config),
                (self.candidate,self.candidate_result,self.candidate_provenance,self.changed,self.candidate_config)):
            path.write_text(json.dumps(config))
            provenance.update(config_sha256=sha(path),training_recipe=config['recipe'])
            (folder/'provenance.json').write_text(json.dumps(provenance))
            (folder/'evaluation.json').write_text(json.dumps(result))

    def run_comparison(self,factor='reduction'):
        return compare(self.baseline,self.candidate,self.base_config,self.candidate_config,factor)

    def tearDown(self): self.temp.cleanup()

    def test_matched_common_protocol_has_paired_case_and_tooth_deltas(self):
        result=self.run_comparison()
        self.assertEqual(result['delta_mm'],-.5)
        self.assertEqual(result['successful_pair_count'],2)
        self.assertEqual(len(result['teeth']),4)
        self.assertEqual(result['teeth'][0]['delta_observed_to_mesh_sample_mean_mm'],-.5)
        self.assertEqual(result['status'],'COMPARED_NOT_QUALITY_ACCEPTED')

    def test_epoch_fit_runtime_and_point_partition_mismatch_rejected(self):
        original=copy.deepcopy(self.candidate_provenance)
        for key,value in [('training_epoch',20),('fit_recipe_sha256','other'),('runtime',{'device_type':'cpu'})]:
            with self.subTest(key=key):
                self.candidate_provenance=copy.deepcopy(original)
                self.candidate_provenance[key]=value
                self.save()
                with self.assertRaisesRegex(ValueError,'protocol mismatch'): self.run_comparison()
        self.candidate_provenance=original
        self.candidate_result['cases'][0]['fit_index_sha256']='different'
        self.save()
        with self.assertRaisesRegex(ValueError,'point partition mismatch'): self.run_comparison()

    def test_extra_factor_and_unequal_budget_rejected(self):
        self.changed['seed']=43
        self.save()
        with self.assertRaisesRegex(ValueError,'more than'): self.run_comparison()
        self.changed['seed']=42
        self.candidate_result['budget']['fit_steps']=100
        self.save()
        with self.assertRaisesRegex(ValueError,'budget/fit recipe'): self.run_comparison()

    def test_failed_case_retains_full_denominator_and_null_delta(self):
        failed=self.candidate_result['cases'][1]
        failed.update(status='FAILED',error='no finite zero surface')
        failed.pop('geometry')
        self.candidate_result.update(successful_case_count=1,failed_case_count=1,failures=['b'],score_mm=None)
        self.save()
        result=self.run_comparison()
        self.assertEqual(result['requested_case_count'],2)
        self.assertEqual(result['successful_pair_count'],1)
        self.assertIsNone(result['delta_mm'])
        self.assertIsNone(result['cases'][1]['delta_mm'])
        self.assertEqual(result['candidate_failed_case_count'],1)

    def test_test_split_rejected_before_metric_read_and_latent_control_supported(self):
        self.candidate_provenance['split']='test'
        self.save()
        (self.candidate/'evaluation.json').unlink()
        with self.assertRaisesRegex(ValueError,'val only'): self.run_comparison()
        self.candidate_provenance['split']='val'
        self.changed=copy.deepcopy(self.config)
        self.changed['recipe']['weights']['latent']=1e5
        self.save()
        self.assertEqual(self.run_comparison('latent_weight')['delta_mm'],-.5)


if __name__=='__main__': unittest.main()
