"""Evaluation isolation and honest failure accounting, not anatomical quality."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_dual_arch import training_fixture
from data.arch_dataset import TrainingCase
from dmm.validation import ContractError, read_json, write_json, reference, sha256
from networks.dmm_net import DMM
from training.arch_training import ArchTrainingSystem, load_training_config, train_arch
from training.evaluation import evaluation_cases, evaluate_cases, evaluate_checkpoint_cases, evaluate_checkpoint_batch, budget_digest, load_fit_recipe, resolve_fit_protocol, fit_budget
from training.validation import validation_samples, mesh_metrics

torch.set_num_threads(1)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path, manifest = training_fixture(self.root, "upper")
        data = read_json(manifest)
        for index, split in ((3, "val"), (4, "test")):
            row = copy.deepcopy(data['cases'][0])
            row.update(case_id=f'case{index}', patient_id=f'patient{index}', split=split, embedding_row=index)
            data['cases'].append(row)
        write_json(manifest, data)
        config = read_json(self.path)
        config['manifest'] = reference(manifest, self.path)
        config['validation'] = dict(enabled=True, fit_steps=2, points_per_component=1,
            offsurface_points=1, evaluation_points_per_component=1, mesh_resolution=8, mesh_samples_per_component=8)
        write_json(self.path, config)
        self.config, self.data, centers, specs = load_training_config(self.path)
        self.system = ArchTrainingSystem(DMM(specs, arch='upper'), self.data, centers, self.config['recipe'])

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_and_training_subset_are_explicit(self):
        self.data.selected = self.data.selected[:1]
        self.assertEqual([c.metadata['case_id'] for c in evaluation_cases(self.data, 'train')], ['case0'])
        for split, wrong in (('train', 'case1'), ('val', 'case4'), ('test', 'case3')):
            with self.assertRaisesRegex(ContractError, 'original'):
                evaluation_cases(self.data, split, [wrong])
        case = self.data.cases[4]
        fit, held = validation_samples(case, 'upper', self.config['validation'], split='test')
        self.assertFalse(set(fit['surface_indices']) & set(held['surface_indices']))
        self.assertFalse(set(fit['offsurface_indices']) & set(held['offsurface_indices']))
        repeated, _ = validation_samples(case, 'upper', self.config['validation'], split='test')
        np.testing.assert_array_equal(fit['surface_indices'], repeated['surface_indices'])
        with self.assertRaisesRegex(ContractError, 'original val'):
            validation_samples(case, 'upper', self.config['validation'])
        mirror = TrainingCase(dict(case.metadata, augmentation_parent_id='case0'), case.presence, case.centers, case.sample_path)
        with self.assertRaisesRegex(ContractError, 'original test'):
            validation_samples(mirror, 'upper', self.config['validation'], split='test')

    def test_train_uses_learned_codes_without_fitting(self):
        before = {k: v.clone() for k,v in self.system.state_dict().items()}
        def metric(model, codes, sample, box, config, **kwargs):
            for label, code in codes.items():
                torch.testing.assert_close(code, self.system.latents[str(label)].weight[0], rtol=0, atol=0)
                self.assertFalse(code.requires_grad)
            return dict(score_mm=1.25)
        with patch('training.evaluation.torch.optim.Adam', side_effect=AssertionError('must not refit train')), \
             patch('training.evaluation.mesh_metrics', side_effect=metric):
            result = evaluate_cases(self.system, self.data, self.config['validation'], 'train', ['case0'])
        self.assertEqual(result['score_mm'], 1.25)
        self.assertEqual(result['cases'][0]['fit_steps'], 0)
        for k,v in self.system.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_heldout_freezes_system_and_keeps_all_failures(self):
        before = {k: v.clone() for k,v in self.system.state_dict().items()}
        flags = [p.requires_grad for p in self.system.parameters()]
        for split in ('val', 'test'):
            with patch('training.evaluation.mesh_metrics', side_effect=ContractError('controlled missing tooth')):
                result = evaluate_cases(self.system, self.data, self.config['validation'], split)
            self.assertEqual(result['failed_case_count'], 1)
            self.assertEqual(result['requested_case_count'], 1)
            self.assertIsNone(result['score_mm'])
            self.assertIn('components', result['cases'][0])
            self.assertIn('controlled missing tooth', result['cases'][0]['error'])
        for k,v in self.system.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertEqual(flags, [p.requires_grad for p in self.system.parameters()])
        self.assertTrue(self.system.training)

    def test_final_test_requires_exact_protocol_and_complete_original_cohort(self):
        with patch('training.validation.mesh_metrics', return_value=dict(score_mm=1.)):
            train_arch(self.path, self.root/'training')
        checkpoint = self.root/'training/final.pth'
        with self.assertRaisesRegex(ContractError, 'frozen final protocol'):
            evaluate_checkpoint_cases(self.path, checkpoint, self.root/'missing', split='test')
        protocol = dict(stage='R4_FINAL_FROZEN', checkpoint_sha256=sha256(checkpoint),
            manifest_sha256=sha256(self.data.manifest_path), evaluation_budget_sha256=budget_digest(self.config['validation']),
            case_ids=['case4'])
        protocol_path = self.root/'protocol.json'
        write_json(protocol_path, protocol)
        with patch('training.evaluation.mesh_metrics', return_value=dict(score_mm=2.)):
            result = evaluate_checkpoint_cases(self.path, checkpoint, self.root/'valid', split='test', test_protocol=protocol_path)
        self.assertEqual(result['score_mm'], 2.)
        with self.assertRaisesRegex(ContractError, 'all original test'):
            evaluate_checkpoint_cases(self.path, checkpoint, self.root/'subset', split='test', case_ids=['case4'], test_protocol=protocol_path)
        protocol['checkpoint_sha256'] = '0'*64
        write_json(protocol_path, protocol)
        with self.assertRaisesRegex(ContractError, 'protocol mismatch'):
            evaluate_checkpoint_cases(self.path, checkpoint, self.root/'wrong', split='test', test_protocol=protocol_path)

    def test_missing_zero_surface_preserves_field_diagnostics(self):
        class ConstantField(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(()))
            def query(self, points, codes):
                return dict(sdf=torch.ones(len(points), device=points.device))
        artifact = self.root/'failed_field'
        code = torch.tensor([.25, -.5], requires_grad=True)
        with self.assertRaisesRegex(ContractError, 'no finite zero surface'):
            mesh_metrics(ConstantField(), {0: code}, {}, [[-1]*3,[1]*3], self.config['validation'], artifact_dir=artifact)
        field = read_json(artifact/'field.json')
        self.assertEqual(field['minimum'], 1.)
        self.assertEqual(field['negative_fraction'], 0.)
        self.assertFalse((artifact/'mesh.npz').exists())
        self.assertEqual(field['codes_sha256'], sha256(artifact/'codes.npz'))
        with np.load(artifact/'codes.npz', allow_pickle=False) as codes:
            self.assertEqual(set(codes.files), {'model_unit_mm', 'code_0'})
            self.assertEqual(codes['model_unit_mm'].item(), 50)
            np.testing.assert_array_equal(codes['code_0'], code.detach().numpy())
        torch.testing.assert_close(code, torch.tensor([.25, -.5]), rtol=0, atol=0)

    def test_common_fit_recipe_is_used_and_original_recipe_and_parameters_restored(self):
        recipe = copy.deepcopy(self.system.recipe)
        recipe['weights']['surface'] *= 2
        original = self.system.recipe
        before = {k: v.clone() for k,v in self.system.state_dict().items()}
        loss = self.system.loss
        observed = []
        def inspect(sample, codes):
            observed.append(copy.deepcopy(self.system.recipe))
            return loss(sample, codes)
        with patch.object(self.system, 'loss', side_effect=inspect), \
             patch('training.evaluation.mesh_metrics', side_effect=ContractError('controlled geometry failure')):
            result = evaluate_cases(self.system, self.data, self.config['validation'], 'val', fit_recipe=recipe)
        self.assertEqual(len(observed), self.config['validation']['fit_steps'])
        self.assertTrue(all(value == recipe for value in observed))
        self.assertEqual(result['fit_recipe'], recipe)
        self.assertIs(self.system.recipe, original)
        for key,value in self.system.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ContractError, 'does not fit'):
            evaluate_cases(self.system, self.data, self.config['validation'], 'train', fit_recipe=recipe)

    def test_explicit_fit_recipe_is_fully_resolved_and_test_protocol_bound(self):
        recipe = copy.deepcopy(self.system.recipe)
        recipe['weights']['surface'] *= 2
        fit_path = self.root/'fit_recipe.json'
        write_json(fit_path, dict(schema='dmm_latent_fit_recipe_v1', arch='upper', recipe=recipe))
        self.assertEqual(load_fit_recipe(fit_path, 'upper'), recipe)
        with self.assertRaisesRegex(ContractError, 'arch mismatch'):
            load_fit_recipe(fit_path, 'lower')
        with patch('training.validation.mesh_metrics', return_value=dict(score_mm=1.)):
            train_arch(self.path, self.root/'training')
        checkpoint = self.root/'training/final.pth'
        protocol = dict(stage='R4_FINAL_FROZEN', checkpoint_sha256=sha256(checkpoint),
            manifest_sha256=sha256(self.data.manifest_path), evaluation_budget_sha256=budget_digest(self.config['validation']),
            case_ids=['case4'])
        protocol_path = self.root/'protocol.json'
        write_json(protocol_path, protocol)
        with self.assertRaisesRegex(ContractError, 'fit recipe file mismatch'):
            evaluate_checkpoint_cases(self.path, checkpoint, self.root/'missing_binding', split='test',
                test_protocol=protocol_path, fit_recipe_path=fit_path)
        protocol['fit_recipe_sha256'] = sha256(fit_path)
        protocol['fit_recipe_resolved_sha256'] = budget_digest(recipe)
        write_json(protocol_path, protocol)
        with patch('training.evaluation.mesh_metrics', return_value=dict(score_mm=2.)):
            result = evaluate_checkpoint_cases(self.path, checkpoint, self.root/'bound', split='test',
                test_protocol=protocol_path, fit_recipe_path=fit_path)
        self.assertEqual(result['fit_recipe'], recipe)
        del recipe['normal_epsilon']
        write_json(fit_path, dict(schema='dmm_latent_fit_recipe_v1', arch='upper', recipe=recipe))
        with self.assertRaisesRegex(ContractError, 'every field'):
            load_fit_recipe(fit_path, 'upper')

    def test_baseline_common_recipe_reproduces_default_fitting_exactly(self):
        with patch('training.evaluation.mesh_metrics', return_value=dict(score_mm=1.)):
            native = evaluate_cases(self.system, self.data, self.config['validation'], 'val')
            common = evaluate_cases(self.system, self.data, self.config['validation'], 'val',
                                    fit_recipe=copy.deepcopy(self.system.recipe))
        for result in (native, common):
            for case in result['cases']: case.pop('seconds')
        self.assertEqual(native, common)

    def test_explicit_native_fit_protocol_preserves_default_numerics(self):
        policy=dict(schema='dmm_code_fit_protocol_v1',arch='upper',fit_steps=2,gum_code_mode='temporary_fitted')
        with patch('training.evaluation.mesh_metrics',return_value=dict(score_mm=1.)):
            native=evaluate_cases(self.system,self.data,self.config['validation'],'val')
            explicit=evaluate_cases(self.system,self.data,self.config['validation'],'val',fit_protocol=policy)
        self.assertEqual(explicit['fit_protocol'],policy)
        self.assertEqual(explicit['budget'].pop('gum_code_mode'),'temporary_fitted')
        explicit['fit_protocol']=None
        for result in (native,explicit):
            for case in result['cases']:
                case.pop('seconds')
                case.pop('fitted_component_labels',None)
        self.assertEqual(native,explicit)

    def test_fixed_gum_uses_export_mean_and_only_teeth_fit(self):
        policy=dict(schema='dmm_code_fit_protocol_v1',arch='upper',fit_steps=3,gum_code_mode='fixed_train_mean')
        before={k:v.clone() for k,v in self.system.state_dict().items()}
        rows=[c.metadata['embedding_row'] for c in self.data.selected if c.metadata['augmentation_parent_id'] is None]
        expected=self.system.latents['0'].weight[rows].detach().numpy().astype(np.float64).mean(0).astype(np.float32)
        calls=[]
        def loss(sample,codes):
            np.testing.assert_array_equal(codes[0].numpy(),expected)
            self.assertFalse(codes[0].requires_grad)
            calls.append([k for k,c in codes.items() if c.requires_grad])
            return sum((c-0.01).square().sum() for c in codes.values()),{}
        def metric(model,codes,*args,**kwargs):
            np.testing.assert_array_equal(codes[0].numpy(),expected)
            self.assertTrue(all(c.requires_grad for k,c in codes.items() if k))
            return dict(score_mm=1.)
        with patch.object(self.system,'loss',side_effect=loss),patch('training.evaluation.mesh_metrics',side_effect=metric):
            result=evaluate_cases(self.system,self.data,self.config['validation'],'val',fit_protocol=policy)
        self.assertEqual(len(calls),3)
        self.assertTrue(all(0 not in labels for labels in calls))
        self.assertEqual(result['cases'][0]['code_mode'],'temporary_teeth_fixed_train_mean_gum')
        self.assertEqual(result['cases'][0]['fit_steps'],3)
        self.assertNotIn(0,result['cases'][0]['fitted_component_labels'])
        for k,v in self.system.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(self.system.training)
        self.assertTrue(all(p.requires_grad for p in self.system.parameters()))

    def test_fit_protocol_schema_and_train_early_rejection(self):
        policy=dict(schema='dmm_code_fit_protocol_v1',arch='upper',fit_steps=2,gum_code_mode='fixed_train_mean')
        for change in ({'fit_steps':True},{'fit_steps':0},{'arch':'lower'},{'gum_code_mode':'guessed'},{'extra':1}):
            with self.assertRaises(ContractError): resolve_fit_protocol(dict(policy,**change),'upper')
        with patch('training.arch_training.load_training_config') as loader:
            with self.assertRaisesRegex(ContractError,'does not fit'):
                evaluate_checkpoint_cases('missing.json','missing.pth',self.root/'rejected',split='train',fit_protocol_path='missing.json')
            loader.assert_not_called()
        with self.assertRaisesRegex(ContractError,'does not fit'):
            evaluate_cases(self.system,self.data,self.config['validation'],'train',fit_protocol=policy)

    def test_final_test_binds_fit_protocol_file_and_resolved_budget(self):
        checkpoint=self.batch_checkpoints()[0]
        policy=dict(schema='dmm_code_fit_protocol_v1',arch='upper',fit_steps=3,gum_code_mode='fixed_train_mean')
        fit_path=self.root/'code_fit.json';write_json(fit_path,policy)
        protocol=dict(stage='R4_FINAL_FROZEN',checkpoint_sha256=sha256(checkpoint),manifest_sha256=sha256(self.data.manifest_path),
                      evaluation_budget_sha256=budget_digest(fit_budget(self.config['validation'],policy)),case_ids=['case4'])
        protocol_path=self.root/'final.json';write_json(protocol_path,protocol)
        with self.assertRaisesRegex(ContractError,'code fit protocol file mismatch'):
            evaluate_checkpoint_cases(self.path,checkpoint,self.root/'unbound',split='test',test_protocol=protocol_path,fit_protocol_path=fit_path)
        protocol.update(fit_protocol_sha256=sha256(fit_path),fit_protocol_resolved_sha256=budget_digest(policy))
        write_json(protocol_path,protocol)
        with patch('training.evaluation.mesh_metrics',return_value=dict(score_mm=2.)):
            result=evaluate_checkpoint_cases(self.path,checkpoint,self.root/'bound',split='test',test_protocol=protocol_path,fit_protocol_path=fit_path)
        self.assertEqual(result['score_mm'],2.)
        self.assertEqual(result['budget']['fit_steps'],3)
        self.assertEqual(result['cases'][0]['code_mode'],'temporary_teeth_fixed_train_mean_gum')
        protocol['fit_protocol_resolved_sha256']='0'*64;write_json(protocol_path,protocol)
        with self.assertRaisesRegex(ContractError,'resolved code fit protocol mismatch'):
            evaluate_checkpoint_cases(self.path,checkpoint,self.root/'wrong',split='test',test_protocol=protocol_path,fit_protocol_path=fit_path)

    def batch_checkpoints(self):
        import shutil
        with patch('training.validation.mesh_metrics', return_value=dict(score_mm=1.)):
            train_arch(self.path,self.root/'training')
        first=self.root/'training/final.pth'
        second=self.root/'checkpoint_copy.pth'
        shutil.copyfile(first,second)
        return [first,second]

    def test_batch_reuses_validation_but_reconstructs_each_checkpoint_exactly(self):
        checkpoints=self.batch_checkpoints()
        hashes=[sha256(path) for path in checkpoints]
        fit_path=self.root/'fit.json'
        write_json(fit_path,dict(schema='dmm_latent_fit_recipe_v1',arch='upper',recipe=self.config['recipe']))
        with patch('training.evaluation.mesh_metrics', return_value=dict(score_mm=2.)):
            expected=evaluate_checkpoint_cases(self.path,checkpoints[0],self.root/'single',split='val',fit_recipe_path=fit_path)
            with patch('training.arch_training.load_training_config',wraps=load_training_config) as loader:
                batch=evaluate_checkpoint_batch(self.path,checkpoints,self.root/'batch',split='val',fit_recipe_path=fit_path)
                self.assertEqual(loader.call_count,1)
        self.assertEqual(batch['status'],'EVALUATION_BATCH_COMPLETED_NOT_QUALITY_ACCEPTED')
        for item in batch['completed']:
            actual=read_json(self.root/'batch'/item['directory']/'evaluation.json')
            for result in (actual,expected):
                for case in result['cases']: case.pop('seconds',None)
            self.assertEqual(actual,expected)
        self.assertEqual(hashes,[sha256(path) for path in checkpoints])

    def test_batch_geometry_failures_are_retained_for_each_checkpoint(self):
        checkpoints=self.batch_checkpoints()
        with patch('training.evaluation.mesh_metrics',side_effect=ContractError('controlled geometry failure')):
            batch=evaluate_checkpoint_batch(self.path,checkpoints,self.root/'batch',split='val')
        self.assertEqual(len(batch['completed']),2)
        self.assertTrue(all(item['failed_case_count']==1 and item['score_mm'] is None for item in batch['completed']))

    def test_batch_rejects_config_drift_and_preserves_first_result(self):
        from training import evaluation
        checkpoints=self.batch_checkpoints()
        original=evaluation._evaluate_checkpoint_cases_loaded
        def mutate_after_first(*args,**kwargs):
            result=original(*args,**kwargs)
            self.path.write_text(self.path.read_text()+'\n')
            return result
        with patch('training.evaluation.mesh_metrics',return_value=dict(score_mm=2.)), \
             patch('training.evaluation._evaluate_checkpoint_cases_loaded',side_effect=mutate_after_first):
            with self.assertRaisesRegex(ContractError,'config/source drift'):
                evaluate_checkpoint_batch(self.path,checkpoints,self.root/'batch',split='val')
        batch=read_json(self.root/'batch/batch.json')
        self.assertEqual(batch['status'],'EVALUATION_BATCH_PROCESS_FAILED')
        self.assertEqual(len(batch['completed']),1)
        self.assertTrue((self.root/'batch/checkpoint_0000/evaluation.json').is_file())

    def test_batch_excludes_test_before_loading_any_inputs(self):
        with patch('training.arch_training.load_training_config') as loader:
            with self.assertRaisesRegex(ContractError,'excludes test'):
                evaluate_checkpoint_batch(self.path,['missing.pth'],self.root/'batch',split='test')
            loader.assert_not_called()

    def test_batch_fit_protocol_is_independent_and_drift_checked(self):
        checkpoints=self.batch_checkpoints()
        policy=dict(schema='dmm_code_fit_protocol_v1',arch='upper',fit_steps=3,gum_code_mode='fixed_train_mean')
        fit_path=self.root/'fit_policy.json';write_json(fit_path,policy)
        with patch('training.evaluation.mesh_metrics',return_value=dict(score_mm=1.)):
            expected=evaluate_checkpoint_cases(self.path,checkpoints[0],self.root/'single_policy',split='val',fit_protocol_path=fit_path)
            batch=evaluate_checkpoint_batch(self.path,checkpoints,self.root/'batch_policy',split='val',fit_protocol_path=fit_path)
        for row in batch['completed']:
            actual=read_json(self.root/'batch_policy'/row['directory']/'evaluation.json')
            for result in (actual,expected):
                for case in result['cases']: case.pop('seconds',None)
            self.assertEqual(actual,expected)
        from training import evaluation
        original=evaluation._evaluate_checkpoint_cases_loaded
        def mutate(*args,**kwargs):
            result=original(*args,**kwargs);fit_path.write_text(fit_path.read_text()+'\n');return result
        with patch('training.evaluation.mesh_metrics',return_value=dict(score_mm=1.)),patch('training.evaluation._evaluate_checkpoint_cases_loaded',side_effect=mutate):
            with self.assertRaisesRegex(ContractError,'code fit protocol drift'):
                evaluate_checkpoint_batch(self.path,checkpoints,self.root/'drift_policy',split='val',fit_protocol_path=fit_path)
        saved=read_json(self.root/'drift_policy/batch.json')
        self.assertEqual(len(saved['completed']),1)
        self.assertEqual(saved['status'],'EVALUATION_BATCH_PROCESS_FAILED')


if __name__ == '__main__':
    unittest.main()
