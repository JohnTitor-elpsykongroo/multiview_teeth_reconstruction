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
from training.evaluation import evaluation_cases, evaluate_cases, evaluate_checkpoint_cases, budget_digest
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
        with self.assertRaisesRegex(ContractError, 'no finite zero surface'):
            mesh_metrics(ConstantField(), {}, {}, [[-1]*3,[1]*3], self.config['validation'], artifact_dir=artifact)
        field = read_json(artifact/'field.json')
        self.assertEqual(field['minimum'], 1.)
        self.assertEqual(field['negative_fraction'], 0.)
        self.assertFalse((artifact/'mesh.npz').exists())


if __name__ == '__main__':
    unittest.main()
