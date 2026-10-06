"""Opt-in workflow and collision contracts; synthetic fixtures only."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rendering_fixtures import fixture_scene
from test_dual_arch import training_fixture
from test_fitting import observed_scene, FIT, SURFACE, MATCH, GPU
from dmm.collision import CollisionConfig, collision_loss
from dmm.recovery import RecoveryConfig
from dmm.fitting import fit_scene
from dmm.validation import ContractError, read_json, write_json, reference, sha256
from training.arch_training import train_arch
from training.dual_training import make_dual_config, load_dual_config, train_dual


class DualWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.upper, _ = training_fixture(self.root, 'upper')
        self.lower, self.lower_manifest = training_fixture(self.root, 'lower')
        self.config = self.root / 'dual.json'
        make_dual_config(self.upper, self.lower, self.config)

    def tearDown(self):
        self.tmp.cleanup()

    def test_validate_both_before_output_and_reject_split_leak(self):
        _, _, report = load_dual_config(self.config)
        self.assertEqual(report['cases'], {'upper': 3, 'lower': 3})
        data = read_json(self.lower_manifest)
        data['cases'][0]['split'] = 'test'
        # Use a non-reference case to keep the canonical training-only contract.
        canonical = self.lower.parent / 'canonical.json'
        canon = read_json(canonical); canon['training_case_ids'] = ['case1', 'case2']
        write_json(canonical, canon)
        write_json(self.lower_manifest, data)
        child = read_json(self.lower)
        child['manifest'] = reference(self.lower_manifest, self.lower)
        child['canonical_reference'] = reference(canonical, self.lower)
        write_json(self.lower, child)
        config = read_json(self.config); config['arches']['lower'] = reference(self.lower, self.config)
        write_json(self.config, config)
        with self.assertRaisesRegex(ContractError, 'patient leaks'):
            train_dual(self.config, self.root / 'bad')
        self.assertFalse((self.root / 'bad').exists())

    def test_independent_tiny_jobs_resume_without_retraining_completed_arch(self):
        first = self.root / 'first'
        def interrupted(config, output, device, resume):
            if Path(config) == self.lower: raise RuntimeError('controlled interruption')
            return train_arch(config, output, device, resume)
        with patch('training.dual_training.train_arch', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'controlled interruption'):
                train_dual(self.config, first)
        self.assertEqual(read_json(first/'workflow.json')['status'], 'INTERRUPTED')
        frozen = sha256(first/'upper/final.pth')
        with patch('training.dual_training.train_arch', wraps=train_arch) as call:
            result = train_dual(self.config, self.root/'second', resume=first/'workflow.json')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(Path(call.call_args.args[0]), self.lower)
        self.assertEqual(set(result['completed']), {'upper', 'lower'})
        self.assertEqual(frozen, sha256(first/'upper/final.pth'))
        self.assertFalse((self.root/'second/upper').exists())
        upper = torch.load(first/'upper/final.pth', weights_only=True)
        lower = torch.load(self.root/'second/lower/final.pth', weights_only=True)
        self.assertEqual((upper['arch'], lower['arch']), ('upper', 'lower'))
        self.assertNotEqual(upper['training_signature'], lower['training_signature'])
        # Completed artifacts are immutable resume inputs.
        (first/'upper/report.json').write_text('{}')
        with self.assertRaisesRegex(ContractError, 'SHA256'):
            train_dual(self.config, self.root/'tampered', resume=first/'workflow.json')
        self.assertFalse((self.root/'tampered').exists())

    def test_unknown_or_completed_resume_rejected(self):
        with self.assertRaisesRegex(ContractError, 'unknown arch'):
            train_dual(self.config, self.root/'bad', arch_resumes={'other': 'x'})
        invalid = read_json(self.config)
        invalid['arches']['lower'] = invalid['arches']['upper']
        write_json(self.config, invalid)
        with self.assertRaisesRegex(ContractError, 'arch/config mismatch'):
            load_dual_config(self.config)


class CollisionTests(unittest.TestCase):
    def test_disabled_is_no_query_and_configuration_roundtrip(self):
        scene = fixture_scene()
        mesh, _ = scene.differentiable_mesh(SURFACE)
        with patch.object(scene.arches['upper'], 'query_model', side_effect=AssertionError('disabled queried field')):
            loss, report = collision_loss(scene, mesh)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(report, {'enabled': False})
        config = CollisionConfig(enabled=True)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'collision.json'; write_json(path, config.resolved())
            self.assertEqual(CollisionConfig.load(path), config)
        with self.assertRaises(ContractError): CollisionConfig(weight=-1)
        with self.assertRaises(ContractError): CollisionConfig(tolerance_mm=float('nan'))

    def test_separated_surfaces_have_no_attraction(self):
        scene = fixture_scene()
        with torch.no_grad(): scene.arches['lower'].base_pose[0, 3] += 100
        mesh, _ = scene.differentiable_mesh(SURFACE)
        loss, report = collision_loss(scene, mesh, CollisionConfig(enabled=True))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertFalse(report['unresolved'])
        for state in scene.arches.values(): self.assertEqual(float(state.pose_delta.grad.norm()), 0.)

    def test_bidirectional_pose_gradient_and_field_scale_correction(self):
        scene = fixture_scene()
        with torch.no_grad():
            scene.arches['lower'].base_pose.copy_(scene.arches['upper'].base_pose)
            scene.arches['lower'].base_pose[0, 3] += 10.
        config = CollisionConfig(enabled=True, samples_per_arch=128)
        mesh, charts = scene.differentiable_mesh(SURFACE)
        loss, report = collision_loss(scene, mesh, config)
        grad = torch.autograd.grad(loss, scene.arches['lower'].pose_delta)[0][3]
        self.assertLess(float(grad), 0.)
        self.assertTrue(all(row['penetrating'] > 0 for row in report['directions']))
        # Local pose derivative with fixed surface topology; target field norm is sqrt(3).
        def measure(value):
            with torch.no_grad(): scene.arches['lower'].pose_delta[3] = value
            mesh, _ = scene.differentiable_mesh(SURFACE, charts)
            return float(collision_loss(scene, mesh, config)[0].detach())
        fd = (measure(1e-4)-measure(-1e-4))/2e-4
        self.assertAlmostEqual(float(grad), fd, places=5)
        self.assertGreater(report['max_proxy_mm'], .2)
        self.assertLess(report['max_proxy_mm'], 20.)
        # Reversed ModuleDict insertion order must not swap mesh ownership.
        with torch.no_grad(): scene.arches['lower'].pose_delta.zero_()
        scene.arches = torch.nn.ModuleDict({'lower': scene.arches['lower'], 'upper': scene.arches['upper']})
        reversed_mesh, _ = scene.differentiable_mesh(SURFACE, charts)
        reverse_loss, _ = collision_loss(scene, reversed_mesh, config)
        torch.testing.assert_close(reverse_loss, loss)

    @unittest.skipUnless(GPU, 'requires real CUDA')
    def test_gpu_fit_opt_in_default_and_checkpoint_binding(self):
        config = replace(FIT, pose_iterations=1, shape_iterations=0, joint_iterations=0)
        with tempfile.TemporaryDirectory() as root:
            scene, renderer = observed_scene('cuda', offset=.2)
            enabled = fit_scene(scene, Path(root)/'enabled', config, SURFACE, renderer, MATCH,
                                collision_config=CollisionConfig(enabled=True, samples_per_arch=64))
            self.assertTrue(enabled['collision']['enabled'])
            self.assertIn('T_upper_from_lower', enabled)
            rows = [json.loads(r) for r in (Path(root)/'enabled/progress.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['after'] <= r['before'] for r in rows))
            scene, renderer = observed_scene('cuda', offset=.2)
            default = fit_scene(scene, Path(root)/'default', config, SURFACE, renderer, MATCH)
            self.assertEqual(default['collision'], {'enabled': False})
            overlap, renderer = observed_scene('cuda')
            with torch.no_grad():
                overlap.arches['lower'].base_pose.copy_(overlap.arches['upper'].base_pose)
                overlap.arches['lower'].base_pose[0, 3] += 10.
            overlap_result = fit_scene(overlap, Path(root)/'overlap',
                replace(config, recovery=RecoveryConfig(enabled=False)), SURFACE, renderer, MATCH,
                collision_config=CollisionConfig(enabled=True, samples_per_arch=64))
            overlap_rows = [json.loads(r) for r in (Path(root)/'overlap/progress.jsonl').read_text().splitlines()]
            self.assertGreater(overlap_rows[0]['collision_loss'], 0.)
            self.assertTrue(all(r['after'] <= r['before'] for r in overlap_rows))
            self.assertNotEqual(overlap_result['status'], 'IMAGE_FIT_CONVERGED')
            checkpoint = read_json(Path(root)/'enabled/checkpoint.json'); checkpoint['status'] = 'RUNNING'
            write_json(Path(root)/'unfinished.json', checkpoint)
            scene, renderer = observed_scene('cuda', offset=.2)
            with self.assertRaisesRegex(ContractError, 'config mismatch'):
                fit_scene(scene, Path(root)/'resume', config, SURFACE, renderer, MATCH, Path(root)/'unfinished.json')


if __name__ == '__main__':
    unittest.main()
