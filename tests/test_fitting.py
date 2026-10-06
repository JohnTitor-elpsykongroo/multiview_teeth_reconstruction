from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rendering_fixtures import fixture_scene
from dmm import CHANNEL_FDI
from dmm.fitting import FitConfig, fit_scene, parameter_state
from dmm.recovery import RecoveryConfig, missing_regions, recovery_loss, bootstrap_translation
from dmm.rendering import RenderConfig
from dmm.surface import SurfaceConfig
from dmm.semanticxy import SemanticXYConfig
from dmm.validation import ContractError

torch.set_num_threads(1)
SURFACE = SurfaceConfig(resolution=14, chunk_size=2048)
MATCH = SemanticXYConfig(samples_per_tooth=4, warmup_iterations=10, epsilon_scaling_steps=2,
                         epsilon_start=.1, max_iterations=500)
FIT = FitConfig(pose_iterations=3, shape_iterations=3, joint_iterations=6, backtracks=5,
                rotation_step_rad=.001, translation_step_mm=.7, latent_step=.06, patience=2)
GPU = torch.cuda.is_available() and importlib.util.find_spec('nvdiffrast') is not None


def observed_scene(device='cpu', offset=0.):
    scene = fixture_scene(torch.float32 if device == 'cuda' else torch.float64).to(device)
    config = RenderConfig(supersample=4) if device == 'cuda' else RenderConfig(backend='torch_reference')
    with torch.no_grad(): images, _, _ = scene.render_views(SURFACE, config)
    fdi = np.array([0, *CHANNEL_FDI], np.uint8)
    for obs in scene.observations:
        obs.labels = fdi[images[obs.view_id].semantics.argmax(-1).cpu().numpy()]
    with torch.no_grad():
        for state in scene.arches.values(): state.base_pose[0, 3] += offset
    return scene, config


class RecoveryTests(unittest.TestCase):
    def test_reference_resume_matches_continuous_parameters(self):
        config = replace(FIT, pose_iterations=1, shape_iterations=1, joint_iterations=2, diagnostic_reference=True)
        scene, renderer = observed_scene(offset=2.)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = scene.render_views
            def interrupt(*args, **kwargs):
                checkpoint = root/'interrupted/checkpoint.json'
                if checkpoint.exists() and len(json.loads(checkpoint.read_text())['history']) >= 1:
                    raise RuntimeError('simulated interruption')
                return original(*args, **kwargs)
            with patch.object(scene, 'render_views', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'simulated'):
                fit_scene(scene, root/'interrupted', config, SURFACE, renderer, MATCH)
            fresh, _ = observed_scene(offset=2.)
            resumed = fit_scene(fresh, root/'resumed', config, SURFACE, renderer, MATCH, root/'interrupted/checkpoint.json')
            control, _ = observed_scene(offset=2.)
            completed = fit_scene(control, root/'control', config, SURFACE, renderer, MATCH)
            for arch in fresh.arches:
                np.testing.assert_allclose(resumed['parameters'][arch]['T_world_from_arch'], completed['parameters'][arch]['T_world_from_arch'], atol=1e-10, rtol=0)
                for label in resumed['parameters'][arch]['q']:
                    np.testing.assert_allclose(resumed['parameters'][arch]['q'][label], completed['parameters'][arch]['q'][label], atol=1e-10, rtol=0)

    def test_configuration_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fit.json'
            path.write_text(json.dumps(FIT.resolved()))
            self.assertEqual(FitConfig.load(path), FIT)
        with self.assertRaises(ContractError): FitConfig(pose_iterations=0, shape_iterations=0, joint_iterations=0)

    def test_per_tooth_detection_and_hidden_component_gradient(self):
        scene, renderer = observed_scene()
        # Hide just the lower arch by moving it outside the image. The upper remains.
        with torch.no_grad(): scene.arches['lower'].base_pose[0, 3] += 100
        images, _, _ = scene.render_views(SURFACE, renderer)
        missing = missing_regions(scene, images, RecoveryConfig())
        self.assertEqual({row['label'] for row in missing}, {31})
        loss, _ = recovery_loss(scene, images, missing, SURFACE, renderer, RecoveryConfig())
        grad = torch.autograd.grad(loss, scene.arches['lower'].pose_delta)[0]
        self.assertGreater(float(grad[3]), 0.)  # descent returns it toward the image
        # A bootstrap uses observed camera rays only and keeps the field frozen.
        pose, detail = bootstrap_translation(scene, 'lower', missing, SURFACE, RecoveryConfig())
        self.assertIsNotNone(pose)
        self.assertLess(detail['translation_mm'][0], -80)

    def test_underconstrained_recovery_reports_failure(self):
        scene, renderer = observed_scene(offset=100)
        scene.observations = scene.observations[:1]
        rendered, _, _ = scene.render_views(SURFACE, renderer)
        active = missing_regions(scene, rendered, RecoveryConfig())
        pose, report = bootstrap_translation(scene, 'upper', active, SURFACE, RecoveryConfig())
        self.assertIsNone(pose)
        self.assertEqual(report['status'], 'UNDERCONSTRAINED_RAYS')

    def test_actual_occluder_provides_depth_recovery_gradient(self):
        scene, renderer = observed_scene()
        with torch.no_grad():
            scene.arches['lower'].base_pose[:3, 3] = torch.tensor([-7*120/85, -6*120/85, 120.], dtype=torch.float64)
        # Controlled visible-target patch over the current upper occluder.
        for obs in scene.observations:
            obs.labels[:] = 0
            obs.labels[13:16, 13:16] = 31
        rendered, _, _ = scene.render_views(SURFACE, renderer)
        active = missing_regions(scene, rendered, RecoveryConfig())
        self.assertTrue(active)
        self.assertTrue(torch.isfinite(rendered['front'].depth_mm[14, 14]))
        loss, _ = recovery_loss(scene, rendered, active, SURFACE, renderer, RecoveryConfig())
        grad = torch.autograd.grad(loss, scene.arches['lower'].pose_delta)[0]
        self.assertGreater(float(grad[5]), 0.)  # Descent moves the tooth toward the camera.
        # The physical render still contains the upper occluder; no hidden render is substituted.
        self.assertEqual(int(rendered['front'].semantics[14, 14].argmax()), 1)

    def test_cpu_fitter_requires_explicit_diagnostic_mode(self):
        scene, renderer = observed_scene()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ContractError, 'CUDA'):
                fit_scene(scene, Path(root)/'fit', FIT, SURFACE, renderer, MATCH)

    def test_no_observation_and_absent_fdi_are_not_invented(self):
        scene, renderer = observed_scene()
        for obs in scene.observations: obs.labels[:] = 0
        rendered, _, _ = scene.render_views(SURFACE, renderer)
        self.assertEqual(missing_regions(scene, rendered, RecoveryConfig()), [])
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ContractError, 'no observed teeth'):
                fit_scene(scene, Path(root)/'fit', replace(FIT, diagnostic_reference=True), SURFACE, renderer, MATCH)


@unittest.skipUnless(GPU, 'native staged integration requires real CUDA nvdiffrast')
class GPUFittingTests(unittest.TestCase):
    def test_three_stages_use_all_views_and_frozen_decoders(self):
        scene, renderer = observed_scene('cuda', offset=2.)
        frozen = {name: value.detach().clone() for name, value in scene.named_parameters() if 'decoder' in name}
        with tempfile.TemporaryDirectory() as root:
            result = fit_scene(scene, Path(root)/'fit', FIT, SURFACE, renderer, MATCH)
            checkpoint = json.loads((Path(root)/'fit'/'checkpoint.json').read_text())
            self.assertEqual({row['stage'] for row in checkpoint['history']}, {'pose', 'shape', 'joint'})
            self.assertTrue(all(row['after'] <= row['before'] for row in checkpoint['history']))
            for row in checkpoint['history']:
                self.assertTrue(all(('pose_delta' in name) == (row['block'] == 'pose') for name in row['parameter_updates']))
            self.assertEqual(set(result['final_metrics']['per_tooth_view'][i]['view_id'] for i in range(len(result['final_metrics']['per_tooth_view']))), {'front', 'side'})
            self.assertGreater(result['final_metrics']['min_tooth_iou'], result['initial_metrics']['min_tooth_iou'])
            self.assertFalse(result['formal_training'])
            for name, value in scene.named_parameters():
                if name in frozen: torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)

    def test_out_of_frame_bootstrap_and_unresolved_state(self):
        scene, renderer = observed_scene('cuda', offset=100.)
        with tempfile.TemporaryDirectory() as root:
            result = fit_scene(scene, Path(root)/'recovered', replace(FIT, pose_iterations=2, shape_iterations=0, joint_iterations=0), SURFACE, renderer, MATCH)
            self.assertTrue(any(row['status'] == 'ACCEPTED_BY_FULL_RENDER' for row in result['bootstrap']))
            self.assertEqual(result['unresolved'], [])
            self.assertGreater(result['final_metrics']['min_tooth_iou'], .5)
            bad, _ = observed_scene('cuda', offset=100.)
            disabled = replace(FIT, pose_iterations=1, shape_iterations=0, joint_iterations=0, recovery=RecoveryConfig(enabled=False))
            failure = fit_scene(bad, Path(root)/'unresolved', disabled, SURFACE, renderer, MATCH)
            self.assertEqual(failure['status'], 'VISIBILITY_RECOVERY_UNRESOLVED')

    def test_resume_restores_exact_state_and_rejects_changed_input(self):
        config = replace(FIT, pose_iterations=1, shape_iterations=1, joint_iterations=2)
        scene, renderer = observed_scene('cuda', offset=2.)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            original = scene.render_views
            def interrupt(*args, **kwargs):
                if (root/'interrupted'/'checkpoint.json').exists():
                    saved = json.loads((root/'interrupted'/'checkpoint.json').read_text())
                    if len(saved['history']) >= 1: raise RuntimeError('simulated interruption')
                return original(*args, **kwargs)
            with patch.object(scene, 'render_views', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'simulated'):
                fit_scene(scene, root/'interrupted', config, SURFACE, renderer, MATCH)
            checkpoint = root/'interrupted'/'checkpoint.json'
            saved = json.loads(checkpoint.read_text())
            fresh, _ = observed_scene('cuda', offset=2.)
            resumed = fit_scene(fresh, root/'resumed', config, SURFACE, renderer, MATCH, checkpoint)
            # Float32 CUDA raster reductions and refreshed non-smooth MC surfaces
            # can diverge even between two uninterrupted controls. Test exact
            # persistence separately from trajectory replay (reference test above).
            starting = json.loads((root/'resumed/starting_parameters.json').read_text())
            self.assertEqual(starting, saved['parameters'])
            final = json.loads((root/'resumed/checkpoint.json').read_text())
            self.assertEqual(final['history'][:len(saved['history'])], saved['history'])
            self.assertTrue(all(r['after'] <= r['before'] for r in final['history']))
            self.assertTrue(np.isfinite(resumed['final_image_loss']))
            wrong, _ = observed_scene('cuda', offset=2.)
            wrong.observations[0].labels[0, 0] = 11
            with self.assertRaisesRegex(ContractError, 'mismatch'):
                fit_scene(wrong, root/'wrong', config, SURFACE, renderer, MATCH, checkpoint)


if __name__ == '__main__': unittest.main()
