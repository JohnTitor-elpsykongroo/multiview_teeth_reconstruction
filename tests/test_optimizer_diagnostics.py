"""Prove scalar logging preserves seeded training, including Adam and RNG."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from test_dual_arch import training_fixture
from training.arch_training import train_arch
from training.diagnostics import optimizer_summary
from dmm.validation import read_json


class OptimizerDiagnosticsTests(unittest.TestCase):
    def test_clipping_distribution_and_unclipped_denominator(self):
        rows = [dict(preclip_norm=n, clip_scale=s, group_preclip_norm={'a': n})
                for n, s in ((10., 1.), (20., .5), (40., .25))]
        result = optimizer_summary(rows)
        self.assertEqual(result['steps'], 3)
        self.assertEqual(result['preclip_norm']['median'], 20.)
        self.assertEqual(result['clip_scale']['min'], .25)
        self.assertAlmostEqual(result['clipped_step_fraction'], 2/3)

    def test_logging_does_not_change_training_state_or_rng(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config, _ = training_fixture(root, 'upper')
            train_arch(config, root/'logged')
            with patch('training.arch_training.optimizer_summary', return_value={}):
                train_arch(config, root/'control')
            logged = torch.load(root/'logged/final.pth', weights_only=True)
            control = torch.load(root/'control/final.pth', weights_only=True)
            for key, value in logged['system_state'].items():
                torch.testing.assert_close(value, control['system_state'][key], rtol=0, atol=0)
            torch.testing.assert_close(logged['rng_state'], control['rng_state'], rtol=0, atol=0)
            for key, values in logged['optimizer_state']['state'].items():
                for field, value in values.items():
                    torch.testing.assert_close(value, control['optimizer_state']['state'][key][field], rtol=0, atol=0)
            records = [__import__('json').loads(line) for line in (root/'logged/optimizer_steps.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), len(logged['training_case_ids']))
            self.assertEqual({r['case_id'] for r in records}, set(logged['training_case_ids']))
            progress = __import__('json').loads((root/'logged/progress.jsonl').read_text().splitlines()[0])
            self.assertGreater(progress['training_seconds'], 0)
            self.assertEqual(progress['optimizer_diagnostics']['steps'], len(records))
