import unittest
from unittest.mock import patch
import torch
from test_dual_arch import training_fixture
from networks.mlp import MLPNet
from training.recipe import resolve_recipe
from dmm.provenance import source_fingerprint


class InitializationTests(unittest.TestCase):
    def test_siren_output_has_nonzero_spatial_and_finite_second_derivatives(self):
        torch.manual_seed(13)
        model=MLPNet([32,32],activation='sine',output_initialization='siren')
        points=torch.randn(16,3,requires_grad=True)
        sdf=model(points)
        gradient=torch.autograd.grad(sdf.sum(),points,create_graph=True)[0]
        self.assertGreater(float(gradient.detach().norm()),1e-3)
        (gradient.square().sum()+sdf.square().sum()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        legacy=MLPNet([32,32],activation='sine')
        self.assertEqual(float(legacy(points).detach().abs().sum()),0.)

    def test_normal_denominator_bounds_zero_field_gradient(self):
        cfg=resolve_recipe(dict(learning_rate=.0001,recipe=dict(normal_epsilon=.001)))
        gradient=torch.zeros((3,3),requires_grad=True)
        loss=(1-torch.nn.functional.cosine_similarity(gradient,torch.eye(3),dim=-1,eps=cfg['normal_epsilon'])).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(gradient.grad).all())
        self.assertLessEqual(float(gradient.grad.norm()),1000.)

    def test_source_hash_is_independent_of_file_enumeration_order(self):
        from pathlib import Path
        original=Path.rglob
        expected=source_fingerprint()
        with patch.object(Path,'rglob',lambda p,pattern:iter(reversed(list(original(p,pattern))))):
            self.assertEqual(source_fingerprint(),expected)


if __name__=='__main__':unittest.main()
