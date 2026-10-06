"""SemanticXY tensor/optimization tests. No DMM training or legacy run inputs."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "DMM"))
from dmm import CHANNEL_FDI
from dmm.scene import Observation
from dmm.semanticxy import (PointMeasure, RenderedSemanticXY, SemanticXYConfig, SemanticXYMatcher,
                            normalize_xy, partial_transport, sample_measure, warp_loss_from_plan)
from dmm.validation import ContractError


def observation(entries=(), shape=(6, 10), view_id="front", ignored=()):
    labels = np.zeros(shape, dtype=np.uint8)
    for y, x, fdi in entries:
        labels[y, x] = fdi
    for y, x in ignored:
        labels[y, x] = 255
    return Observation(view_id, np.eye(3), np.eye(4), labels, labels != 255, {})


def rendered(entries=(), shape=(6, 10), grad=True):
    semantics = torch.zeros(*shape, 29, dtype=torch.float64)
    semantics[..., 0] = 1
    for y, x, fdi in entries:
        semantics[y, x, 0] = 0
        semantics[y, x, 1 + CHANNEL_FDI.index(fdi)] = 1
    semantics.requires_grad_(grad)
    y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
    xy = torch.stack((x, y), -1).double().requires_grad_(grad)
    return RenderedSemanticXY(semantics, xy)


def measure(xy, mass, fdi=11):
    xy = torch.tensor(xy, dtype=torch.float64)
    semantic = torch.zeros(len(xy), 28, dtype=torch.float64)
    semantic[:, CHANNEL_FDI.index(fdi)] = 1
    return PointMeasure(semantic, xy, torch.tensor(mass, dtype=torch.float64), torch.arange(len(xy)), len(xy))


class SemanticXYTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_config_file_is_complete_and_pins_defaults(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "semanticxy_static_v1.json"
        self.assertEqual(SemanticXYConfig.load(path), SemanticXYConfig())
        with self.assertRaises(ContractError):
            SemanticXYConfig(semantic_weight=float("nan"))
        with self.assertRaises(ContractError):
            SemanticXYConfig(samples_per_tooth=0)

    def test_xy_pixel_centers_and_uniform_resize(self):
        pixels = torch.tensor([[0., 0.], [9., 5.], [4.5, 2.5]], dtype=torch.float64, requires_grad=True)
        low = normalize_xy(pixels, 10, 6)
        high = normalize_xy((pixels + .5) * 2 - .5, 20, 12)
        torch.testing.assert_close(low, high)
        torch.testing.assert_close(low[-1], torch.zeros(2, dtype=torch.float64))
        low.sum().backward()
        torch.testing.assert_close(pixels.grad, torch.ones_like(pixels) / np.hypot(10, 6))
        obs = observation()
        torch.testing.assert_close(normalize_xy(rendered().xy_pixels, 10, 6).float(), torch.tensor(obs.normalized_xy()))

    def test_distant_same_tooth_has_geometric_gradient(self):
        pred = rendered([(2, 1, 11)])
        result = SemanticXYMatcher().match_view(pred, observation([(2, 8, 11)]))
        self.assertEqual(result.diagnostics["status"], "MATCHED")
        self.assertGreater(result.diagnostics["matched_mass"], .99 / 60)
        result.loss.backward()
        self.assertLess(float(pred.xy_pixels.grad[2, 1, 0]), 0)
        self.assertAlmostEqual(float(pred.xy_pixels.grad[2, 1, 1]), 0, places=10)

    def test_float32_renderer_contract_keeps_finite_gradients(self):
        pred = rendered([(2, 1, 31)])
        pred = RenderedSemanticXY(pred.semantics.detach().float().requires_grad_(), pred.xy_pixels.detach().float().requires_grad_())
        result = SemanticXYMatcher().match_view(pred, observation([(2, 8, 31)]))
        result.loss.backward()
        self.assertTrue(torch.isfinite(pred.semantics.grad).all())
        self.assertTrue(torch.isfinite(pred.xy_pixels.grad).all())
        self.assertLess(float(pred.xy_pixels.grad[2, 1, 0]), 0)

    def test_different_fdi_prefers_unmatched_even_at_same_position(self):
        result = SemanticXYMatcher().match_view(rendered([(2, 1, 31)]), observation([(2, 1, 11)]))
        self.assertEqual(result.diagnostics["status"], "LOW_MATCH_MASS")
        self.assertGreater(result.diagnostics["source_unmatched_fraction"], .999)
        self.assertGreater(float(result.loss.detach()), 0)

    def test_soft_semantics_are_not_hardened(self):
        pred = rendered([(2, 1, 11)], grad=False)
        pred.semantics[2, 1, 1] = .7
        pred.semantics[2, 1, 2] = .3
        result = SemanticXYMatcher().match_view(pred, observation([(2, 1, 11)]), mode="evaluate")
        torch.testing.assert_close(result.source.semantic[0, :2], torch.tensor([.7, .3], dtype=torch.float64))
        self.assertAlmostEqual(float(result.source.mass.sum()), 1 / 60)

    def test_sampling_conserves_area_and_retains_small_tooth(self):
        pred = rendered([(y, x, 11) for y in range(6) for x in range(9)] + [(0, 9, 31)], grad=False)
        pred.semantics[:, :9, 0] = .8
        pred.semantics[:, :9, 1] = .2
        valid = torch.ones(6, 10, dtype=torch.bool)
        config = SemanticXYConfig(samples_per_tooth=3)
        sampled = sample_measure(pred.semantics, pred.xy_pixels, valid, "front", "prediction", config)
        again = sample_measure(pred.semantics, pred.xy_pixels, valid, "front", "prediction", config)
        self.assertEqual(len(sampled.mass), 4)
        self.assertEqual(set(sampled.semantic.argmax(-1).tolist()), {0, 14})
        self.assertAlmostEqual(float(sampled.mass.sum()), (54 * .2 + 1) / 60)
        self.assertTrue(torch.equal(sampled.pixel_indices, again.pixel_indices))
        self.assertAlmostEqual(float(sampled.mass[sampled.semantic[:, 14] == 1].sum()), 1 / 60)

    def test_point_duplication_does_not_change_transport_mass_or_loss(self):
        a = measure([[.1, 0]], [.4])
        duplicated = measure([[.1, 0], [.1, 0]], [.2, .2])
        b = measure([[.2, 0]], [.3])
        config = SemanticXYConfig()
        original = partial_transport(a, b, config)
        repeated = partial_transport(duplicated, b, config)
        self.assertAlmostEqual(float(original.real.sum()), float(repeated.real.sum()), places=6)
        left = sum(warp_loss_from_plan(a, b, original, config))
        right = sum(warp_loss_from_plan(duplicated, b, repeated, config))
        self.assertAlmostEqual(float(left), float(right), places=6)

    def test_unequal_area_is_not_normalized_away(self):
        a = measure([[.1, 0]], [.4])
        b = measure([[.1, 0]], [.1])
        plan = partial_transport(a, b, SemanticXYConfig())
        self.assertGreater(float(plan.source_unmatched.sum()), .299)
        self.assertLessEqual(float(plan.real.sum()), .100001)
        self.assertLess(float(plan.target_unmatched.sum()), .001)

    def test_entropic_optimality_cycle_and_marginals(self):
        config = SemanticXYConfig()
        a = measure([[0., 0.]], [.4])
        b = measure([[1.3, 0.]], [.3])
        plan = partial_transport(a, b, config)
        m = float(plan.real.sum())
        source_unmatched = float(plan.source_unmatched.sum())
        target_unmatched = float(plan.target_unmatched.sum())
        self.assertAlmostEqual(m + source_unmatched, .4, places=6)
        self.assertAlmostEqual(m + target_unmatched, .3, places=6)
        log_cycle = np.log(m * (.7 + m) / (source_unmatched * target_unmatched))
        self.assertAlmostEqual(log_cycle, (2 * config.unmatched_cost - 1.3) / config.entropy_epsilon, places=4)

    def test_all_28_channels_at_default_sample_budget(self):
        rng = torch.Generator().manual_seed(123)
        n = 28 * 16
        semantic = torch.eye(28, dtype=torch.float64).repeat_interleave(16, dim=0)
        xy = torch.rand(n, 2, generator=rng, dtype=torch.float64) * .4 - .2
        mass = torch.rand(n, generator=rng, dtype=torch.float64) + .1
        mass = mass / mass.sum() * .4
        source = PointMeasure(semantic, xy, mass, torch.arange(n), n)
        target = PointMeasure(semantic, xy + .1, mass * .8, torch.arange(n), n)
        plan = partial_transport(source, target, SemanticXYConfig())
        torch.testing.assert_close(plan.real.sum(1) + plan.source_unmatched, mass, atol=3e-6, rtol=0)
        torch.testing.assert_close(plan.real.sum(0) + plan.target_unmatched, mass * .8, atol=3e-6, rtol=0)
        self.assertGreater(float(plan.source_unmatched.sum()), .079)
        wrong_label = semantic.argmax(-1)[:, None] != semantic.argmax(-1)[None, :]
        self.assertLess(float(plan.real[wrong_label].sum()), 1e-8)

    def test_empty_prediction_keeps_loss_and_blocks_visibility_convergence(self):
        pred = rendered()
        result = SemanticXYMatcher().match_view(pred, observation([(2, 5, 11)]))
        self.assertTrue(result.diagnostics["needs_visibility_recovery"])
        self.assertEqual(result.diagnostics["status"], "EMPTY_PREDICTION_REQUIRES_VISIBILITY_RECOVERY")
        self.assertAlmostEqual(float(result.unmatched_loss.detach()), .75 / 60)
        self.assertGreater(float(result.loss.detach()), 0)
        result.loss.backward()
        self.assertLess(float(pred.semantics.grad[2, 5, 1]), 0)

    def test_empty_target_penalizes_prediction_and_both_empty_is_zero(self):
        result = SemanticXYMatcher().match_view(rendered([(1, 1, 11)]), observation())
        self.assertEqual(result.diagnostics["status"], "EMPTY_TARGET_NEGATIVE_EVIDENCE")
        self.assertAlmostEqual(float(result.loss.detach()), 1.75 / 60)
        empty = SemanticXYMatcher().match_view(rendered(), observation())
        self.assertEqual(empty.diagnostics["status"], "BOTH_EMPTY_VALID_BACKGROUND")
        self.assertEqual(float(empty.loss.detach()), 0)
        empty.loss.backward()

    def test_ignore_pixels_have_no_loss_gradient_or_mass(self):
        obs = observation(ignored=[(1, 1)])
        pred = rendered([(1, 1, 31)])
        result = SemanticXYMatcher().match_view(pred, obs)
        self.assertEqual(float(result.loss.detach()), 0)
        self.assertEqual(result.diagnostics["source_area_fraction"], 0)
        result.loss.backward()
        self.assertTrue(torch.equal(pred.semantics.grad[1, 1], torch.zeros(29, dtype=torch.float64)))

    def test_constant_grid_rejected_for_fit_but_explicit_diagnostic_allowed(self):
        pred = rendered([(2, 2, 11)], grad=False)
        obs = observation([(2, 3, 11)])
        with self.assertRaisesRegex(ContractError, "projected-geometry"):
            SemanticXYMatcher().match_view(pred, obs)
        pred.xy_source = "fixed_grid_diagnostic"
        result = SemanticXYMatcher().match_view(pred, obs, mode="evaluate")
        self.assertEqual(result.diagnostics["mode"], "evaluate")

    def test_frozen_correspondence_surrogate_finite_difference(self):
        a = measure([[.1, .2], [.2, .25]], [.2, .3])
        b = measure([[.35, .3], [.4, .4]], [.3, .2])
        config = SemanticXYConfig()
        plan = partial_transport(a, b, config)
        points = a.xy.clone().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(lambda xy: sum(warp_loss_from_plan(replace(a, xy=xy), b, plan, config)), (points,)))

    def test_view_mean_includes_empty_negative_and_requires_all_views(self):
        matcher = SemanticXYMatcher()
        observations = [observation([(1, 6, 11)], view_id="a"), observation(view_id="b"), observation(view_id="c")]
        predictions = {"a": rendered([(1, 1, 11)]), "b": rendered([(1, 1, 31)]), "c": rendered()}
        result = matcher.match_scene(predictions, observations)
        expected = sum(r.loss for r in result["views"].values()) / 3
        torch.testing.assert_close(result["loss"], expected)
        self.assertEqual(len(result["views"]), 3)
        with self.assertRaisesRegex(ContractError, "ALL"):
            matcher.match_scene({"a": predictions["a"]}, observations)
        with self.assertRaises(ContractError):
            matcher.match_scene(predictions, [observations[0], observations[0]])

    def test_ot_nonconvergence_and_budget_fail_explicitly(self):
        a = measure([[.1, 0], [.9, 0]], [.2, .5])
        b = measure([[.2, 0], [.4, 0]], [.1, .2])
        with self.assertRaisesRegex(ContractError, "did not converge"):
            partial_transport(a, b, SemanticXYConfig(max_iterations=1, check_every=1, marginal_tolerance=1e-16))
        with self.assertRaisesRegex(ContractError, "pair budget"):
            partial_transport(a, b, SemanticXYConfig(max_pair_entries=4))

    def test_invalid_probabilities_unknown_fdi_and_all_ignore_fail(self):
        pred = rendered(grad=False)
        pred.semantics[0, 0, 0] = .9
        with self.assertRaises(ContractError):
            SemanticXYMatcher().match_view(pred, observation(), mode="evaluate")
        with self.assertRaises(ContractError):
            SemanticXYMatcher().match_view(rendered(), observation([(0, 0, 18)]))
        with self.assertRaisesRegex(ContractError, "no valid pixels"):
            SemanticXYMatcher().match_view(rendered(), observation(ignored=[(y, x) for y in range(6) for x in range(10)]))

    def test_correspondence_refresh_translates_projected_attribute(self):
        # A renderer-contract fixture, not a full rasterizer/geometry reconstruction.
        obs = observation([(2, 8, 11)])
        fixed = rendered([(2, 1, 11)], grad=False)
        shift = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        matcher = SemanticXYMatcher()
        optimizer = torch.optim.SGD([shift], lr=1)
        before = None
        for _ in range(15):
            optimizer.zero_grad()
            xy = fixed.xy_pixels + torch.stack((shift, shift * 0))
            result = matcher.match_view(RenderedSemanticXY(fixed.semantics, xy), obs)
            if before is None:
                before = float(result.warp_loss.detach())
            (result.loss * 60 * np.hypot(6, 10)).backward()
            optimizer.step()
        self.assertGreater(float(shift.detach()), 5)
        self.assertLess(float(result.warp_loss.detach()), before * .3)


if __name__ == "__main__":
    unittest.main()
