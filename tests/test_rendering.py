import json
import importlib.util
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import numpy as np
import torch

from rendering_fixtures import fixture_scene, observation
from dmm.rendering import RenderConfig, clip_coordinates, render_mesh
from dmm.surface import DifferentiableMesh, SurfaceConfig, attach_chart
from dmm.semanticxy import SemanticXYMatcher, SemanticXYConfig, warp_loss_from_plan
from dmm.validation import ContractError

torch.set_num_threads(1)
REFERENCE = RenderConfig(backend="torch_reference")
SURFACE = SurfaceConfig(resolution=18, chunk_size=2048, root_tolerance=1e-8)


def triangle(z=80., background=False):
    vertices = torch.tensor([[-10., -10., z], [10., -10., z], [0., 10., z]], dtype=torch.float64, requires_grad=True)
    semantic = torch.zeros(3, 29, dtype=vertices.dtype)
    semantic[:, 0 if background else 1] = 1
    return DifferentiableMesh(vertices, torch.tensor([[0, 1, 2]]), semantic, torch.tensor([0]), {})


class RenderingTests(unittest.TestCase):
    def test_configs_roundtrip_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            for cls, config in ((RenderConfig, REFERENCE), (SurfaceConfig, SURFACE)):
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(config.resolved()))
                self.assertEqual(cls.load(path), config)
        with self.assertRaises(ContractError):
            RenderConfig(near_mm=10., far_mm=1.)

    def test_native_cli_artifact_export_and_backward(self):
        import contextlib
        import io
        from dmm.__main__ import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, config in (("surface", SURFACE), ("render", REFERENCE), ("matching", SemanticXYConfig())):
                (root / f"{name}.json").write_text(json.dumps(config.resolved()))
            arguments = ["dmm_cli.py", "render-scene", "fixture.json", "--device", "cpu", "--output", str(root / "result"),
                         "--surface-config", str(root / "surface.json"), "--render-config", str(root / "render.json"),
                         "--matching-config", str(root / "matching.json"), "--backward-check"]
            # Bundle loading is tested separately; this test exercises the real renderer/CLI/artifact path.
            with patch("sys.argv", arguments), patch("dmm.scene.load_scene", return_value=fixture_scene()), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads((root / "result" / "report.json").read_text())
            self.assertEqual(set(report["views"]), {"front", "side"})
            self.assertEqual(len(report["gradients"]), 4)
            self.assertTrue((root / "result" / "preview.png").exists())

    def test_projection_integer_centers_and_xy_derivative(self):
        mesh, obs = triangle(), observation()
        result = render_mesh(mesh, obs, REFERENCE)
        y, x = torch.meshgrid(torch.arange(40), torch.arange(40), indexing="ij")
        hit = result.coverage.bool()
        torch.testing.assert_close(result.xy_pixels[hit], torch.stack((x, y), -1).double()[hit])
        gradient = torch.autograd.grad(result.xy_pixels[20, 20, 0], mesh.vertices_world_mm)[0]
        self.assertAlmostEqual(float(gradient[:, 0].sum()), 65 / 80, places=10)
        def shifted(amount):
            moved = DifferentiableMesh(mesh.vertices_world_mm + mesh.vertices_world_mm.new_tensor([amount, 0, 0]), mesh.faces, mesh.semantics, mesh.face_arch, {})
            return render_mesh(moved, obs, REFERENCE, result.context).xy_pixels[20, 20, 0]
        fd = (shifted(1e-4) - shifted(-1e-4)) / 2e-4
        self.assertAlmostEqual(float(fd.detach()), float(gradient[:, 0].sum()), places=8)

    def test_gum_occludes_other_arch(self):
        front, back = triangle(70., True), triangle(90.)
        mesh = DifferentiableMesh(torch.cat((front.vertices_world_mm, back.vertices_world_mm)),
                                  torch.cat((front.faces, back.faces + 3)), torch.cat((front.semantics, back.semantics)), torch.tensor([0, 1]), {})
        result = render_mesh(mesh, observation(), REFERENCE)
        self.assertEqual(float(result.semantics[20, 20, 0].detach()), 1.)
        self.assertAlmostEqual(float(result.depth_mm[20, 20].detach()), 70.)
        self.assertEqual(int(result.face_index[20, 20]), 0)

    def test_perspective_depth(self):
        mesh = triangle()
        with torch.no_grad():
            mesh.vertices_world_mm[:, 2] = torch.tensor([70., 90., 100.])
        result = render_mesh(mesh, observation(), REFERENCE)
        hit = result.coverage.bool()
        torch.testing.assert_close(result.context.material_weights[hit].sum(-1), torch.ones_like(result.depth_mm[hit]))
        expected = (mesh.vertices_world_mm[:, 2] * result.context.material_weights[20, 20]).sum()
        torch.testing.assert_close(result.depth_mm[20, 20], expected)

    def test_clip_orientation_and_half_pixel(self):
        obs = observation()
        camera = torch.tensor([[0., 0., 80.], [0., 1., 80.]], dtype=torch.float64)
        clip = clip_coordinates(camera, camera.new_tensor(obs.K), 40, 40, REFERENCE)
        torch.testing.assert_close(clip[0, :2], torch.zeros(2, dtype=camera.dtype))
        self.assertLess(float(clip[1, 1]), 0.)

    def test_context_binding_and_near_crossing_fail(self):
        mesh = triangle()
        result = render_mesh(mesh, observation(), REFERENCE)
        with self.assertRaisesRegex(ContractError, "mismatch"):
            render_mesh(mesh, observation(offset=1.), REFERENCE, result.context)
        with torch.no_grad():
            mesh.vertices_world_mm[0, 2] = .01
        with self.assertRaisesRegex(ContractError, "clip"):
            render_mesh(mesh, observation(), REFERENCE)

    def test_no_silent_gpu_fallback(self):
        with self.assertRaisesRegex(ContractError, "CUDA"):
            render_mesh(triangle(), observation(), RenderConfig())

    def test_actual_dmm_surface_and_latent_ift(self):
        scene = fixture_scene()
        mesh, charts = scene.differentiable_mesh(SURFACE)
        self.assertEqual(set(charts), {"upper", "lower"})
        self.assertEqual(set(mesh.face_arch.tolist()), {0, 1})
        self.assertTrue(mesh.vertices_world_mm.requires_grad)
        state, chart = scene.arches["upper"], charts["upper"]
        q = state.q["11"]
        direction = chart.normals.to(mesh.vertices_world_mm)
        value = (mesh.vertices_world_mm[:len(direction)] * direction).sum()
        analytical = torch.autograd.grad(value, q)[0][0]
        values = []
        for delta in (1e-4, -1e-4):
            with torch.no_grad(): q[0] = delta
            world, _, diag = attach_chart(state, chart, SURFACE)
            self.assertLessEqual(diag["max_sdf_residual"], SURFACE.root_tolerance)
            values.append((world * direction).sum().detach())
        with torch.no_grad(): q[0] = 0
        numeric = (values[0] - values[1]) / 2e-4
        torch.testing.assert_close(analytical, numeric, rtol=1e-5, atol=1e-6)
        self.assertGreater(float(analytical.abs()), 1.)
        self.assertTrue(all(p.grad is None for p in state.decoder.parameters()))

    def test_chart_trust_region_and_missing_domain_surface(self):
        scene = fixture_scene()
        _, charts = scene.differentiable_mesh(SURFACE)
        with torch.no_grad(): scene.arches["upper"].q["11"][0] = 5.
        with self.assertRaisesRegex(ContractError, "trust region"):
            attach_chart(scene.arches["upper"], charts["upper"], SURFACE)

    def test_actual_dmm_render_semanticxy_backward_and_fd(self):
        scene = fixture_scene()
        renders, _, charts = scene.render_views(SURFACE, REFERENCE)
        self.assertEqual(set(renders), {v.view_id for v in scene.observations})
        obs, initial = scene.observations[0], renders["front"]
        # Target is a known displaced mask for this isolated interface test.
        labels = np.array([0, *(__import__("dmm").CHANNEL_FDI)], dtype=np.uint8)[initial.semantics.detach().argmax(-1).numpy()]
        obs.labels = np.roll(labels, 2, axis=1)
        matcher = SemanticXYMatcher(SemanticXYConfig(samples_per_tooth=8))
        baseline = matcher.match_view(initial, obs)
        state, q = scene.arches["upper"], scene.arches["upper"].q["11"]
        analytical = torch.autograd.grad(baseline.warp_loss, (q, state.pose_delta), retain_graph=True)
        self.assertTrue(torch.isfinite(analytical[0]).all() and torch.isfinite(analytical[1]).all())
        self.assertGreater(float(analytical[0][0].abs()), 1e-8)
        context = {key: value.context for key, value in renders.items()}
        values = []
        for delta in (1e-4, -1e-4):
            with torch.no_grad(): q[0] = delta
            changed, _, _ = scene.render_views(SURFACE, REFERENCE, charts, context)
            sampled = matcher.match_view(changed["front"], obs)
            self.assertTrue(torch.equal(sampled.source.pixel_indices, baseline.source.pixel_indices))
            values.append(warp_loss_from_plan(sampled.source, baseline.target, baseline.plan, matcher.config)[0].detach())
        with torch.no_grad(): q[0] = 0
        numeric = (values[0] - values[1]) / 2e-4
        torch.testing.assert_close(analytical[0][0], numeric, atol=1e-7, rtol=2e-4)
        baseline.loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for arch in scene.arches.values() for p in arch.q.values()))

    def test_soft_semantic_latent_gradient(self):
        scene = fixture_scene()
        rendered, _, charts = scene.render_views(SURFACE, REFERENCE)
        q = scene.arches["upper"].q["11"]
        value = rendered["front"].semantics[..., 1].sum()
        gradient = torch.autograd.grad(value, q)[0][1]
        context = {key: value.context for key, value in rendered.items()}
        values = []
        for delta in (1e-4, -1e-4):
            with torch.no_grad(): q[1] = delta
            changed, _, _ = scene.render_views(SURFACE, REFERENCE, charts, context)
            values.append(changed["front"].semantics[..., 1].sum().detach())
        numeric = (values[0] - values[1]) / 2e-4
        self.assertGreater(float(gradient.abs()), 1e-7)
        torch.testing.assert_close(gradient, numeric, atol=1e-8, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("nvdiffrast") is not None,
                         "CUDA nvdiffrast runtime unavailable; GPU coverage gradients NOT validated")
    def test_cuda_orientation_occlusion_and_antialias_backward(self):
        mesh = triangle()
        cpu = render_mesh(mesh, observation(), REFERENCE)
        gpu_mesh = DifferentiableMesh(mesh.vertices_world_mm.detach().float().cuda().requires_grad_(),
                                      mesh.faces.cuda(), mesh.semantics.float().cuda(), mesh.face_arch.cuda(), {})
        gpu = render_mesh(gpu_mesh, observation(), RenderConfig())
        interior = cpu.coverage.bool() & (cpu.context.material_weights.min(-1).values > .1)
        torch.testing.assert_close(gpu.xy_pixels.cpu()[interior].double(), cpu.xy_pixels[interior], atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(gpu.semantics.cpu()[interior].double(), cpu.semantics[interior])
        self.assertTrue(bool(((gpu.coverage > 0) & (gpu.coverage < 1)).any()))
        # Area derivative is unavailable in the hard-coverage reference backend.
        grad = torch.autograd.grad(gpu.semantics[..., 1].sum(), gpu_mesh.vertices_world_mm)[0]
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad.norm()), 1e-5)

    def test_empty_prediction_requests_recovery(self):
        mesh, obs = triangle(), observation()
        with torch.no_grad(): mesh.vertices_world_mm[:, 0] += 1000
        obs.labels[15:20, 15:20] = 11
        rendered = render_mesh(mesh, obs, REFERENCE)
        result = SemanticXYMatcher().match_view(rendered, obs)
        self.assertGreater(float(result.loss.detach()), 0)
        self.assertTrue(result.diagnostics["needs_visibility_recovery"])


if __name__ == "__main__":
    unittest.main()
