"""Reproducible untrained native-DMM bridge diagnostic; never runs training."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "third_party" / "DMM"))
from rendering_fixtures import fixture_scene
from dmm import CHANNEL_FDI
from dmm.rendering import RenderConfig
from dmm.render_io import write_render_artifacts
from dmm.surface import SurfaceConfig
from dmm.semanticxy import SemanticXYConfig, SemanticXYMatcher, warp_loss_from_plan
from dmm.validation import require, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / ("render_bridge_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    require(not output.exists(), "diagnostic output already exists")
    torch.set_num_threads(1)
    suite = unittest.TestSuite()
    for pattern in ("test_rendering.py", "test_semanticxy.py"):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern))
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not tests.wasSuccessful():
        output.mkdir(parents=True)
        (output / "tests.log").write_text(log.getvalue(), encoding="utf-8")
        raise RuntimeError(f"bridge/regression tests failed: {output}")
    scene = fixture_scene()
    surface = SurfaceConfig(resolution=18, chunk_size=2048, root_tolerance=1e-8)
    render = RenderConfig(backend="torch_reference")
    rendered, mesh, charts = scene.render_views(surface, render)
    fdi = np.array([0, *CHANNEL_FDI], dtype=np.uint8)
    for obs in scene.observations:
        labels = fdi[rendered[obs.view_id].semantics.detach().argmax(-1).numpy()]
        obs.labels = np.roll(labels, 2, axis=1)
    matcher = SemanticXYMatcher(SemanticXYConfig(samples_per_tooth=8))
    match = matcher.match_scene(rendered, scene.observations)
    baseline = match["views"]["front"]
    state = scene.arches["upper"]
    parameters = [("upper.q11[0]", state.q["11"], 0, 1e-4),
                  ("upper.pose_translation_x_mm", state.pose_delta, 3, 1e-4)]
    contexts = {k: v.context for k, v in rendered.items()}
    derivatives = {}
    for name, parameter, index, step in parameters:
        autodiff = float(torch.autograd.grad(baseline.warp_loss, parameter, retain_graph=True)[0][index])
        values = []
        original = float(parameter[index].detach())
        for sign in (1, -1):
            with torch.no_grad(): parameter[index] = original + sign * step
            perturbed, _, _ = scene.render_views(surface, render, charts, contexts)
            source = matcher.match_view(perturbed["front"], scene.observations[0]).source
            require(torch.equal(source.pixel_indices, baseline.source.pixel_indices), "FD sampling changed")
            values.append(float(warp_loss_from_plan(source, baseline.target, baseline.plan, matcher.config)[0].detach()))
        with torch.no_grad(): parameter[index] = original
        fd = (values[0] - values[1]) / (2 * step)
        error = abs(fd - autodiff)
        require(error <= 1e-7 + 2e-4 * abs(fd) and abs(autodiff) > 1e-8, f"gradient check failed: {name}")
        derivatives[name] = dict(autodiff=autodiff, central_difference=fd, absolute_error=error, step=step)
    # Rebuild after in-place finite-difference probes before the full backward.
    rendered, mesh, _ = scene.render_views(surface, render, charts)
    match = matcher.match_scene(rendered, scene.observations)
    match["loss"].backward()
    gradients = {name: None if param.grad is None else float(param.grad.norm())
                 for name, param in scene.named_parameters() if param.requires_grad}
    require(all(value is not None and np.isfinite(value) for value in gradients.values()), "missing/nonfinite image gradient")
    matching = dict(loss=float(match["loss"].detach()), views={k: v.diagnostics for k, v in match["views"].items()},
                    resolved_config=match["resolved_config"], needs_visibility_recovery=match["needs_visibility_recovery"])
    write_render_artifacts(output, scene, rendered, mesh, matching, gradients)
    (output / "tests.log").write_text(log.getvalue(), encoding="utf-8")
    report = dict(status="CPU_REFERENCE_BRIDGE_VALIDATED_GPU_ANTIALIAS_PENDING", formal_training_started=False,
                  fixture="deterministic untrained original DMM: octahedral fields and two independent arches",
                  torch_version=torch.__version__, device="cpu", tests_run=tests.testsRun,
                  tests_passed=tests.testsRun - len(tests.skipped), skipped=[dict(test=str(test), reason=reason) for test, reason in tests.skipped],
                  finite_differences=derivatives, frozen_for_fd=["surface topology, anchors and normal chart", "raster face IDs and XY material weights", "OT transport and sample indices"],
                  limitations=["CPU hard coverage has no silhouette or occlusion-boundary derivative", "CUDA nvdiffrast forward/backward not executed", "no photo reconstruction or trained-model quality validation"],
                  evidence_sha256={p.name: sha256(p) for p in (output / "report.json", output / "tests.log", output / "preview.png")})
    write_json(output / "validation.json", report)
    print(json.dumps(dict(output=str(output.resolve()), **report), ensure_ascii=False))


if __name__ == "__main__":
    main()
