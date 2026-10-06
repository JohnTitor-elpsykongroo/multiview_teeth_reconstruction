"""Required real-CUDA rendering gates, never replaced by skipped/CPU tests."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'third_party' / 'DMM'))
from rendering_fixtures import fixture_scene, observation
from test_rendering import triangle
from dmm.surface import DifferentiableMesh, SurfaceConfig
from dmm.rendering import RenderConfig, render_mesh
from dmm.render_io import write_render_artifacts
from dmm.semanticxy import SemanticXYMatcher, SemanticXYConfig
from dmm import CHANNEL_FDI
from dmm.validation import require, write_json
from dmm.gpu_validation import validate_cuda_backend


def main():
    require(torch.cuda.is_available(), 'CUDA is required for this gate')
    import nvdiffrast.torch  # fail instead of skip
    torch.set_num_threads(1)
    runtime = validate_cuda_backend(torch.device('cuda:0'))
    output = ROOT / 'runs' / ('gpu_render_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_rendering.py')
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (output / 'tests.log').write_text(log.getvalue(), encoding='utf-8')
    require(tests.wasSuccessful() and not tests.skipped, f'GPU test gate failed; see {output}')
    # Same depth buffer: the background-semantic gum must still hide the other arch.
    front, back = triangle(70., True), triangle(90.)
    occluder = DifferentiableMesh(torch.cat((front.vertices_world_mm, back.vertices_world_mm)).detach().float().cuda().requires_grad_(),
                                 torch.cat((front.faces, back.faces + 3)).cuda(), torch.cat((front.semantics, back.semantics)).float().cuda(),
                                 torch.tensor([0, 1], device='cuda'), {})
    occluded = render_mesh(occluder, observation(), RenderConfig())
    require(float(occluded.semantics[20, 20, 0].detach()) > .999 and abs(float(occluded.depth_mm[20, 20].detach()) - 70) < 1e-3, 'GPU gum occlusion failed')
    scene = fixture_scene(torch.float32).cuda()
    surface = SurfaceConfig(resolution=18, chunk_size=2048)
    rendered, mesh, charts = scene.render_views(surface, RenderConfig())
    parameters = [('q11', scene.arches['upper'].q['11'], 0, .002),
                  ('upper_translation_x', scene.arches['upper'].pose_delta, 3, .002)]
    # Interior material-point derivative: the same local map as the CPU gate.
    # Dense marching-cubes AA has discrete edge-selection changes (recorded below).
    def image_loss(images): return images['front'].xy_pixels[12, 12].sum()
    initial = image_loss(rendered)
    contexts = {key: value.context for key, value in rendered.items()}
    differences = {}
    for name, parameter, index, epsilon in parameters:
        analytic = float(torch.autograd.grad(initial, parameter, retain_graph=True)[0][index])
        values = []
        for sign in (1, -1):
            with torch.no_grad(): parameter[index] = sign * epsilon
            changed, _, _ = scene.render_views(surface, RenderConfig(), charts, contexts)
            values.append(float(image_loss(changed).detach()))
        with torch.no_grad(): parameter[index] = 0
        finite = (values[0] - values[1]) / (2 * epsilon)
        error = abs(analytic - finite)
        require(abs(analytic) > 1e-6 and error <= .001 + .05 * abs(finite), f'GPU AA finite difference failed: {name} {analytic} vs {finite}')
        differences[name] = dict(analytic=analytic, finite_difference=finite, absolute_error=error, epsilon=epsilon, tolerance='.001 + .05*abs(fd)')
    # Separate AA derivative gate on one large triangle, away from discrete
    # silhouette-candidate changes. Vary its top vertex, not constant-area translation.
    base = triangle()
    v = base.vertices_world_mm.detach().float().cuda()
    displacement = torch.tensor(0., device='cuda', requires_grad=True)
    direction = torch.zeros_like(v)
    direction[2, 1] = 1
    def area(value):
        item = DifferentiableMesh(v + value * direction, base.faces.cuda(), base.semantics.float().cuda(), base.face_arch.cuda(), {})
        return render_mesh(item, observation(), RenderConfig()).semantics[..., 1].sum()
    derivative = float(torch.autograd.grad(area(displacement), displacement)[0])
    finite = float(((area(displacement + .002) - area(displacement - .002)) / .004).detach())
    require(abs(derivative - finite) <= .001 + .05 * abs(finite) and abs(derivative) > 1e-6, f'large-triangle AA derivative failed: {derivative} vs {finite}')
    differences['large_triangle_antialias'] = dict(analytic=derivative, finite_difference=finite, absolute_error=abs(derivative-finite), epsilon=.002)
    rendered, mesh, _ = scene.render_views(surface, RenderConfig(), charts)
    fdi = np.array([0, *CHANNEL_FDI], np.uint8)
    for obs in scene.observations:
        obs.labels = np.roll(fdi[rendered[obs.view_id].semantics.detach().argmax(-1).cpu().numpy()], 2, axis=1)
    result = SemanticXYMatcher(SemanticXYConfig(samples_per_tooth=8)).match_scene(rendered, scene.observations)
    result['loss'].backward()
    grads = {name: None if value.grad is None else float(value.grad.norm()) for name, value in scene.named_parameters() if value.requires_grad}
    require(all(value is not None and np.isfinite(value) and value > 0 for value in grads.values()), 'GPU image-to-DMM gradient missing/nonfinite/zero')
    report = write_render_artifacts(output / 'render', scene, rendered, mesh, gradients=grads)
    write_json(output / 'validation.json', dict(status='GPU_RENDER_AND_IMPLICIT_BRIDGE_VALIDATED', torch=torch.__version__,
               gpu=torch.cuda.get_device_name(), tests_passed=tests.testsRun, skipped=0, gpu_gum_occlusion=True,
               finite_differences=differences, finite_difference_scope='DMM interior material XY with fixed raster and chart; large-triangle AA separately',
               limitation='Dense marching-cubes AA failed strict global finite difference: discrete silhouette-candidate changes. This gate does NOT certify that derivative.',
               gradients=grads, runtime=runtime, source_sha256=report['source_sha256'], formal_training=False))
    print(json.dumps(dict(output=str(output), differences=differences, gradients=grads)))


if __name__ == '__main__': main()
