"""Persist real-GPU integration evidence using deterministic untrained DMMs."""
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tests'), str(ROOT/'third_party'/'DMM')]
from test_fitting import observed_scene, FIT, SURFACE, MATCH
from dmm.fitting import fit_scene
from dmm.recovery import RecoveryConfig
from dmm.provenance import source_fingerprint
from dmm.validation import write_json, require, sha256


def main():
    require(torch.cuda.is_available(), 'real CUDA required')
    output = ROOT/'runs'/('staged_fit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    suite = unittest.TestSuite()
    for pattern in ('test_rendering.py', 'test_semanticxy.py', 'test_fitting.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern))
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (output/'tests.log').write_text(log.getvalue(), encoding='utf-8')
    require(tests.wasSuccessful() and not tests.skipped, f'integration tests failed; see {output}')
    specifications = [('staged', 2., FIT),
                      ('out_of_frame', 100., replace(FIT, pose_iterations=3, shape_iterations=2, joint_iterations=4)),
                      ('occluded_lower', 0., replace(FIT, pose_iterations=3, shape_iterations=2, joint_iterations=4)),
                      ('recovery_disabled', 100., replace(FIT, pose_iterations=1, shape_iterations=0, joint_iterations=0,
                                                        recovery=RecoveryConfig(enabled=False)))]
    # Fixed cases declared before any fits; no selection based on outcomes.
    write_json(output/'cases.json', [dict(name=name, offset_mm=offset, config=config.resolved()) for name, offset, config in specifications])
    results = {}
    for name, offset, config in specifications:
        scene, renderer = observed_scene('cuda', offset)
        if name == 'occluded_lower':
            with torch.no_grad(): scene.arches['lower'].base_pose[:3, 3] = torch.tensor([-7*120/85, -6*120/85, 120.], device='cuda')
        result = fit_scene(scene, output/name, config, SURFACE, renderer, MATCH)
        results[name] = dict(status=result['status'], initial_min_iou=result['initial_metrics']['min_tooth_iou'],
                             final_min_iou=result['final_metrics']['min_tooth_iou'], unresolved=len(result['unresolved']),
                             iterations=result['iterations'], bootstrap=result['bootstrap'])
        print(name, results[name], flush=True)
    write_json(output/'validation.json', dict(status='STAGED_FITTER_INTERFACE_VALIDATED', tests_passed=tests.testsRun,
               skipped=len(tests.skipped), source_sha256=source_fingerprint(), formal_training=False,
               fixture='untrained native DMM octahedra; known cameras; no historical sanity data', cases=results,
               evidence={f'{name}/result.json': sha256(output/name/'result.json') for name in results}))
    print(json.dumps(dict(output=str(output), tests_passed=tests.testsRun, cases=results)))


if __name__ == '__main__': main()
