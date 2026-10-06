"""Persist optional dual workflow/GPU regression evidence; no real data training."""
from dataclasses import replace
from datetime import datetime, timezone
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tests'), str(ROOT/'third_party/DMM')]
import torch
from test_fitting import observed_scene, FIT, SURFACE, MATCH
from dmm.collision import CollisionConfig
from dmm.recovery import RecoveryConfig
from dmm.fitting import fit_scene
from dmm.provenance import source_fingerprint
from dmm.validation import require, write_json, sha256


def main():
    require(torch.cuda.is_available(), 'real CUDA required')
    output = ROOT/'runs'/('dual_extension_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    suite = unittest.TestSuite()
    for pattern in ('test_rendering.py', 'test_semanticxy.py', 'test_fitting.py', 'test_dual_extension.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern))
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (output/'tests.log').write_text(log.getvalue(), encoding='utf-8')
    report = dict(source_sha256=source_fingerprint(), tests_run=tests.testsRun, skipped=len(tests.skipped),
                  failures=len(tests.failures), errors=len(tests.errors), formal_training=False,
                  training_scope='two temporary 1-epoch independent unit-test jobs on synthetic points only',
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    write_json(output/'validation.json', report)
    require(tests.wasSuccessful() and not tests.skipped, f'tests failed: {output}')
    cases = {}
    config = replace(FIT, pose_iterations=2, shape_iterations=1, joint_iterations=2)
    for name, collision in [('disabled', CollisionConfig()), ('enabled', CollisionConfig(enabled=True, samples_per_arch=64)),
                            ('overlap_enabled', CollisionConfig(enabled=True, samples_per_arch=64))]:
        scene, renderer = observed_scene('cuda', offset=.2)
        case_config = config
        if name == 'overlap_enabled':
            with torch.no_grad():
                scene.arches['lower'].base_pose.copy_(scene.arches['upper'].base_pose)
                scene.arches['lower'].base_pose[0, 3] += 10.
            case_config = replace(config, recovery=RecoveryConfig(enabled=False))
        result = fit_scene(scene, output/name, case_config, SURFACE, renderer, MATCH, collision_config=collision)
        cases[name] = dict(status=result['status'], collision=result['collision'],
                           min_iou=result['final_metrics']['min_tooth_iou'],
                           result_sha256=sha256(output/name/'result.json'))
    report.update(status='OPTIONAL_DUAL_EXTENSION_INTERFACE_VALIDATED', cases=cases)
    write_json(output/'validation.json', report)
    print(output, flush=True)


if __name__ == '__main__': main()
