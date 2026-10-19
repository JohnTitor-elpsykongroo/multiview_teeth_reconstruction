"""Exercise real decoder updates/reset and compare cached blending to direct R4 queries."""
from pathlib import Path
import gzip
import json
import time
import urllib.request
import urllib.error
import numpy as np
import torch
from engine import Engine, ROOT

OUT = ROOT / 'runs/dmm_latent_viewer_verification_20261016'
OUT.mkdir(exist_ok=True)
URL = 'http://127.0.0.1:8877'


def api(path, body=None):
    request = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        if response.headers.get('Content-Encoding') == 'gzip':
            raw = gzip.decompress(raw)
        return json.loads(raw)


def submit(arch, q):
    job_id = api('/api/decode', {'arch': arch, 'q': q})['id']
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status = api('/api/job?id=' + job_id)
        if status['status'] == 'error':
            raise AssertionError(status)
        if status['status'] == 'done':
            return api('/api/mesh?job=' + job_id)
        time.sleep(.4)
    raise TimeoutError(job_id)


def main():
    status = api('/api/status')
    assert status['ready'], status
    report = {'model_metadata': status['metadata'], 'api_updates': {}, 'cache_direct_field_comparison': {}}
    for arch, label, dim, value in [('upper', '11', 0, 2), ('lower', '36', 2, -2)]:
        base = api('/api/mesh?arch=' + arch)
        q = {key: [0] * 10 for key in base['q']}
        q[label][dim] = value
        changed = submit(arch, q)
        assert changed['q'] == q
        vertices = np.asarray(changed['vertices']); faces = np.asarray(changed['faces'])
        assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices)
        assert changed['vertices'] != base['vertices'], 'latent update did not change mesh'
        for key in q:
            q[key] = [0] * 10
        restored = submit(arch, q)
        assert restored['vertices'] == base['vertices'] and restored['faces'] == base['faces'], 'zero reset differs'
        report['api_updates'][arch] = {'label': label, 'dimension': dim + 1, 'value': value,
            'decode_seconds': changed['seconds'], 'mesh_changed': True, 'zero_reset_exact': True,
            'triangles': changed['count']['triangles']}
        print(arch, report['api_updates'][arch], flush=True)
    try:
        api('/api/decode', {'arch': 'upper', 'q': {'11': [100] * 10}})
        raise AssertionError('malformed parameters were accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 400
        report['invalid_parameters_rejected'] = True
    engine = Engine(lambda *args: None, status['metadata']['resolution'])
    engine.prepare()
    rng = np.random.default_rng(20261016)
    with torch.no_grad():
        for arch in ('upper', 'lower'):
            bundle = engine.models[arch]
            ids = rng.choice(engine.n ** 3, size=4096, replace=False)
            box = np.asarray(bundle.metadata['sampling_domain_model'])
            axes = [torch.linspace(float(box[0, i]), float(box[1, i]), engine.n, device=engine.device) for i in range(3)]
            index = torch.tensor(ids, device=engine.device)
            points = torch.stack([axes[0][index // engine.n ** 2], axes[1][index // engine.n % engine.n], axes[2][index % engine.n]], -1)
            parts = list(engine.bases[arch].values())
            logs = np.stack([part[1, ids] for part in parts]); fields = np.stack([part[0, ids] for part in parts])
            weights = np.exp(logs - logs.max(0, keepdims=True))
            cached = (fields * weights).sum(0) / weights.sum(0)
            direct = bundle.model.query(points, bundle.means)['sdf'].cpu().numpy()
            error = float(np.abs(cached - direct).max())
            assert error < 3e-6, (arch, error)
            report['cache_direct_field_comparison'][arch] = {'points': len(ids), 'maximum_absolute_error_model_units': error}
    report['status'] = 'PASS_REAL_MODEL_UPDATES_RESET_AND_CACHE_EQUIVALENCE'
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
