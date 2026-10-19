"""Read-only identity/order/transform preflight for later 3D evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh

project = Path(__file__).resolve().parents[1]
run = Path((project / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
source = Path(json.loads((run / 'plan.json').read_text('utf-8'))['source']) / 'dataset/Patient_L428SD7J'
config = json.loads((source / 'source_resolved_config.json').read_text('utf-8'))
metadata = json.loads((source / 'source_scene_metadata.json').read_text('utf-8'))
result = dict(scope='Post-fit reference preflight only; not supplied to optimizer', arches={})
for arch in ('upper', 'lower'):
    obj = Path(config['source_root']) / arch / 'L428SD7J' / ('L428SD7J_'+arch+'.obj')
    label_path = obj.with_suffix('.json')
    hashes = {}
    for path in (obj, label_path):
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        assert h == config['expected_source_files'][str(path)]
        hashes[path.name] = h
    mesh = trimesh.load(obj, process=False, force='mesh', maintain_order=True)
    labels = np.array(json.loads(label_path.read_text('utf-8'))['labels'])
    assert len(labels) == len(mesh.vertices)
    transform = np.array(metadata['transforms'][arch]['source_to_world_mm'])
    errors = {}
    for label in sorted(set(labels)-{0}):
        if label%10 == 8:
            continue
        matrix = np.array(metadata['object_transforms_blender_m']['Tooth_'+str(label)])
        a = np.asarray(mesh.vertices)[:10] @ transform[:3, :3].T + transform[:3, 3]
        b = 1000*(np.asarray(mesh.vertices)[:10] @ matrix[:3, :3].T + matrix[:3, 3])
        errors[str(label)] = float(np.max(np.abs(a-b)))
    result['arches'][arch] = dict(vertices=len(mesh.vertices), faces=len(mesh.faces), hashes=hashes,
        transform_max_difference_mm=max(errors.values()), per_tooth_transform_difference_mm=errors)
    assert max(errors.values()) < .002
(run / 'geometry_reference_preflight.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
