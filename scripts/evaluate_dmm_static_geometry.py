"""Post-fit sampled surface error to hash-verified synthetic source teeth.

Uses the renderer's world transforms only for evaluation; no alignment/refit.
Distances from sampled points to triangles are exact, including degenerate-edge
fallbacks. This is a finite area sample, not an exact whole-surface Hausdorff.
"""
import colorsys
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT = Path(__file__).resolve().parents[1]
RUN = Path((PROJECT / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
FIT = RUN / sys.argv[1]
BASELINE = RUN / 'fit_pilot_004' if 'resume' in FIT.name else FIT
assert (FIT / 'result.json').exists(), 'Post-fit evaluation only'
OUT = RUN / ('geometry_evaluation_' + FIT.name)
PLOT_ONLY = '--plot-only' in sys.argv[2:]
if PLOT_ONLY:
    assert (OUT / 'geometry_metrics.json').is_file(), 'Plot-only mode requires completed distance metrics'
OUT.mkdir(exist_ok=PLOT_ONLY)
PLAN = json.loads((RUN / 'plan.json').read_text('utf-8'))
DATA = Path(PLAN['source']) / 'dataset/Patient_L428SD7J'
metadata = json.loads((DATA / 'source_scene_metadata.json').read_text('utf-8'))
config = json.loads((DATA / 'source_resolved_config.json').read_text('utf-8'))
FDI = [*range(11, 18), *range(21, 28), *range(31, 38), *range(41, 48)]
palette = {label: colorsys.hsv_to_rgb((i*.618)%1, .65, .95) for i, label in enumerate(FDI)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sample(triangles, count, seed):
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    assert np.isfinite(area).all() and area.sum() > 0
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(triangles), count, p=area/area.sum())
    uv = rng.random((count, 2))
    u = np.sqrt(uv[:, :1])
    weights = np.concatenate((1-u, u*(1-uv[:, 1:]), u*uv[:, 1:]), axis=1)
    return (triangles[selected]*weights[..., None]).sum(1)


def distances(points, triangles):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    n2 = (normal*normal).sum(1)
    d00, d01, d11 = (ab*ab).sum(1), (ab*ac).sum(1), (ac*ac).sum(1)
    determinant = d00*d11-d01*d01
    output = []
    for first in range(0, len(points), 64):
        p = points[first:first+64, None]
        ap = p-a
        d20, d21 = (ap*ab).sum(-1), (ap*ac).sum(-1)
        safe = np.where(determinant > 1e-30, determinant, 1.)
        u = (d11*d20-d01*d21)/safe
        v = (d00*d21-d01*d20)/safe
        inside = (determinant > 1e-30) & (u >= 0) & (v >= 0) & (u+v <= 1)
        plane = (ap*normal).sum(-1)**2/np.where(n2 > 1e-30, n2, 1.)
        best = np.where(inside, plane, np.inf)
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end-start
            length = (edge*edge).sum(1)
            t = ((p-start)*edge).sum(-1)/np.where(length > 1e-30, length, 1.)
            delta = p - (start+np.clip(t, 0, 1)[..., None]*edge)
            best = np.minimum(best, (delta*delta).sum(-1))
        output.append(np.sqrt(best.min(1)))
    return np.concatenate(output)


# Independent analytic checks validate face-interior, edge and degenerate cases.
test_triangle = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]])
assert np.allclose(distances(np.array([[.25, .25, 2.], [2., 0, 0], [1., 1, 0]]), test_triangle), [2., 1., np.sqrt(.5)])
assert np.allclose(distances(np.array([[0., 0, 2.]]), np.zeros((1, 3, 3))), [2.])

sources, truth, truth_all = {}, {}, []
for arch in ('upper', 'lower'):
    obj = Path(config['source_root']) / arch / 'L428SD7J' / ('L428SD7J_' + arch + '.obj')
    annotation = obj.with_suffix('.json')
    for path in (obj, annotation):
        expected = config['expected_source_files'][str(path)]
        actual = digest(path)
        assert actual == expected, path
        sources[str(path)] = actual
    mesh = trimesh.load(obj, process=False, force='mesh', maintain_order=True)
    labels = np.array(json.loads(annotation.read_text('utf-8'))['labels'])
    assert len(labels) == len(mesh.vertices), 'Source vertex order/labels mismatch'
    transform = np.array(metadata['transforms'][arch]['source_to_world_mm'])
    vertices = np.asarray(mesh.vertices) @ transform[:3, :3].T + transform[:3, 3]
    face_labels = labels[np.asarray(mesh.faces)]
    for label in FDI:
        if (label < 30) != (arch == 'upper'):
            continue
        selected = (face_labels == label).sum(1) >= 2
        triangles = vertices[np.asarray(mesh.faces)[selected]]
        assert len(triangles), label
        matrix = np.array(metadata['object_transforms_blender_m']['Tooth_' + str(label)])
        verify = 1000*(np.asarray(mesh.vertices)[:10] @ matrix[:3, :3].T + matrix[:3, 3])
        assert np.max(np.abs(verify-vertices[:10])) < .002, 'Object/world transform mismatch'
        truth[label] = triangles
        truth_all.append((triangles, np.array(palette[label])))

rows = json.loads((OUT / 'geometry_metrics.json').read_text('utf-8'))['per_tooth'] if PLOT_ONLY else []
phase_mesh = {}
for phase in ('initial', 'final'):
    render_root = BASELINE if phase == 'initial' else FIT
    with np.load(render_root / (phase + '_render/mesh.npz')) as data:
        vertices, faces, probabilities = data['vertices_world_mm'].astype(float), data['faces'], data['semantics']
    # A component label from the blended surface is an evaluation partition,
    # not a claim of anatomical tooth segmentation.
    face_class = np.mean(probabilities[faces], axis=1).argmax(1)
    channel_labels = np.array([0, *FDI])
    assigned = channel_labels[face_class]
    phase_mesh[phase] = vertices[faces], assigned
    if PLOT_ONLY:
        continue
    for label in FDI:
        predicted = vertices[faces[assigned == label]]
        row = dict(phase=phase, label=label, predicted_faces=len(predicted), truth_faces=len(truth[label]))
        if not len(predicted):
            row['status'] = 'MISSING_PREDICTED_SURFACE'
        else:
            before = sample(predicted, 1000, 42000+label)
            reference = sample(truth[label], 1000, 84000+label)
            p2g, g2p = distances(before, truth[label]), distances(reference, predicted)
            pair = np.concatenate((p2g, g2p))
            row.update(status='SAMPLED_BIDIRECTIONAL_SURFACE_DISTANCE', mean_mm=float(pair.mean()),
                rms_mm=float(np.sqrt(np.mean(pair**2))), median_mm=float(np.median(pair)), p95_mm=float(np.percentile(pair, 95)),
                predicted_to_truth_mean_mm=float(p2g.mean()), truth_to_predicted_mean_mm=float(g2p.mean()))
        rows.append(row)
        print(phase, label, row.get('rms_mm', row['status']), flush=True)

summary = dict(scope='Post-fit evaluation only; no ICP/Sim3/alignment; renderer world transforms in assumed source mm',
    source_files=sources, samples_per_direction_per_tooth=1000, distance='Exact point-to-triangle distances at finite area-weighted samples',
    partition='Prediction face average soft-semantic argmax; GT majority source-vertex FDI; third molars excluded',
    phase_metrics={}, per_tooth=rows, independent_test=False, quality_accepted=False)
for phase in ('initial', 'final'):
    group = [r for r in rows if r['phase'] == phase and 'rms_mm' in r]
    summary['phase_metrics'][phase] = dict(valid_teeth=len(group), expected_teeth=28,
        macro_rms_mm=float(np.mean([r['rms_mm'] for r in group])) if group else None,
        macro_mean_mm=float(np.mean([r['mean_mm'] for r in group])) if group else None,
        macro_p95_mm=float(np.mean([r['p95_mm'] for r in group])) if group else None)
paired = sorted(set(r['label'] for r in rows if r['phase'] == 'initial' and 'rms_mm' in r)
                & set(r['label'] for r in rows if r['phase'] == 'final' and 'rms_mm' in r))
summary['paired_teeth'] = paired
summary['paired_macro_rms_mm'] = {phase: float(np.mean([r['rms_mm'] for r in rows
    if r['phase'] == phase and r['label'] in paired])) if paired else None for phase in ('initial', 'final')}
(OUT / 'geometry_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
with (OUT / 'per_tooth_surface_error.csv').open('w', encoding='utf-8', newline='') as stream:
    keys = sorted(set().union(*(set(row) for row in rows)))
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)

figure = plt.figure(figsize=(14, 10))
all_vertices = np.concatenate([t.reshape(-1, 3) for t, _ in truth_all])
center = (all_vertices.min(0)+all_vertices.max(0))/2
radius = np.ptp(all_vertices, axis=0).max()*.62
for i, phase in enumerate(('reference', 'initial', 'final')):
    if phase == 'reference':
        points = np.concatenate([sample(t, 700, 25000+i) for i, (t, _) in enumerate(truth_all)])
        color = np.concatenate([np.tile(c[None], (700, 1)) for _, c in truth_all])
    else:
        tri, assigned = phase_mesh[phase]
        groups = [(label, tri[assigned == label]) for label in FDI if (assigned == label).any()]
        points = np.concatenate([sample(t, 700, 25000+label) for label, t in groups])
        color = np.concatenate([np.tile(palette[label], (700, 1)) for label, _ in groups])
    for j, (elev, azim, title) in enumerate(((0, -90, 'Anterior'), (25, -125, 'Oblique'), (80, -90, 'Occlusal'))):
        ax = figure.add_subplot(3, 3, i*3+j+1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=.45, linewidths=0, rasterized=True, depthshade=True)
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
        ax.set_box_aspect((1, 1, 1), zoom=1.6)
        ax.set_proj_type('ortho')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(phase.title()+' | '+title)
        ax.set_axis_off()
figure.suptitle('Evaluation only | area-sampled source/DMM tooth surfaces | same world frame; no alignment')
figure.tight_layout()
figure.savefig(OUT / 'reference_initial_final_geometry.png', dpi=140)
plt.close(figure)
print(json.dumps(summary['phase_metrics'], indent=2))
