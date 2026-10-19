"""Use completed render artifacts to isolate the actual pose-stage effect."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parents[1]
run = Path((project / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
before = run / 'fit_pilot_004/initial_render'
after = run / 'fit_pilot_005_resume/initial_render'
reports = [json.loads((folder / 'report.json').read_text('utf-8')) for folder in (before, after)]
spec = json.loads((run / 'pilot_scene_spec.json').read_text('utf-8'))
fdi = np.array([0, *range(11, 18), *range(21, 28), *range(31, 38), *range(41, 48)], np.uint8)
cache, scores, foreground = {}, [[], []], [[], []]
for view in reports[0]['views']:
    phase = []
    for index, folder in enumerate((before, after)):
        with np.load(folder / reports[index]['views'][view]['file']) as data:
            pred = fdi[data['semantics'].argmax(-1)]
            labels, valid = data['labels'].copy(), data['valid'].copy()
        for label in sorted(set(np.unique(labels[valid]))-{0, 255}):
            a, b = (labels == label)&valid, (pred == label)&valid
            scores[index].append(float((a&b).sum()/(a|b).sum()))
        a, b = (labels != 0)&valid, (pred != 0)&valid
        foreground[index].append(float((a&b).sum()/(a|b).sum()))
        phase.append(pred)
    cache[view] = phase, labels, valid

def overlay(rgb, labels, pred, valid):
    image = rgb.astype(float)/255
    def edge(mask):
        return (mask != np.roll(mask, 1, 0)) | (mask != np.roll(mask, 1, 1))
    image[~valid] = image[~valid]*.65 + np.array([.5, .1, .55])*.35
    image[edge(labels)&(labels != 0)&valid] = [1, .85, .05]
    image[edge(pred)&(pred != 0)&valid] = [.05, 1, 1]
    return image

figure, axes = plt.subplots(3, 4, figsize=(14, 10))
for row, view in enumerate(('oral007', 'oral008', 'oral009')):
    rgb_path = next(item['rgb'] for item in spec['views'] if item['view_id'] == view)
    rgb = np.array(Image.open(rgb_path))
    (initial, fitted), labels, valid = cache[view]
    error = np.zeros((*labels.shape, 3))
    error[valid&(fitted == labels)&(labels != 0)] = [.2, .7, .35]
    error[valid&(fitted != labels)] = [.95, .22, .2]
    error[~valid] = [.35, .2, .4]
    panels = (rgb, overlay(rgb, labels, initial, valid), overlay(rgb, labels, fitted, valid), error)
    titles = ('Synthetic RGB', 'Before pose', 'After 10 pose steps', 'After pose: red = label error')
    for column, (image, title) in enumerate(zip(panels, titles)):
        axes[row, column].imshow(image)
        axes[row, column].set_title(view+' | '+title)
        axes[row, column].axis('off')
figure.suptitle('Pose-stage diagnostic only | target yellow / model cyan | latent shape fitting not evaluated here')
figure.tight_layout()
figure.savefig(run / 'pose_stage_comparison.png', dpi=150)
result = dict(scope='Completed 10-step pose stage only, before any accepted latent update',
    observed_tooth_view_mean_iou=dict(initial=float(np.mean(scores[0])), after_pose=float(np.mean(scores[1]))),
    tooth_foreground_mean_iou=dict(initial=float(np.mean(foreground[0])), after_pose=float(np.mean(foreground[1]))),
    observed_tooth_view_pairs=len(scores[0]), views=len(cache))
(run / 'pose_stage_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
