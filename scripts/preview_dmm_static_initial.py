"""Preview the actual saved initial alignment while optimization continues."""
import colorsys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parents[1]
run = Path((project / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
fit = run / 'fit_pilot_004'
report = json.loads((fit / 'initial_render/report.json').read_text('utf-8'))
spec = json.loads((run / 'pilot_scene_spec.json').read_text('utf-8'))
palette = np.array([[.07, .07, .09], *[colorsys.hsv_to_rgb((i*.618)%1, .65, .95) for i in range(28)]])
label_order = np.array([0, *range(11, 18), *range(21, 28), *range(31, 38), *range(41, 48)], np.uint8)
figure, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, view in enumerate(('oral007', 'oral008', 'oral009')):
    rgb_path = next(row['rgb'] for row in spec['views'] if row['view_id'] == view)
    rgb = np.array(Image.open(rgb_path))
    with np.load(fit / 'initial_render' / report['views'][view]['file']) as data:
        pred = label_order[data['semantics'].argmax(-1)]
        labels, valid = data['labels'], data['valid']
        colors = data['semantics'] @ palette
    def edge(mask):
        return (mask != np.roll(mask, 1, 0)) | (mask != np.roll(mask, 1, 1))
    overlay = rgb.astype(float)/255
    overlay[edge(labels) & (labels != 0) & valid] = [1, .85, .05]
    overlay[edge(pred) & (pred != 0) & valid] = [.05, 1, 1]
    for j, (image, title) in enumerate(((rgb, 'Synthetic RGB'), (colors, 'Initial model semantics'), (overlay, 'Target yellow / model cyan'))):
        axes[i, j].imshow(image)
        axes[i, j].set_title(view + ' | ' + title)
        axes[i, j].axis('off')
figure.suptitle('Actual initial alignment: observation-only centroid/Kabsch | no optimized shape yet')
figure.tight_layout()
figure.savefig(run / 'initial_alignment.png', dpi=150)
print(run / 'initial_alignment.png')
