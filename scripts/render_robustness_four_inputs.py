"""Visualize prepared masks without opening geometry truth."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_forward_check import PROJECT_ROOT, sha256


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('batch_root', type=Path); args = parser.parse_args()
    root = args.batch_root.resolve(strict=True)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()) or (root/'input_review.png').exists():
        raise ValueError('Invalid root or existing figure')
    state = json.loads((root/'status.json').read_text())
    selected = [j for j in state['jobs'] if j['axis']['kind'] in ['clean_replay', 'mask_boundary', 'view_drop']]
    variants = []
    for job in selected:
        fit = Path(job['directory'])/'case/fit_input'
        variants.append((job['id'], fit, json.loads((fit/'manifest.json').read_text())))
    _, source, sm = variants[0]
    fig, axes = plt.subplots(3, len(variants), figsize=(14, 10), squeeze=False)
    for row, mask in enumerate(sm['masks']):
        name = mask['camera']; original = np.asarray(Image.open(source/mask['path']))
        yy, xx = np.nonzero(original); x0, x1 = max(0, xx.min()-8), xx.max()+9; y0, y1 = max(0, yy.min()-8), yy.max()+9
        for col, (identifier, fit, manifest) in enumerate(variants):
            ax = axes[row, col]; ax.axis('off')
            item = next((m for m in manifest['masks'] if m['camera'] == name), None)
            if item is None:
                ax.text(.5, .5, 'NOT PROVIDED TO FITTER', ha='center', va='center'); ax.set_title(identifier+'\n'+name); continue
            if sha256(fit/item['path']) != item['sha256']:
                raise ValueError('Mask changed')
            target = np.asarray(Image.open(fit/item['path']))
            rgb = np.full((*original.shape, 3), 20, dtype=np.uint8)
            rgb[(original == target) & (target > 0)] = (220, 220, 220)
            rgb[(original > 0) & (target == 0)] = (242, 70, 83)
            rgb[(original == 0) & (target > 0)] = (60, 146, 245)
            rgb[(original > 0) & (target > 0) & (original != target)] = (250, 190, 48)
            ax.imshow(rgb[y0:y1, x0:x1], interpolation='nearest')
            ax.set_title(f'{identifier}\n{name}: {np.count_nonzero(original != target)} pixels', fontsize=10)
    fig.suptitle('Prepared four-tooth observations vs clean original\nWhite: unchanged | Red: removed | Blue: added | Yellow: relabeled', fontsize=13)
    fig.subplots_adjust(top=.86, bottom=.03, left=.01, right=.99, hspace=.35, wspace=.13)
    fig.savefig(root/'input_review.png', dpi=150); plt.close(fig)
    print(root/'input_review.png')


if __name__ == '__main__':
    main()
