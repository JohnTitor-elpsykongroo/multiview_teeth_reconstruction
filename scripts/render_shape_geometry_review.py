"""Post-fit sampled surface-distance views; geometry truth never enters fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from evaluate_shape_only import sampled_points
from surface_metric_utils import finite_nearest_triangle_distances as nearest_triangle_distances
from run_forward_check import PROJECT_ROOT, load_ply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--output', default='surface_review.png')
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()) or Path(args.output).name != args.output:
        raise ValueError('output must be a filename in a project run')
    output = root/args.output
    if output.exists():
        raise FileExistsError(output)
    evaluation = json.loads((root/'evaluation.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    truth = json.loads((Path(provenance['fit_input']).parent/'truth/manifest.json').read_text())
    source = Path(truth['source_run'])
    points, distances = {}, {}
    for stage in ['initial', 'final']:
        point_parts, distance_parts = [], []
        for label in evaluation['active_labels']:
            mesh = load_ply(root/'meshes'/stage/f'tooth{label}.ply')
            target = load_ply(source/'meshes/training_case'/f'tooth{label}.ply')
            sampled = sampled_points(mesh, 1500, 510+label)
            point_parts.append(sampled)
            distance_parts.append(nearest_triangle_distances(sampled, target))
        points[stage] = np.concatenate(point_parts)
        distances[stage] = np.concatenate(distance_parts)
    all_points = np.concatenate(list(points.values()))
    center = (all_points.min(axis=0)+all_points.max(axis=0))/2
    radius = np.max(np.ptp(all_points, axis=0))/2*1.05
    maximum = float(np.percentile(distances['initial'], 95))
    norm = Normalize(0, max(maximum, 1e-8))
    fig = plt.figure(figsize=(12, 9), layout='constrained')
    for row, stage in enumerate(['initial', 'final']):
        for column, (elevation, azimuth) in enumerate([(65, -90), (20, -60)]):
            ax = fig.add_subplot(2, 2, row*2+column+1, projection='3d')
            p = points[stage]
            artist = ax.scatter(*p.T, c=distances[stage], cmap='magma', norm=norm,
                                s=2, depthshade=False, rasterized=True)
            for axis, value in zip('xyz', center):
                getattr(ax, 'set_'+axis+'lim')(value-radius, value+radius)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elevation, azim=azimuth)
            ax.set_title(f'{stage} | elevation {elevation} deg', fontsize=12)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            ax.tick_params(labelsize=8)
    fig.colorbar(artist, ax=fig.axes, shrink=.65, label='Candidate surface to GT distance (DMM units)', extend='max')
    fig.suptitle(f'{len(evaluation["active_labels"])} active teeth: initial/final surface error\n'
                 '1500 area-sampled points per tooth; one-way review, shared initial P95 color scale', fontsize=13)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    (output.with_suffix('.json')).write_text(json.dumps({
        'status': 'POSTFIT_GEOMETRY_VISUAL_REVIEW', 'active_labels': evaluation['active_labels'],
        'samples_per_tooth': 1500, 'distance': 'one-way candidate-to-GT nearest-32-candidate triangle distance',
        'shared_color_max_dmm': maximum, 'evaluation_surface_metric': 'separate bidirectional 6000-point metric'
    }, indent=2))
    print(json.dumps({'figure': str(output), 'color_max_dmm': maximum}))


if __name__ == '__main__':
    main()
