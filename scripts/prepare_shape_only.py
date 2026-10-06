"""Synthetic generator: package perturbed codes separately from active truth."""

from __future__ import annotations
import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from run_forward_check import PROJECT_ROOT, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/shape_only.json')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    source = Path(config['source_run']).resolve(strict=True)
    source_provenance = json.loads((source/'provenance.json').read_text())
    truth = json.loads((source/'truth/manifest.json').read_text())
    truth_latents = source/'truth'/truth['latent_file']
    if sha256(truth_latents) != truth['latent_sha256']:
        raise ValueError('truth latent hash mismatch')
    exp = Path(config['experiment'])
    latent_path = exp/'LatentCodes'/f"latent_vecs_{config['checkpoint']}.pth"
    model_path = exp/'ModelParameters'/f"dmm_{config['checkpoint']}.pth"
    if sha256(latent_path) != source_provenance['input_sha256'][str(latent_path)]:
        raise ValueError('latent checkpoint changed')
    if sha256(model_path) != source_provenance['input_sha256'][str(model_path)]:
        raise ValueError('model checkpoint changed')
    saved = torch.load(latent_path, map_location='cpu', weights_only=True)
    if saved['epoch'] != truth['checkpoint_epoch']:
        raise ValueError('epoch mismatch')
    arrays = dict(np.load(truth_latents))
    active = set(config['active_labels'])
    labels = [v for v in source_provenance['labels'] if v != 0]
    if not active.issubset(labels):
        raise ValueError('active labels are outside upper teeth')
    rng = np.random.default_rng(config['perturbation_seed'])
    initial, scales = {}, {}
    for label in labels:
        matrix = saved['latent_codes'][f'{label}.module.weight'].numpy()
        scale = matrix.std(axis=0, ddof=1).astype(np.float64)
        if np.any(scale <= 0):
            raise ValueError('invalid training code scales')
        scales[f'label_{label}'] = scale
        initial[f'label_{label}'] = arrays[f'label_{label}'].astype(np.float64).copy()
        if label in active:
            label_rng = np.random.default_rng(np.random.SeedSequence([config['perturbation_seed'],label])) if config.get('perturbation_rng')=='per_label_seedsequence' else rng
            initial[f'label_{label}'] += label_rng.normal(size=len(scale)) * scale * config['perturbation_training_std']
    root = args.output_root.resolve() if args.output_root else PROJECT_ROOT/'runs'/(dt.datetime.now(dt.timezone.utc).strftime('shape_case_%Y%m%dT%H%M%SZ_') + sha256(args.config)[:8])
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('output root must be inside project runs')
    root.mkdir(parents=True, exist_ok=False)
    fit = root/'fit_input'
    shutil.copytree(source/'fit_input', fit)
    np.savez_compressed(fit/'initial_codes.npz', **initial)
    np.savez_compressed(fit/'training_scales.npz', **scales)
    mesh_metrics = json.loads((source/'mesh_metrics.json').read_text())
    boxes = {str(label): {k: mesh_metrics['zero'][str(label)][k]
                         for k in ['grid_origin', 'grid_top', 'grid_n']} for label in labels}
    padding=int(config.get('sampling_padding_cells',0))
    for label in active:
        box=boxes[str(label)]
        origin,top=np.asarray(box['grid_origin']),np.asarray(box['grid_top'])
        spacing=(top-origin)/(box['grid_n']-1)
        box.update({'grid_origin':(origin-padding*spacing).tolist(),
                    'grid_top':(top+padding*spacing).tolist(),'grid_n':box['grid_n']+2*padding})
    (fit/'sampling_boxes.json').write_text(json.dumps(boxes, indent=2))
    manifest = json.loads((fit/'manifest.json').read_text())
    manifest.update({
        'status': 'SHAPE_ONLY_SYNTHETIC_INPUT_READY',
        'active_labels': sorted(active), 'frozen_labels': [v for v in labels if v not in active],
        'fixed_arch_to_world': np.eye(4).tolist(),
        'diagnostic_oracle_conditions': ['known cameras', 'true upper arch pose',
            'inactive teeth fixed at generator codes', 'active initialization perturbed from generator codes'],
        'active_truth_codes_available_to_fitter': False,
        'initial_codes': 'initial_codes.npz', 'training_scales': 'training_scales.npz',
        'sampling_boxes': 'sampling_boxes.json',
        'extra_input_sha256': {name: sha256(fit/name) for name in
                              ['initial_codes.npz', 'training_scales.npz', 'sampling_boxes.json']},
        'model_sha256': sha256(model_path), 'specs_sha256': sha256(exp/'specs.json'),
        'checkpoint_epoch': truth['checkpoint_epoch'],
        'perturbation_rng':config.get('perturbation_rng','legacy_sequential'),
        'sampling_padding_cells':padding,
    })
    (fit/'manifest.json').write_text(json.dumps(manifest, indent=2))
    isolated = root/'truth'
    isolated.mkdir()
    shutil.copyfile(truth_latents, isolated/'latents.npz')
    shutil.copyfile(source/'review_camera.json', isolated/'heldout_camera.json')
    shutil.copyfile(source/'renders/training_case/occlusal_65deg_teeth_labels.png', isolated/'heldout_labels.png')
    truth.update({'source_run': str(source), 'latent_file': 'latents.npz',
                  'heldout_camera_file': 'heldout_camera.json', 'heldout_mask_file': 'heldout_labels.png',
                  'heldout_mask_sha256': sha256(isolated/'heldout_labels.png'),
                  'heldout_camera_sha256': sha256(isolated/'heldout_camera.json'),
                  'preparation_config_sha256': sha256(args.config)})
    (isolated/'manifest.json').write_text(json.dumps(truth, indent=2))
    (root/'resolved_config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps({'status': manifest['status'], 'prepared_root': str(root), 'fit_input': str(fit)}))


if __name__ == '__main__':
    main()
