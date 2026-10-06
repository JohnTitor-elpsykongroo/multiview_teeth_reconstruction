"""Post-fit image-Jacobian spectrum; uses initial fit geometry, never target codes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_forward_check import load_ply
from shape_only_common import (load_model,fixed_render_cache,render_scene,make_pairs,SurfaceImageObjective)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_root',type=Path)
    args=parser.parse_args()
    root=args.run_root.resolve(strict=True)
    if not (root/'evaluation.json').exists():
        raise ValueError('diagnose only after independent evaluation completed')
    output=root/'observability.json'
    if output.exists():
        raise FileExistsError(output)
    provenance=json.loads((root/'provenance.json').read_text())
    config=json.loads((root/'resolved_config.json').read_text())
    fit=Path(provenance['fit_input'])
    manifest=json.loads((fit/'manifest.json').read_text())
    cameras=json.loads((fit/manifest['camera_file']).read_text())
    initial=dict(np.load(root/'initial_codes.npz'))
    final=dict(np.load(root/'final_codes.npz'))
    scales=dict(np.load(fit/manifest['training_scales']))
    targets={v['camera']:np.asarray(Image.open(fit/v['path'])) for v in manifest['masks']}
    active=manifest['active_labels']
    pose=manifest['fixed_arch_to_world']
    meshes={k:load_ply(root/'meshes/initial'/f'tooth{k}.ply') for k in manifest['visible_fdi_ids']}
    cache=fixed_render_cache({k:v for k,v in meshes.items() if k not in active},cameras,pose)
    masks,maps=render_scene({k:meshes[k] for k in active},cache,cameras,pose,True)
    pairs=make_pairs(masks,maps,targets,cameras,active,pose,config['boundary_samples_per_direction'])
    net=load_model(config,manifest)
    result={}
    for label in active:
        key=f'label_{label}'
        q0=np.zeros_like(initial[key])
        objective=SurfaceImageObjective(net,label,initial[key],scales[key],q0,pairs[label],cameras,pose,0.0)
        J=objective.jacobian(q0)[:-len(q0)]
        _,s,vt=np.linalg.svd(J,full_matrices=False)
        rank=int(np.count_nonzero(s>=s[0]*.01))
        delta=(final[key]-initial[key])/scales[key]
        coefficients=vt@delta
        result[str(label)]={'image_residual_count':len(J),'latent_dimensions':len(q0),
            'singular_values_pixels_per_training_std':s.tolist(),
            'directions_above_one_percent_of_max':rank,'condition_number':float(s[0]/max(s[-1],1e-15)),
            'final_code_update_energy_in_weaker_directions_fraction':float(np.sum(coefficients[rank:]**2)/max(np.sum(coefficients**2),1e-15))}
    output.write_text(json.dumps({'scope':'local initial visible-contour Jacobian, damping excluded; not proof of global identifiability',
                                  'per_tooth':result},indent=2))
    print(json.dumps({'observability':str(output),'ranks':{k:v['directions_above_one_percent_of_max'] for k,v in result.items()}}))


if __name__=='__main__':
    main()
