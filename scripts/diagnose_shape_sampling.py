"""Diagnose initialization-domain truncation without target meshes or codes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from shape_only_common import load_model,decode
from run_forward_check import save_ply


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job_root',type=Path)
    parser.add_argument('--label',type=int,default=21)
    args=parser.parse_args()
    root=args.job_root.resolve(strict=True)
    fit=root/'prepared/fit_input'
    manifest=json.loads((fit/'manifest.json').read_text())
    config=json.loads((root/'config.json').read_text())
    codes=dict(np.load(fit/manifest['initial_codes']))
    box=json.loads((fit/manifest['sampling_boxes']).read_text())[str(args.label)]
    output=root/'sampling_diagnostic'
    output.mkdir(exist_ok=False)
    net=load_model(config,manifest)
    records=[]
    for pad in [0,8,16]:
        origin,top=np.asarray(box['grid_origin']),np.asarray(box['grid_top'])
        spacing=(top-origin)/(box['grid_n']-1)
        expanded={'grid_origin':(origin-pad*spacing).tolist(),'grid_top':(top+pad*spacing).tolist(),
                  'grid_n':box['grid_n']+2*pad}
        record={'padding_cells':pad,'sampling_box':expanded}
        try:
            vertices,faces=decode(net,args.label,codes[f'label_{args.label}'],expanded)
            mesh=trimesh.Trimesh(vertices,faces,process=False)
            record.update({'status':'BOUNDED_SURFACE','watertight':bool(mesh.is_watertight),
                           'connected_components':len(mesh.split(only_watertight=False)),
                           'bounds_min':vertices.min(0).tolist(),'bounds_max':vertices.max(0).tolist()})
            save_ply(output/f'pad{pad}_tooth{args.label}.ply',vertices,faces)
        except ValueError as error:
            record.update({'status':'DOMAIN_CHECK_FAILED','reason':str(error)})
        records.append(record)
    (output/'report.json').write_text(json.dumps({'label':args.label,'target_truth_used':False,'records':records},indent=2))
    print(json.dumps({'report':str(output/'report.json'),'records':records}))


if __name__=='__main__':
    main()
