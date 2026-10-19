"""Read-only decoder probes; write isolated diagnostic meshes and field volumes."""
from pathlib import Path
import sys, json, time, hashlib, contextlib, io, argparse
import numpy as np
import torch
from skimage.measure import marching_cubes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/dmm_mean_quality_audit_20261016'
OUT.mkdir(exist_ok=True)
ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['r4','old'],required=True)
ap.add_argument('--mode',choices=['mean','zero'],default='mean');ap.add_argument('--resolution',type=int,default=128)
args=ap.parse_args();torch.set_num_threads(4)
start=time.monotonic()
if args.model=='r4':
    source=ROOT/'.runtime/dmm_r4_best_val_20261015/package/third_party/DMM'
    sys.path.insert(0,str(source))
    from dmm.bundle import load_bundle
    bundle=load_bundle(ROOT/'.runtime/dmm_r4_best_val_20261015/package/model_bundle/upper/model.json','upper','cuda:0')
    net=bundle.model;box=np.asarray(bundle.metadata['sampling_domain_model'])
    codes=bundle.means if args.mode=='mean' else {k:torch.zeros_like(v) for k,v in bundle.means.items()}
    query=lambda x:net.query(x,codes)['sdf']
    metadata={'model_id':bundle.metadata['model_id'],'code_mode':args.mode,
              'latent_mean_norms':{str(k):float(v.norm()) for k,v in bundle.means.items()},'model_unit_mm':50}
else:
    source=Path('D:/WorkSpace/Dental/DMM');sys.path[:0]=[str(source),str(source/'third_party')]
    from networks.dmm_net import DMM
    exp=Path('D:/WorkSpace/Dental/experiments/DMM/20260821')
    specs=json.loads((exp/'specs.json').read_text())
    with contextlib.redirect_stdout(io.StringIO()):net=DMM(specs).cuda()
    checkpoint=exp/'ModelParameters/dmm_latest.pth'
    saved=torch.load(checkpoint,map_location='cuda:0',weights_only=True)
    net.load_state_dict(saved['model_state_dict'],strict=True);net.eval().requires_grad_(False)
    codes={k:torch.zeros(1,10,device='cuda:0') for k in specs['labels']}
    query=lambda x:net.inference(codes,x).reshape(-1)
    box=np.array([[-1,-1,-1],[1,1,1]])
    metadata={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'epoch':saved['epoch'],'code_mode':'raw_zero','model_unit_mm':None}
n=args.resolution;axes=[torch.linspace(float(box[0,i]),float(box[1,i]),n,device='cuda') for i in range(3)]
volume=np.empty(n**3,dtype=np.float32);last=time.monotonic()
with torch.no_grad():
    for begin in range(0,n**3,8192):
        ids=torch.arange(begin,min(begin+8192,n**3),device='cuda');x=torch.stack([axes[0][ids//(n*n)],axes[1][ids//n%n],axes[2][ids%n]],-1)
        volume[begin:begin+len(ids)]=query(x).cpu().numpy()
        if time.monotonic()-last>20:
            print(f'{args.model} {args.mode} N={n} {100*(begin+len(ids))/n**3:.1f}% elapsed {time.monotonic()-start:.1f}s',flush=True);last=time.monotonic()
volume=volume.reshape(n,n,n)
v,f,_,_=marching_cubes(volume,0,spacing=(box[1]-box[0])/(n-1),allow_degenerate=False);v+=box[0]
stem=f'{args.model}_{args.mode}_n{n}'
np.savez_compressed(OUT/f'{stem}.npz',vertices=v,faces=f,volume=volume,box=box)
with (OUT/f'{stem}.obj').open('w') as out:
    out.write('# Native model coordinates; marching cubes without smoothing\n')
    for row in v:out.write('v %.8f %.8f %.8f\n'%tuple(row))
    for row in f+1:out.write('f %d %d %d\n'%tuple(row))
boundary=np.concatenate([volume[0].ravel(),volume[-1].ravel(),volume[:,0].ravel(),volume[:,-1].ravel(),volume[:,:,0].ravel(),volume[:,:,-1].ravel()])
metadata.update(resolution=n,box=box.tolist(),grid_spacing=((box[1]-box[0])/(n-1)).tolist(),vertices=len(v),faces=len(f),seconds=time.monotonic()-start,
                boundary_min=float(boundary.min()),source=str(source),extraction='Full-domain direct decoder sampling; marching cubes, no smoothing or root refinement')
(OUT/f'{stem}.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print(json.dumps(metadata),flush=True)
