"""Bounded real-record CUDA training preflight; never a model quality certificate."""
import argparse
import copy
import json
import platform
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'third_party/DMM'))
import torch
from data.arch_dataset import ArchDataset
from networks.dmm_net import DMM
from training.arch_training import ArchTrainingSystem
from training.recipe import resolve_recipe
from dmm.validation import read_json, resolve_ref, write_json, require, sha256
from dmm.provenance import source_fingerprint


def single_case(config_path, quota=16, offsurface=32):
    path=Path(config_path).resolve();config=read_json(path)
    manifest_path=resolve_ref(path,config['manifest']);manifest=read_json(manifest_path)
    ids=config.get('training_case_ids')
    row=next(r for r in manifest['cases'] if r['split']=='train' and r['augmentation_parent_id'] is None and (ids is None or r['case_id'] in ids))
    data=ArchDataset.__new__(ArchDataset)
    data.arch=config['arch'];data.manifest_path=manifest_path
    data.points_per_component=quota;data.gum_points=quota;data.offsurface_points=offsurface
    data.seed=config['seed'];data.epoch=0;data.cases=manifest['cases']
    data.selected=[data._load(row)]
    centers={int(k):v for k,v in read_json(resolve_ref(path,config['canonical_reference']))['centers_model'].items()}
    specs=read_json(resolve_ref(path,config['specs']))
    return config,data,centers,specs


def run(config_path, steps=20, quota=16, offsurface=32, device='cuda', override=None):
    config,data,centers,specs=single_case(config_path,quota,offsurface)
    recipe=resolve_recipe(config)
    if override:
        specs=copy.deepcopy(specs)
        specs['NetworkSpecsRef']['output_initialization']=override['output_initialization']
        recipe['normal_epsilon']=override['normal_epsilon']
    torch.manual_seed(config['seed']);torch.set_num_threads(2)
    start=time.perf_counter()
    system=ArchTrainingSystem(DMM(specs,arch=config['arch']),data,centers,recipe).to(device)
    groups=[dict(params=list(system.model.deform_nets_dict.parameters()),lr=recipe['deformation_lr'],name='deformation'),
            dict(params=list(system.model.ref_nets_dict.parameters()),lr=recipe['reference_lr'],name='reference'),
            dict(params=list(system.latents.parameters()),lr=recipe['latent_lr'],name='latent')]
    optimizer=torch.optim.Adam(groups)
    sample=data[0]
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    history=[]
    for step in range(steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss,terms=system.loss(sample)
        require(bool(torch.isfinite(loss)),'nonfinite preflight loss')
        loss.backward()
        norms={}
        for group in groups:
            grads=[p.grad for p in group['params'] if p.grad is not None]
            require(bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads),'nonfinite/missing preflight gradient')
            norms[group['name']]=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
        record=dict(step=step,loss=float(loss.detach()),gradient_norms=norms,terms=terms)
        history.append(record)
        if step==steps:break
        if recipe['gradient_clip_norm']:
            torch.nn.utils.clip_grad_norm_(system.parameters(),recipe['gradient_clip_norm'],error_if_nonfinite=True)
        before=[p.detach().clone() for p in system.parameters()]
        optimizer.step()
        record['max_absolute_update']=max(float((p-old).detach().abs().max()) for p,old in zip(system.parameters(),before))
        require(all(bool(torch.isfinite(p).all()) for p in system.parameters()),'nonfinite parameters after Adam')
    return dict(status='REAL_CASE_OPTIMIZER_SMOKE_NOT_QUALITY_ACCEPTED',arch=config['arch'],case_id=sample['case_id'],
                config_sha256=sha256(config_path),sample_sha256=sha256(data.selected[0].sample_path),
                source_sha256=source_fingerprint(),recipe=recipe,specs=specs,steps=steps,history=history,
                seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
                loss_ratio=history[-1]['loss']/history[0]['loss'],gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None,
                torch_version=str(torch.__version__),platform=platform.platform(),
                scope='one real original train record; fixed sampled points; temporary in-memory Adam; no checkpoint/quality claim')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=20);parser.add_argument('--points',type=int,default=16)
    parser.add_argument('--offsurface',type=int,default=32);parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    require(args.steps>0 and args.points>0 and args.offsurface>0,'positive smoke budgets required')
    output=Path(args.output);require(not output.exists(),'smoke output already exists')
    result=run(args.config,args.steps,args.points,args.offsurface,args.device)
    write_json(output,result)
    print(json.dumps({k:result[k] for k in ('status','arch','case_id','loss_ratio','seconds','peak_allocated_bytes')}))
