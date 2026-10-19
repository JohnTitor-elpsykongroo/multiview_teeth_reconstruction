"""Verify the actual local DMM contract without changing datasets or weights."""
import contextlib, io, json, sys, hashlib, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(r'D:\WorkSpace\Dental\DMM')
DATA=Path(r'D:\WorkSpace\Dental\Teeth3DS_DualArch_Preprocess\output\latest\data')
OUT=Path(r'D:\WorkSpace\Dental\multiview_teeth_reconstruction\runs\space_sampling_audit_20261018_01')
sys.path[:0]=[str(ROOT),str(ROOT/'third_party')]
from data.data_with_labels import SDFSamples
from networks.dmm_net import DMM
from networks.loss import overall_sdf_normal_loss

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

torch.set_num_threads(3)
results=[];start=time.time()
for arch in ['upper','lower']:
    torch.manual_seed(20261018);np.random.seed(20261018)
    folder=DATA/arch;split=read(folder/'splits'/'train_computational_pass.json')[:1]
    dataset=SDFSamples(str(folder),split=split,subsample=512,load_ram=True)
    (batch,indices)=next(iter(torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=0)))
    x,mask,normals,centers=batch
    specs=read(folder/'specs.json');avg=torch.from_numpy(np.loadtxt(folder/'avg_centroids.txt',dtype=np.float32))
    with contextlib.redirect_stdout(io.StringIO()):network=DMM(specs,avg)
    codes=torch.nn.ModuleDict()
    for k in specs['labels']:
        key='GumDeformNetworkSpecs' if k==0 else 'TeethDeformNetworkSpecs'
        emb=torch.nn.Embedding(1,specs[key]['latent_dim']);torch.nn.init.normal_(emb.weight,std=.002);codes[str(k)]=torch.nn.DataParallel(emb)
    params_before=[p.detach().clone() for p in network.parameters()]
    values=[]
    for variant in ['all_zero','opposite_slot_signs','reversed_slot_signs']:
        sample=x.clone();off=(mask[...,0]==-1)
        if variant!='all_zero':
            offsets=torch.nonzero(off[0],as_tuple=True)[0];half=len(offsets)//2
            sample[0,offsets[:half],3]=17 if variant=='opposite_slot_signs' else -17
            sample[0,offsets[half:],3]=-17 if variant=='opposite_slot_signs' else 17
        loss,terms=network(codes,sample,mask,normals,centers,indices)
        values.append(dict(variant=variant,total=float(loss.detach()),terms={k:float(v.detach() if hasattr(v,'detach') else v) for k,v in terms.items()}))
        del loss
    delta=max(abs(v['total']-values[0]['total']) for v in values)
    assert delta==0
    assert all(torch.equal(p.detach(),q) for p,q in zip(network.parameters(),params_before))
    results.append(dict(arch=arch,filename=dataset.npyfiles[0],surface_count=int((mask==0).sum()),space_count=int((mask==-1).sum()),space_label_values=torch.unique(x[...,4][mask[...,0]==-1]).tolist(),max_total_loss_difference=delta,values=values,network_parameters_unchanged=True))

coords=torch.tensor([[[.1,.2,.3],[.4,-.2,.1]]],dtype=torch.float32,requires_grad=True)
pred=(coords**2).sum(-1,keepdim=True)+.2
mask=torch.full((1,2,1),-1.);normals=torch.full((1,2,3),-1.)
pos=overall_sdf_normal_loss(pred,coords,mask,normals)
neg=overall_sdf_normal_loss(-pred,coords,mask,normals)
symmetry={k:dict(positive=float(pos[k].detach()),negative=float(neg[k].detach())) for k in pos}
assert all(torch.equal(pos[k],neg[k]) for k in pos)
sources={str(p):digest(p) for p in [ROOT/'train_dmm.py',ROOT/'data/data_with_labels.py',ROOT/'networks/dmm_net.py',ROOT/'networks/loss.py']}
report=dict(scope='actual local train_dmm.py RAM-loader path; two randomly initialized official DMM networks, same coordinates/labels/mask/weights, only finite fourth-column SDF values changed; no optimizer/training',arches=results,offsurface_only_loss_sign_symmetry=symmetry,source_sha256=sources,elapsed_seconds=time.time()-start)
(OUT/'dmm_space_contract.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
