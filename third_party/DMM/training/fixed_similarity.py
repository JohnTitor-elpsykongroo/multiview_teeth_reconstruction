"""Training adapter for fixed-reference similarity exports, using DMM.forward losses.

This checkpoint family deliberately cannot masquerade as the legacy 50 mm bundle.
No data conversion, surface clipping, loss reweighting or inside/outside relabeling.
"""
import hashlib
import json
from pathlib import Path
import platform
import os
import shutil
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from data.fixed_similarity_dataset import FixedSimilarityDataset, portable_path
from dmm.provenance import source_fingerprint
from dmm.validation import read_json, require, write_json

FORMAT='dmm_fixed_similarity_training/1'


def configuration(path):
    path=Path(path).resolve();cfg=read_json(path)
    allowed={'schema','dataset_root','arch','cohort','samples_per_scene','epochs','batch_size','learning_rate',
             'seed','checkpoint_every','checkpoint_every_steps','cache_cases','num_workers','case_ids'}
    require(cfg.get('schema')==FORMAT and not set(cfg)-allowed, 'expected fixed-similarity training config')
    defaults=dict(cohort='computational_pass',samples_per_scene=4096,epochs=100,batch_size=1,learning_rate=.0001,
                  seed=20261018,checkpoint_every=10,checkpoint_every_steps=100,cache_cases=1,num_workers=0,case_ids=None)
    cfg=dict(defaults,**cfg)
    for key in ('epochs','batch_size','checkpoint_every','checkpoint_every_steps','samples_per_scene'):
        require(type(cfg[key]) is int and cfg[key]>0, f'invalid {key}')
    for key in ('seed','cache_cases','num_workers'):
        require(type(cfg[key]) is int and cfg[key]>=0, f'invalid {key}')
    require(type(cfg['learning_rate']) in (float,int) and np.isfinite(cfg['learning_rate']) and cfg['learning_rate']>0, 'invalid learning rate')
    cfg['dataset_root']=str(portable_path(cfg['dataset_root'],path))
    return cfg


def dataset_for(cfg,split='train',case_ids=None):
    return FixedSimilarityDataset(cfg['dataset_root'],cfg['arch'],split,cfg['cohort'],cfg['samples_per_scene'],
                                  cfg['seed'],cfg['cache_cases'],cfg['case_ids'] if case_ids is None and split=='train' else case_ids)


def validate_latest(config_path,verify_arrays=False,cohort=None):
    cfg=configuration(config_path);reports={}
    if cohort is not None:cfg['cohort']=cohort
    for split in ('train','val','test'):
        ds=dataset_for(cfg,split)
        if verify_arrays:
            for i in range(len(ds)):
                ds._arrays(i)
                if (i+1)%100==0:print(json.dumps(dict(stage='verify_arrays',arch=cfg['arch'],split=split,done=i+1,total=len(ds))),flush=True)
        reports[split]=dict(ds.summary(),array_verification='all selected files' if verify_arrays else 'deferred to authenticated on-demand load')
    return dict(status='LATEST_INPUTS_VALIDATED',splits=reports,full_training_started=False)


class LatestSystem(nn.Module):
    def __init__(self,dataset,device):
        super().__init__()
        from networks.dmm_net import DMM
        avg=torch.from_numpy(dataset.avg_centers.copy()).to(device)
        self.model=DMM(dataset.specs,avg_centers=avg,arch=dataset.arch)
        self.latents=nn.ModuleDict({str(k):nn.Embedding(len(dataset),dim) for k,dim in self.model.dimensions.items()})
        for embedding in self.latents.values():nn.init.normal_(embedding.weight,std=.002)
        self.to(device)

    def forward(self,batch,indices):return self.model(self.latents,*batch,indices)


def atomic_checkpoint(path,payload):
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def checkpoint_alias(source,target):
    """Identical best/latest/final states share storage; future writes replace files atomically."""
    temp=target.with_suffix('.link.tmp')
    try:os.link(source,temp)
    except OSError:shutil.copyfile(source,temp)
    temp.replace(target)


def runtime(device):
    gpu=str(device).startswith('cuda')
    return dict(python=platform.python_version(),system=platform.system(),torch=str(torch.__version__),
                numpy=np.__version__,cuda=torch.version.cuda if gpu else None,
                device_type='cuda' if gpu else 'cpu',gpu=torch.cuda.get_device_name(device) if gpu else None,
                capability=list(torch.cuda.get_device_capability(device)) if gpu else None,
                matmul_precision=torch.get_float32_matmul_precision(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),torch_threads=torch.get_num_threads())


def train_latest(config_path,output_dir,device='cuda',resume=None,max_steps=None,epochs=None,case_ids=None,samples_per_scene=None,cohort=None):
    output=Path(output_dir).resolve();require(not output.exists(),'training output must be a new directory')
    cfg=configuration(config_path)
    if epochs is not None:cfg['epochs']=epochs
    if case_ids is not None:cfg['case_ids']=case_ids
    if samples_per_scene is not None:cfg['samples_per_scene']=samples_per_scene
    if cohort is not None:cfg['cohort']=cohort
    require(type(cfg['epochs']) is int and cfg['epochs']>0,'invalid epoch override')
    require(max_steps is None or (type(max_steps) is int and max_steps>0),'max_steps must be positive')
    ds=dataset_for(cfg);ds.assert_current()
    require(device=='cpu' or str(device).startswith('cuda'),'device must be cpu or cuda')
    if str(device).startswith('cuda'):require(torch.cuda.is_available(),'CUDA unavailable; CPU is not a CUDA validation')
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']%2**32)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(cfg['seed']);torch.cuda.reset_peak_memory_stats(device)
    system=LatestSystem(ds,device)
    optimizer=torch.optim.Adam(system.parameters(),lr=cfg['learning_rate'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=.5)
    source=source_fingerprint();env=runtime(device)
    signed_config={k:v for k,v in cfg.items() if k not in ('epochs','checkpoint_every','checkpoint_every_steps','dataset_root','num_workers','cache_cases')}
    signature=hashlib.sha256(json.dumps(dict(config=signed_config,binding=ds.binding,source=source),sort_keys=True).encode()).hexdigest()
    epoch=cursor=global_step=0;epoch_sum=0.;epoch_batches=0;best=float('inf');previous_updates=0
    if resume:
        # Restore through CPU so Adam's scalar step counters stay on CPU while
        # load_state_dict moves parameter moments to the selected device.
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        require(saved.get('format')==FORMAT,'incompatible checkpoint: old 50mm/embedding checkpoints cannot resume this dataset')
        require(saved.get('training_signature')==signature,'resume source/data/cohort/case order/sampling/optimizer mismatch')
        require(saved.get('runtime')==env,'resume must use the same numerical runtime/device')
        system.load_state_dict(saved['system_state'],strict=True);optimizer.load_state_dict(saved['optimizer_state']);scheduler.load_state_dict(saved['scheduler_state'])
        epoch,cursor,global_step=saved['next_epoch'],saved['next_batch'],saved['global_step']
        epoch_sum,epoch_batches,best=saved['epoch_loss_sum'],saved['epoch_batches'],saved['best_train_loss']
        torch.set_rng_state(saved['rng_state'].cpu())
        if str(device).startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng_state']])
        previous_updates=global_step
        del saved
    require(epoch<cfg['epochs'],'no remaining epochs')
    output.mkdir(parents=True)
    write_json(output/'config.json',cfg);write_json(output/'runtime.json',env)
    # Keep the small executable source alongside checkpoints, including local
    # uncommitted changes. No dataset or environment is copied into a run.
    from dmm import DMM_ROOT
    sources=[DMM_ROOT/'train_dmm.py',DMM_ROOT/'dmm_cli.py']
    for module in ('dmm','networks','training','data','utils','third_party/torchmeta'):
        sources.extend((DMM_ROOT/module).rglob('*.py'))
    if (DMM_ROOT/'UPSTREAM.json').is_file():sources.append(DMM_ROOT/'UPSTREAM.json')
    for source_file in sources:
        target=output/'source_snapshot'/source_file.relative_to(DMM_ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source_file,target)
    binding=dict(dataset=ds.summary(),manifest_files=ds.dataset_files,
                 training_rows=[dict(case_id=r['case_id'],embedding_row=r['embedding_row'],patient_id=r['patient_id'],split=r['split'],parent_id=r.get('parent_id'),qc_status=r['qc_status'],
                    presence={str(k):str(k) in r['centers'] for k in ds.specs['labels'] if k},
                    samples_sha256=r['samples_sha256'],centers_sha256=r['centers_sha256'],source_mm_to_model=r['source_mm_to_model'],model_to_source_mm=r['model_to_source_mm']) for r in ds.rows],
                 reference=ds.reference,model_unit_mm=None,per_case_inverse_required=True,source_sha256=source,training_signature=signature,
                 resume=str(Path(resume).resolve()) if resume else None)
    write_json(output/'data_binding.json',binding)
    started=time.time();initial={n:p.detach().clone() for n,p in system.named_parameters()};step_limit=False;last_saved=None
    def payload():
        value=dict(format=FORMAT,implementation_version='fixed_similarity_npz_v2',arch=ds.arch,specs=ds.specs,
            training_signature=signature,dataset_binding=ds.binding,source_sha256=source,runtime=env,
            next_epoch=epoch,next_batch=cursor,global_step=global_step,epoch_loss_sum=epoch_sum,epoch_batches=epoch_batches,best_train_loss=best,
            system_state=system.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
            rng_state=torch.get_rng_state(),case_ids=[r['case_id'] for r in ds.rows],training_rows=binding['training_rows'],coordinate_contract=binding['dataset']['coordinate_contract'],
            model_unit_mm=None,per_case_inverse_required=True,resolved_config=cfg,reference=ds.reference,
            avg_centers=torch.from_numpy(ds.avg_centers.copy()),formal_quality_accepted=False)
        if str(device).startswith('cuda'):value['cuda_rng_state']=torch.cuda.get_rng_state_all()
        return value
    try:
        while epoch<cfg['epochs']:
            ds.assert_current();ds.set_epoch(epoch)
            order=np.random.default_rng(cfg['seed']+epoch).permutation(len(ds)).tolist()
            batches=(len(order)+cfg['batch_size']-1)//cfg['batch_size']
            require(0<=cursor<batches,'invalid resume position')
            remaining=order[cursor*cfg['batch_size']:]
            generator=torch.Generator().manual_seed(cfg['seed']+epoch)
            loader=DataLoader(Subset(ds,remaining),batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers'],generator=generator)
            for batch,indices in loader:
                step_start=time.time();batch=[x.to(device) for x in batch];indices=indices.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss,terms=system(batch,indices)
                require(bool(torch.isfinite(loss)),'nonfinite training loss')
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in system.parameters()),'nonfinite training gradient')
                # No clipping or loss changes are introduced by the data adapter.
                optimizer.step()
                require(all(bool(torch.isfinite(p).all()) for p in system.parameters()),'nonfinite optimizer parameters')
                value=float(loss.detach());global_step+=1;cursor+=1;epoch_sum+=value;epoch_batches+=1
                record=dict(epoch=epoch+1,batch=cursor,global_step=global_step,case_ids=[ds.rows[int(i)]['case_id'] for i in indices.cpu()],
                    loss=value,terms={k:float(v.detach() if hasattr(v,'detach') else v) for k,v in terms.items()},seconds=time.time()-step_start)
                with (output/'steps.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
                print(json.dumps(record,allow_nan=False),flush=True)
                del loss
                complete=cursor==batches
                if complete:
                    mean=epoch_sum/epoch_batches;improved=mean<best;best=min(best,mean)
                    epoch+=1;cursor=0;epoch_sum=0.;epoch_batches=0;scheduler.step()
                    with (output/'epochs.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(epoch=epoch,mean_train_loss=mean,best_train_loss=best))+'\n')
                    atomic_checkpoint(output/'latest.pth',payload())
                    last_saved=(global_step,epoch,cursor)
                    if improved:checkpoint_alias(output/'latest.pth',output/'best_train.pth')
                    if epoch%cfg['checkpoint_every']==0:checkpoint_alias(output/'latest.pth',output/f'epoch_{epoch:06d}.pth')
                elif global_step%cfg['checkpoint_every_steps']==0:
                    atomic_checkpoint(output/'latest.pth',payload())
                    last_saved=(global_step,epoch,cursor)
                if max_steps is not None and global_step-previous_updates>=max_steps:
                    step_limit=True;break
            if step_limit:break
        ds.assert_current()
        if last_saved!=(global_step,epoch,cursor):atomic_checkpoint(output/'latest.pth',payload())
        checkpoint_alias(output/'latest.pth',output/'final.pth')
        changes={group:sum(not torch.equal(p.detach(),initial[n]) for n,p in system.named_parameters() if n.startswith(group+'.')) for group in ('model','latents')}
        report=dict(status='TRAINING_COMPLETED_NOT_QUALITY_ACCEPTED' if epoch==cfg['epochs'] else 'BOUNDED_TRAINING_CHECK_COMPLETED',arch=ds.arch,
            completed_epochs=epoch,next_batch=cursor,global_step=global_step,optimizer_steps_this_run=global_step-previous_updates,
            updated_parameter_tensors=changes,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
            dataset=ds.summary(),elapsed_seconds=time.time()-started,training_signature=signature,formal_quality_accepted=False,
            validation_policy='No held-out patient enters optimizer; best_train is training loss, not validation/quality acceptance')
        write_json(output/'report.json',report);return report
    except BaseException as exc:
        write_json(output/'failure.json',dict(error=str(exc),type=type(exc).__name__,global_step=global_step,next_epoch=epoch,next_batch=cursor))
        raise
