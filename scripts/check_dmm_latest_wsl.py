"""Bounded real WSL CUDA train/save/resume and configured point-budget checks."""
import argparse,gc,json,os,platform,resource,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party'/'DMM'))
import torch
from data.fixed_similarity_dataset import FixedSimilarityDataset
from training.fixed_similarity import configuration,dataset_for,validate_latest,train_latest,LatestSystem,runtime
from dmm.validation import write_json,require

def capacity(config,ids,device):
    cfg=configuration(config);cfg['case_ids']=ids;ds=dataset_for(cfg)
    torch.manual_seed(cfg['seed']);system=LatestSystem(ds,device)
    optimizer=torch.optim.Adam(system.parameters(),lr=cfg['learning_rate'])
    (batch,index)=ds[0];batch=[x[None].to(device) for x in batch];indices=torch.tensor([index],device=device)
    components=sorted(int(k) for k in torch.unique(batch[0][...,4]).cpu().tolist() if k!=-1)
    require(components==ds.specs['labels'],'configured-budget probe must include all components')
    torch.cuda.reset_peak_memory_stats(device);start=time.time()
    loss,terms=system(batch,indices);require(bool(torch.isfinite(loss)),'capacity loss is nonfinite')
    loss.backward();require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in system.parameters()),'capacity gradients nonfinite')
    before=system.latents['0'].weight.detach().clone();optimizer.step()
    require(not torch.equal(before,system.latents['0'].weight.detach()),'capacity optimizer did not update latent')
    require(all(bool(torch.isfinite(p).all()) for p in system.parameters()),'capacity parameters nonfinite')
    return dict(samples_per_scene=cfg['samples_per_scene'],batch_size=1,space_points=int((batch[1]==-1).sum()),loss=float(loss.detach()),
                components=components,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),seconds=time.time()-start,optimizer_update_verified=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=True);require(not (out/'WSL_VALIDATION.json').exists(),'validation report exists')
    require(platform.system()=='Linux' and 'microsoft' in platform.release().lower(),'this check requires actual WSL')
    require(torch.cuda.is_available(),'actual CUDA required');torch.set_num_threads(4)
    results={};start=time.time()
    write_json(out/'runtime_wsl.json',runtime('cuda'))
    for arch in ('upper','lower'):
        config=ROOT/'configs'/'dmm_latest_v2'/(arch+'.json')
        inputs=validate_latest(config,cohort='all_exported');write_json(out/(arch+'_wsl_input_validation.json'),inputs)
        cfg=configuration(config);ds=dataset_for(cfg);lookup={e['case_id']:e for e in ds.rows}
        original=next(e['case_id'] for e in ds.rows if not e.get('parent_id') and len(e['centers'])==14 and e['case_id']+'__mirror' in lookup)
        ids=[original,original+'__mirror']
        del ds
        first=train_latest(config,out/(arch+'_cuda_first'),'cuda',max_steps=1,epochs=1,case_ids=ids,samples_per_scene=512)
        require(first['global_step']==1 and first['next_batch']==1,'first step position mismatch')
        require(all(v>0 for v in first['updated_parameter_tensors'].values()),'model or latents did not update')
        gc.collect();torch.cuda.empty_cache()
        resumed=train_latest(config,out/(arch+'_cuda_resumed'),'cuda',resume=out/(arch+'_cuda_first')/'latest.pth',epochs=1,case_ids=ids,samples_per_scene=512)
        require(resumed['global_step']==2 and resumed['completed_epochs']==1 and resumed['optimizer_steps_this_run']==1,'resume failed')
        require(all(v>0 for v in resumed['updated_parameter_tensors'].values()),'resumed optimizer did not update model and latents')
        checkpoint=torch.load(out/(arch+'_cuda_resumed')/'final.pth',map_location='cpu',weights_only=True)
        require(checkpoint['case_ids']==sorted(ids) and checkpoint['model_unit_mm'] is None,'checkpoint coordinate or latent-row binding lost')
        require(checkpoint['per_case_inverse_required'] is True,'per-case inverse lost')
        require(len(checkpoint['training_rows'])==2 and all('model_to_source_mm' in r and 'presence' in r for r in checkpoint['training_rows']),'checkpoint per-case units/presence missing')
        del checkpoint;gc.collect();torch.cuda.empty_cache()
        probe=capacity(config,ids,'cuda');gc.collect();torch.cuda.empty_cache()
        results[arch]=dict(case_ids=ids,first=first,resumed=resumed,configured_budget=probe)
        write_json(out/'WSL_VALIDATION_PARTIAL.json',dict(arches=results))
        print(json.dumps(dict(stage='WSL_ARCH_PASS',arch=arch,capacity=probe)),flush=True)
    report=dict(status='PASS_WSL_CUDA_LOAD_TRAIN_SAVE_RESUME',runtime=runtime('cuda'),arches=results,
        elapsed_seconds=time.time()-start,peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        full_training_started=False,formal_quality_accepted=False,scope='two real cases per arch, 512 point optimizer/save/reload/resume; extra configured 4096 point optimizer step per arch; no original data/old weights modified')
    write_json(out/'WSL_VALIDATION.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
