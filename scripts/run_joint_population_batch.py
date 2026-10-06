"""Leave-source-out population prior, fixed unit weight; no gate tuning."""
import concurrent.futures as cf
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from run_forward_check import sha256,PROJECT_ROOT as ROOT

def read(p): return json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,indent=2))
def main():
    source=Path(sys.argv[1]).resolve(strict=True)
    root=ROOT/'runs'/('joint_population_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')); root.mkdir()
    checkpoint=Path('D:/WorkSpace/Dental/experiments/DMM/20260824/LatentCodes/latent_vecs_best.pth')
    source_provenance=read(ROOT/'runs/forward_20260929T092217Z_b45f45ab/provenance.json')
    if sha256(checkpoint)!=source_provenance['input_sha256'][str(checkpoint)]: raise ValueError('latent checkpoint changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['epoch']!=295: raise ValueError('wrong epoch')
    split=Path('D:/WorkSpace/Dental/data/Teeth3DS_DMM/05_dataset_profiles/upper_v1.3_strict_fixed_offsurface/splits/train_split.json')
    if sha256(split)!=source_provenance['input_sha256'][str(split)]: raise ValueError('training split changed')
    names=read(split)
    excluded=[i for i,n in enumerate(names) if n.replace('__mirror','')==source_provenance['case']]
    if excluded!=[0,1]: raise ValueError('unexpected source case mapping')
    population={}; details={}
    for label in [11,21]:
        key=f'label_{label}'; matrix=saved['latent_codes'][f'{label}.module.weight'].numpy().astype(np.float64)
        if len(matrix)!=len(names): raise ValueError('training row map differs')
        matrix=np.delete(matrix,excluded,axis=0)
        mean=matrix.mean(0); cov=np.cov(matrix,rowvar=False); scale=np.sqrt(np.diag(cov))
        corr=cov/np.outer(scale,scale)
        # Fixed 5% diagonal shrinkage controls sampling instability, independent of GT errors.
        shrink=.95*corr+.05*np.eye(len(mean)); eig,U=np.linalg.eigh(shrink)
        W=(U/np.sqrt(eig)).T/scale[None,:]
        population[key+'_mean']=mean; population[key+'_whitener']=W
        details[key]={'rows':len(matrix),'excluded_rows':excluded,'excluded_cases':[names[i] for i in excluded],'split_sha256':sha256(split),'shrinkage':.05,'correlation_eigen_min':float(eig.min())}
    jobs=[]
    for name in ['clean_block','seed202_block','erode_band','dilate_band']:
        old=source/'jobs'/name; job=root/'jobs'/name; job.mkdir(parents=True); shutil.copytree(old/'case',job/'case')
        fit=job/'case/fit_input'; manifest=read(fit/'manifest.json'); manifest['warm_start_manifest_sha256']=sha256(fit/'manifest.json')
        np.savez_compressed(fit/'population_prior.npz',**population)
        write(fit/'population_prior_provenance.json',{'checkpoint_sha256':sha256(checkpoint),'epoch':295,'aggregate_only':True,'source_case_excluded':True,'details':details,'weight':1})
        for n in ['population_prior.npz','population_prior_provenance.json']: manifest['extra_input_sha256'][n]=sha256(fit/n)
        write(fit/'manifest.json',manifest)
        config=read(old/'config.json'); config['fit_input']=str(fit); config['initial_code_damping']=0
        config['robustness']['axis']['prior']='leave-source-out population Mahalanobis, unit weight'
        write(job/'config.json',config); jobs.append(job)
    snapshot=root/'code_snapshot'; snapshot.mkdir()
    for p in (ROOT/'scripts').glob('*.py'): shutil.copyfile(p,snapshot/p.name)
    write(root/'protocol.json',{'status':'PREPARED','jobs':[p.name for p in jobs],'prior':'leave-source-out population; unit weight; 5% covariance shrinkage; no truth selection','selection':'full-contour bidirectional boundary-band score plus prior; all observed views','original_acceptance_unchanged':True,'budget':'24 outer iterations, 24 nfev per block, patience 8'})
    print(str(root),flush=True)
    def run(job):
        with (job/'process.log').open('w') as log:
            code=subprocess.call([sys.executable,str(ROOT/'scripts/run_joint_population.py'),'--config',str(job/'config.json')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            if code==0: code=subprocess.call([sys.executable,str(ROOT/'scripts/evaluate_joint_robust.py'),str(job/'fit_attempt_01')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        state={'job':job.name,'exit_code':code}; p=job/'fit_attempt_01/evaluation.json'
        if p.exists(): state['evaluation']=read(p)
        write(job/'completion.json',state); print(job.name,code,flush=True); return state
    with cf.ThreadPoolExecutor(max_workers=2) as executor: results=list(executor.map(run,jobs))
    write(root/'summary.json',{'status':'COMPLETE','jobs':results})
if __name__=='__main__': main()
