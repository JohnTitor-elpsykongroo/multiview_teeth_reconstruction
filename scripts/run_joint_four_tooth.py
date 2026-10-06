"""One preregistered four-tooth conditioning diagnosis, no population expansion claim."""
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from run_forward_check import PROJECT_ROOT as ROOT,sha256
def read(p): return json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,indent=2))
def call(script,args,log):
    result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,capture_output=True,text=True)
    log.write(result.stdout+result.stderr); log.flush()
    if result.returncode: raise RuntimeError(f'{script} failed; see process.log')
    return result.stdout
def main():
    root=ROOT/'runs'/dt.datetime.now(dt.timezone.utc).strftime('joint_four_tooth_%Y%m%dT%H%M%SZ'); root.mkdir()
    config=read(ROOT/'configs/joint_warmup.json'); config['active_labels']=[11,12,21,22]
    config.update(outer_iterations=24,stop_after_no_best_iterations=8)
    write(root/'generator_config.json',config)
    snapshot=root/'code_snapshot'; snapshot.mkdir()
    for p in (ROOT/'scripts').glob('*.py'): shutil.copyfile(p,snapshot/p.name)
    write(root/'protocol.json',{'status':'PREPARED','labels':[11,12,21,22],'seed':101,'perturbation_sigma':.25,'known_cameras':'all three original fitting cameras','heldout':'65 degrees, post-fit only','scope':'one clean four-tooth conditioning diagnosis; two-tooth failures retained; no large batch','acceptance':config['acceptance'],'method':'alternating blocks, leave-patient-out population prior, zero boundary band; 24 outer, 24 nfev, patience 8'})
    print(str(root),flush=True)
    try:
        with (root/'process.log').open('w') as log:
            output=call('prepare_joint.py',['--config',root/'generator_config.json'],log)
            generated=Path(json.loads(output.strip().splitlines()[-1])['prepared_root'])
            job=root/'jobs/four_clean'; job.mkdir(parents=True); shutil.copytree(generated,job/'case'); fit=job/'case/fit_input'; truth=job/'case/truth'
            manifest=read(fit/'manifest.json'); manifest['observation_type']='hard FDI labels; four active teeth; no gum or lip occluder'; write(fit/'manifest.json',manifest)
            (truth/'clean_fit/masks').mkdir(parents=True); shutil.copyfile(fit/manifest['camera_file'],truth/'clean_fit/cameras.json')
            masks=[]
            for m in manifest['masks']:
                target='clean_fit/masks/'+Path(m['path']).name; shutil.copyfile(fit/m['path'],truth/target); masks.append(dict(m,path=target))
            write(truth/'clean_fit_manifest.json',{'camera_file':'clean_fit/cameras.json','camera_sha256':manifest['camera_sha256'],'masks':masks})
            warm=root/'warmup'; warm.mkdir(); shutil.copytree(fit,warm/'fit_input')
            config.update(fit_input=str(warm/'fit_input'),robustness={'axis':{'id':'four_clean','kind':'conditioning_diagnosis'}})
            write(warm/'config.json',config); call('run_pose_warmup_only.py',['--config',warm/'config.json'],log)
            manifest['warm_start_manifest_sha256']=sha256(fit/'manifest.json')
            checkpoint=Path(config['experiment'])/'LatentCodes/latent_vecs_best.pth'; forward=read(Path(config['source_run'])/'provenance.json')
            split=Path('D:/WorkSpace/Dental/data/Teeth3DS_DMM/05_dataset_profiles/upper_v1.3_strict_fixed_offsurface/splits/train_split.json')
            if sha256(checkpoint)!=forward['input_sha256'][str(checkpoint)] or sha256(split)!=forward['input_sha256'][str(split)]: raise ValueError('training inputs changed')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True); names=read(split); excluded=[i for i,n in enumerate(names) if n.replace('__mirror','')==forward['case']]
            if excluded!=[0,1] or saved['epoch']!=295: raise ValueError('row mapping or epoch differs')
            population={}; details={}
            for label in config['active_labels']:
                key=f'label_{label}'; matrix=saved['latent_codes'][f'{label}.module.weight'].numpy().astype(float)
                if len(matrix)!=len(names): raise ValueError('training rows differ')
                matrix=np.delete(matrix,excluded,0); mean=matrix.mean(0); cov=np.cov(matrix,rowvar=False); scale=np.sqrt(np.diag(cov))
                eig,U=np.linalg.eigh(.95*cov/np.outer(scale,scale)+.05*np.eye(len(mean)))
                population[key+'_mean']=mean; population[key+'_whitener']=(U/np.sqrt(eig)).T/scale[None,:]
                details[key]={'rows':len(matrix),'excluded_rows':excluded,'excluded_cases':[names[i] for i in excluded],'shrinkage':.05}
            np.savez_compressed(fit/'population_prior.npz',**population)
            write(fit/'population_prior_provenance.json',{'checkpoint_sha256':sha256(checkpoint),'split_sha256':sha256(split),'epoch':295,'aggregate_only':True,'source_case_excluded':True,'details':details,'weight':1})
            for n in ['population_prior.npz','population_prior_provenance.json']: manifest['extra_input_sha256'][n]=sha256(fit/n)
            write(fit/'manifest.json',manifest)
            config.update(fit_input=str(fit),warm_start_run=str(warm/'fit_attempt_01'),initial_code_damping=0,stabilization={'band_radius_pixels':0,'band_transition_pixels':.25})
            write(job/'config.json',config); call('run_joint_population.py',['--config',job/'config.json'],log)
            call('evaluate_joint_robust.py',[job/'fit_attempt_01'],log)
            write(root/'summary.json',{'status':'COMPLETE','evaluation':read(job/'fit_attempt_01/evaluation.json')})
    except Exception as error:
        write(root/'failure.json',{'status':'FOUR_TOOTH_DIAGNOSIS_FAILED','error':str(error)}); raise
if __name__=='__main__': main()
