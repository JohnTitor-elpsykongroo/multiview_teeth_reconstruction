"""Frozen-input ablations, two CPU workers, durable logs and post-fit evaluation."""
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,indent=2))
def main():
    batch=ROOT/'runs'/('joint_stabilization_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    batch.mkdir(exist_ok=False)
    pilot=ROOT/'runs/joint_robustness_20261004T044624Z_1d578127'
    clean=ROOT/'runs/joint_case_20260930T101534Z_ec4c7cfe'
    control=ROOT/'runs/joint_warmup_20261004T043110Z_ddd095ce'
    matrix=[('clean_block',None,0),('seed202_block','init_seed202',0),('erode_block','mask_erode1',0),('erode_band','mask_erode1',2**.5),('dilate_band','mask_dilate1',2**.5)]
    jobs=[]
    for name,axis,radius in matrix:
        job=batch/'jobs'/name; job.mkdir(parents=True)
        source=pilot/'jobs'/axis/'case' if axis else clean
        warm=pilot/'jobs'/axis/'fit_attempt_01' if axis else control
        shutil.copytree(source,job/'case')
        if not axis:
            truth=job/'case/truth'; fit=job/'case/fit_input'; manifest=read(fit/'manifest.json')
            (truth/'clean_fit/masks').mkdir(parents=True)
            shutil.copyfile(fit/manifest['camera_file'],truth/'clean_fit/cameras.json')
            masks=[]
            for m in manifest['masks']:
                path='clean_fit/masks/'+Path(m['path']).name
                shutil.copyfile(fit/m['path'],truth/path)
                masks.append(dict(m,path=path))
            write(truth/'clean_fit_manifest.json',{'camera_file':'clean_fit/cameras.json','camera_sha256':manifest['camera_sha256'],'masks':masks})
        config=read(warm/'resolved_config.json')
        config.update(fit_input=str(job/'case/fit_input'),warm_start_run=str(warm),outer_iterations=24,stop_after_no_best_iterations=8)
        config['stabilization']={'band_radius_pixels':radius,'band_transition_pixels':.25,'schedule':'shape then pose with refreshed correspondences'}
        config['robustness']=dict(config.get('robustness',{}),axis={'id':name,'input_axis':axis or 'clean','band_radius_pixels':radius})
        write(job/'config.json',config); jobs.append(job)
    snapshot=batch/'code_snapshot'; snapshot.mkdir()
    for p in (ROOT/'scripts').glob('*.py'): shutil.copyfile(p,snapshot/p.name)
    write(batch/'protocol.json',{'status':'PREPARED','jobs':[p.name for p in jobs],'max_workers':2,'cpu_threads_per_worker':4,'original_acceptance_unchanged':True,'selection':'observed fitting IoU only','budget':'24 outer iterations, 24 nfev per block, patience 8; greater budget than prior coupled control'})
    print(str(batch),flush=True)
    def run(job):
        with (job/'process.log').open('w') as log:
            code=subprocess.call([sys.executable,str(ROOT/'scripts/run_joint_stabilized.py'),'--config',str(job/'config.json')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            if code==0:
                code=subprocess.call([sys.executable,str(ROOT/'scripts/evaluate_joint_robust.py'),str(job/'fit_attempt_01')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        state={'job':job.name,'exit_code':code}
        p=job/'fit_attempt_01/evaluation.json'
        if p.exists(): state['evaluation']=read(p)
        write(job/'completion.json',state)
        print(job.name,code,flush=True)
        return state
    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(run,jobs))
    write(batch/'summary.json',{'status':'COMPLETE','jobs':results})
if __name__=='__main__': main()
