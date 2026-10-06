"""Resumable staged Shape-only experiments; each fit has isolated truth evaluation."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from run_forward_check import PROJECT_ROOT, sha256


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def run_process(root,job,phase,arguments,state):
    plan=json.loads((root/'plan.json').read_text())
    for name,digest in plan.get('code_sha256',{}).items():
        if sha256(PROJECT_ROOT/'scripts'/name)!=digest:
            raise RuntimeError(f'batch code changed since plan: {name}; create a new batch for the new solver')
    output_path=job/f'{phase}.log'
    state.update({'current_job':job.name,'phase':phase,'log':str(output_path),
                  'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    env=os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE']='1'
    with output_path.open('a',encoding='utf-8') as log:
        command=[sys.executable,'-u',*arguments]
        child=subprocess.Popen(command,cwd=PROJECT_ROOT,stdout=log,stderr=subprocess.STDOUT,
                               env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        state['child_pid']=child.pid
        write_json(root/'status.json',state)
        code=child.wait()
    state['child_pid']=None
    write_json(root/'status.json',state)
    return code


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,default=PROJECT_ROOT/'configs/shape_expansion.json')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stage-limit',type=int)
    args=parser.parse_args()
    if args.resume:
        root=args.resume.resolve(strict=True)
        plan=json.loads((root/'plan.json').read_text())
    else:
        definition=json.loads(args.plan.read_text())
        base_path=PROJECT_ROOT/definition['base_config']
        base=json.loads(base_path.read_text())
        root=PROJECT_ROOT/'runs'/(dt.datetime.now(dt.timezone.utc).strftime('shape_expansion_%Y%m%dT%H%M%SZ_')+sha256(args.plan)[:8])
        root.mkdir(parents=True,exist_ok=False)
        plan={'definition':definition,'base_config':base,'definition_sha256':sha256(args.plan),
              'base_config_sha256':sha256(base_path),'stage_gate':'all seeds pass; failures are preserved; dependent stages are skipped',
              'runner_code_sha256':sha256(Path(__file__)),'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        code_names=['prepare_shape_only.py','run_shape_only.py','shape_only_common.py','evaluate_shape_only.py',
                    'run_forward_check.py','run_pose_only.py','run_shape_expansion.py','surface_metric_utils.py']
        plan['code_sha256']={name:sha256(PROJECT_ROOT/'scripts'/name) for name in code_names}
        snapshot=root/'code_snapshot'
        snapshot.mkdir()
        for name in code_names:
            shutil.copyfile(PROJECT_ROOT/'scripts'/name,snapshot/name)
        write_json(root/'plan.json',plan)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('batch root must be inside project runs')
    definition=plan['definition']
    state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'batch_root':str(root),'stages':{},'jobs':{}}
    print(json.dumps({'status':'EXPANSION_STARTED','batch_root':str(root)}),flush=True)
    for stage_index,stage in enumerate(definition['stages']):
        sid=stage['id']
        if args.stage_limit is not None and stage_index>=args.stage_limit:
            state['status']='STAGE_LIMIT_REACHED'
            break
        dependency=stage.get('requires')
        if dependency and state['stages'].get(dependency)!='PASS':
            state['stages'][sid]='SKIPPED_DEPENDENCY_NOT_PASSED'
            write_json(root/'status.json',state)
            continue
        outcomes=[]
        for seed in stage['seeds']:
            job_id=f'{sid}_seed{seed}'
            if state['jobs'].get(job_id,{}).get('terminal'):
                outcomes.append(state['jobs'][job_id]['status']=='PASS')
                continue
            job=root/'jobs'/job_id
            job.mkdir(parents=True,exist_ok=True)
            config=dict(plan['base_config'])
            config.update({k:definition[k] for k in ['perturbation_rng','outer_iterations','stop_after_no_best_iterations','acceptance']})
            config.update(definition.get('config_overrides',{}))
            config.update({'active_labels':stage['active_labels'],'perturbation_seed':seed,'perturbation_training_std':stage['sigma']})
            config_path=job/'config.json'
            if config_path.exists():
                if json.loads(config_path.read_text())!=config:
                    raise ValueError('resume config differs')
            else:
                write_json(config_path,config)
            prepared=job/'prepared'
            if not (prepared/'fit_input/manifest.json').exists():
                if prepared.exists():
                    raise ValueError('incomplete preparation requires an explicit new attempt')
                code=run_process(root,job,'prepare',['scripts/prepare_shape_only.py','--config',str(config_path),'--output-root',str(prepared)],state)
                if code:
                    state['jobs'][job_id]={'status':'PREPARATION_FAILED','terminal':True,'job_root':str(job)}
                    outcomes.append(False)
                    write_json(root/'status.json',state)
                    continue
            attempts=sorted(job.glob('fit_attempt_*'))
            fit=next((p for p in reversed(attempts) if (p/'fit_report.json').exists()),None)
            if fit is None:
                fit=job/f'fit_attempt_{len(attempts)+1:02d}'
                code=run_process(root,job,'fit',['scripts/run_shape_only.py','--config',str(config_path),
                     '--fit-input',str(prepared/'fit_input'),'--run-root',str(fit)],state)
                if code:
                    state['jobs'][job_id]={'status':'FIT_FAILED','terminal':True,'job_root':str(job),'fit_root':str(fit)}
                    outcomes.append(False)
                    write_json(root/'status.json',state)
                    print(json.dumps({'job':job_id,'status':'FIT_FAILED'}),flush=True)
                    continue
            if not (fit/'evaluation.json').exists():
                code=run_process(root,job,'evaluate',['scripts/evaluate_shape_only.py',str(fit)],state)
                if code:
                    raise RuntimeError(f'evaluator failed; inspect {job}/evaluate.log and resume after repair')
            evaluation=json.loads((fit/'evaluation.json').read_text())
            passed=evaluation['status']=='SHAPE_ONLY_SUBSET_PASS'
            state['jobs'][job_id]={'status':'PASS' if passed else 'EVALUATION_FAIL','terminal':True,
                'job_root':str(job),'fit_root':str(fit),'active_labels':stage['active_labels'],'sigma':stage['sigma'],'seed':seed,
                'initial_iou':evaluation['initial_active_mean_tooth_iou'],'final_iou':evaluation['final_active_mean_tooth_iou'],
                'surface_improvement':evaluation['surface_mean_improvement_fraction'],
                'heldout_initial_iou':evaluation['heldout']['initial']['active_mean_tooth_iou'],
                'heldout_final_iou':evaluation['heldout']['final']['active_mean_tooth_iou'],
                'failed_checks':[k for k,v in evaluation['checks'].items() if not v]}
            outcomes.append(passed)
            write_json(root/'status.json',state)
            print(json.dumps({'job':job_id,**state['jobs'][job_id]}),flush=True)
        state['stages'][sid]='PASS' if all(outcomes) and len(outcomes)==len(stage['seeds']) else 'FAIL'
        write_json(root/'status.json',state)
        print(json.dumps({'stage':sid,'status':state['stages'][sid]}),flush=True)
    else:
        state['status']='EXPANSION_COMPLETED'
    state.update({'current_job':None,'phase':'idle','child_pid':None})
    write_json(root/'status.json',state)
    print(json.dumps({'status':state['status'],'batch_root':str(root),'stages':state['stages']}),flush=True)


if __name__=='__main__':
    main()
