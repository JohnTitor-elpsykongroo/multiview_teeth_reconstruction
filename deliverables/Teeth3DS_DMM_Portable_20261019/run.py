"""Self-contained Teeth3DS reproduction. sample cleans its owned workspace on PASS."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sys,tempfile,time
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[variable]='1'
ROOT=Path(__file__).resolve().parent
import pipeline as p
from reproduction import compare,runtime

def check_package():
    manifest=p.read(ROOT/'PACKAGE_MANIFEST.json')
    for relative,digest in manifest['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or p.sha(path)!=digest:
            raise ValueError('Package source/assets modified or missing: '+relative)
    return p.sha(ROOT/'PACKAGE_MANIFEST.json')

def mapped_candidates(raw,locate,required):
    source=p.read(ROOT/'assets/candidates.json');rows=source['cases'];index=None;missing=[]
    for row in rows:
        for field in ('obj','annotation'):
            relative=row[field+'_relative'];path=(raw/relative).resolve()
            if not path.is_relative_to(raw):raise ValueError('Unsafe relative raw path')
            if row['case_id'] in required and not path.is_file() and locate:
                if index is None:
                    names={r['case_id']+ext for r in rows for ext in ('.obj','.json')};index={}
                    # Discovery changes paths only; membership remains the bundled 1793 IDs.
                    for candidate in raw.rglob('*'):
                        if candidate.is_file() and candidate.name in names:index.setdefault(candidate.name,[]).append(candidate)
                found=index.get(Path(relative).name,[])
                if len(found)>1:raise ValueError('Ambiguous raw basename: '+Path(relative).name)
                if found:path=found[0].resolve()
            if not path.is_relative_to(raw):raise ValueError('Raw symlink escapes supplied root')
            if row['case_id'] in required and not path.is_file():missing.append(str(path))
            row[field]=str(path)
    if missing:raise ValueError('Raw files missing; supply the directory containing upper/lower or use --locate-files: '+str(missing[:6]))
    return source

def make_config(raw,work,required,workers,locate,package_id):
    if not raw.is_dir():raise ValueError('Raw root does not exist')
    out=(work/'output/latest').resolve()
    if out.is_relative_to(raw) or raw.is_relative_to(out):raise ValueError('Output must be separate from raw data')
    source=mapped_candidates(raw,locate,required)
    out.parent.mkdir(parents=True,exist_ok=True)
    owner=out/'.portable_owner.json'
    identity=dict(schema='portable_teeth3ds_output/1',package_manifest_sha256=package_id,raw_root=str(raw),selected_ids=sorted(required))
    if out.exists():
        if not owner.is_file() or p.read(owner)!=identity:raise ValueError('Existing output has different ownership/input scope; use a new --work-dir')
    else:out.mkdir()
    p.write(owner,identity)
    inputs=work/'.inputs';inputs.mkdir(parents=True,exist_ok=True)
    p.write(inputs/'candidates.json',source)
    config=dict(p.read(ROOT/'assets/recipe.json'),project_root=str(work),output_root=str(out),
                legacy_root=str(ROOT/'vendor'),archive_root=str(ROOT/'assets'),
                candidate_manifest=str(inputs/'candidates.json'),exclusion_policy=str(ROOT/'assets/exclusions.json'),workers=workers)
    p.write(inputs/'config.json',config)
    return p.load_config(inputs/'config.json')

def safe_cleanup_sample(work,parent,token):
    work=work.resolve();parent=parent.resolve();marker=work/'.sample_owner.json'
    if work.parent!=parent or not work.name.startswith('sample_') or not marker.is_file() or p.read(marker).get('token')!=token:
        raise ValueError('Refusing unowned sample cleanup')
    # Both absolute target and allowed parent are checked before recursive deletion.
    count=0;size=0
    for f in work.rglob('*'):
        if f.is_symlink():raise ValueError('Refusing cleanup containing symlinks')
        if f.is_file():count+=1;size+=f.stat().st_size
    record=dict(removed_directory=str(work),files=count,bytes=size,absolute_target_checked=True)
    shutil.rmtree(work)
    if work.exists():raise ValueError('Sample cleanup incomplete')
    record['confirmed_removed']=True
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('sample','batch','verify','check-package'))
    parser.add_argument('--raw-root',help='Unprocessed Teeth3DS root; read only')
    parser.add_argument('--work-dir',help='Batch workspace; outputs written to output/latest within it')
    parser.add_argument('--workers',type=int,default=None)
    parser.add_argument('--locate-files',action='store_true',help='Resolve known basenames in alternate raw layouts; never add cases')
    parser.add_argument('--allow-environment-mismatch',action='store_true',help='Record an alternative runtime; strict content verification still required')
    parser.add_argument('--keep-work',action='store_true',help='Keep an owned sample workspace for debugging')
    args=parser.parse_args();package_id=check_package();start=time.time()
    if args.command=='check-package':
        print(json.dumps(dict(status='PASS_PACKAGE_INTEGRITY',package_manifest_sha256=package_id,runtime=runtime()),indent=2));return
    if args.workers is not None and args.workers<1:parser.error('--workers must be positive')
    if args.command=='sample' and args.work_dir:parser.error('sample uses an automatically owned .work/sample_* directory; --work-dir is for batch/verify')
    if sys.version_info<(3,10):raise ValueError('Python >=3.10 is required; the reference processing runtime is Python 3.12')
    env=runtime();baseline=p.read(ROOT/'assets/provenance.json')['runtime']
    same=all(env[k]==baseline[k] for k in ('numpy','scipy')) and env['python'].split('.')[:2]==baseline['python'].split('.')[:2]
    print(json.dumps(dict(stage='environment',actual=env,reference=baseline,reference_packages_and_python_minor_match=same)),flush=True)
    if not same and args.command=='batch' and not args.allow_environment_mismatch:
        raise ValueError('Use reference preprocessing environment, or first pass sample then explicitly record the alternative with --allow-environment-mismatch')
    full=args.command!='sample'
    candidate_rows=p.read(ROOT/'assets/candidates.json')['cases']
    if len(candidate_rows)!=1793:raise ValueError('Bundled candidate membership changed')
    required={r['case_id'] for r in candidate_rows} if full else set(p.read(ROOT/'assets/sample_arrays.json')['case_ids'])
    work=None;token=None;parent=(ROOT/'.work').resolve()
    if parent.parent!=ROOT:raise ValueError('Sample workspace parent must remain inside this project')
    if args.command=='verify':
        work=Path(args.work_dir).resolve() if args.work_dir else ROOT
        config=p.load_config(work/'.inputs/config.json')
        owner=p.read(Path(config['output_root'])/'.portable_owner.json')
        if owner['package_manifest_sha256']!=package_id:raise ValueError('Output source/asset binding differs')
        required=set(owner['selected_ids']);full=len(required)==1793
    else:
        if not args.raw_root:parser.error('--raw-root is required for sample and batch')
        raw=Path(args.raw_root).expanduser().resolve()
        if args.command=='sample':
            parent.mkdir(exist_ok=True);work=Path(tempfile.mkdtemp(prefix='sample_',dir=parent)).resolve()
            token=os.urandom(16).hex();p.write(work/'.sample_owner.json',dict(token=token))
        else:work=Path(args.work_dir).expanduser().resolve() if args.work_dir else ROOT
        work.mkdir(parents=True,exist_ok=True)
        try:config=make_config(raw,work,required,args.workers or (2 if args.command=='sample' else 6),args.locate_files,package_id)
        except BaseException as exc:
            record=dict(error=f'{type(exc).__name__}: {exc}',workspace=str(work),processing_started=False)
            if args.command=='sample':record['cleanup']=safe_cleanup_sample(work,parent,token)
            print(json.dumps(record,ensure_ascii=False),file=sys.stderr,flush=True)
            raise
    try:
        if args.command!='verify':p.run(config,None if full else sorted(required))
        from geometry_verify import verify
        verification=verify(config,expected=len(required))
        report=compare(config,required,full=full)
        report.update(command=args.command,package_manifest_sha256=package_id,geometry_verification=verification,
                      reference_environment_matches=same,total_elapsed_seconds=time.time()-start,workspace=str(work))
        report['numeric_dependency_paths']={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
            if name.startswith('teeth3ds_dmm.') and getattr(module,'__file__',None)}
        if report['status']!='PASS_REPRODUCED_DATA_CONTENT':
            print(json.dumps(report,ensure_ascii=False,indent=2),file=sys.stderr,flush=True)
            raise ValueError('Reproduction mismatch; see terminal diagnostics above')
        if args.command=='sample' and not args.keep_work:
            report['cleanup']=safe_cleanup_sample(work,parent,token)
        else:report['cleanup']=dict(retained=True,reason='explicit --keep-work' if args.command=='sample' else 'batch deliverable or verification target')
        print(json.dumps(dict(status=report['status'],checks=report['checks'],
            dataset_root=str(Path(config['output_root'])),cleanup=report['cleanup']),indent=2),flush=True)
    except BaseException as exc:
        print(json.dumps(dict(error=f'{type(exc).__name__}: {exc}',workspace=str(work),preserved_for_diagnosis=True),ensure_ascii=False),file=sys.stderr,flush=True)
        raise

if __name__=='__main__':main()
