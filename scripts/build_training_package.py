"""Create a portable source archive, separate data inventory, and SHA256."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/DMM'))
from dmm.provenance import source_fingerprint,decoder_contract,upstream_commit
from dmm.validation import read_json,sha256,write_json,require

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output).resolve()
    require(not out.exists(),'package output already exists');out.mkdir(parents=True)
    # Exact training source, bundled torchmeta, git revision, local tests and docs.
    source=[]
    for folder in ('third_party/DMM','configs/training_handoff_v1'):
        source.extend((ROOT/folder).rglob('*'))
    source.extend(ROOT/'scripts'/name for name in ('training_smoke.py','training_target_preflight.py','verify_training_package.py',
                                                   'build_training_package.py','setup_training_wsl.sh','run_training_wsl.sh'))
    source.extend(ROOT/'tests'/name for name in ('test_dual_arch.py','test_training_readiness.py','test_training_handoff.py',
                         'test_fitting.py','test_rendering.py','rendering_fixtures.py','test_semanticxy.py','test_dual_extension.py'))
    source.extend(ROOT/'docs'/name for name in ('TRAINING_HANDOFF_20261006.md','WSL_RTX5090_environment_and_training.md','training_readiness_v1.md'))
    source=[path for path in source if path.is_file() and not path.is_symlink()
            and not any(part in ('__pycache__','build','dist') for part in path.relative_to(ROOT).parts)
            and path.suffix not in ('.pyc','.pyd','.so') and '.git/hooks' not in path.relative_to(ROOT).as_posix()]
    source=sorted(set(source),key=lambda path:path.relative_to(ROOT).as_posix())
    data={}
    def add(path,digest=None):
        path=path.resolve();require(path.is_relative_to(ROOT.parent/'data'),'data outside portable root')
        name=path.relative_to(ROOT.parent).as_posix()
        item=dict(path=name,sha256=digest or sha256(path),bytes=path.stat().st_size)
        require(name not in data or data[name]==item,'inconsistent resource digest')
        data[name]=item
    for arch in ('upper','lower'):
        path=ROOT/'configs/training_handoff_v1'/f'{arch}_pilot.json';config=read_json(path)
        for key in ('manifest','canonical_reference'):
            file=(path.parent/config[key]['path']).resolve();add(file,config[key]['sha256'])
        manifest=(path.parent/config['manifest']['path']).resolve()
        for row in read_json(manifest)['cases']:
            for key in ('source_geometry','source_annotation','samples','centers'):
                add(manifest.parent/row[key]['path'],row[key]['sha256'])
    manifest=dict(format='dmm_training_handoff_v1',source_sha256=source_fingerprint(),decoder_contract=decoder_contract(),
                  upstream_commit=upstream_commit(),data_bytes=sum(x['bytes'] for x in data.values()),
                  files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p)) for p in source],
                  data_files=sorted(data.values(),key=lambda x:x['path']),
                  scope='source archive; data separate; target runtime unverified; geometry quality unverified')
    write_json(out/'training_package_manifest.json',manifest)
    archive=out/'dmm_training_source_v1.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in source:z.write(path,'multiview_teeth_reconstruction/'+path.relative_to(ROOT).as_posix())
        z.write(out/'training_package_manifest.json','multiview_teeth_reconstruction/training_package_manifest.json')
    write_json(out/'delivery.json',dict(archive=archive.name,sha256=sha256(archive),bytes=archive.stat().st_size,
                                      data_files=len(data),data_bytes=manifest['data_bytes'],source_sha256=manifest['source_sha256']))
    (out/'data_files.txt').write_text('\n'.join(sorted(data))+'\n',encoding='utf-8')
    print(json.dumps(read_json(out/'delivery.json')))

if __name__=='__main__':main()
