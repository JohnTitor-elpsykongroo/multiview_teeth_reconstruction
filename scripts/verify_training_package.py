"""Verify portable source/data inventory and a matching target preflight."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/DMM'))
from dmm.provenance import source_fingerprint
from dmm.validation import sha256,read_json,require

def git_identity():
    command=['git','-c',f'safe.directory={ROOT.as_posix()}','-C',str(ROOT)]
    def git(*args):
        return subprocess.check_output(command+list(args),text=True,stderr=subprocess.PIPE).strip()
    require(Path(git('rev-parse','--show-toplevel')).resolve()==ROOT.resolve(),'not a project Git checkout')
    # Untracked code/config can alter imports too. Outputs belong in ignored directories.
    require(not git('status','--porcelain','--untracked-files=all'),'commit project changes before target preflight/training')
    require(not git('ls-files','--others','--exclude-standard'),'untracked project inputs')
    return dict(kind='git',commit=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'))

def verify(data=False,proof=None):
    inventory=ROOT/'training_package_manifest.json'
    if not inventory.exists():
        identity=git_identity()
        result=dict(status='GIT_SOURCE_VERIFIED',source_identity=identity,source_sha256=source_fingerprint(),
                    data_files='full hashes checked by target preflight ArchDataset loaders')
        if proof:
            verify_proof(proof,identity)
        return result
    manifest=read_json(inventory)
    identity=dict(kind='package',manifest_sha256=sha256(inventory))
    require(manifest['source_sha256']==source_fingerprint(),'DMM source fingerprint mismatch')
    for item in manifest['files']:
        path=(ROOT/item['path']).resolve()
        require(path.is_relative_to(ROOT),'package path escape')
        require(path.is_file() and sha256(path)==item['sha256'],f'package file mismatch: {item["path"]}')
    if data:
        for index,item in enumerate(manifest['data_files']):
            path=(ROOT.parent/item['path']).resolve()
            require(path.is_relative_to(ROOT.parent/'data'),'data path outside portable data tree')
            require(path.is_file() and sha256(path)==item['sha256'],f'data file mismatch: {item["path"]}')
            if (index+1)%250==0:print(f'Hashed {index+1}/{len(manifest["data_files"])} data resources',flush=True)
    if proof:verify_proof(proof,identity)
    return dict(status='PACKAGE_VERIFIED',source_sha256=manifest['source_sha256'],
                source_identity=identity,
                source_files=len(manifest['files']),data_files=len(manifest['data_files']) if data else 'checked by training loader')

def verify_proof(proof,identity):
    from dmm.provenance import training_runtime
    result=read_json(proof)
    require(result['status']=='TARGET_READY_FOR_BOUNDED_PILOT','target preflight did not pass')
    require(result['source_sha256']==source_fingerprint(),'preflight source mismatch')
    recorded=result.get('source_identity')
    if recorded is None and identity['kind']=='package':
        recorded=dict(kind='package',manifest_sha256=result.get('package_manifest_sha256'))
    require(recorded==identity,'preflight project revision/config mismatch; rerun preflight')
    require(result['runtime']==training_runtime('cuda'),'preflight target runtime mismatch')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-inventory',action='store_true');p.add_argument('--preflight')
    a=p.parse_args();print(json.dumps(verify(a.data_inventory,a.preflight)))
