"""Optional orchestration of two independent arch training jobs; no paired loss."""
from pathlib import Path

from data.arch_dataset import validate_patient_splits
from networks.dmm_net import DMM
from training.arch_training import load_training_config, train_arch
from dmm.provenance import source_fingerprint
from dmm.validation import header, read_json, reference, require, resolve_ref, stamped, write_json


def load_dual_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    header(config, 'dual_training_config')
    require(config.get('mode') == 'independent_arches', 'only independent arch training is supported')
    require(set(config.get('arches', {})) == {'upper', 'lower'}, 'both arch configs required')
    paths, datasets = {}, {}
    for arch in ('upper', 'lower'):
        paths[arch] = resolve_ref(path, config['arches'][arch])
        child, dataset, _, specs = load_training_config(paths[arch])
        require(child['arch'] == arch, 'dual training arch/config mismatch')
        DMM(specs, arch=arch)  # Validate both architectures before any job starts.
        datasets[arch] = dataset
    splits = validate_patient_splits(*datasets.values())
    return config, paths, dict(status='DUAL_TRAINING_INPUT_VALIDATED',
                             cases={k: len(v) for k, v in datasets.items()}, patients=len(splits),
                             paired_geometry_required=False, cross_arch_loss=False)


def make_dual_config(upper, lower, output):
    output = Path(output).resolve()
    require(not output.exists(), 'dual config already exists')
    # Validate without creating a potentially misleading config on failure.
    datasets = []
    for arch, path in [('upper', upper), ('lower', lower)]:
        config, dataset, _, specs = load_training_config(path)
        require(config['arch'] == arch, 'dual training arch/config mismatch')
        DMM(specs, arch=arch)
        datasets.append(dataset)
    validate_patient_splits(*datasets)
    data = stamped('dual_training_config', mode='independent_arches',
                   arches={k: reference(p, output) for k, p in [('upper', upper), ('lower', lower)]})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, data)
    return dict(status='DUAL_CONFIG_CREATED', config=str(output))


def train_dual(config_path, output_dir, device='cpu', resume=None, arch_resumes=None):
    """Fresh run roots; completed jobs are hash-bound references, never overwritten.

    A failed job can resume its ordinary arch checkpoint via arch_resumes.
    Without one, only that unfinished job restarts. No weight sharing or pairing.
    """
    config, paths, validation = load_dual_config(config_path)
    output = Path(output_dir).resolve()
    require(not output.exists(), 'dual output must be a fresh directory')
    arch_resumes = arch_resumes or {}
    require(set(arch_resumes).issubset(paths), 'unknown arch resume')
    identity = dict(config=config, source_sha256=source_fingerprint(), device=str(device))
    completed = {}
    if resume:
        old = read_json(resume)
        require(old['identity'] == identity, 'dual resume source/config/device mismatch')
        require(old['status'] in ('RUNNING', 'INTERRUPTED'), 'dual run already completed')
        require(set(old['completed']).issubset(paths), 'invalid completed arches')
        for arch, artifacts in old['completed'].items():
            require(set(artifacts) == {'checkpoint', 'report'}, 'invalid completed artifacts')
            completed[arch] = {k: resolve_ref(resume, ref) for k, ref in artifacts.items()}
    require(not (set(completed) & set(arch_resumes)), 'cannot resume an already completed arch')
    output.mkdir(parents=True)
    owner = output / 'workflow.json'

    def save(status, active=None, error=None):
        data = dict(status=status, identity=identity, validation=validation, active_arch=active,
                    completed={a: {k: reference(p, owner) for k, p in row.items()} for a, row in completed.items()},
                    error=error, scope='independent training execution; not model quality acceptance')
        temporary = output / 'workflow.tmp.json'
        write_json(temporary, data)
        temporary.replace(owner)
        return data

    save('RUNNING')
    active = None
    try:
        for arch in ('upper', 'lower'):
            if arch in completed:
                continue
            active = arch
            save('RUNNING', arch)
            train_arch(paths[arch], output / arch, device, arch_resumes.get(arch))
            completed[arch] = dict(checkpoint=output / arch / 'final.pth', report=output / arch / 'report.json')
            save('RUNNING')
        return save('DUAL_TRAINING_COMPLETED_NOT_QUALITY_ACCEPTED')
    except BaseException as exc:
        save('INTERRUPTED', active, str(exc))
        raise
