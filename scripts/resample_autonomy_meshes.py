"""CPU mesh-sampling repeatability with fixed meshes, fitted codes and scan points."""
import argparse
import json
from pathlib import Path
import platform
import statistics

import numpy as np
import scipy
from scipy.spatial import cKDTree
from summarize_autonomy_run import checked_evaluation, sha, write_table


def sample_metrics(path, labels, samples, seed):
    with np.load(path, allow_pickle=False) as arrays:
        vertices, faces, face_labels = arrays['vertices_model'], arrays['faces'], arrays['face_labels']
        observed, observed_labels = arrays['observed_points_model'], arrays['observed_labels']
        unit = float(arrays['model_unit_mm'])
    if unit != 50:
        raise ValueError('unexpected model unit; protocol requires 50mm/unit')
    if samples <= 0 or seed < 0 or not labels:
        raise ValueError('invalid mesh-sampling budget')
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=-1)/2
    rng = np.random.default_rng(seed)
    result = {}
    for label in labels:
        label = int(label)
        pool = np.flatnonzero((face_labels == label) & (area > 0))
        truth = observed[observed_labels == label]*50
        if not len(pool) or not len(truth):
            raise ValueError(f'missing present component {label}')
        chosen = rng.choice(pool, samples, p=area[pool]/area[pool].sum())
        uv = rng.random((len(chosen),2)); root = np.sqrt(uv[:,0])
        bary = np.column_stack((1-root, root*(1-uv[:,1]), root*uv[:,1]))
        points = (triangles[chosen]*bary[:,:,None]).sum(1)*50
        forward = cKDTree(points).query(truth)[0]
        reverse = cKDTree(truth).query(points)[0]
        result[str(label)] = dict(observed_to_mesh_sample_mean_mm=float(forward.mean()),
            observed_to_mesh_sample_p95_mm=float(np.quantile(forward,.95)),
            generated_to_observed_sample_mean_mm=float(reverse.mean()),
            generated_to_observed_sample_p95_mm=float(np.quantile(reverse,.95)))
    return dict(score_mm=float(np.mean([row['observed_to_mesh_sample_mean_mm'] for row in result.values()])),
                per_tooth=result)


def resample(folder, output, samples=(1024,4096), seeds=(20261006,20261007,20261008,20261009,20261010)):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('output must be fresh')
    if not samples or not seeds or len(set(samples)) != len(samples) or len(set(seeds)) != len(seeds):
        raise ValueError('budgets/seeds must be nonempty and unique')
    if any(type(value) is not int or value <= 0 for value in samples) or any(type(value) is not int or value < 0 for value in seeds):
        raise ValueError('invalid mesh-sampling budget')
    original, provenance = checked_evaluation(folder)
    rows, tooth_rows, diagnostics, hashes = [], [], [], {}
    hashes[str(folder/'evaluation.json')] = sha(folder/'evaluation.json')
    hashes[str(folder/'provenance.json')] = sha(folder/'provenance.json')
    budget = original['budget']
    # Reproduce all stored tooth metrics first; changing seeds only changes mesh samples.
    for index, case in enumerate(original['cases']):
        if case['status'] == 'FAILED':
            diagnostics.append(dict(case_id=case['case_id'], status='FAILED', error=case['error'], score_mm=None))
            continue
        mesh = folder/case['artifact_directory']/'mesh.npz'
        hashes[str(mesh)] = sha(mesh)
        labels = list(case['geometry']['per_tooth'])
        expected = set(case['components']) - {'0'}
        if set(labels) != expected:
            raise ValueError('stored mesh metric omits a present tooth')
        parity = sample_metrics(mesh, labels, budget['mesh_samples_per_component'], budget['seed'])
        for label, metrics in parity['per_tooth'].items():
            for name, value in metrics.items():
                if abs(value-case['geometry']['per_tooth'][label][name]) > 1e-8:
                    raise ValueError(f'stored metric parity failed for {case["case_id"]}/{label}/{name}')
        if abs(parity['score_mm']-case['geometry']['score_mm']) > 1e-8:
            raise ValueError('stored score parity failed')
        case_rows = []
        for count in samples:
            for seed in seeds:
                metrics = sample_metrics(mesh, labels, count, seed)
                row = dict(case_id=case['case_id'], patient_id=case['patient_id'], split=original['split'],
                    mesh_samples_per_component=count, mesh_seed=seed, score_mm=metrics['score_mm'])
                rows.append(row); case_rows.append(row)
                for label, tooth in metrics['per_tooth'].items():
                    tooth_rows.append(dict(case_id=case['case_id'], fdi=label, mesh_samples_per_component=count,
                                           mesh_seed=seed, **tooth))
        diagnostics.append(dict(case_id=case['case_id'], status='RESAMPLED_NOT_QUALITY_ACCEPTED',
            original_score_mm=case['geometry']['score_mm'], stored_metric_parity='PASS',
            budgets=[dict(samples=count, seed_count=len(seeds),
                mean_mm=statistics.mean(r['score_mm'] for r in case_rows if r['mesh_samples_per_component']==count),
                seed_score_min_mm=min(r['score_mm'] for r in case_rows if r['mesh_samples_per_component']==count),
                seed_score_max_mm=max(r['score_mm'] for r in case_rows if r['mesh_samples_per_component']==count),
                seed_score_std_mm=statistics.pstdev(r['score_mm'] for r in case_rows if r['mesh_samples_per_component']==count))
                for count in samples]))
    cohorts = []
    for count in samples:
        for seed in seeds:
            values = [r['score_mm'] for r in rows if r['mesh_samples_per_component']==count and r['mesh_seed']==seed]
            cohorts.append(dict(samples=count, mesh_seed=seed, requested=original['requested_case_count'],
                successful=len(values), failed=original['failed_case_count'],
                score_mm=statistics.mean(values) if values and not original['failures'] else None))
    summary = dict(schema='dmm_mesh_repeatability_v1', evaluation=str(folder), split=original['split'],
        checkpoint_sha256=provenance['checkpoint_sha256'], budget=budget, samples=list(samples), mesh_seeds=list(seeds),
        scope='mesh samples vary only; same extracted mesh, fitted/learned codes, face labels and evaluated scan points; no mesh-resolution or code-fitting uncertainty measured',
        quality_accepted=False, requested=original['requested_case_count'], failures=original['failures'],
        cases=diagnostics, cohorts=cohorts, input_sha256=hashes, source_script_sha256=sha(__file__),
        source_summary_helper_sha256=sha(Path(__file__).with_name('summarize_autonomy_run.py')),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,device='cpu'),
        uncertainty='empirical mesh-seed spread; not a patient confidence interval or clinical tolerance')
    output.mkdir(parents=True,exist_ok=False)
    write_table(output/'case_resamples.csv',rows)
    write_table(output/'tooth_resamples.csv',tooth_rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Fixed-mesh sampling repeatability','',summary['scope'],'',summary['uncertainty'],'',
           '| Case | Samples / tooth | Mean mm | Seed min–max mm | Std mm |',
           '| --- | ---: | ---: | ---: | ---: |']
    for case in diagnostics:
        for item in case.get('budgets',[]):
            lines.append(f"| {case['case_id']} | {item['samples']} | {item['mean_mm']:.6f} | {item['seed_score_min_mm']:.6f}–{item['seed_score_max_mm']:.6f} | {item['seed_score_std_mm']:.6f} |")
    lines += ['',f"Requested {summary['requested']}; failures retained: {', '.join(summary['failures']) or 'none'}.",
              '', 'Area-weighted surface samples; equal-tooth/equal-case distances. Reverse distances include unsupported generated surface.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--samples',type=int,nargs='+',default=[1024,4096])
    parser.add_argument('--mesh-seeds',type=int,nargs='+',default=[20261006,20261007,20261008,20261009,20261010])
    args=parser.parse_args()
    result=resample(args.evaluation,args.output,tuple(args.samples),tuple(args.mesh_seeds))
    print(json.dumps(dict(requested=result['requested'],failures=result['failures'],output=args.output)))
