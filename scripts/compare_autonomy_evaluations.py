"""Compare matched-epoch, common-objective val evaluations without selecting a model."""
import argparse
import copy
import json
from pathlib import Path
import statistics

from summarize_autonomy_run import checked_evaluation, finite, read, sha, write_table


def compare(baseline, candidate, baseline_config, candidate_config, factor):
    baseline,candidate = Path(baseline).resolve(),Path(candidate).resolve()
    configs = [Path(baseline_config).resolve(),Path(candidate_config).resolve()]
    # Reject test/train before opening either evaluation's metrics.
    provenance = [read(folder/'provenance.json') for folder in (baseline,candidate)]
    if any(p.get('split') != 'val' for p in provenance):
        raise ValueError('controlled comparison accepts val only; test remains sealed')
    results = [checked_evaluation(folder)[0] for folder in (baseline,candidate)]
    raw = [read(path) for path in configs]
    for p,c,path in zip(provenance,raw,configs):
        if p['config_sha256'] != sha(path) or p['training_recipe'] != c['recipe']:
            raise ValueError('comparison config/recipe binding mismatch')
        if p['manifest_sha256'] != c['manifest']['sha256']:
            raise ValueError('comparison manifest/config binding mismatch')
        if not isinstance(p.get('training_epoch'),int) or p['training_epoch'] < 1:
            raise ValueError('missing valid training epoch')
        if not p.get('fit_recipe_sha256') or not isinstance(p.get('fit_recipe'),dict):
            raise ValueError('controlled comparison requires explicit common fit recipe')
    for key in ('training_epoch','manifest_sha256','canonical_reference_id','evaluation_source_sha256',
                'runtime','fit_recipe_sha256','fit_recipe'):
        if provenance[0][key] != provenance[1][key]:
            raise ValueError(f'comparison protocol mismatch: {key}')
    expected = copy.deepcopy(raw[0])
    if factor == 'reduction':
        if (raw[0]['recipe']['version'],raw[0]['recipe']['reduction']) != ('masked_mean_v1','all_points') or (
                raw[1]['recipe']['version'],raw[1]['recipe']['reduction']) != ('effective_count_v1','effective'):
            raise ValueError('unexpected reduction control')
        expected['recipe'].update(version='effective_count_v1',reduction='effective')
    elif factor == 'latent_weight':
        before,after = raw[0]['recipe']['weights']['latent'],raw[1]['recipe']['weights']['latent']
        if before != 1e6 or after != 1e5:
            raise ValueError('unexpected latent weight control')
        expected['recipe']['weights']['latent'] = 1e5
    else:
        raise ValueError('unsupported controlled factor')
    if raw[1] != expected:
        raise ValueError('candidate changes more than the declared single factor')
    if results[0]['budget'] != results[1]['budget'] or results[0]['fit_recipe'] != results[1]['fit_recipe'] or results[0]['fit_recipe'] != provenance[0]['fit_recipe']:
        raise ValueError('comparison result budget/fit recipe mismatch')
    maps = [{c['case_id']:c for c in result['cases']} for result in results]
    if set(maps[0]) != set(maps[1]) or not maps[0]:
        raise ValueError('comparison cohort mismatch')
    cases,tooth_rows = [],[]
    for case_id in sorted(maps[0]):
        left,right = maps[0][case_id],maps[1][case_id]
        if left['patient_id'] != right['patient_id']:
            raise ValueError('comparison patient mismatch')
        row = dict(case_id=case_id,patient_id=left['patient_id'],baseline_status=left['status'],candidate_status=right['status'],
            baseline_error=left.get('error'),candidate_error=right.get('error'),delta_mm=None)
        valid = left['status'] != 'FAILED' and right['status'] != 'FAILED'
        for field in ('fit_index_sha256','evaluation_index_sha256'):
            if field in left and field in right:
                if left[field] != right[field]: raise ValueError(f'comparison point partition mismatch: {field}')
            elif valid:
                raise ValueError('successful comparison requires point partition digests')
        row['point_partition_comparison'] = 'verified' if all(field in left and field in right for field in
            ('fit_index_sha256','evaluation_index_sha256')) else 'unavailable_due_to_failed_case'
        if valid:
            lg,rg = left['geometry'],right['geometry']
            for key in ('metric','resolution','samples_per_component'):
                if lg[key] != rg[key]: raise ValueError(f'comparison geometry budget mismatch: {key}')
            lt,rt = lg['per_tooth'],rg['per_tooth']
            if set(lt) != set(rt): raise ValueError('comparison present tooth mismatch')
            row.update(baseline_score_mm=lg['score_mm'],candidate_score_mm=rg['score_mm'],delta_mm=rg['score_mm']-lg['score_mm'])
            for label in sorted(lt,key=int):
                tooth = dict(case_id=case_id,patient_id=left['patient_id'],fdi=int(label))
                for key in ('observed_to_mesh_sample_mean_mm','observed_to_mesh_sample_p95_mm'):
                    before,after = finite(lt[label][key],key),finite(rt[label][key],key)
                    tooth.update({f'baseline_{key}':before,f'candidate_{key}':after,f'delta_{key}':after-before})
                tooth_rows.append(tooth)
        cases.append(row)
    complete = all(row['delta_mm'] is not None for row in cases)
    successful = [row for row in cases if row['delta_mm'] is not None]
    result = dict(schema='dmm_controlled_comparison_v1',status='COMPARED_NOT_QUALITY_ACCEPTED',factor=factor,
        epoch=provenance[0]['training_epoch'],requested_case_count=len(cases),successful_pair_count=len(successful),
        baseline_failed_case_count=results[0]['failed_case_count'],candidate_failed_case_count=results[1]['failed_case_count'],
        baseline_score_mm=results[0]['score_mm'],candidate_score_mm=results[1]['score_mm'],
        delta_mm=results[1]['score_mm']-results[0]['score_mm'] if complete else None,
        successful_pair_diagnostics=dict(scope='successful_pairs_only_not_selection_score',
            mean_delta_mm=statistics.mean(row['delta_mm'] for row in successful) if successful else None,
            improved_case_count=sum(row['delta_mm']<0 for row in successful)),
        baseline_checkpoint_sha256=provenance[0]['checkpoint_sha256'],candidate_checkpoint_sha256=provenance[1]['checkpoint_sha256'],
        common_fit_recipe_sha256=provenance[0]['fit_recipe_sha256'],budget=results[0]['budget'],cases=cases,teeth=tooth_rows,
        input_sha256={str(path):sha(path) for path in [*configs,*[folder/name for folder in (baseline,candidate)
                      for name in ('provenance.json','evaluation.json')]]},
        limitations='Paired fixed-cohort development diagnosis. No bootstrap, grid/fit/training-seed uncertainty or clinical acceptance. Negative delta favors candidate at this protocol only.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','baseline-config','candidate-config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--factor',choices=('reduction','latent_weight'),required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('output must be fresh')
    result=compare(args.baseline,args.candidate,args.baseline_config,args.candidate_config,args.factor)
    result['script_sha256']=sha(__file__)
    result['summary_helper_sha256']=sha(Path(__file__).with_name('summarize_autonomy_run.py'))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_table(args.output/'cases.csv',result['cases'])
    write_table(args.output/'teeth.csv',result['teeth'])


if __name__=='__main__': main()
