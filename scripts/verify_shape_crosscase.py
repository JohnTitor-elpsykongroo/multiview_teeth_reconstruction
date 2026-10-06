"""Read-only provenance verification of a completed, uniformly reviewed campaign."""
import argparse
import json
from pathlib import Path
from run_forward_check import PROJECT_ROOT, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('campaign_root', type=Path)
    args = parser.parse_args()
    root = args.campaign_root.resolve(strict=True)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('only project runs may be verified')
    plan = json.loads((root/'plan.json').read_text())
    state = json.loads((root/'status.json').read_text())
    if state['status'] != 'CROSSCASE_COMPLETED':
        raise ValueError('campaign has not completed')
    output = root/'review/provenance_verification.json'
    if output.exists():
        raise FileExistsError(output)
    checks, verified = {}, {}

    def check_hash(path, digest):
        key = str(path)
        if key not in verified:
            verified[key] = sha256(path) == digest
        else:
            verified[key] = verified[key] and sha256(path) == digest
        return verified[key]

    checks['execution_core_unchanged'] = all([
        check_hash(PROJECT_ROOT/'scripts'/name, digest)
        for name, digest in plan['code_sha256'].items()])
    checks['execution_snapshots_unchanged'] = all([
        check_hash(root/'code_snapshot'/name, digest)
        for name, digest in plan['code_sha256'].items()])
    checks['frozen_dmm_source_unchanged'] = all([
        check_hash(Path(plan['forward_config']['dmm_root'])/name, digest)
        for name, digest in plan['dmm_source_sha256'].items()])
    original_states, review_states, metrics = {}, {}, []
    original_preserved, gates_unchanged, finite_unchanged = [], [], []
    source_checks, child_checks, fit_checks, input_checks, control_checks = [], [], [], [], []
    for name, case in state['cases'].items():
        source = Path(case['source_run'])
        provenance = json.loads((source/'provenance.json').read_text())
        source_checks += [check_hash(Path(p), digest) for p, digest in provenance['input_sha256'].items()]
        batch = Path(case['shape_batch'])
        child = json.loads((batch/'plan.json').read_text())
        child_checks += [check_hash(batch/'code_snapshot'/p, digest) for p, digest in child['code_sha256'].items()]
        original_states[name] = case['status']
        reviewed_jobs = []
        for job in case['jobs'].values():
            fit_root = Path(job['fit_root'])
            original_path = fit_root/'evaluation.json'
            original = json.loads(original_path.read_text())
            review = json.loads((fit_root/'surface_review_v1/evaluation.json').read_text())
            fit_provenance = json.loads((fit_root/'provenance.json').read_text())
            fit_report = json.loads((fit_root/'fit_report.json').read_text())
            fit_input = Path(fit_provenance['fit_input'])
            manifest = json.loads((fit_input/'manifest.json').read_text())
            input_checks.append(check_hash(fit_input/'manifest.json', fit_provenance['fit_manifest_sha256']))
            input_checks.append(check_hash(fit_input/manifest['camera_file'], manifest['camera_sha256']))
            input_checks += [check_hash(fit_input/v['path'], v['sha256']) for v in manifest['masks']]
            input_checks += [check_hash(fit_input/p, digest) for p,digest in manifest['extra_input_sha256'].items()]
            truth_path = fit_input.parent/'truth/manifest.json'
            truth = json.loads(truth_path.read_text())
            input_checks.append(check_hash(truth_path, original['truth_manifest_sha256']))
            input_checks += [check_hash(truth_path.parent/truth[p], truth[h]) for p,h in [
                ('latent_file','latent_sha256'), ('heldout_camera_file','heldout_camera_sha256'),
                ('heldout_mask_file','heldout_mask_sha256')]]
            control_checks.append(not fit_provenance['active_truth_codes_read'] and original['truth_read_after_fit']
                and fit_report['pose_and_cameras_unchanged'] and not manifest['frozen_labels']
                and original['active_latent_dimensions']==280)
            original_preserved.append(check_hash(original_path, review['original_evaluation_sha256']))
            gates_unchanged.append(original['acceptance_limits'] == review['acceptance_limits'] == plan['expansion_config']['acceptance'])
            finite_unchanged.append(review['finite_original_metrics_unchanged'])
            fit_checks += [check_hash(Path(p), digest) for p, digest in review['mesh_input_sha256'].items()]
            fit_checks += [check_hash(PROJECT_ROOT/'scripts'/p, digest) for p, digest in review['metric_code_sha256'].items()]
            reviewed_jobs.append(review['status'])
            metrics.append({'case': name, 'seed': job['seed'], 'original_status': original['status'],
                'reviewed_status': review['status'], 'evaluation': str(fit_root/'surface_review_v1/evaluation.json'),
                'gradient': review['gradient_max_relative_error'],
                'surface_gain': review['surface_mean_improvement_fraction'],
                'minimum_tooth_surface_gain': min(v['mean_improvement_fraction'] for v in review['surface_per_tooth'].values()),
                'maximum_tooth_p95_ratio': max(v['final']['symmetric_sampled_surface_p95_dmm']/v['initial']['symmetric_sampled_surface_p95_dmm'] for v in review['surface_per_tooth'].values()),
                'latent_error_increased_teeth': sum(v['final_rms_error_training_std'] > v['initial_rms_error_training_std'] for v in review['latent_error_diagnostic_only'].values()),
                'finite_candidate_repairs': sum(d['nonfinite_candidate_distances_repaired'] for v in review['surface_per_tooth'].values() for stage in ['initial','final'] for d in v[stage]['distance_diagnostics']),
                'failed_checks': [k for k,v in review['checks'].items() if not v]})
        review_states[name] = 'PASS' if len(reviewed_jobs)==len(plan['definition']['seeds']) and all(s=='SHAPE_ONLY_SUBSET_PASS' for s in reviewed_jobs) else 'FAIL'
    checks.update({'frozen_and_source_inputs_unchanged': all(source_checks), 'child_code_snapshots_unchanged': all(child_checks),
        'review_meshes_and_code_unchanged': all(fit_checks), 'original_evaluations_preserved': all(original_preserved),
        'acceptance_limits_unchanged': all(gates_unchanged), 'finite_original_metrics_unchanged': all(finite_unchanged),
        'fit_and_truth_packages_unchanged': all(input_checks), 'known_pose_camera_and_truth_isolation_recorded': all(control_checks),
        'all_planned_cases_and_seeds_evaluated': len(metrics)==len(plan['selection']['selected'])*len(plan['definition']['seeds'])})
    report = {'status': 'PROVENANCE_VERIFIED' if all(checks.values()) else 'PROVENANCE_FAILED',
        'checks': checks, 'hash_checks': verified, 'cases_original': original_states, 'cases_reviewed': review_states,
        'metrics': metrics, 'fit_count': len(metrics), 'tooth_fit_count': len(metrics)*len(plan['definition']['active_labels']),
        'max_gradient_relative_error': max(m['gradient'] for m in metrics),
        'min_tooth_surface_gain': min(m['minimum_tooth_surface_gain'] for m in metrics),
        'max_tooth_p95_ratio': max(m['maximum_tooth_p95_ratio'] for m in metrics),
        'latent_error_increased_teeth': sum(m['latent_error_increased_teeth'] for m in metrics),
        'scope': 'Hashes verified before any future evaluator integration; original evaluation and mesh artifacts retained.'}
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ['status','checks','fit_count','tooth_fit_count','max_gradient_relative_error','min_tooth_surface_gain','max_tooth_p95_ratio','latent_error_increased_teeth']}))


if __name__ == '__main__':
    main()
