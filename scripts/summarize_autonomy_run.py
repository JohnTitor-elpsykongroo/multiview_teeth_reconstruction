"""CPU-only train/val research tables. Does not select or approve a model."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error_family(error):
    for text, family in (
        ('no finite zero surface', 'no_zero_surface'),
        ('non-positive domain boundary', 'nonpositive_domain_boundary'),
        ('missing blended surface component', 'missing_present_tooth'),
        ('nonfinite', 'nonfinite_numeric'),
        ('gradient', 'invalid_gradient'),
        ('hash', 'integrity_mismatch'),
    ):
        if text in error:
            return family
    return 'other_preserved_error'


def finite(value, field):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f'nonfinite/invalid {field}')
    return value


def checked_evaluation(folder):
    # Check split before reading any metric content; this R2/R3 tool excludes test.
    provenance = read(folder/'provenance.json')
    if provenance.get('split') not in ('train', 'val'):
        raise ValueError('R2/R3 summary excludes test evaluations')
    result = read(folder/'evaluation.json')
    if result['split'] != provenance['split']:
        raise ValueError('evaluation/provenance split mismatch')
    cases = result['cases']
    ids = [c['case_id'] for c in cases]
    if len(ids) != len(set(ids)) or result['requested_case_count'] != len(cases):
        raise ValueError('duplicate cases or requested denominator mismatch')
    failures = []
    scores = []
    for case in cases:
        if case['split'] != result['split']:
            raise ValueError('mixed split evaluation')
        if case['status'] == 'FAILED':
            failures.append(case['case_id'])
            continue
        if case['status'] != 'EVALUATED_NOT_QUALITY_ACCEPTED':
            raise ValueError('unknown case status')
        geometry = case['geometry']
        scores.append(finite(geometry['score_mm'], 'case score'))
        teeth = geometry['per_tooth']
        if not teeth:
            raise ValueError('valid geometry has no tooth metrics')
        means = [finite(t['observed_to_mesh_sample_mean_mm'], 'tooth score') for t in teeth.values()]
        if abs(statistics.mean(means)-geometry['score_mm']) > 1e-9:
            raise ValueError('case score is not equal-tooth mean')
    if set(result['failures']) != set(failures) or result['failed_case_count'] != len(failures) or result['successful_case_count'] != len(scores):
        raise ValueError('failure denominator mismatch')
    if failures:
        if result['score_mm'] is not None:
            raise ValueError('failed cohort cannot have a selection score')
    elif not scores or abs(finite(result['score_mm'], 'cohort score')-statistics.mean(scores)) > 1e-9:
        raise ValueError('cohort score is not equal-case mean')
    return result, provenance


def write_table(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mesh_shape_statistics(path):
    """Generated-area diagnostics, including gum/caps; no anatomical threshold."""
    import numpy as np
    with np.load(path, allow_pickle=False) as arrays:
        vertices = arrays['vertices_model'] * float(arrays['model_unit_mm'])
        faces, labels = arrays['faces'], arrays['face_labels']
    if not np.isfinite(vertices).all() or len(faces) != len(labels):
        raise ValueError('invalid mesh artifact')
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1)/2
    total = float(area.sum())
    if total <= 0:
        raise ValueError('empty generated mesh area')
    parents = list(range(len(vertices)))
    def find(vertex):
        vertex = int(vertex)
        while parents[vertex] != vertex:
            parents[vertex] = parents[parents[vertex]]
            vertex = parents[vertex]
        return vertex
    for a,b,c in faces:
        owner = find(a)
        parents[find(b)] = owner
        parents[find(c)] = owner
    components = {}
    for face, value in zip(faces, area):
        owner = find(face[0])
        components[owner] = components.get(owner, 0.) + float(value)
    sizes = sorted(components.values(), reverse=True)
    return dict(mesh_area_mm2=total, mesh_gum_area_fraction=float(area[labels == 0].sum())/total,
        mesh_largest_component_area_fraction=sizes[0]/total, mesh_face_component_count=len(sizes),
        mesh_component_areas_mm2=json.dumps(sizes), mesh_extent_mm=json.dumps(np.ptp(vertices, axis=0).tolist()),
        mesh_area_by_fdi_mm2=json.dumps({str(int(label)):float(area[labels == label].sum()) for label in np.unique(labels)}))


def summarize(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if not run.is_dir() or output.exists():
        raise ValueError('run must exist and output must be fresh')
    cases, teeth, curves, evaluations, inputs = [], [], [], [], {}
    failure_counts = {}
    # Only completed evaluations; an active writer's partial records are not silently scored.
    for path in sorted(run.glob('*/evaluation.json')):
        result, provenance = checked_evaluation(path.parent)
        inputs[str(path)] = sha(path)
        inputs[str(path.parent/'provenance.json')] = sha(path.parent/'provenance.json')
        evaluations.append(dict(evaluation=path.parent.name, split=result['split'],
            requested=result['requested_case_count'], successful=result['successful_case_count'],
            failed=result['failed_case_count'], score_mm=result['score_mm'],
            training_epoch=provenance['training_epoch'], checkpoint_sha256=provenance['checkpoint_sha256'],
            config_sha256=provenance['config_sha256'], budget=result['budget']))
        for record in result['cases']:
            geometry = record.get('geometry', {})
            prefix = dict(evaluation=path.parent.name, split=result['split'], case_id=record['case_id'],
                patient_id=record['patient_id'], status=record['status'],
                checkpoint_sha256=provenance['checkpoint_sha256'])
            family = error_family(record.get('error', '')) if record['status'] == 'FAILED' else ''
            if family:
                failure_counts[family] = failure_counts.get(family, 0) + 1
            artifact = path.parent/record.get('artifact_directory', '__no_artifact__')
            for artifact_name in ('field.json', 'mesh.npz'):
                if (artifact/artifact_name).is_file():
                    inputs[str(artifact/artifact_name)] = sha(artifact/artifact_name)
            field = read(artifact/'field.json') if (artifact/'field.json').is_file() else {}
            mesh_shape = mesh_shape_statistics(artifact/'mesh.npz') if (artifact/'mesh.npz').is_file() else {}
            cases.append(dict(prefix, error=record.get('error', ''), failure_family=family,
                code_mode=record.get('code_mode'), fit_steps=record.get('fit_steps'),
                fit_points=record.get('fit_points'), evaluation_points=record.get('evaluation_points'),
                fit_index_sha256=record.get('fit_index_sha256'), evaluation_index_sha256=record.get('evaluation_index_sha256'),
                score_mm=geometry.get('score_mm'), connected_components=geometry.get('connected_components'),
                boundary_edges=geometry.get('boundary_edges'), nonmanifold_edges=geometry.get('nonmanifold_edges'),
                sdf_abs_mean_model=record.get('sdf_abs_mean_model'), normal_cosine_error=record.get('normal_cosine_error'),
                semantic_error=record.get('semantic_error'),
                gradient_norm_quantiles=json.dumps(record.get('gradient_norm_quantiles')),
                field_min=field.get('minimum'), field_max=field.get('maximum'),
                field_negative_fraction=field.get('negative_fraction'),
                field_boundary_nonpositive_count=field.get('boundary_nonpositive_count'), seconds=record.get('seconds'), **mesh_shape))
            for label in sorted(set(record.get('components', {})) | set(geometry.get('per_tooth', {})), key=int):
                component = record.get('components', {}).get(label, {})
                mesh = geometry.get('per_tooth', {}).get(label, {})
                teeth.append(dict(prefix, fdi=label, geometry_metric_available=bool(mesh),
                    observed_mean_mm=mesh.get('observed_to_mesh_sample_mean_mm'),
                    observed_point_p95_mm=mesh.get('observed_to_mesh_sample_p95_mm'),
                    generated_mean_mm_diagnostic=mesh.get('generated_to_observed_sample_mean_mm'),
                    generated_p95_mm_diagnostic=mesh.get('generated_to_observed_sample_p95_mm'),
                    points=component.get('points'), own_sdf_abs_mean_model=component.get('own_sdf_abs_mean_model'),
                    own_blend_mean=component.get('own_blend_mean'), own_semantic_error=component.get('own_semantic_error'),
                    own_gradient_norm_median=component.get('own_gradient_norm_median')))
    for path in sorted(run.glob('*/progress.jsonl')):
        content = path.read_text(encoding='utf-8-sig')
        # Snapshot only complete JSONL records; skip an unfinished append, retain its source hash.
        lines = content.splitlines()
        if content and not content.endswith('\n'):
            lines = lines[:-1]
        inputs[str(path)] = hashlib.sha256(content.encode()).hexdigest()
        for line in lines:
            record = json.loads(line)
            validation = record.get('validation') or {}
            diagnostics = record.get('optimizer_diagnostics', {})
            curve = dict(training=path.parent.name, epoch=record['epoch'], mean_train_loss=record['mean_train_loss'],
                best_train_loss=record['best_train_loss'], best_val_score=record.get('best_val_score'),
                val_score_mm=validation.get('score_mm'), val_failed_cases=len(validation['failures']) if validation else None,
                training_seconds=record.get('training_seconds'), epoch_seconds_before_checkpoint=record.get('epoch_seconds_before_checkpoint'),
                clipped_step_fraction=diagnostics.get('clipped_step_fraction'))
            for key, value in record['mean_terms'].items():
                curve['mean/'+key] = value
                curve['denominator/'+key] = record['term_case_counts'][key]
            for key in ('preclip_norm', 'clip_scale'):
                for quantile, value in diagnostics.get(key, {}).items():
                    curve[key+'/'+quantile] = value
            curves.append(curve)
    summary = dict(schema='dmm_autonomy_summary_v1', run=str(run),
        scope='train/val only; no test; all failed requested cases retained; no automatic quality approval',
        evaluations=evaluations, completed_evaluation_count=len(evaluations),
        failure_counts=failure_counts, training_epochs_observed=len(curves), input_sha256=inputs,
        source_script_sha256=sha(__file__), quality_accepted=False,
        metric_notes='equal-tooth then equal-case observed-to-mesh-sample mm; reverse diagnostic includes unsupported generated surface; field SDF uses model units (50mm/unit); normals are cosine errors')
    output.mkdir(parents=True, exist_ok=False)
    write_table(output/'cases.csv', cases)
    write_table(output/'teeth.csv', teeth)
    write_table(output/'training_curves.csv', curves)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# Autonomous research evidence snapshot', '', summary['scope'], '',
             '| Evaluation | Split | Valid / requested | Mean mm |', '| --- | --- | ---: | ---: |']
    for evaluation in evaluations:
        score = 'null (failed cohort)' if evaluation['score_mm'] is None else f"{evaluation['score_mm']:.6f}"
        lines.append(f"| {evaluation['evaluation']} | {evaluation['split']} | {evaluation['successful']} / {evaluation['requested']} | {score} |")
    lines += ['', 'No completed evaluations yet.' if not evaluations else 'See cases.csv and teeth.csv for every failure and tooth diagnostic.',
              '', summary['metric_notes'], '', 'This snapshot does not select a checkpoint, rank different protocols or authorize R3/R4.']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summary = summarize(args.run, args.output)
    print(json.dumps(dict(completed_evaluations=summary['completed_evaluation_count'],
                         training_epochs=summary['training_epochs_observed'], output=str(Path(args.output).resolve()))))
