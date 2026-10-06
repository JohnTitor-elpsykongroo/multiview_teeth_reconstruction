"""CPU train/val code telemetry in the exported empirical-prior coordinates.

No fitting, decoder queries, test access, bundle export or quality selection.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/DMM'))
from inspect_autonomy_latents import original_rows, probe_binding, sha
from summarize_autonomy_run import checked_evaluation


def export_statistics(z):
    """Same shrinkage/floor as arch_training.export_bundle; checked against it."""
    z = np.asarray(z,dtype=np.float64)
    if z.ndim != 2 or z.shape[1] == 0 or len(z) < z.shape[1]+1 or not np.isfinite(z).all():
        raise ValueError('insufficient original present train support or invalid codes')
    dimension = z.shape[1]
    covariance = np.atleast_2d(np.cov(z,rowvar=False,ddof=1))
    variance = np.trace(covariance)/dimension
    if variance <= 0 or not np.all(np.diag(covariance)>0):
        raise ValueError('zero original latent variance')
    covariance = .95*covariance+.05*np.diag(np.diag(covariance))
    eigenvalues,eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors*np.maximum(eigenvalues,1e-6*variance))@eigenvectors.T
    return z.mean(0),covariance,np.linalg.cholesky(covariance)


def component_prior(saved,label):
    rows = [row for row in original_rows(saved) if label == 0 or row['presence'][str(label)]]
    table = saved['system_state'][f'latents.{label}.weight'].detach().cpu().numpy()
    z = table[[row['embedding_row'] for row in rows]].astype(np.float64)
    mu,covariance,factor = export_statistics(z)
    return mu,covariance,factor,[row['case_id'] for row in rows]


def inspect(checkpoint,evaluation,probe_config=None):
    folder = Path(evaluation).resolve()
    provenance = json.loads((folder/'provenance.json').read_text())
    if provenance.get('split') not in ('train','val'):
        raise ValueError('code telemetry excludes test before checkpoint/metric access')
    result,provenance = checked_evaluation(folder)
    if sha(checkpoint) != provenance['checkpoint_sha256']:
        raise ValueError('code telemetry checkpoint SHA mismatch')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    hashes = {str(Path(checkpoint).resolve()):sha(checkpoint),str(folder/'evaluation.json'):sha(folder/'evaluation.json'),
              str(folder/'provenance.json'):sha(folder/'provenance.json')}
    hashes.update(probe_binding(provenance,saved,probe_config))
    eligible = {row['case_id']:row for row in saved['case_rows']
                if row['split'] == result['split'] and row['augmentation_parent_id'] is None
                and (result['split']!='train' or row['case_id'] in saved['training_case_ids'])}
    records = []
    for case in result['cases']:
        record = dict(case_id=case['case_id'],geometry_status=case['status'],status='CODE_TELEMETRY_FAILED',components={})
        try:
            if case['case_id'] not in eligible:
                raise ValueError('case is not an original eligible checkpoint row')
            expected_mode = 'learned_train' if result['split']=='train' else 'temporary_fitted'
            if case.get('code_mode') != expected_mode:
                raise ValueError('unexpected code fitting mode')
            artifact = (folder/case['artifact_directory']).resolve()
            if not artifact.is_relative_to(folder):
                raise ValueError('code artifact outside evaluation directory')
            if not (artifact/'codes.npz').exists():
                raise ValueError('no saved codes: fresh evaluation required; old fit cannot be recovered')
            field = json.loads((artifact/'field.json').read_text())
            if field.get('codes_sha256') != sha(artifact/'codes.npz'):
                raise ValueError('field/code SHA mismatch')
            hashes.update({str(artifact/name):sha(artifact/name) for name in ('field.json','codes.npz')})
            with np.load(artifact/'codes.npz',allow_pickle=False) as arrays:
                labels = {0,*[int(k) for k,v in eligible[case['case_id']]['presence'].items() if v]}
                if arrays['model_unit_mm'].item()!=50 or set(arrays.files)!={'model_unit_mm',*[f'code_{label}' for label in labels]}:
                    raise ValueError('code artifact units/presence mismatch')
                for label in sorted(labels):
                    mu,covariance,factor,support = component_prior(saved,label)
                    code = np.asarray(arrays[f'code_{label}'],dtype=np.float64)
                    if code.shape!=mu.shape or not np.isfinite(code).all():
                        raise ValueError('invalid code dimension/numeric')
                    if result['split']=='train':
                        learned=saved['system_state'][f'latents.{label}.weight'][eligible[case['case_id']]['embedding_row']].detach().cpu().numpy().astype(np.float64)
                        if not np.array_equal(code,learned):
                            raise ValueError('saved train code differs from its checkpoint learned row')
                    q = np.linalg.solve(factor,code-mu)
                    if not np.isfinite(q).all(): raise ValueError('nonfinite standardized code')
                    record['components'][str(label)] = dict(original_present_trained_count=len(support),support_case_ids=support,
                        raw_mean_squared_code=float(np.square(code).mean()),
                        raw_mean_squared_delta_from_train_mean=float(np.square(code-mu).mean()),
                        standardized_q_rms=float(np.sqrt(np.square(q).mean())),
                        standardized_q=q.tolist(),bundle_gum_fixed=(label==0),
                        empirical_covariance_trace=float(np.trace(covariance)))
            record['status']='CODE_TELEMETRY_NOT_QUALITY_ACCEPTED'
        except (KeyError,OSError,ValueError,RuntimeError) as exc:
            record['error']=str(exc)
        records.append(record)
    if any(sha(path)!=digest for path,digest in hashes.items()): raise ValueError('input drift during code telemetry')
    return dict(schema='dmm_code_prior_telemetry_v1',split=result['split'],arch=saved['arch'],epoch=saved['epoch'],
                quality_accepted=False,requested=len(records),failed=sum(r['status']=='CODE_TELEMETRY_FAILED' for r in records),
                cases=records,input_sha256=hashes,script_sha256=sha(__file__),
                helper_sha256={name:sha(Path(__file__).with_name(name)) for name in
                               ('inspect_autonomy_latents.py','summarize_autonomy_run.py')},
                runtime=dict(torch=torch.__version__,numpy=np.__version__,device='cpu'),
                scope='Float64 exported-statistical-array coordinates; bundle runtime converts means/factors to float32. Gum q is hypothetical because bundle gum is fixed. No tolerance or code-collapse claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--evaluation',required=True)
    parser.add_argument('--probe-config')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('fresh output required')
    value=inspect(args.checkpoint,args.evaluation,args.probe_config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(requested=value['requested'],failed=value['failed'],output=str(args.output))))
