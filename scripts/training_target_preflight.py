"""Run on the actual training host before a pilot. No rendering dependency."""
import argparse,json,os,platform,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/DMM'))
sys.path.insert(0,str(ROOT/'tests'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from dmm.validation import require,write_json,sha256,read_json
from dmm.provenance import source_fingerprint,training_runtime
from training.arch_training import load_training_config
from data.arch_dataset import validate_patient_splits
from training_smoke import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--expected-capability',default='12.0')
    args=p.parse_args();output=Path(args.output).resolve()
    require(not output.exists(),'preflight output must be fresh');output.mkdir(parents=True)
    report=dict(status='RUNNING',source_sha256=source_fingerprint(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    from verify_training_package import verify
    try:
        report['source_identity']=verify()['source_identity']
        require(torch.cuda.is_available(),'CUDA unavailable')
        capability='.'.join(map(str,torch.cuda.get_device_capability()))
        require(capability==args.expected_capability,'unexpected GPU capability')
        report['runtime']=training_runtime('cuda')
        x=torch.randn(512,512,device='cuda',requires_grad=True)
        loss=(x@x.T).square().mean();loss.backward();torch.cuda.synchronize()
        require(bool(torch.isfinite(x.grad).all()),'GPU basic backward failed')
        del x,loss
        # Scope only training tests: no nvdiffrast prerequisite or skipped GPU gates.
        suite=unittest.TestSuite()
        for name in ('test_dual_arch','test_training_handoff.InitializationTests','test_training_readiness.TrainingTests'):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
        with (output/'tests.log').open('w',encoding='utf-8') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        require(result.wasSuccessful() and not result.skipped,'training tests failed or skipped')
        report['tests_passed']=result.testsRun
        datasets=[]
        for arch in ('upper','lower'):
            config=ROOT/'configs/training_handoff_v1'/f'{arch}_pilot.json'
            print(f'Validating all {arch} input records and hashes...',flush=True)
            resolved,data,_,_=load_training_config(config);datasets.append(data)
            record=run(config,steps=20,quota=16,offsurface=32)
            write_json(output/f'{arch}_optimizer_smoke.json',record)
            require(record['loss_ratio']<1.,f'{arch} fixed-sample smoke loss did not decrease')
            full=run(config,steps=2,quota=resolved['points_per_component'],offsurface=resolved['offsurface_points'])
            write_json(output/f'{arch}_configured_sampling_smoke.json',full)
            require(full['loss_ratio']<1.,f'{arch} configured-sampling smoke loss did not decrease')
            print(f'{arch} optimizer smoke passed; loss ratio {record["loss_ratio"]:.4f}',flush=True)
        report['patients']=len(validate_patient_splits(*datasets))
        report.update(status='TARGET_READY_FOR_BOUNDED_PILOT',formal_training_quality_accepted=False,
                      scope='full input integrity; actual target GPU; training tests; two fixed-sample optimizer smokes; no geometric/generalization acceptance')
    except BaseException as exc:
        report.update(status='TARGET_PREFLIGHT_FAILED',error=str(exc))
        raise
    finally:write_json(output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
