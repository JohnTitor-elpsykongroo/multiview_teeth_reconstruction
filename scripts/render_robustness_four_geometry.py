"""Run the frozen diagnostic renderer with accurate dynamic captions only."""
import argparse
from pathlib import Path
import sys
from run_forward_check import PROJECT_ROOT
from run_joint_robustness_four import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_root', type=Path)
    parser.add_argument('job_id')
    args = parser.parse_args()
    batch = args.batch_root.resolve(strict=True)
    if not batch.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('Outside project runs')
    verify(batch)
    script = batch/'engine/scripts/render_joint_review.py'
    run = batch/'engine/runs/jobs'/args.job_id/'fit_attempt_01'
    if not run.resolve(strict=True).is_relative_to((batch/'engine/runs/jobs').resolve()):
        raise ValueError('Invalid job')
    source = script.read_text(encoding='utf-8')
    old = 'Joint: 40 latent + 6 pose variables'
    new = 'Observed targets + holdout | {evaluation["active_latent_dimensions"]} latent + {evaluation["total_dimensions"]-evaluation["active_latent_dimensions"]} pose variables'
    if source.count(old) != 1:
        raise ValueError('Unexpected frozen caption')
    source = source.replace(old, new)
    # A fresh process isolates frozen modules from imports in this wrapper.
    import subprocess
    command = 'import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); p=Path(sys.argv[2]); source=p.read_text(encoding="utf-8"); source=source.replace(sys.argv[3],sys.argv[4]); sys.argv=[str(p),sys.argv[5]]; exec(compile(source,str(p),"exec"),{"__name__":"__main__","__file__":str(p)})'
    subprocess.run([sys.executable, '-B', '-c', command, str(script.parent), str(script), old, new, str(run)], check=True)


if __name__ == '__main__':
    main()
