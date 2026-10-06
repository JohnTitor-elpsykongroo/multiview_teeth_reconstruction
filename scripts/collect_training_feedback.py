"""Collect bounded training diagnostics, failures and checkpoint hashes, without weights/data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'.json', '.jsonl', '.log', '.txt', '.png', '.jpg'}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def collect(runs, logs, output, max_file_mb=64):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('feedback output must be fresh')
    if max_file_mb <= 0:
        raise ValueError('max-file-mb must be positive')
    entries, run_records, omitted = [], [], []
    for index, raw in enumerate(runs):
        run = Path(raw).resolve()
        if not run.is_dir():
            raise ValueError(f'run directory missing: {run}')
        record = dict(archive_root=f'run_{index:02d}', original_path=str(run), checkpoints=[])
        # Only direct run files: do not recursively collect datasets, truth or snapshots.
        for path in sorted(run.iterdir()):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix in {'.pth', '.pt', '.ckpt'}:
                record['checkpoints'].append(dict(name=path.name, bytes=path.stat().st_size, sha256=digest(path)))
            elif path.suffix.lower() in ALLOWED:
                entries.append((path, f'run_{index:02d}/{path.name}'))
        if not any(name.startswith(f'run_{index:02d}/') for _, name in entries):
            raise ValueError(f'no diagnostic files in {run}')
        run_records.append(record)
    for index, raw in enumerate(logs):
        path = Path(raw).resolve()
        if not path.is_file():
            raise ValueError(f'log missing: {path}')
        entries.append((path, f'logs/{index:02d}_{path.name}'))
    installed = ROOT / '.venv/installed-requirements.txt'
    if installed.is_file():
        entries.append((installed, 'collector_environment/installed-requirements.txt'))
    try:
        revision = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', '-C', str(ROOT),
                                             'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = dict(schema='dmm_training_feedback_v1', collector_commit=revision, runs=run_records,
                  training_revision_source='Use each run provenance.json/checkpoint; collector commit is not training identity.',
                  weights_included=False, files=[], omitted=omitted)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path, name in entries:
            size = path.stat().st_size
            if size > max_file_mb * 1024 * 1024:
                omitted.append(dict(path=name, bytes=size, reason='exceeds max-file-mb; send separately or raise limit'))
                continue
            archive.write(path, name)
            report['files'].append(dict(path=name, bytes=size, sha256=digest(path)))
        report['status'] = 'INCOMPLETE_OVERSIZE_FILES' if omitted else 'COLLECTED'
        archive.writestr('feedback.json', json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='Repeat for preflight, smoke and pilot directories.')
    parser.add_argument('--log', action='append', default=[], help='Repeat for launcher console logs.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-file-mb', type=int, default=64)
    args = parser.parse_args()
    report = collect(args.run, args.log, args.output, args.max_file_mb)
    print(json.dumps(dict(status=report['status'], output=str(Path(args.output).resolve()),
                         runs=len(report['runs']), files=len(report['files']), omitted=report['omitted']), ensure_ascii=False))
    raise SystemExit(2 if report['omitted'] else 0)
