"""Download verified NVIDIA compiler components into this workspace only."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1] / '.runtime'
BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'
VERSION = '13.2.1'


def download(url, path, size):
    if size < 40_000_000:
        subprocess.run(['curl.exe', '--silent', '--show-error', '--fail', '--location', '--connect-timeout', '20',
                        '--max-time', '240', '--retry', '2', '--output', str(path), url], check=True)
        return
    chunk = (size + 15) // 16
    def part(index):
        start, end = index * chunk, min(size, (index + 1) * chunk) - 1
        target = path.with_suffix(f'.part{index:02d}')
        if not target.exists() or target.stat().st_size != end - start + 1:
            subprocess.run(['curl.exe', '--silent', '--show-error', '--fail', '--location', '--connect-timeout', '20',
                            '--max-time', '240', '--retry', '2', '--range', f'{start}-{end}', '--output', str(target), url], check=True)
        assert target.stat().st_size == end - start + 1, 'server did not honor range'
        print('Downloaded segment', index, flush=True)
        return target
    with ThreadPoolExecutor(max_workers=16) as pool:
        parts = list(pool.map(part, range(16)))
    with path.open('wb') as output:
        for piece in parts:
            with piece.open('rb') as source: shutil.copyfileobj(source, output)


def main():
    cache, dest = ROOT / 'downloads', ROOT / 'cuda-13.2.1'
    cache.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    metadata = cache / f'redistrib_{VERSION}.json'
    if not metadata.exists():
        urllib.request.urlretrieve(BASE + metadata.name, metadata)
    manifest = json.loads(metadata.read_text())
    selected = {}
    for key in ('cuda_nvcc', 'cuda_crt', 'libnvvm', 'cuda_cudart', 'cuda_cccl'):
        row = manifest[key]['windows-x86_64']
        path = cache / Path(row['relative_path']).name
        if not path.exists() or path.stat().st_size != int(row['size']):
            print('Download', key, row['size'], flush=True)
            download(BASE + row['relative_path'], path, int(row['size']))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], key
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                parts = Path(entry.filename).parts[1:]
                if not parts or entry.is_dir(): continue
                # Preserve separate vendor licenses when flattening component roots.
                relative = Path(*parts)
                if len(parts) == 1: relative = Path('licenses') / key / relative
                target = (dest / relative).resolve()
                assert target.is_relative_to(dest.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
        selected[key] = row
        print('Verified/extracted', key, flush=True)
    (dest / 'provenance.json').write_text(json.dumps(dict(source=BASE + metadata.name, components=selected), indent=2))
    print(dest)


if __name__ == '__main__': main()
