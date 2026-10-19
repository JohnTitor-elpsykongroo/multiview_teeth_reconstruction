"""Loopback-only HTML viewer and a serial, coalescing CUDA decode worker."""
import argparse
from collections import OrderedDict
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import traceback
from urllib.parse import urlparse, parse_qs
import uuid

from engine import Engine

STATIC = Path(__file__).parent / 'static'


class Service:
    def __init__(self, resolution):
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.ready = False
        self.error = None
        self.message = '启动模型服务'
        self.progress = 0
        self.pending = OrderedDict()
        self.jobs = OrderedDict()
        self.initial = {}
        self.engine = Engine(self.update, resolution)
        threading.Thread(target=self.worker, daemon=True).start()

    def update(self, message, progress):
        with self.lock:
            self.message, self.progress = message, float(progress)

    def worker(self):
        try:
            self.engine.prepare()
            for arch in ('upper', 'lower'):
                q = {str(k): [0.] * 10 for k in self.engine.models[arch].model.labels[1:]}
                self.initial[arch] = self.compress(self.engine.decode(arch, q))
            with self.lock:
                self.ready = True
                self.message, self.progress = '模型已就绪', 1
            print('DMM_VIEWER_READY', flush=True)
        except Exception as error:
            with self.lock:
                self.error = str(error)
            traceback.print_exc()
            return
        while True:
            with self.condition:
                self.condition.wait_for(lambda: bool(self.pending))
                arch, job_id = self.pending.popitem(last=False)
                job = self.jobs[job_id]
                job['status'] = 'running'
            try:
                result = self.engine.decode(arch, job['q'])
                with self.lock:
                    job.update(status='done', data=self.compress(result), seconds=result['seconds'], count=result['count'])
            except Exception as error:
                traceback.print_exc()
                with self.lock:
                    job.update(status='error', error=str(error))
            with self.lock:
                self.message = '模型已就绪'
                self.progress = 1
                # Bound retained payload memory; active jobs are never removed.
                done = [key for key, value in self.jobs.items() if value['status'] in ('done', 'error', 'superseded')]
                for key in done[:-16]:
                    del self.jobs[key]

    @staticmethod
    def compress(value):
        return gzip.compress(json.dumps(value, separators=(',', ':'), allow_nan=False).encode(), compresslevel=1)

    def submit(self, value):
        with self.condition:
            if not self.ready:
                raise ValueError('模型正在准备')
            arch, q = value.get('arch'), value.get('q')
            if arch not in ('upper', 'lower') or not isinstance(q, dict):
                raise ValueError('无效颌或 latent')
            expected = {str(k) for k in self.engine.models[arch].model.labels[1:]}
            if set(q) != expected:
                raise ValueError('必须提交该颌全部 14 颗牙的 latent')
            for vector in q.values():
                if not isinstance(vector, list) or len(vector) != 10 or any(type(v) not in (int, float) or not -5 <= v <= 5 for v in vector):
                    raise ValueError('每颗牙需要 10 个有限数值，范围 [-5,5]')
            q = {key: [float(v) for v in vector] for key, vector in q.items()}
            if arch in self.pending:
                self.jobs[self.pending[arch]]['status'] = 'superseded'
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = dict(status='queued', arch=arch, q=q)
            self.pending[arch] = job_id
            self.condition.notify()
            return job_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8877)
    parser.add_argument('--resolution', type=int, choices=[128, 160, 192, 256], default=160)
    args = parser.parse_args()
    service = Service(args.resolution)

    class Handler(BaseHTTPRequestHandler):
        def allowed(self):
            allowed = {f'127.0.0.1:{args.port}', f'localhost:{args.port}'}
            if self.headers.get('Host') not in allowed:
                return False
            origin = self.headers.get('Origin')
            return not origin or origin in {f'http://{h}' for h in allowed}

        def send(self, value, status=200, content_type='application/json; charset=utf-8', compressed=False):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if compressed:
                self.send_header('Content-Encoding', 'gzip')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.allowed():
                return self.send({'error': 'Loopback origin required'}, 403)
            url = urlparse(self.path)
            if url.path == '/api/status':
                with service.lock:
                    return self.send(dict(app='dmm-latent-viewer', ready=service.ready, error=service.error,
                                          message=service.message, progress=service.progress,
                                          metadata=service.engine.metadata() if service.ready else None))
            if url.path == '/api/job':
                job_id = parse_qs(url.query).get('id', [''])[0]
                with service.lock:
                    job = service.jobs.get(job_id)
                    return self.send({k: v for k, v in job.items() if k != 'data'} if job else {'error': 'job not found'}, 200 if job else 404)
            if url.path == '/api/mesh':
                query = parse_qs(url.query)
                with service.lock:
                    job_id = query.get('job', [None])[0]
                    data = service.jobs.get(job_id, {}).get('data') if job_id else service.initial.get(query.get('arch', [''])[0])
                return self.send(data, compressed=True) if data else self.send({'error': 'mesh not ready'}, 404)
            static = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if url.path in static:
                name, mime = static[url.path]
                return self.send((STATIC / name).read_bytes(), content_type=mime)
            return self.send({'error': 'not found'}, 404)

        def do_POST(self):
            if not self.allowed():
                return self.send({'error': 'Loopback origin required'}, 403)
            if self.path != '/api/decode':
                return self.send({'error': 'not found'}, 404)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 65536 or not self.headers.get('Content-Type', '').startswith('application/json'):
                    raise ValueError('Expected bounded JSON body')
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError('Expected object')
                return self.send({'id': service.submit(value)}, 202)
            except (ValueError, TypeError) as error:
                return self.send({'error': str(error)}, 400)

        def log_message(self, fmt, *values):
            if self.path not in ('/api/status',) and not self.path.startswith('/api/job'):
                print(time.strftime('%H:%M:%S'), fmt % values, flush=True)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'DMM viewer listening on http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
