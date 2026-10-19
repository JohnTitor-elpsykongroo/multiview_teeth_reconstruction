"""Exact frozen R4 component decoding with reusable, provenance-bound grid caches."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / '.runtime/dmm_r4_best_val_20261015/package'
sys.path.insert(0, str(PACKAGE / 'third_party/DMM'))
from dmm.bundle import load_bundle


class Engine:
    def __init__(self, progress, resolution=160, device='cuda:0'):
        self.progress = progress
        self.n = resolution
        self.device = device
        self.models = {}
        self.bases = {}
        self.current = {}
        self.codes = {}
        self.placement = {}
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False

    def metadata(self):
        return dict(resolution=self.n, device=self.device,
                    models={a: dict(id=b.metadata['model_id'], epoch=b.metadata['training_epoch'],
                                     labels=list(b.model.labels[1:]), dimensions=10)
                            for a, b in self.models.items()},
                    latent_mapping='z = training_mean + Cholesky @ q',
                    gum_policy='fixed_training_mean', model_unit_mm=50,
                    placement=self.placement, default_opening_degrees=6)

    def prepare(self):
        for arch in ('upper', 'lower'):
            self.progress(f'加载{"上颌" if arch == "upper" else "下颌"} R4 模型', 0)
            bundle = load_bundle(PACKAGE / 'model_bundle' / arch / 'model.json', arch, self.device)
            self.models[arch] = bundle
            binding = dict(weights=bundle.metadata['weights']['sha256'],
                           statistics=bundle.metadata['latent_statistics']['sha256'],
                           source=bundle.metadata['local_source_sha256'],
                           resolution=self.n, version='component-sdf-logsigmoid-v1')
            signature = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:20]
            folder = ROOT / '.runtime/dmm_latent_viewer/cache' / signature
            folder.mkdir(parents=True, exist_ok=True)
            self.bases[arch] = {}
            self.current[arch] = {}
            self.codes[arch] = {str(k): [0.] * 10 for k in bundle.model.labels[1:]}
            for index, label in enumerate(bundle.model.labels):
                filename = folder / f'component_{label}.npy'
                if not filename.exists():
                    value = self.component_grid(arch, label, bundle.means[label],
                                                f'{arch} · 初次准备组件 {label}')
                    temp = filename.with_suffix('.tmp.npy')
                    np.save(temp, value)
                    temp.replace(filename)
                array = np.load(filename, mmap_mode='r', allow_pickle=False)
                if array.shape != (2, self.n ** 3) or array.dtype != np.float32:
                    raise ValueError(f'缓存格式错误：{filename}')
                self.bases[arch][label] = array
                self.current[arch][label] = array
                self.progress(f'{arch} · 组件 {label} 已就绪', (index + 1) / 15)
            (folder / 'binding.json').write_text(json.dumps(binding, indent=2), encoding='utf-8')

    def points(self, arch, begin, end):
        box = np.asarray(self.models[arch].metadata['sampling_domain_model'])
        # Match the deployment's torch.linspace grid exactly.
        axes = [torch.linspace(float(box[0, i]), float(box[1, i]), self.n, device=self.device)
                for i in range(3)]
        ids = torch.arange(begin, end, device=self.device)
        return torch.stack([axes[0][ids // (self.n * self.n)],
                            axes[1][ids // self.n % self.n], axes[2][ids % self.n]], -1)

    @torch.no_grad()
    def component_grid(self, arch, label, code, message):
        result = np.empty((2, self.n ** 3), dtype=np.float32)
        size = self.n ** 3
        last = 0
        for begin in range(0, size, 32768):
            end = min(size, begin + 32768)
            c = self.models[arch].model.component(self.points(arch, begin, end), code, label)
            result[0, begin:end] = c['sdf'].cpu().numpy()
            result[1, begin:end] = torch.nn.functional.logsigmoid(c['logit']).cpu().numpy()
            if time.monotonic() - last > .4:
                self.progress(message, end / size)
                last = time.monotonic()
        if not np.isfinite(result).all():
            raise ValueError('模型输出含非有限值')
        return result

    def blend(self, arch):
        components = list(self.current[arch].values())
        result = np.empty(self.n ** 3, dtype=np.float32)
        for begin in range(0, len(result), 131072):
            end = min(len(result), begin + 131072)
            fields = np.stack([a[0, begin:end] for a in components])
            logs = np.stack([a[1, begin:end] for a in components])
            weights = np.exp(logs - logs.max(axis=0, keepdims=True))
            result[begin:end] = (fields * weights).sum(axis=0) / weights.sum(axis=0)
        return result.reshape((self.n,) * 3)

    @torch.no_grad()
    def decode(self, arch, q):
        start = time.monotonic()
        bundle = self.models[arch]
        for label in bundle.model.labels[1:]:
            requested = np.asarray(q[str(label)], dtype=np.float32)
            if np.array_equal(requested, np.asarray(self.codes[arch][str(label)], dtype=np.float32)):
                continue
            if not np.any(requested):
                self.current[arch][label] = self.bases[arch][label]
            else:
                code = bundle.means[label] + bundle.cholesky[label] @ torch.as_tensor(requested, device=self.device)
                self.current[arch][label] = self.component_grid(arch, label, code, f'正在解码牙位 {label}')
            self.codes[arch][str(label)] = requested.tolist()
        self.progress('合成整颌隐式场并提取网格', .88)
        volume = self.blend(arch)
        if not (np.isfinite(volume).all() and volume.min() < 0 < volume.max()):
            raise ValueError('当前 latent 无有效零等值面；请减小滑块幅度')
        boundary = np.concatenate([volume[0].ravel(), volume[-1].ravel(), volume[:, 0].ravel(),
                                   volume[:, -1].ravel(), volume[:, :, 0].ravel(), volume[:, :, -1].ravel()])
        if np.any(boundary <= 0):
            raise ValueError('形状超出模型采样范围；保留上次网格，请减小 latent 幅度')
        box = np.asarray(bundle.metadata['sampling_domain_model'])
        v, f, _, _ = marching_cubes(volume, 0, spacing=(box[1] - box[0]) / (self.n - 1), allow_degenerate=False)
        v = np.ascontiguousarray(v + box[0], dtype=np.float32)
        codes = {0: bundle.means[0]}
        codes.update({k: bundle.means[k] + bundle.cholesky[k] @ torch.tensor(q[str(k)], dtype=torch.float32, device=self.device)
                      for k in bundle.model.labels[1:]})
        gum, labels = [], []
        for begin in range(0, len(v), 8192):
            result = bundle.model.query(torch.as_tensor(v[begin:begin + 8192], device=self.device), codes)
            weights = result['weights'].cpu().numpy()
            gum.append(weights[:, 0])
            labels.append(np.asarray(result['labels'])[weights.argmax(axis=1)])
        normal = np.zeros_like(v)
        cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        for k in range(3):
            np.add.at(normal, f[:, k], cross)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        mm = v * 50
        if arch not in self.placement:
            # Display-only placement: align anterior extent, separate occlusal extremes.
            # Freeze this transform at the mean so latent edits do not move the whole arch.
            self.placement[arch] = dict(x=-float((mm[:, 0].min() + mm[:, 0].max()) / 2),
                                        y=-float(mm[:, 1].min()),
                                        z=-float(mm[:, 2].min() if arch == 'upper' else mm[:, 2].max()),
                                        hinge_y=float(np.ptp(mm[:, 1])), hinge_z=0.)
        return dict(arch=arch, model_id=bundle.metadata['model_id'], q=q, resolution=self.n,
                    vertices=mm.round(5).tolist(), normals=normal.round(6).tolist(), faces=f.tolist(),
                    gum=np.concatenate(gum).round(5).tolist(), labels=np.concatenate(labels).tolist(),
                    placement=self.placement[arch], seconds=round(time.monotonic() - start, 3),
                    count=dict(vertices=len(v), triangles=len(f)),
                    boundary_min=float(boundary.min()))
