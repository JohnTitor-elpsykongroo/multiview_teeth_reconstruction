"""Isolated diagnostic utilities; original fitting run is read-only."""
import hashlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
OUT = Path((PROJECT / 'tmp/dmm_diagnostic_current.txt').read_text('utf-8-sig'))
PLAN = json.loads((OUT / 'plan.json').read_text('utf-8-sig'))
OLD = Path(PLAN['source_run'])
sys.path.insert(0, str(OLD / 'runtime_source/DMM'))
from dmm.scene import load_scene
from dmm.surface import SurfaceConfig
from dmm.rendering import RenderConfig
from dmm.semanticxy import SemanticXYConfig, SemanticXYMatcher
from dmm.fitting import FitConfig, restore_parameters, _direction
from dmm.provenance import source_fingerprint
import dmm.surface as surface
import dmm.recovery as recovery
import dmm.root_solver as root_module

SURFACE = SurfaceConfig.load(OLD / 'configs/surface_static_v1.json')
RENDER = RenderConfig.load(OLD / 'configs/render_pilot256.json')
FIT = FitConfig.load(OLD / 'configs/fit_pilot40.json')
MATCH = SemanticXYConfig.load(OLD / 'configs/semanticxy_static_v1.json')
MANIFEST = OLD / 'scenes/L428SD7J_pilot256/fit_input/manifest.json'

def write(name, data):
    path = OUT / name
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temp.replace(path)

def event(**data):
    data['time'] = time.time()
    with (OUT / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(data)+'\n')
    print(json.dumps(data), flush=True)

def scene_at(which='final'):
    scene = load_scene(MANIFEST, 'cuda:0')
    if which == 'final':
        restore_parameters(scene, json.loads((OLD / 'fit_pilot_008_resume_persistent/checkpoint.json').read_text('utf-8'))['parameters'])
    for state in scene.arches.values():
        state.decoder.requires_grad_(False)
        state.precise_decoder.requires_grad_(False)
        state.pose_delta.requires_grad_(False)
        for q in state.q.values():
            q.requires_grad_(True)
    return scene

def install_execution_helpers(scene):
    """Lossless saved-tensor offload plus exact root/component memoization."""
    frozen = {(p.device.index, p.untyped_storage().data_ptr()) for p in scene.parameters() if p.is_cuda and not p.requires_grad}
    def pack(t):
        resident = not t.is_cuda or (t.device.index, t.untyped_storage().data_ptr()) in frozen
        return t.device, t.detach() if resident else t.detach().cpu()
    def unpack(saved):
        return saved[1].to(saved[0])
    original_gradient = surface._field_gradient
    def field_gradient(*args, **kwargs):
        with torch.autograd.graph.saved_tensors_hooks(lambda t:t.detach(), lambda t:t):
            return original_gradient(*args, **kwargs)
    surface._field_gradient = field_gradient
    original_solver = root_module.solve_normal_roots
    root_source = source_fingerprint()+str(torch.__version__)
    cache = OUT / 'root_cache'
    cache.mkdir(exist_ok=True)
    counts = dict(old_root_hits=0, new_root_hits=0, solved=0, failures=0)
    def solve(state, anchors, normal, config, field_gradient, box):
        metadata = state.bundle.metadata
        key = hashlib.sha256((root_source+metadata['model_id']+metadata['weights']['sha256']+metadata['latent_statistics']['sha256']+repr(config)).encode())
        for value in (anchors, normal, box):
            a = value.detach().cpu().contiguous().numpy()
            key.update(str((a.dtype, a.shape)).encode()); key.update(a.tobytes())
        for label, value in sorted(state.codes().items()):
            key.update(str(label).encode()); key.update(value.detach().cpu().contiguous().numpy().tobytes())
        stem = key.hexdigest()+'.npz'
        for directory, label in ((cache, 'new_root_hits'), (OLD / 'exact_root_cache', 'old_root_hits')):
            path = directory / stem
            if path.exists():
                counts[label] += 1
                with np.load(path, allow_pickle=False) as data:
                    return tuple(torch.from_numpy(data[k].copy()).to(anchors.device) for k in ('roots','offset'))
        counts['solved'] += 1
        try:
            roots, offsets = original_solver(state, anchors, normal, config, field_gradient, box)
        except Exception as exc:
            counts['failures'] += 1
            details = dict(model=metadata['model_id'], points=len(anchors), error=str(exc))
            tb = exc.__traceback__
            while tb:
                if tb.tb_frame.f_code.co_name == 'solve_normal_roots':
                    values = tb.tb_frame.f_locals
                    if 'bound' in values:
                        bound = values['bound']; details['unbracketed_points'] = int((~torch.isfinite(bound)).sum())
                    if 'residual' in values:
                        residual = values['residual']; details['max_residual'] = float(residual.max()); details['not_converged'] = int((residual > config.root_tolerance).sum())
                tb = tb.tb_next
            event(root_failure=details)
            raise
        np.savez_compressed(cache / stem, roots=roots.detach().cpu().numpy(), offset=offsets.detach().cpu().numpy())
        return roots, offsets
    root_module.solve_normal_roots = solve
    fields, memo = {}, None
    def component_points(scene, arch, label, config, charts):
        key = arch, label
        if memo is not None and key in memo:
            return memo[key]
        if key not in fields:
            fields[key] = recovery._ComponentField(scene.arches[arch], label)
        field = fields[key]
        if key not in charts:
            charts[key] = surface.extract_chart(field, config)
        points, _, _ = surface.attach_chart(field, charts[key], config)
        if memo is not None:
            memo[key] = points
        return points
    recovery.component_points = component_points
    @contextmanager
    def component_memo():
        nonlocal memo
        previous, memo = memo, {}
        try:
            yield
        finally:
            memo = previous
    return pack, unpack, component_memo, counts
