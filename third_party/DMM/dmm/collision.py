"""Optional bidirectional mixed-surface penetration proxy, including gingiva.

The field is not an exact signed distance. Normalize by its detached spatial
gradient locally; never interpret the result as a certified distance or occlusion.
"""
from dataclasses import asdict, dataclass
import math

import torch

from dmm import MODEL_UNIT_MM
from dmm.validation import header, read_json, require, stamped


@dataclass(frozen=True)
class CollisionConfig:
    enabled: bool = False
    weight: float = .1
    samples_per_arch: int = 512
    tolerance_mm: float = .2
    scale_mm: float = 1.
    min_gradient_norm: float = 1e-6

    def __post_init__(self):
        require(type(self.enabled) is bool, 'collision enabled must be boolean')
        require(type(self.samples_per_arch) is int and self.samples_per_arch > 0, 'invalid collision sample count')
        for key in ('weight', 'scale_mm', 'min_gradient_norm'):
            value = getattr(self, key)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0, f'invalid collision {key}')
        require(type(self.tolerance_mm) in (int, float) and math.isfinite(self.tolerance_mm)
                and self.tolerance_mm >= 0, 'invalid collision tolerance')

    def resolved(self):
        return stamped('collision_config', algorithm='bidirectional_local_field_proxy_v1', **asdict(self))

    @classmethod
    def load(cls, path):
        data = read_json(path)
        header(data, 'collision_config')
        require(data.pop('algorithm') == 'bidirectional_local_field_proxy_v1', 'unknown collision algorithm')
        for key in ('contract_id', 'version', 'artifact_kind', 'representation_profile'): data.pop(key)
        require(set(data) == set(cls.__dataclass_fields__), 'collision config must be fully resolved')
        return cls(**data)


def _sample_surface(mesh, owner, count):
    triangles = mesh.vertices_world_mm[mesh.faces[mesh.face_arch == owner].long()]
    require(len(triangles) > 0, 'collision requires both arch surfaces')
    area = torch.linalg.vector_norm(torch.cross(triangles[:, 1]-triangles[:, 0],
                                               triangles[:, 2]-triangles[:, 0], dim=-1), dim=-1).detach()
    require(bool(torch.isfinite(area).all()) and float(area.sum()) > 0, 'degenerate collision surface')
    # Equal-mass deterministic quadrature; barycentres remain differentiable.
    cumulative = area.cumsum(0)
    quantiles = (torch.arange(count, device=area.device, dtype=area.dtype) + .5) / count * cumulative[-1]
    ids = torch.searchsorted(cumulative, quantiles).clamp_max(len(area)-1)
    return triangles[ids].mean(1)


def collision_loss(scene, mesh, config=None):
    config = config or CollisionConfig()
    zero = mesh.vertices_world_mm.sum() * 0
    if not config.enabled:
        return zero, dict(enabled=False)
    require(set(scene.arches) == {'upper', 'lower'}, 'collision requires upper and lower')
    terms, rows = [], []
    # build_scene_mesh always assigns 0=upper, 1=lower, regardless of dict order.
    for owner, source in enumerate(('upper', 'lower')):
        target = 'lower' if source == 'upper' else 'upper'
        state = scene.arches[target]
        points = _sample_surface(mesh, owner, config.samples_per_arch)
        pose = state.T_world_from_arch
        local = ((points - pose[:3, 3]) @ pose[:3, :3]) / MODEL_UNIT_MM
        box = local.new_tensor(state.bundle.metadata['sampling_domain_model'])
        inside = ((local.detach() >= box[0]) & (local.detach() <= box[1])).all(-1)
        # The positive-boundary surface contract contains the target solid.
        # Never extrapolate a neural SDF outside its verified extraction domain.
        if not bool(inside.any()):
            terms.append(zero)
            rows.append(dict(source=source, target=target, queried=0, samples=len(points), max_proxy_mm=0., penetrating=0))
            continue
        x = local[inside]
        with torch.enable_grad():
            probe = x.detach().requires_grad_(True)
            field = state.query_model(probe)['sdf']
            grad = torch.autograd.grad(field.sum(), probe)[0].detach()
        sdf = state.query_model(x)['sdf']
        norm = grad.norm(dim=-1)
        require(bool(torch.isfinite(sdf).all()) and bool(torch.isfinite(norm).all()), 'nonfinite collision field')
        require(bool(((norm >= config.min_gradient_norm) | (sdf.detach() >= 0)).all()), 'singular interior collision field')
        depth = torch.relu(-sdf) * MODEL_UNIT_MM / norm.clamp_min(config.min_gradient_norm)
        penalty = torch.relu(depth - config.tolerance_mm).square() / config.scale_mm**2
        terms.append(penalty.sum() / len(points))
        rows.append(dict(source=source, target=target, queried=int(inside.sum()), samples=len(points),
                         max_proxy_mm=float(depth.detach().max()),
                         penetrating=int((depth.detach() > config.tolerance_mm).sum())))
    return torch.stack(terms).mean(), dict(enabled=True, directions=rows,
        max_proxy_mm=max(r['max_proxy_mm'] for r in rows), unresolved=any(r['penetrating'] for r in rows),
        scope='sampled local field penetration proxy; not certified mesh intersection or contact')
