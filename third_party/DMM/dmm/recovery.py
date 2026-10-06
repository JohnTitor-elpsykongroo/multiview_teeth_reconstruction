"""Observed-but-not-rendered FDI recovery without changing physical visibility."""
from dataclasses import asdict, dataclass
from types import SimpleNamespace
import math

import numpy as np
import torch

from dmm import CHANNEL_FDI, TEETH
from dmm.rendering import camera_coordinates, project
from dmm.surface import extract_chart, attach_chart
from dmm.validation import require


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = True
    mass_ratio: float = .05
    min_pixels: int = 1
    source_samples: int = 128
    target_samples: int = 64
    depth_margin_mm: float = .5
    depth_scale_mm: float = 50.
    bootstrap_max_translation_mm: float = 200.
    bootstrap_condition_limit: float = 1e6

    def __post_init__(self):
        require(type(self.enabled) is bool, 'recovery enabled must be Boolean')
        require(0 < self.mass_ratio < 1, 'invalid recovery mass ratio')
        for name in ('min_pixels', 'source_samples', 'target_samples'):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0, f'invalid {name}')
        for name in ('depth_margin_mm', 'depth_scale_mm', 'bootstrap_max_translation_mm', 'bootstrap_condition_limit'):
            require(math.isfinite(getattr(self, name)) and getattr(self, name) > 0, f'invalid {name}')


def missing_regions(scene, rendered, config):
    """Test each observed FDI, including cases where other teeth remain visible."""
    missing = []
    for obs in scene.observations:
        probabilities = rendered[obs.view_id].semantics.detach()
        valid = torch.as_tensor(obs.valid, device=probabilities.device)
        dominant = probabilities.argmax(-1)
        for label in obs.visible_fdi:
            channel = CHANNEL_FDI.index(label) + 1
            target_pixels = int(((obs.labels == label) & obs.valid).sum())
            mass = float(probabilities[..., channel][valid].sum())
            predicted_pixels = int(((dominant == channel) & valid).sum())
            if mass < config.mass_ratio * target_pixels or predicted_pixels < config.min_pixels:
                arch = 'upper' if label in TEETH['upper'] else 'lower'
                missing.append(dict(view_id=obs.view_id, arch=arch, label=label, target_pixels=target_pixels,
                                    predicted_pixels=predicted_pixels, predicted_mass=mass))
    return missing


class _ComponentField:
    """Use a native component's zero surface only for auxiliary correspondences.

    This never replaces the complete blended arch mesh in the physical renderer.
    """
    def __init__(self, state, label):
        require(state.presence.get(label, False), 'cannot recover an absent tooth')
        self.state, self.label = state, label
        self.decoder, self.base_pose = state.decoder, state.base_pose
        self.bundle = SimpleNamespace(metadata=dict(state.bundle.metadata, model_id=state.bundle.metadata['model_id'] + f':component:{label}'))

    @property
    def T_world_from_arch(self): return self.state.T_world_from_arch

    def codes(self): return {self.label: self.state.codes()[self.label]}

    def query_model(self, points):
        result = self.decoder.component(points, self.codes()[self.label], self.label)
        sem = points.new_zeros(len(points), 29)
        sem[:, CHANNEL_FDI.index(self.label) + 1] = 1
        return dict(sdf=result['sdf'], semantics=sem)


def component_points(scene, arch, label, surface_config, charts):
    field = _ComponentField(scene.arches[arch], label)
    key = (arch, label)
    if key not in charts: charts[key] = extract_chart(field, surface_config)
    points, _, _ = attach_chart(field, charts[key], surface_config)
    return points


def _sample(points, count):
    indices = torch.linspace(0, len(points)-1, min(count, len(points)), device=points.device).long()
    return points[indices]


def recovery_loss(scene, rendered, active, surface_config, render_config, config, charts=None):
    charts = {} if charts is None else charts
    zero = sum(state.pose_delta.sum() * 0 for state in scene.arches.values())
    if not active or not config.enabled: return zero, charts
    observations = {obs.view_id: obs for obs in scene.observations}
    terms = []
    for row in active:
        obs = observations[row['view_id']]
        points = _sample(component_points(scene, row['arch'], row['label'], surface_config, charts), config.source_samples)
        camera = camera_coordinates(points, obs)
        # Positive projection denominator plus an explicit behind-camera penalty.
        safe = torch.cat((camera[:, :2], camera[:, 2:3].clamp_min(render_config.near_mm)), -1)
        xy = project(safe, points.new_tensor(obs.K))
        yx = torch.as_tensor(np.argwhere((obs.labels == row['label']) & obs.valid), device=points.device)
        target_yx = _sample(yx, config.target_samples)
        from dmm.pixels import pixel_offset
        target_xy = target_yx[:, [1, 0]].to(points.dtype) + pixel_offset(obs.metadata)
        diagonal = math.hypot(*obs.labels.shape)
        distance = (target_xy[:, None] - xy[None]).square().sum(-1) / diagonal**2
        depth = rendered[obs.view_id].depth_mm.detach()[target_yx[:, 0], target_yx[:, 1]]
        finite = torch.isfinite(depth)
        depth = torch.where(finite, depth, torch.zeros_like(depth))
        barrier = torch.relu(camera[None, :, 2] - depth[:, None] + config.depth_margin_mm) / config.depth_scale_mm
        barrier = torch.where(finite[:, None], barrier, torch.zeros_like(barrier))
        near = torch.relu(render_config.near_mm - camera[:, 2]) / config.depth_scale_mm
        cost = distance + barrier.square() + near[None].square()
        # Discrete correspondence fixed in this backward; visible target only.
        nearest = cost.detach().argmin(-1)
        terms.append(cost[torch.arange(len(target_yx), device=points.device), nearest].mean())
    return torch.stack(terms).sum() / len(scene.observations), charts


@torch.no_grad()
def bootstrap_translation(scene, arch, active, surface_config, config):
    """Translation-only least squares from known-camera rays; no truth input.

    Only attempt when every observed region of this arch is missing. Existing
    rotation, scale, latent and other arch remain fixed. Caller validates render.
    """
    labels = set(TEETH[arch])
    expected = {(obs.view_id, k) for obs in scene.observations for k in obs.visible_fdi if k in labels}
    missing = {(row['view_id'], row['label']) for row in active if row['arch'] == arch}
    if not config.enabled or not expected or not expected.issubset(missing):
        return None, dict(status='NOT_APPLICABLE', arch=arch)
    charts, matrices, rhs = {}, [], []
    for obs in scene.observations:
        for label in obs.visible_fdi:
            if label not in labels: continue
            points = component_points(scene, arch, label, surface_config, charts)
            center = points.mean(0).double().cpu().numpy()
            yx = np.argwhere((obs.labels == label) & obs.valid).mean(0)
            from dmm.pixels import pixel_offset
            offset = pixel_offset(obs.metadata)
            ray = np.linalg.solve(obs.K, [yx[1] + offset, yx[0] + offset, 1.])
            rows = np.array([[1., 0., -ray[0]], [0., 1., -ray[1]]])
            camera = obs.T_camera_from_world[:3, :3] @ center + obs.T_camera_from_world[:3, 3]
            matrices.append(rows @ obs.T_camera_from_world[:3, :3])
            rhs.append(-rows @ camera)
    A, b = np.concatenate(matrices), np.concatenate(rhs)
    singular = np.linalg.svd(A, compute_uv=False)
    if len(singular) < 3 or singular[-1] <= 1e-12 or singular[0] / singular[-1] > config.bootstrap_condition_limit:
        return None, dict(status='UNDERCONSTRAINED_RAYS', arch=arch)
    delta = np.linalg.lstsq(A, b, rcond=None)[0]
    if np.linalg.norm(delta) > config.bootstrap_max_translation_mm:
        return None, dict(status='BOOTSTRAP_TRUST_BOUND', arch=arch, translation_mm=delta.tolist())
    pose = scene.arches[arch].T_world_from_arch.detach().clone()
    pose[:3, 3] += pose.new_tensor(delta)
    return pose, dict(status='PROPOSED', arch=arch, translation_mm=delta.tolist(),
                      condition=float(singular[0]/singular[-1]), ray_residual=float(np.linalg.norm(A @ delta-b)))
