"""Local implicit-function surface charts for the blended DMM field.

Marching cubes supplies detached topology/anchors. A normal-line root solve and
the implicit function theorem attach those roots to component latents. Charts
are local and must be rebuilt if the root/offset/domain checks fail.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from dmm import MODEL_UNIT_MM
from dmm.validation import domain, header, read_json, require, stamped


@dataclass(frozen=True)
class SurfaceConfig:
    resolution: int = 48
    chunk_size: int = 4096
    root_iterations: int = 15
    root_tolerance: float = 1e-6
    min_gradient_norm: float = 1e-6
    min_normal_derivative: float = 1e-6
    max_normal_offset: float = 0.1
    max_vertices: int = 200_000

    def __post_init__(self):
        for key in ("resolution", "chunk_size", "root_iterations", "max_vertices"):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0, f"invalid {key}")
        require(8 <= self.resolution <= 256, "surface resolution must be in [8,256]")
        for key in ("root_tolerance", "min_gradient_norm", "min_normal_derivative", "max_normal_offset"):
            require(type(getattr(self, key)) in (int, float) and math.isfinite(getattr(self, key)) and getattr(self, key) > 0, f"invalid {key}")

    def resolved(self):
        return stamped("surface_config", algorithm="normal_chart_ift_v1", **asdict(self))

    @classmethod
    def load(cls, path):
        data = read_json(path)
        header(data, "surface_config")
        require(data.pop("algorithm") == "normal_chart_ift_v1", "unknown surface algorithm")
        for key in ("contract_id", "version", "artifact_kind", "representation_profile"):
            data.pop(key)
        require(set(data) == set(cls.__dataclass_fields__), "surface config must be fully resolved")
        return cls(**data)


@dataclass
class SurfaceChart:
    arch: str
    anchors: torch.Tensor
    normals: torch.Tensor
    faces: torch.Tensor
    model_id: str
    present_labels: tuple
    sampling_domain: np.ndarray


@dataclass
class DifferentiableMesh:
    vertices_world_mm: torch.Tensor
    faces: torch.Tensor
    semantics: torch.Tensor
    face_arch: torch.Tensor
    diagnostics: dict


def _field_gradient(state, points):
    # Local coordinate derivatives are needed even for an evaluation-only render.
    with torch.enable_grad():
        x = points.detach().requires_grad_(True)
        phi = state.query_model(x)["sdf"]
        grad = torch.autograd.grad(phi.sum(), x, create_graph=False)[0]
    require(bool(torch.isfinite(phi).all()) and bool(torch.isfinite(grad).all()), "nonfinite implicit field/gradient")
    return phi.detach(), grad.detach()


def extract_chart(state, config):
    from skimage.measure import marching_cubes
    box = domain(state.bundle.metadata["sampling_domain_model"])
    dtype, device = state.base_pose.dtype, state.base_pose.device
    size = config.resolution
    axes = [torch.linspace(float(box[0, i]), float(box[1, i]), size, dtype=dtype, device=device) for i in range(3)]
    # Generate coordinate chunks by flat index, not a full CUDA Nx3 grid.
    values = []
    with torch.no_grad():
        for start in range(0, size**3, config.chunk_size):
            ids = torch.arange(start, min(start + config.chunk_size, size**3), device=device)
            points = torch.stack((axes[0][ids // (size * size)], axes[1][ids // size % size], axes[2][ids % size]), -1)
            values.append(state.query_model(points)["sdf"].cpu().numpy())
    volume = np.concatenate(values).reshape((size,) * 3)
    require(np.isfinite(volume).all() and volume.min() < 0 < volume.max(), "no finite mixed-field zero surface")
    boundary = np.concatenate([volume[0].ravel(), volume[-1].ravel(), volume[:, 0].ravel(),
                               volume[:, -1].ravel(), volume[:, :, 0].ravel(), volume[:, :, -1].ravel()])
    require(bool(np.all(boundary > 0)), "surface intersects domain boundary; expand verified domain instead of clipping")
    vertices, faces, _, _ = marching_cubes(volume, 0., spacing=(box[1] - box[0]) / (size - 1), allow_degenerate=False)
    require(0 < len(vertices) <= config.max_vertices, "surface vertex budget exceeded or empty")
    anchors = torch.tensor(np.ascontiguousarray(vertices + box[0]), dtype=dtype, device=device)
    normals = []
    for points in anchors.split(config.chunk_size):
        _, grad = _field_gradient(state, points)
        length = grad.norm(dim=-1)
        require(bool((length >= config.min_gradient_norm).all()), "singular surface gradient; cannot build chart")
        normals.append(grad / length[:, None])
    return SurfaceChart(state.decoder.arch, anchors, torch.cat(normals),
                        torch.tensor(np.ascontiguousarray(faces), dtype=torch.long, device=device),
                        state.bundle.metadata["model_id"], tuple(state.codes()), box)


def attach_chart(state, chart, config):
    require(chart.arch == state.decoder.arch and chart.model_id == state.bundle.metadata["model_id"]
            and chart.present_labels == tuple(state.codes()), "surface chart model/presence mismatch")
    require(chart.anchors.device == state.base_pose.device and chart.anchors.dtype == state.base_pose.dtype,
            "rebuild surface charts after moving model dtype/device")
    require(chart.anchors.shape == chart.normals.shape and chart.anchors.ndim == 2 and chart.anchors.shape[1] == 3
            and bool(torch.isfinite(chart.anchors).all()) and bool(torch.isfinite(chart.normals).all()), "invalid chart anchors/normals")
    require(bool(torch.allclose(chart.normals.norm(dim=-1), torch.ones_like(chart.normals[:, 0]), atol=1e-5, rtol=0)), "chart normals must be unit vectors")
    box = torch.as_tensor(chart.sampling_domain, dtype=chart.anchors.dtype, device=chart.anchors.device)
    vertices, semantics, residuals, offsets = [], [], [], []
    for anchors, normal in zip(chart.anchors.split(config.chunk_size), chart.normals.split(config.chunk_size)):
        offset = torch.zeros(len(anchors), dtype=anchors.dtype, device=anchors.device)
        for _ in range(config.root_iterations):
            roots = anchors + offset[:, None] * normal
            phi, grad = _field_gradient(state, roots)
            denom = (grad * normal).sum(-1)
            require(bool((denom.abs() >= config.min_normal_derivative).all()), "singular normal-line derivative; refresh surface chart")
            if float(phi.abs().max()) <= config.root_tolerance:
                break
            offset -= phi / denom
            require(bool(torch.isfinite(offset).all()) and bool((offset.abs() <= config.max_normal_offset).all()),
                    "normal offset exceeds chart trust region; refresh surface chart")
        roots = (anchors + offset[:, None] * normal).detach()
        phi, grad = _field_gradient(state, roots)
        require(float(phi.abs().max()) <= config.root_tolerance, "normal-line roots failed to converge; refresh surface chart")
        require(bool(((roots >= box[0]) & (roots <= box[1])).all()), "surface root left declared domain")
        denominator = (grad * normal).sum(-1)
        require(bool((denominator.abs() >= config.min_normal_derivative).all()), "singular implicit derivative at root")
        # Forward geometry is the solved root. Backward: dx/dq = -n*dPhi/dq/(gradPhi.n).
        live_phi = state.query_model(roots)["sdf"]
        attached = roots - normal * ((live_phi - live_phi.detach()) / denominator)[:, None]
        prediction = state.query_model(attached)
        vertices.append(attached)
        semantics.append(prediction["semantics"])
        residuals.append(float(phi.abs().max()))
        offsets.append(float(offset.abs().max()))
    points = torch.cat(vertices)
    pose = state.T_world_from_arch
    world = (MODEL_UNIT_MM * points) @ pose[:3, :3].T + pose[:3, 3]
    return world, torch.cat(semantics), dict(vertices=len(points), faces=len(chart.faces),
                                           max_sdf_residual=max(residuals), max_normal_offset=max(offsets))


def build_scene_mesh(scene, config=None, charts=None):
    config = SurfaceConfig() if config is None else config
    require(set(scene.arches) == {"upper", "lower"}, "both arch states required")
    if charts is None:
        charts = {name: extract_chart(state, config) for name, state in scene.arches.items()}
    require(set(charts) == set(scene.arches), "surface charts must cover both arches")
    vertices, faces, semantic, owners, details = [], [], [], [], {}
    offset = 0
    for arch_index, name in enumerate(("upper", "lower")):
        state, chart = scene.arches[name], charts[name]
        points, probabilities, diagnostic = attach_chart(state, chart, config)
        vertices.append(points)
        faces.append(chart.faces + offset)
        semantic.append(probabilities)
        owners.append(torch.full((len(chart.faces),), arch_index, dtype=torch.long, device=points.device))
        details[name] = diagnostic
        offset += len(points)
    return DifferentiableMesh(torch.cat(vertices), torch.cat(faces), torch.cat(semantic), torch.cat(owners),
                              dict(arches=details, surface_config=config.resolved(), derivative="local_normal_chart_ift")), charts
