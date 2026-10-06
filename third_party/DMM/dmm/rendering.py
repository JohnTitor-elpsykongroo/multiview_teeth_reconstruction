"""Known-camera rendering of one depth buffer containing both complete arches.

The torch reference backend has hard visibility (no silhouette derivative).
nvdiffrast supplies CUDA antialiasing and silhouette/occlusion derivatives.
XY follows a material point with detached raster weights during each backward.
"""
from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np
import torch

from dmm.validation import header, read_json, require, stamped
from dmm.pixels import pixel_offset


@dataclass(frozen=True)
class RenderConfig:
    backend: str = "nvdiffrast"
    near_mm: float = .1
    far_mm: float = 10000.
    tile_size: int = 16
    face_chunk: int = 128
    max_pixels: int = 1_048_576
    supersample: int = 1

    def __post_init__(self):
        require(self.backend in ("torch_reference", "nvdiffrast"), "unknown raster backend")
        require(all(type(x) in (int, float) and math.isfinite(x) for x in (self.near_mm, self.far_mm))
                and 0 < self.near_mm < self.far_mm, "invalid depth interval")
        for key in ("tile_size", "face_chunk", "max_pixels"):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0, f"invalid {key}")
        require(type(self.supersample) is int and 1 <= self.supersample <= 8, "supersample must be in [1,8]")
        require(self.backend != "torch_reference" or self.supersample == 1, "reference backend requires supersample=1")

    def resolved(self):
        return stamped("render_config", algorithm="material_xy_v1", **asdict(self))

    @classmethod
    def load(cls, path):
        data = read_json(path)
        header(data, "render_config")
        require(data.pop("algorithm") == "material_xy_v1", "unknown render algorithm")
        for key in ("contract_id", "version", "artifact_kind", "representation_profile"):
            data.pop(key)
        require(set(data) == set(cls.__dataclass_fields__), "render config must be fully resolved")
        return cls(**data)


@dataclass
class RasterContext:
    face_index: torch.Tensor
    material_weights: torch.Tensor
    faces: torch.Tensor
    camera_signature: str


@dataclass
class CudaRasterContext:
    raster: torch.Tensor
    faces: torch.Tensor
    camera_signature: str
    supersample: int


@dataclass
class RenderOutput:
    semantics: torch.Tensor
    xy_pixels: torch.Tensor
    depth_mm: torch.Tensor
    coverage: torch.Tensor
    face_index: torch.Tensor
    context: object
    metadata: dict
    xy_source: str = "projected_geometry"


def camera_coordinates(vertices, observation):
    transform = vertices.new_tensor(observation.T_camera_from_world)
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def project(camera, K):
    homogeneous = camera @ K.T
    # Background samples are zero; avoid NaNs without changing foreground depth.
    z = homogeneous[..., 2:3]
    safe = torch.where(z.abs() > 1e-12, z, torch.ones_like(z))
    return homogeneous[..., :2] / safe


def clip_coordinates(camera, K, height, width, config, metadata=None):
    homogeneous = camera @ K.T
    z = camera[:, 2]
    near, far = config.near_mm, config.far_mm
    shift = .5 - pixel_offset(metadata or {})
    return torch.stack((2 * (homogeneous[:, 0] + shift * z) / width - z,
                        z - 2 * (homogeneous[:, 1] + shift * z) / height,
                        (far + near) / (far - near) * z - 2 * far * near / (far - near), z), -1)


def _signature(observation, config):
    return hashlib.sha256(np.asarray(observation.K).tobytes() + np.asarray(observation.T_camera_from_world).tobytes()
                          + str((observation.view_id, observation.labels.shape, config.near_mm, config.far_mm, pixel_offset(getattr(observation, "metadata", {})))).encode()).hexdigest()


def _barycentric(triangles, pixels):
    # Broadcast [...,3,2] triangles and [...,2] pixel centers.
    a, b, c = triangles.unbind(-2)
    den = (b[..., 1] - c[..., 1]) * (a[..., 0] - c[..., 0]) + (c[..., 0] - b[..., 0]) * (a[..., 1] - c[..., 1])
    safe = torch.where(den.abs() > 1e-12, den, torch.ones_like(den))
    u = ((b[..., 1] - c[..., 1]) * (pixels[..., 0] - c[..., 0]) + (c[..., 0] - b[..., 0]) * (pixels[..., 1] - c[..., 1])) / safe
    v = ((c[..., 1] - a[..., 1]) * (pixels[..., 0] - c[..., 0]) + (a[..., 0] - c[..., 0]) * (pixels[..., 1] - c[..., 1])) / safe
    return torch.stack((u, v, 1 - u - v), -1), den


@torch.no_grad()
def _reference_context(camera, pixels, faces, observation, config):
    h, w = observation.labels.shape
    xy = project(camera, camera.new_tensor(observation.K))
    triangles, zs = xy[faces], camera[faces, 2]
    near, far = config.near_mm, config.far_mm
    crossing = ((zs.min(-1).values < near) & (zs.max(-1).values >= near)) | ((zs.min(-1).values <= far) & (zs.max(-1).values > far))
    require(not bool(crossing.any()), "torch_reference does not clip depth-crossing triangles; use nvdiffrast")
    eligible = (zs >= near).all(-1) & (zs <= far).all(-1)
    lower, upper = triangles.min(-2).values, triangles.max(-2).values
    index = torch.full((h, w), -1, device=camera.device, dtype=torch.long)
    weights = camera.new_zeros(h, w, 3)
    for y in range(0, h, config.tile_size):
        for x in range(0, w, config.tile_size):
            patch = pixels[y:y + config.tile_size, x:x + config.tile_size]
            points = patch.reshape(-1, 2)
            candidates = torch.where(eligible & (lower[:, 0] <= points[:, 0].max()) & (upper[:, 0] >= x)
                                     & (lower[:, 1] <= points[:, 1].max()) & (upper[:, 1] >= y))[0]
            best = camera.new_full((len(points),), float("inf"))
            ids = torch.full_like(best, -1, dtype=torch.long)
            bary = camera.new_zeros(len(points), 3)
            for offset in range(0, len(candidates), config.face_chunk):
                selected = candidates[offset:offset + config.face_chunk]
                screen, den = _barycentric(triangles[selected, None], points[None])
                material = screen / zs[selected, None]
                inv_depth = material.sum(-1)
                inside = (screen >= -1e-7).all(-1) & (den.abs() > 1e-12) & (inv_depth > 0)
                depth = torch.where(inside, inv_depth.clamp_min(1e-20).reciprocal(), float("inf"))
                values, local = depth.min(0)
                update = values < best
                best = torch.where(update, values, best)
                ids = torch.where(update, selected[local], ids)
                normalized = material / inv_depth[..., None].clamp_min(1e-20)
                bary = torch.where(update[:, None], normalized[local, torch.arange(len(points), device=camera.device)], bary)
            index[y:y + patch.shape[0], x:x + patch.shape[1]] = ids.reshape(patch.shape[:2])
            weights[y:y + patch.shape[0], x:x + patch.shape[1]] = bary.reshape(*patch.shape[:2], 3)
    return RasterContext(index, weights, faces.detach().clone(), _signature(observation, config))


def _reference(mesh, observation, config, context):
    camera = camera_coordinates(mesh.vertices_world_mm, observation)
    K = camera.new_tensor(observation.K)
    h, w = observation.labels.shape
    y, x = torch.meshgrid(torch.arange(h, device=camera.device), torch.arange(w, device=camera.device), indexing="ij")
    pixels = torch.stack((x, y), -1).to(camera.dtype) + pixel_offset(getattr(observation, "metadata", {}))
    if context is None:
        context = _reference_context(camera, pixels, mesh.faces, observation, config)
    require(context.camera_signature == _signature(observation, config) and torch.equal(context.faces, mesh.faces),
            "frozen raster context camera/topology mismatch")
    require(context.face_index.device == camera.device, "raster context device mismatch")
    hit = context.face_index >= 0
    indices = mesh.faces[context.face_index.clamp_min(0)]
    tri_camera = camera[indices]
    screen, _ = _barycentric(project(tri_camera, K), pixels)
    # Recompute ordinary perspective interpolation for semantic attributes.
    z = tri_camera[..., 2]
    z = torch.where(z.abs() > 1e-12, z, torch.ones_like(z))
    bary = screen / z
    total = bary.sum(-1, keepdim=True)
    bary = bary / torch.where(total.abs() > 1e-12, total, torch.ones_like(total))
    # Numerical edge tolerance must not create negative semantic probabilities.
    bary = bary.clamp_min(0)
    bary = bary / bary.sum(-1, keepdim=True).clamp_min(1e-12)
    semantic = (mesh.semantics[indices] * bary[..., None]).sum(-2)
    background = torch.zeros_like(semantic)
    background[..., 0] = 1
    semantic = torch.where(hit[..., None], semantic, background)
    # Floating-point barycentric sums may exceed one by an ulp even for a
    # one-hot triangle. Match the CUDA probability contract exactly.
    semantic = semantic.clamp_min(0)
    semantic = semantic / semantic.sum(-1, keepdim=True).clamp_min(1e-12)
    # Fixed material weights prevent live rasterization cancelling projected XY.
    material_point = (tri_camera * context.material_weights.detach()[..., None]).sum(-2)
    xy = torch.where(hit[..., None], project(material_point, K), 0.)
    depth = torch.where(hit, (tri_camera[..., 2] * bary).sum(-1), float("inf"))
    return RenderOutput(semantic, xy, depth, hit.to(camera.dtype), context.face_index, context,
                        dict(backend="torch_reference", silhouette_gradients=False, frozen_material_xy=True,
                             render_config=config.resolved()))


def _nvdiffrast(mesh, observation, config, context=None):
    require(mesh.vertices_world_mm.is_cuda and mesh.vertices_world_mm.dtype == torch.float32,
            "nvdiffrast requires CUDA float32 scene tensors; no implicit backend fallback")
    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise RuntimeError("nvdiffrast is required for the selected CUDA backend") from exc
    camera = camera_coordinates(mesh.vertices_world_mm, observation)
    K = camera.new_tensor(observation.K)
    h, w = observation.labels.shape
    clip = clip_coordinates(camera, K, h, w, config, getattr(observation, "metadata", {}))[None].contiguous()
    faces = mesh.faces.to(torch.int32).contiguous()
    ctx = dr.RasterizeCudaContext(device=camera.device)
    ss = config.supersample
    if context is None:
        rast, _ = dr.rasterize(ctx, clip, faces, resolution=(h * ss, w * ss), grad_db=False)
        context = CudaRasterContext(rast.detach(), mesh.faces.detach().clone(), _signature(observation, config), ss)
    else:
        require(isinstance(context, CudaRasterContext) and context.supersample == ss
                and context.camera_signature == _signature(observation, config) and torch.equal(context.faces, mesh.faces),
                "frozen CUDA raster context mismatch")
        rast = context.raster
    hit = (rast[..., 3:4] > 0).to(camera.dtype)
    sem, _ = dr.interpolate(mesh.semantics[None].contiguous(), rast, faces)
    bg = torch.zeros_like(sem)
    bg[..., 0] = 1
    sem = sem + (1 - hit) * bg
    material, _ = dr.interpolate(camera[None].contiguous(), rast.detach(), faces)
    xy = project(material, K)
    packed = torch.cat((sem, xy * hit, hit), -1)
    aa = dr.antialias(packed.contiguous(), rast, clip, faces)[0].flip(0)
    if ss > 1:
        aa = aa.reshape(h, ss, w, ss, 32).mean((1, 3))
    coverage = aa[..., 31].clamp(0, 1)
    semantics = aa[..., :29].clamp_min(0)
    semantics = semantics / semantics.sum(-1, keepdim=True).clamp_min(1e-12)
    xy = torch.where(coverage[..., None] > 1e-8, aa[..., 29:31] / coverage[..., None].clamp_min(1e-8), 0.)
    ordinary, _ = dr.interpolate(camera[None].contiguous(), rast, faces)
    depth = torch.where(hit[0, ..., 0].bool(), ordinary[0, ..., 2], float("inf")).flip(0)
    face_index = rast[0, ..., 3].long().flip(0) - 1
    if ss > 1:
        samples = depth.reshape(h, ss, w, ss).permute(0, 2, 1, 3).reshape(h, w, ss * ss)
        depth, nearest = samples.min(-1)
        ids = face_index.reshape(h, ss, w, ss).permute(0, 2, 1, 3).reshape(h, w, ss * ss)
        face_index = ids.gather(-1, nearest[..., None])[..., 0]
    return RenderOutput(semantics, xy, depth, coverage, face_index, context,
                        dict(backend="nvdiffrast", silhouette_gradients=True, frozen_material_xy=True,
                             render_config=config.resolved()))


def render_mesh(mesh, observation, config=None, context=None):
    config = RenderConfig() if config is None else config
    vertices, faces, semantic = mesh.vertices_world_mm, mesh.faces, mesh.semantics
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) > 0 and bool(torch.isfinite(vertices).all()), "invalid mesh vertices")
    require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0 and faces.dtype == torch.long
            and int(faces.min()) >= 0 and int(faces.max()) < len(vertices), "invalid mesh faces")
    require(semantic.shape == (len(vertices), 29) and bool(torch.isfinite(semantic).all()) and bool((semantic >= 0).all())
            and bool(torch.allclose(semantic.sum(-1), torch.ones_like(semantic[:, 0]), atol=1e-5)), "invalid vertex probabilities")
    require(vertices.device == faces.device == semantic.device and vertices.dtype == semantic.dtype, "mesh tensor device/dtype mismatch")
    require(observation.labels.ndim == 2 and 0 < observation.labels.size * config.supersample**2 <= config.max_pixels, "render pixel budget exceeded")
    if config.backend == "torch_reference":
        return _reference(mesh, observation, config, context)
    return _nvdiffrast(mesh, observation, config, context)


def render_scene(scene, surface_config=None, render_config=None, charts=None, contexts=None):
    from dmm.surface import build_scene_mesh
    mesh, charts = build_scene_mesh(scene, surface_config, charts)
    ids = [v.view_id for v in scene.observations]
    require(len(ids) > 0 and len(set(ids)) == len(ids), "all views must have unique identifiers")
    if contexts is not None:
        require(set(contexts) == set(ids), "frozen contexts must include every observation")
    rendered = {v.view_id: render_mesh(mesh, v, render_config, None if contexts is None else contexts[v.view_id])
                for v in scene.observations}
    return rendered, mesh, charts
