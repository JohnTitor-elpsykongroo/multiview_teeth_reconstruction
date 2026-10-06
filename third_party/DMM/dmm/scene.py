"""Known-camera dual-arch observations and separate blended arch surfaces."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image
import torch
from torch import nn

from dmm import CHANNEL_FDI, MODEL_UNIT_MM, TEETH
from dmm.bundle import load_bundle
from dmm.pixels import pixel_offset, image_center
from utils.math import transform_screw
from dmm.validation import (array, header, presence_map, read_json, require, resolve_ref,
                         rigid, sha256, nonempty)


@dataclass
class Observation:
    view_id: str
    K: np.ndarray
    T_camera_from_world: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    metadata: dict

    @property
    def visible_fdi(self):
        return tuple(int(x) for x in np.unique(self.labels[self.valid]) if x != 0)

    def semantic_target(self):
        """HxWx29: background then fixed 28 FDI channels; ignore pixels are all zero."""
        return np.stack([(self.labels == label) & self.valid for label in (0, *CHANNEL_FDI)], -1).astype(np.float32)

    def normalized_xy(self):
        height, width = self.labels.shape
        y, x = np.mgrid[:height, :width]
        offset = pixel_offset(self.metadata)
        center = image_center(width, height, self.metadata)
        return (np.stack((x + offset - center[0], y + offset - center[1]), -1) / np.hypot(width, height)).astype(np.float32)


def _intrinsics(value, name):
    k = array(value, (3, 3), name)
    require(k[0, 0] > 0 and k[1, 1] > 0 and np.allclose(k[2], [0, 0, 1], atol=1e-8, rtol=0)
            and abs(k[1, 0]) < 1e-8, f"invalid pinhole {name}")
    return k


def _png(path, shape):
    with Image.open(path) as im:
        require(im.format == "PNG" and im.mode == "L", "mask must be single-channel uint8 PNG (mode L)")
        values = np.asarray(im).copy()
    require(values.shape == shape and values.dtype == np.uint8, "mask dimensions/dtype mismatch")
    return values


def load_observations(camera_path, expected_ids, presence, allowed_roots):
    cameras = read_json(camera_path)
    header(cameras, "cameras")
    require(isinstance(cameras["views"], list) and len(cameras["views"]) > 0, "no views")
    views, identifiers, signatures = [], set(), set()
    present_labels = {k for arch in presence.values() for k, v in arch.items() if v}
    for row in cameras["views"]:
        row = dict(row)
        if "pixel_convention" in cameras:
            require(row.get("pixel_convention", cameras["pixel_convention"]) == cameras["pixel_convention"], "mixed camera pixel conventions")
            row["pixel_convention"] = cameras["pixel_convention"]
        pixel_offset(row)
        view_id = nonempty(row["view_id"], "view_id")
        require(view_id not in identifiers, "duplicate view_id")
        identifiers.add(view_id)
        for field in ("width", "height", "source_width", "source_height"):
            require(type(row[field]) is int and row[field] > 0, f"invalid {field}")
        require(row["distortion_model"] == "none", "v1 requires undistorted images")
        k, k_source = _intrinsics(row["K"], "K"), _intrinsics(row["K_source"], "K_source")
        affine = array(row["A_fit_from_source_pixels"], (3, 3), "A_fit_from_source_pixels")
        require(np.allclose(affine[2], [0, 0, 1], atol=1e-8, rtol=0) and affine[0, 0] > 0
                and affine[1, 1] > 0 and abs(affine[1, 0]) < 1e-8 and abs(affine[0, 1]) < 1e-8,
                "v1 pixel transform supports positive axis-aligned crop/resize only")
        require(np.allclose(k, affine @ k_source, atol=1e-5, rtol=0), "K != A_fit_from_source_pixels @ K_source")
        pose = rigid(row["T_camera_from_world"], "T_camera_from_world")
        shape = (row["height"], row["width"])
        labels = _png(resolve_ref(camera_path, row["mask"], allowed_roots), shape)
        valid = _png(resolve_ref(camera_path, row["valid_mask"], allowed_roots), shape)
        require(set(np.unique(valid)).issubset({0, 1}), "valid mask must contain 0/1")
        require(np.array_equal(valid == 0, labels == 255), "valid==0 iff label==255 violated")
        require(set(np.unique(labels)).issubset({0, 255} | present_labels), "mask contains unknown or absent tooth")
        signature = hashlib.sha256(k.tobytes() + pose.tobytes() + np.asarray(shape).tobytes()
                                   + labels.tobytes() + valid.tobytes()).hexdigest()
        require(signature not in signatures, "duplicate camera/mask/valid observation")
        signatures.add(signature)
        views.append(Observation(view_id, k, pose, labels, valid.astype(bool), row))
    require([v.view_id for v in views] == expected_ids, "manifest views must match camera order exactly")
    for arch in TEETH:
        distinct = {(v.K.tobytes(), v.T_camera_from_world.tobytes()) for v in views
                    if set(v.visible_fdi) & set(TEETH[arch])}
        require(len(distinct) >= 2, f"{arch} needs two distinct cameras with visible teeth")
    return views


class ArchState(nn.Module):
    def __init__(self, bundle, presence, initial):
        super().__init__()
        self.bundle = bundle
        self.decoder = bundle.model
        self.presence = presence
        device = next(self.decoder.parameters()).device
        for label in self.decoder.labels:
            self.register_buffer(f"mean_{label}", bundle.means[label].clone())
            self.register_buffer(f"factor_{label}", bundle.cholesky[label].clone())
        self.register_buffer("base_pose", torch.tensor(rigid(initial["T_world_from_arch"], "T_world_from_arch"), dtype=torch.float32, device=device))
        require(bool(torch.isfinite(self.base_pose).all()), "pose cannot be represented in float32")
        self.pose_delta = nn.Parameter(torch.zeros(6, device=device))
        expected = {str(k) for k, v in presence.items() if v}
        require(set(initial["q"]) == expected, "initial q must contain exactly present teeth")
        self.q = nn.ParameterDict({k: nn.Parameter(torch.tensor(array(v, (self.decoder.dimensions[int(k)],), f"q {k}"), dtype=torch.float32, device=device))
                                   for k, v in initial["q"].items()})
        require(all(bool(torch.isfinite(q).all()) for q in self.q.values()), "q cannot be represented in float32")

    @property
    def T_world_from_arch(self):
        # Left-multiplied SE(3) increment in world coordinates, rad/mm.
        basis = torch.cat((self.base_pose.new_zeros(1, 3), torch.eye(3, dtype=self.base_pose.dtype, device=self.base_pose.device)))
        moved = transform_screw(basis, self.pose_delta.expand(4, -1))
        upper = torch.cat(((moved[1:] - moved[:1]).T, moved[:1].T), -1)
        increment = torch.cat((upper, self.base_pose.new_tensor([[0, 0, 0, 1]])), 0)
        return increment @ self.base_pose

    def codes(self):
        result = {0: self.mean_0}
        result.update({int(k): getattr(self, f"mean_{k}") + getattr(self, f"factor_{k}") @ q for k, q in self.q.items()})
        return result

    def query_model(self, points):
        result = self.decoder.query(points, self.codes())
        channels = {label: i for i, label in enumerate(result["labels"])}
        zeros = result["sdf"] * 0
        # Gum remains channel 0 background but its geometry is retained.
        result["semantics"] = torch.stack([result["weights"][:, channels[label]] if label in channels else zeros
                                           for label in (0, *CHANNEL_FDI)], -1)
        return result

    def query_world(self, world_mm):
        pose = self.T_world_from_arch
        model_points = ((world_mm - pose[:3, 3]) @ pose[:3, :3]) / MODEL_UNIT_MM
        result = self.query_model(model_points)
        result["sdf_mm"] = result["sdf"] * MODEL_UNIT_MM
        return result


@dataclass
class SceneMesh:
    vertices_world_mm: np.ndarray
    faces: np.ndarray
    semantics: np.ndarray
    face_arch: np.ndarray  # 0 upper, 1 lower


class DualArchScene(nn.Module):
    def __init__(self, manifest, observations, states):
        super().__init__()
        self.manifest, self.observations = manifest, observations
        self.arches = nn.ModuleDict(states)

    @property
    def T_upper_from_lower(self):
        return torch.linalg.inv(self.arches["upper"].T_world_from_arch) @ self.arches["lower"].T_world_from_arch

    def query_world(self, points_mm):
        """Separate fields; never blend upper and lower SDFs into a single DMM."""
        return {arch: state.query_world(points_mm) for arch, state in self.arches.items()}

    def latent_prior(self):
        penalties = [q.square().mean() for arch in self.arches.values() for q in arch.q.values()]
        return torch.stack(penalties).mean() if penalties else self.T_upper_from_lower.sum() * 0

    def differentiable_mesh(self, config=None, charts=None):
        from dmm.surface import build_scene_mesh
        return build_scene_mesh(self, config, charts)

    def render_views(self, surface_config=None, render_config=None, charts=None, contexts=None):
        from dmm.rendering import render_scene
        return render_scene(self, surface_config, render_config, charts, contexts)

    def summary(self):
        seen = set().union(*(set(v.visible_fdi) for v in self.observations))
        return dict(scene_id=self.manifest["scene_id"], views=len(self.observations),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    arches={arch: dict(model_id=state.bundle.metadata["model_id"],
                                       present=[k for k, v in state.presence.items() if v],
                                       unobserved=[k for k, v in state.presence.items() if v and k not in seen])
                            for arch, state in self.arches.items()})

    @torch.no_grad()
    def extract_mesh(self, resolution=64, chunk_size=32768):
        """Evaluation/export only. Marching cubes is NOT the fitting gradient bridge."""
        from skimage.measure import marching_cubes
        require(type(resolution) is int and resolution >= 8 and chunk_size > 0, "invalid mesh sampling settings")
        vertices, triangles, semantics, owners = [], [], [], []
        offset = 0
        for arch_index, (arch, state) in enumerate(self.arches.items()):
            box = np.asarray(state.bundle.metadata["sampling_domain_model"], dtype=float)
            axes = [np.linspace(box[0, i], box[1, i], resolution) for i in range(3)]
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            device = state.base_pose.device
            def query(data, key):
                return np.concatenate([state.query_model(torch.tensor(data[i:i + chunk_size], dtype=torch.float32, device=device))[key].cpu().numpy()
                                       for i in range(0, len(data), chunk_size)])
            volume = query(grid, "sdf").reshape((resolution,) * 3)
            require(np.isfinite(volume).all() and volume.min() < 0 < volume.max(), f"{arch}: no finite zero surface")
            boundary = np.concatenate([volume[0].ravel(), volume[-1].ravel(), volume[:, 0].ravel(), volume[:, -1].ravel(), volume[:, :, 0].ravel(), volume[:, :, -1].ravel()])
            require(np.all(boundary > 0), f"{arch}: non-positive domain boundary; surface may be truncated")
            spacing = (box[1] - box[0]) / (resolution - 1)
            verts, faces, _, _ = marching_cubes(volume, level=0, spacing=spacing)
            verts += box[0]
            probs = query(verts, "semantics")
            pose = state.T_world_from_arch.cpu().numpy()
            world = (verts * MODEL_UNIT_MM) @ pose[:3, :3].T + pose[:3, 3]
            vertices.append(world)
            triangles.append(faces + offset)
            semantics.append(probs)
            owners.append(np.full(len(faces), arch_index, dtype=np.uint8))
            offset += len(verts)
        return SceneMesh(np.concatenate(vertices), np.concatenate(triangles), np.concatenate(semantics), np.concatenate(owners))


def compose_depth_layers(depth_mm, semantics):
    """Shared nearest depth, including gum. Inputs AxHxW and AxHxWx29.

    +inf means no hit. This hard visibility operation is an assembly reference,
    not a differentiable rasterizer. Gradients pass only to the selected layer.
    """
    require(depth_mm.ndim == 3 and semantics.shape == (*depth_mm.shape, 29), "invalid depth/semantics layer shape")
    require(bool(((depth_mm > 0) & ~torch.isnan(depth_mm)).all()), "depth must be positive camera Z or +inf")
    require(bool(torch.isfinite(semantics).all()) and bool((semantics >= 0).all()), "invalid semantic probabilities")
    hit = torch.isfinite(depth_mm)
    require(bool(torch.allclose(semantics.sum(-1)[hit], torch.ones_like(depth_mm[hit]), atol=1e-5, rtol=0)), "hit semantics must sum to one")
    depth, layer = depth_mm.min(0)  # deterministic first-layer tie policy
    result = torch.gather(semantics, 0, layer[None, ..., None].expand(1, *layer.shape, 29))[0]
    background = torch.zeros_like(result)
    background[..., 0] = 1
    result = torch.where(torch.isfinite(depth)[..., None], result, background)
    return depth, result


def load_scene(manifest_path, device="cpu", *, allow_candidate=False):
    path = Path(manifest_path).resolve()
    allowed_names = {"manifest.json", "candidate_manifest.json"} if allow_candidate else {"manifest.json"}
    require(path.name in allowed_names and path.parent.name == "fit_input", "scene entrypoint must be fit_input/manifest.json")
    root = path.parent.parent
    roots = [root / name for name in ("fit_input", "model_bundle", "initialization")]
    require(all(p.resolve() == p for p in roots), "scene allowlist directories may not be symlinks/junctions")
    data = read_json(path)
    header(data, "fit_manifest")
    nonempty(data["scene_id"], "scene_id")
    require(data["length_unit"] == "mm" and set(data["presence"]) == set(TEETH), "invalid scene units/arches")
    presence = {arch: presence_map(data["presence"][arch], arch) for arch in TEETH}
    cameras = resolve_ref(path, data["camera_file"], roots)
    observations = load_observations(cameras, data["views"], presence, roots)
    init_path = resolve_ref(path, data["initialization"], roots)
    initialization = read_json(init_path)
    header(initialization, "scene_initialization")
    require(initialization["method"] in ("training_mean", "independent_estimate"), "truth-perturbed initialization is not accepted")
    nonempty(initialization["origin"], "initialization origin")
    require(set(data["model_bundle"]) == set(TEETH) and set(initialization["arches"]) == set(TEETH), "both arch bundles and initial states required")
    states = {}
    for arch in TEETH:
        model_path = resolve_ref(path, data["model_bundle"][arch], roots)
        bundle = load_bundle(model_path, arch, device, roots)
        initial = initialization["arches"][arch]
        require(initial["model_sha256"] == sha256(model_path)
                and initial["statistics_sha256"] == bundle.metadata["latent_statistics"]["sha256"], "initialization model/statistics hash mismatch")
        if initialization["method"] == "training_mean":
            require(all(np.all(np.asarray(v) == 0) for v in initial["q"].values()), "training_mean initialization requires q=0")
        states[arch] = ArchState(bundle, presence[arch], initial)
    return DualArchScene(data, observations, states)
