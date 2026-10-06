"""Reproducible, read-only DMM forward check; writes only to a fresh local run.

Run from this project root with a Python environment containing torch, numpy,
scikit-image, plyfile, and Pillow.  JSON result goes to stdout; progress goes to
stderr and runs/<id>/progress.log.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import math
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATENT_KEY = re.compile(r"^(\d+)\.(?:module\.)?weight$")
COLORS = {
    0: (20, 25, 31),
    1: (109, 114, 124),
    11: (235, 107, 95), 12: (244, 163, 97), 13: (242, 205, 110),
    14: (172, 207, 124), 15: (102, 191, 143), 16: (85, 177, 169),
    17: (94, 156, 211), 21: (230, 103, 164), 22: (190, 113, 205),
    23: (151, 121, 220), 24: (111, 140, 221), 25: (91, 171, 215),
    26: (115, 197, 184), 27: (154, 210, 150),
}


def progress(run_root: Path, message: str) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {message}"
    print(line, file=sys.stderr, flush=True)
    with (run_root / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xyz = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(np.float32)
    faces = np.vstack(ply["face"]["vertex_indices"]).astype(np.int32)
    if not np.isfinite(xyz).all() or faces.size == 0:
        raise ValueError(f"invalid mesh: {path}")
    if faces.min() < 0 or faces.max() >= len(xyz):
        raise ValueError(f"out-of-range face index: {path}")
    return xyz, faces


def save_ply(path: Path, xyz: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for axis, column in enumerate(("x", "y", "z")):
        vertices[column] = xyz[:, axis]
    face_data = np.empty(len(faces), dtype=[("vertex_indices", "O")])
    face_data["vertex_indices"] = [np.asarray(face, dtype=np.int32) for face in faces]
    PlyData([PlyElement.describe(vertices, "vertex"), PlyElement.describe(face_data, "face")],
            text=False).write(str(path))


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("degenerate camera basis")
    return vector / norm


def make_cameras(meshes: dict[int, tuple[np.ndarray, np.ndarray]], config: dict) -> list[dict]:
    means = {label: xyz.mean(axis=0).astype(np.float64) for label, (xyz, _) in meshes.items()
             if label >= 11}
    front = unit((means[11] + means[21]) / 2 - (means[17] + means[27]) / 2)
    side = unit(means[21] - means[11])
    up = unit(np.cross(side, front))
    if up[2] < 0:
        up = -up
    extent = np.concatenate([meshes[label][0] for label in means], axis=0)
    target = extent.mean(axis=0).astype(np.float64)
    arch_width = float(np.linalg.norm(means[27] - means[17]))
    distance = arch_width * float(config["camera_distance_arch_widths"])
    elevation = math.radians(float(config["camera_elevation_degrees"]))
    width, height = int(config["image_width"]), int(config["image_height"])
    focal = float(config["focal_pixels"])
    cameras = []
    for azimuth_deg in config["camera_azimuth_degrees"]:
        azimuth = math.radians(float(azimuth_deg))
        radial = unit(front * math.cos(azimuth) + side * math.sin(azimuth))
        eye = target + distance * (radial * math.cos(elevation) + up * math.sin(elevation))
        forward = unit(target - eye)
        right = unit(np.cross(forward, up))
        camera_up = unit(np.cross(right, forward))
        # OpenCV-style camera axes: x right, y down, z forward.
        rotation = np.stack([right, -camera_up, forward], axis=0)
        translation = -rotation @ eye
        K = np.array([[focal, 0, (width - 1) / 2],
                      [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=np.float64)
        cameras.append({
            "name": f"az_{int(azimuth_deg):+d}",
            "azimuth_degrees": int(azimuth_deg),
            "K": K.tolist(), "R_world_to_camera": rotation.tolist(),
            "t_world_to_camera": translation.tolist(), "eye_world": eye.tolist(),
            "image_width": width, "image_height": height,
            "coordinate_convention": "X_cam=R*X_world+t; +x right, +y down, +z forward; pixels=K*(X_cam/z)",
        })
    return cameras


def rasterize(meshes: dict[int, tuple[np.ndarray, np.ndarray]], camera: dict,
              include_gum: bool = True, return_world_points: bool = False,
              shade: bool = True):
    width, height = camera["image_width"], camera["image_height"]
    depth = np.full((height, width), np.inf, dtype=np.float32)
    labels = np.zeros((height, width), dtype=np.uint8)
    shaded = np.zeros((height, width), dtype=np.uint8)
    world_points = (np.full((height, width, 3), np.nan, dtype=np.float32)
                    if return_world_points else None)
    R = np.asarray(camera["R_world_to_camera"], dtype=np.float64)
    t = np.asarray(camera["t_world_to_camera"], dtype=np.float64)
    K = np.asarray(camera["K"], dtype=np.float64)
    eye_world = -R.T @ t
    rejected_behind = 0
    for label, (xyz, faces) in meshes.items():
        if label == 0 and not include_gum:
            continue
        cam = xyz.astype(np.float64) @ R.T + t
        if np.any(cam[:, 2] <= 1e-4):
            rejected_behind += int(np.count_nonzero(cam[:, 2] <= 1e-4))
            continue
        uv = np.column_stack([K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2],
                              K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]])
        inverse_z = 1.0 / cam[:, 2]
        semantic = 1 if label == 0 else label
        for face in faces:
            p = uv[face]
            low = np.maximum(np.floor(p.min(axis=0)).astype(int), 0)
            high = np.minimum(np.ceil(p.max(axis=0)).astype(int), [width - 1, height - 1])
            if np.any(high < low):
                continue
            ax, ay = p[0]
            bx, by = p[1]
            cx, cy = p[2]
            determinant = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(determinant) < 1e-10:
                continue
            yy, xx = np.mgrid[low[1]:high[1] + 1, low[0]:high[0] + 1]
            xx = xx + 0.5
            yy = yy + 0.5
            a = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / determinant
            b = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / determinant
            c = 1.0 - a - b
            inside = (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
            if not inside.any():
                continue
            z = 1.0 / (a * inverse_z[face[0]] + b * inverse_z[face[1]] + c * inverse_z[face[2]])
            view = depth[low[1]:high[1] + 1, low[0]:high[0] + 1]
            update = inside & (z < view)
            view[update] = z[update]
            label_view = labels[low[1]:high[1] + 1, low[0]:high[0] + 1]
            label_view[update] = semantic
            if world_points is not None:
                inverse_depth = a * inverse_z[face[0]] + b * inverse_z[face[1]] + c * inverse_z[face[2]]
                weights = np.stack([a * inverse_z[face[0]] / inverse_depth,
                                    b * inverse_z[face[1]] / inverse_depth,
                                    c * inverse_z[face[2]] / inverse_depth], axis=-1)
                surface = weights @ xyz[face]
                point_view = world_points[low[1]:high[1] + 1, low[0]:high[0] + 1]
                point_view[update] = surface[update]
            if shade:
                normal = np.cross(xyz[face[1]] - xyz[face[0]], xyz[face[2]] - xyz[face[0]])
                norm = float(np.linalg.norm(normal))
                if norm > 1e-12:
                    normal /= norm
                light = unit(eye_world - xyz[face].mean(axis=0) +
                             np.array([-0.3, 0.1, 0.7]))
                intensity = 0.28 + 0.72 * abs(float(np.dot(normal, light)))
                gray = int(round(235 * intensity)) if label else int(round(130 * intensity))
                shaded_view = shaded[low[1]:high[1] + 1, low[0]:high[0] + 1]
                shaded_view[update] = gray
    counts = {str(label): int(np.count_nonzero(labels == label)) for label in sorted(meshes)
              if label != 0}
    counts["gum"] = int(np.count_nonzero(labels == 1))
    result = (labels, depth, shaded, {"visible_pixel_counts": counts,
                                      "vertices_behind_camera": rejected_behind})
    return (*result, world_points) if return_world_points else result


def colorize(labels: np.ndarray) -> Image.Image:
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels):
        rgb[labels == label] = COLORS[int(label)]
    return Image.fromarray(rgb, "RGB")


def panel_image(rendered: list[tuple[str, Image.Image]], path: Path) -> None:
    if not rendered:
        return
    width, height = rendered[0][1].size
    sheet = Image.new("RGB", (width * len(rendered), height + 30), (245, 246, 248))
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(rendered):
        sheet.paste(image, (index * width, 30))
        draw.text((index * width + 12, 8), label, fill=(20, 24, 30), font=ImageFont.load_default())
    sheet.save(path)


def decode_mesh(net, label: int, code: torch.Tensor, source_xyz: np.ndarray,
                grid_n: int, query_batch: int) -> tuple[np.ndarray, np.ndarray, dict]:
    source_min = source_xyz.min(axis=0).astype(np.float64)
    source_max = source_xyz.max(axis=0).astype(np.float64)
    extent = source_max - source_min
    margin = np.maximum(extent * 0.22, 0.08)
    origin = source_min - margin
    top = source_max + margin
    axes = [np.linspace(origin[i], top[i], grid_n, dtype=np.float32) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    sdf = np.empty(len(grid), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(grid), query_batch):
            query = torch.from_numpy(grid[start:start + query_batch].copy())
            values = net.inference({label: code}, query).reshape(-1)
            sdf[start:start + len(query)] = values.cpu().numpy()
    cube = sdf.reshape((grid_n,) * 3)
    boundary = np.concatenate([cube[0].ravel(), cube[-1].ravel(), cube[:, 0].ravel(),
                               cube[:, -1].ravel(), cube[:, :, 0].ravel(), cube[:, :, -1].ravel()])
    if cube.min() >= 0 or cube.max() <= 0:
        raise ValueError(f"no zero-level set for component {label}")
    spacing = tuple(float((top[i] - origin[i]) / (grid_n - 1)) for i in range(3))
    verts, faces, _, _ = marching_cubes(cube, level=0.0, spacing=spacing)
    verts = (verts + origin).astype(np.float32)
    faces = faces.astype(np.int32)
    return verts, faces, {
        "grid_n": grid_n, "grid_origin": origin.tolist(), "grid_top": top.tolist(),
        "sdf_min": float(cube.min()), "sdf_max": float(cube.max()),
        "boundary_nonpositive_count": int(np.count_nonzero(boundary <= 0)),
        "vertex_count": len(verts), "face_count": len(faces),
        "bounds_min": verts.min(axis=0).tolist(), "bounds_max": verts.max(axis=0).tolist(),
    }


def sample_sdf_residual(net, label: int, code: torch.Tensor, xyz: np.ndarray) -> dict:
    indices = np.linspace(0, len(xyz) - 1, min(512, len(xyz)), dtype=int)
    with torch.no_grad():
        result = net.inference({label: code}, torch.from_numpy(xyz[indices].copy())).reshape(-1)
    values = result.abs().numpy()
    return {"median_abs_sdf": float(np.median(values)), "p95_abs_sdf": float(np.percentile(values, 95))}


def symmetric_surface_change(first: np.ndarray, second: np.ndarray) -> dict:
    first_sample = first[np.linspace(0, len(first) - 1, min(3000, len(first)), dtype=int)]
    second_sample = second[np.linspace(0, len(second) - 1, min(3000, len(second)), dtype=int)]
    distances = np.concatenate([cKDTree(second).query(first_sample)[0],
                                cKDTree(first).query(second_sample)[0]])
    return {"median_nearest_surface_distance": float(np.median(distances)),
            "p95_nearest_surface_distance": float(np.percentile(distances, 95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "forward_check.json")
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = dt.datetime.now(dt.timezone.utc).strftime("forward_%Y%m%dT%H%M%SZ_") + hashlib.sha256(
        config_path.read_bytes()).hexdigest()[:8]
    run_root = args.run_root.resolve() if args.run_root else PROJECT_ROOT / "runs" / run_id
    if not run_root.is_relative_to((PROJECT_ROOT / "runs").resolve()):
        raise ValueError("run root must be inside project runs")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    started = time.perf_counter()
    try:
        progress(run_root, "START checkpoint and data contract")
        dmm_root = Path(config["dmm_root"]).resolve(strict=True)
        experiment = Path(config["experiment"]).resolve(strict=True)
        dataset = Path(config["dataset"]).resolve(strict=True)
        specs_path = experiment / "specs.json"
        model_path = experiment / "ModelParameters" / f"dmm_{config['checkpoint']}.pth"
        latent_path = experiment / "LatentCodes" / f"latent_vecs_{config['checkpoint']}.pth"
        split_path = dataset / "splits" / "train_split.json"
        case_name = config["case"]
        case_npz = dataset / "SdfSamples" / case_name
        case_pkl = case_npz.with_suffix(".pkl")
        baseline_dir = experiment / "GeneratedMeshes" / "295" / "Meshes"
        expected_labels = [0, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
        baseline_paths = [baseline_dir / ("gum.ply" if label == 0 else f"tooth{label}.ply")
                          for label in expected_labels]
        inputs = [config_path, specs_path, model_path, latent_path, split_path, case_npz, case_pkl,
                  *baseline_paths]
        if any(not path.is_file() for path in inputs):
            raise FileNotFoundError([str(path) for path in inputs if not path.is_file()])
        specs = json.loads(specs_path.read_text(encoding="utf-8"))
        labels = [int(label) for label in specs["labels"]]
        if labels != expected_labels:
            raise ValueError(f"unexpected label set: {labels}")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split_set = set(split)
        ordered_cases = sorted(path.name for path in (dataset / "SdfSamples").glob("*.npz")
                               if path.name in split_set)
        if case_name not in ordered_cases:
            raise ValueError(f"selected case is not in resolved training split: {case_name}")
        case_row = ordered_cases.index(case_name)
        with case_pkl.open("rb") as handle:
            present_labels = {int(label) for label in pickle.load(handle)} | {0}
        if set(labels) != present_labels:
            raise ValueError(f"selected case has labels {sorted(present_labels)}, expected {labels}")
        model_state = torch.load(model_path, map_location="cpu", weights_only=True)
        latent_state = torch.load(latent_path, map_location="cpu", weights_only=True)
        if model_state["epoch"] != latent_state["epoch"]:
            raise ValueError("model and latent checkpoint epochs do not match")
        if int(model_state["epoch"]) != 295:
            raise ValueError("baseline mesh directory is pinned to epoch 295")
        sys.path[:0] = [str(dmm_root), str(dmm_root / "third_party")]
        sys.dont_write_bytecode = True
        from networks.dmm_net import DMM  # local, frozen source
        with contextlib.redirect_stdout(io.StringIO()):
            net = DMM(specs)
        net.load_state_dict(model_state["model_state_dict"], strict=True)
        net.eval()
        torch.set_num_threads(int(config["cpu_threads"]))
        codes = {}
        for key, matrix in latent_state["latent_codes"].items():
            match = LATENT_KEY.fullmatch(key)
            if match:
                label = int(match.group(1))
                if matrix.shape[0] != len(ordered_cases):
                    raise ValueError(f"latent rows disagree with split for {label}")
                expected_dim = int(specs["GumDeformNetworkSpecs" if label == 0 else
                                          "TeethDeformNetworkSpecs"]["latent_dim"])
                if matrix.shape[1] != expected_dim:
                    raise ValueError(f"latent width disagree for {label}")
                codes[label] = matrix[case_row:case_row + 1].clone()
        if set(codes) != set(labels):
            raise ValueError(f"latent components {sorted(codes)} disagree with specs")
        provenance = {
            "status": "PREFLIGHT_PASSED", "run_id": run_id,
            "epoch": int(model_state["epoch"]), "model_metric": model_state.get("metric_name"),
            "model_metric_value": model_state.get("metric_value"),
            "labels": labels, "case": case_name, "case_row": case_row,
            "training_rows": len(ordered_cases), "latent_norms": {str(k): float(v.norm()) for k, v in codes.items()},
            "input_sha256": {str(path): sha256(path) for path in inputs},
            "python": sys.executable, "torch": torch.__version__, "device": "cpu",
            "existing_baseline_dir": str(baseline_dir),
        }
        (run_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        progress(run_root, f"PREFLIGHT_PASSED epoch={provenance['epoch']} case_row={case_row} labels={len(labels)}")

        source_meshes = {}
        for label in labels:
            name = "gum.ply" if label == 0 else f"tooth{label}.ply"
            source_meshes[label] = load_ply(baseline_dir / name)
        zero_codes = {label: torch.zeros_like(code) for label, code in codes.items()}
        residuals = {}
        for label in labels:
            residuals[str(label)] = sample_sdf_residual(net, label, zero_codes[label], source_meshes[label][0])
        (run_root / "existing_zero_mesh_residuals.json").write_text(
            json.dumps(residuals, indent=2), encoding="utf-8")
        progress(run_root, "EXISTING_ZERO_MESH_SDF_CHECK finished")

        variants = {"zero": zero_codes, "training_case": codes}
        mesh_sets = {}
        metrics = {"run_id": run_id, "status": "RUNNING", "variants": {}, "cameras": []}
        for variant, variant_codes in variants.items():
            progress(run_root, f"DECODE_START variant={variant}")
            variant_dir = run_root / "meshes" / variant
            variant_dir.mkdir(parents=True)
            mesh_sets[variant] = {}
            metrics["variants"][variant] = {}
            for label in labels:
                source_xyz = source_meshes[label][0]
                grid_n = int(config["gum_grid_n"] if label == 0 else config["tooth_grid_n"])
                tick = time.perf_counter()
                xyz, faces, item = decode_mesh(net, label, variant_codes[label], source_xyz,
                                               grid_n, int(config["query_batch"]))
                item["elapsed_seconds"] = round(time.perf_counter() - tick, 2)
                item["sdf_residual"] = sample_sdf_residual(net, label, variant_codes[label], xyz)
                geometry = trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
                connected = geometry.split(only_watertight=False)
                item["watertight"] = bool(geometry.is_watertight)
                item["connected_components"] = len(connected)
                item["largest_component_face_fraction"] = (
                    max(len(component.faces) for component in connected) / len(faces))
                metrics["variants"][variant][str(label)] = item
                mesh_sets[variant][label] = (xyz, faces)
                save_ply(variant_dir / ("gum.ply" if label == 0 else f"tooth{label}.ply"), xyz, faces)
                progress(run_root, f"DECODE_DONE variant={variant} label={label} "
                         f"v={len(xyz)} f={len(faces)} s={item['elapsed_seconds']}")
        (run_root / "mesh_metrics.json").write_text(json.dumps(metrics["variants"], indent=2), encoding="utf-8")
        metrics["zero_to_training_case_surface_change"] = {
            str(label): symmetric_surface_change(mesh_sets["zero"][label][0],
                                                 mesh_sets["training_case"][label][0])
            for label in labels
        }

        cameras = make_cameras(mesh_sets["zero"], config)
        camera_qc = {}
        for camera in cameras:
            R = np.asarray(camera["R_world_to_camera"])
            camera_qc[camera["name"]] = {
                "rotation_determinant": float(np.linalg.det(R)),
                "orthonormality_max_abs_error": float(np.abs(R @ R.T - np.eye(3)).max()),
            }
        (run_root / "cameras.json").write_text(json.dumps(cameras, indent=2), encoding="utf-8")
        by_variant = {}
        for variant in variants:
            render_dir = run_root / "renders" / variant
            render_dir.mkdir(parents=True)
            panels = []
            tooth_panels = []
            shade_panels = []
            by_variant[variant] = {}
            for camera in cameras:
                labels_image, depth, _, info = rasterize(mesh_sets[variant], camera)
                tooth_labels, tooth_depth, tooth_shade, tooth_info = rasterize(
                    mesh_sets[variant], camera, include_gum=False)
                name = camera["name"]
                Image.fromarray(labels_image, "L").save(render_dir / f"{name}_labels.png")
                np.savez_compressed(render_dir / f"{name}_depth.npz", depth=depth)
                color = colorize(labels_image)
                color.save(render_dir / f"{name}_semantic.png")
                panels.append((f"{variant} {name}", color))
                Image.fromarray(tooth_labels, "L").save(render_dir / f"{name}_teeth_labels.png")
                np.savez_compressed(render_dir / f"{name}_teeth_depth.npz", depth=tooth_depth)
                tooth_color = colorize(tooth_labels)
                tooth_color.save(render_dir / f"{name}_teeth_semantic.png")
                tooth_shaded_image = Image.fromarray(tooth_shade, "L").convert("RGB")
                tooth_shaded_image.save(render_dir / f"{name}_teeth_shaded.png")
                tooth_panels.append((f"{variant} {name}", tooth_color))
                shade_panels.append((f"{variant} {name}", tooth_shaded_image))
                by_variant[variant][name] = {"with_gum": info, "teeth_only": tooth_info}
                progress(run_root, f"RENDER_DONE variant={variant} camera={name} "
                         f"visible_teeth={sum(v > 0 for v in info['visible_pixel_counts'].values()) - 1}")
            panel_image(panels, run_root / f"{variant}_contact_sheet.png")
            panel_image(tooth_panels, run_root / f"{variant}_teeth_contact_sheet.png")
            panel_image(shade_panels, run_root / f"{variant}_teeth_shaded_contact_sheet.png")
        metrics["cameras"] = cameras
        metrics["camera_qc"] = camera_qc
        metrics["render_checks"] = by_variant
        metrics["existing_zero_mesh_residuals"] = residuals
        metrics["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        metrics["status"] = "FORWARD_RENDERED_VISUAL_REVIEW_REQUIRED"
        (run_root / "report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        progress(run_root, metrics["status"])
        print(json.dumps({"status": metrics["status"], "run_root": str(run_root),
                          "report": str(run_root / "report.json"),
                          "elapsed_seconds": metrics["elapsed_seconds"]}, ensure_ascii=False))
    except Exception as exc:
        failure = {"status": "FORWARD_CHECK_FAILED", "run_root": str(run_root),
                   "error_type": type(exc).__name__, "error": str(exc)}
        (run_root / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        progress(run_root, f"FORWARD_CHECK_FAILED {type(exc).__name__}: {exc}")
        print(json.dumps(failure, ensure_ascii=False))
        raise


if __name__ == "__main__":
    main()
