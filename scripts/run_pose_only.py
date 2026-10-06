"""Fit one rigid upper-arch pose to all known-camera tooth-ID masks.

The fitting code opens only fit_input and fixed_meshes. Generator truth is used
by a separate evaluator after this command has produced a final pose.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_erosion

from run_forward_check import colorize, load_ply, panel_image, rasterize, sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log(run_root: Path, message: str, details: dict | None = None) -> None:
    record = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "message": message}
    if details:
        record.update(details)
    print(record["time_utc"], message, file=sys.stderr, flush=True)
    with (run_root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def transformed(meshes: dict[int, tuple[np.ndarray, np.ndarray]], pose: np.ndarray):
    R = Rotation.from_rotvec(pose[:3]).as_matrix()
    t = pose[3:]
    return {label: (xyz @ R.T + t, faces) for label, (xyz, faces) in meshes.items()}


def project(points: np.ndarray, pose: np.ndarray, camera: dict) -> np.ndarray:
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    world = points @ rotation.T + pose[3:]
    R = np.asarray(camera["R_world_to_camera"])
    t = np.asarray(camera["t_world_to_camera"])
    K = np.asarray(camera["K"])
    cam = world @ R.T + t
    if np.any(cam[:, 2] <= 0):
        raise ValueError("pose projected points behind camera")
    return np.column_stack((K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2],
                            K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]))


def mask_centroid(mask: np.ndarray, label: int):
    yy, xx = np.nonzero(mask == label)
    if len(xx) == 0:
        return None
    return np.array([xx.mean() + 0.5, yy.mean() + 0.5]), len(xx)


def mean_iou(predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray],
             labels: list[int]) -> tuple[float, dict, float]:
    per_camera = {}
    all_scores = []
    total_intersection = 0
    total_union = 0
    for camera_name, prediction in predictions.items():
        target = targets[camera_name]
        scores = {}
        for label in labels:
            actual = target == label
            predicted = prediction == label
            union = int(np.count_nonzero(actual | predicted))
            intersection = int(np.count_nonzero(actual & predicted))
            scores[str(label)] = intersection / union if union else None
            if union:
                all_scores.append(scores[str(label)])
                total_intersection += intersection
                total_union += union
        per_camera[camera_name] = scores
    return float(np.mean(all_scores)), per_camera, total_intersection / total_union


def target_features(masks: dict[str, np.ndarray], cameras: list[dict],
                    labels: list[int]) -> dict[tuple[str, int], tuple[np.ndarray, float]]:
    features = {}
    for camera in cameras:
        name = camera["name"]
        for label in labels:
            item = mask_centroid(masks[name], label)
            if item is not None:
                features[(name, label)] = item
    return features


def centroid_residual(pose: np.ndarray, cameras: list[dict], labels: list[int],
                      mesh_centers: dict[int, np.ndarray], features: dict) -> np.ndarray:
    values = []
    for camera in cameras:
        visible_labels = [label for label in labels if (camera["name"], label) in features]
        points = np.stack([mesh_centers[label] for label in visible_labels])
        pixels = project(points, pose, camera)
        for pixel, label in zip(pixels, visible_labels):
            target, area = features[(camera["name"], label)]
            weight = np.sqrt(min(area, 2000) / 2000)
            values.extend(weight * (pixel - target))
    return np.asarray(values)


def boundary_pixels(mask: np.ndarray, label: int) -> np.ndarray:
    binary = mask == label
    if not binary.any():
        return np.empty((0, 2), dtype=np.int32)
    edge = binary & ~binary_erosion(binary)
    yy, xx = np.nonzero(edge)
    return np.column_stack((xx, yy)).astype(np.int32)


def sample_rows(array: np.ndarray, maximum: int) -> np.ndarray:
    if len(array) <= maximum:
        return array
    return array[np.linspace(0, len(array) - 1, maximum, dtype=int)]


def render_masks(meshes, cameras, pose, include_world_points=False):
    moved = transformed(meshes, pose)
    rendered = {}
    point_maps = {}
    for camera in cameras:
        result = rasterize(moved, camera, include_gum=False,
                           return_world_points=include_world_points, shade=False)
        rendered[camera["name"]] = result[0]
        if include_world_points:
            point_maps[camera["name"]] = result[4]
    return rendered, point_maps


def contour_pairs(pose, meshes, cameras, targets, labels, maximum, distance_limit):
    rendered, points_world = render_masks(meshes, cameras, pose, include_world_points=True)
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    points = []
    pixels = []
    camera_indices = []
    pair_distances = []
    for camera_index, camera in enumerate(cameras):
        name = camera["name"]
        pred = rendered[name]
        target = targets[name]
        world_map = points_world[name]
        for label in labels:
            predicted_edge = boundary_pixels(pred, label)
            target_edge = boundary_pixels(target, label)
            if not len(predicted_edge) or not len(target_edge):
                continue
            predicted_sample = sample_rows(predicted_edge, maximum)
            target_sample = sample_rows(target_edge, maximum)
            # Predicted contour to observed contour.
            distances, indices = cKDTree(target_edge).query(predicted_sample)
            for (x, y), distance, nearest in zip(predicted_sample, distances, indices):
                if distance > distance_limit:
                    continue
                world = world_map[y, x]
                if not np.isfinite(world).all():
                    continue
                points.append((world - pose[3:]) @ rotation)
                pixels.append(target_edge[nearest].astype(float) + 0.5)
                camera_indices.append(camera_index)
                pair_distances.append(float(distance))
            # Observed contour to predicted contour, using the model point at
            # the matched rendered pixel. This prevents simple underfilling.
            distances, indices = cKDTree(predicted_edge).query(target_sample)
            for target_pixel, distance, nearest in zip(target_sample, distances, indices):
                if distance > distance_limit:
                    continue
                x, y = predicted_edge[nearest]
                world = world_map[y, x]
                if not np.isfinite(world).all():
                    continue
                points.append((world - pose[3:]) @ rotation)
                pixels.append(target_pixel.astype(float) + 0.5)
                camera_indices.append(camera_index)
                pair_distances.append(float(distance))
    if len(points) < 100:
        raise ValueError(f"only {len(points)} usable contour pairs")
    return rendered, np.asarray(points), np.asarray(pixels), np.asarray(camera_indices), {
        "pair_count": len(points),
        "pair_distance_median_px": float(np.median(pair_distances)),
        "pair_distance_p95_px": float(np.percentile(pair_distances, 95)),
    }


def contour_residual(pose, points, pixels, camera_indices, cameras,
                     centroid_args, centroid_weight):
    residuals = np.empty((len(points), 2), dtype=float)
    for index, camera in enumerate(cameras):
        selected = camera_indices == index
        residuals[selected] = project(points[selected], pose, camera) - pixels[selected]
    if centroid_weight:
        extra = centroid_weight * centroid_residual(pose, *centroid_args)
        return np.r_[residuals.ravel(), extra]
    return residuals.ravel()


def save_render_comparison(run_root, stage, predictions, targets, cameras, labels):
    stage_dir = run_root / "renders" / stage
    stage_dir.mkdir(parents=True)
    semantic_panels = []
    overlay_panels = []
    for camera in cameras:
        name = camera["name"]
        pred = predictions[name]
        target = targets[name]
        Image.fromarray(pred, "L").save(stage_dir / f"{name}_predicted_labels.png")
        colored = colorize(pred)
        colored.save(stage_dir / f"{name}_predicted_semantic.png")
        semantic_panels.append((name, colored))
        target_teeth = np.isin(target, labels)
        predicted_teeth = np.isin(pred, labels)
        overlay = np.zeros((*pred.shape, 3), dtype=np.uint8)
        overlay[target_teeth & predicted_teeth] = (225, 225, 225)
        overlay[target_teeth & ~predicted_teeth] = (242, 70, 83)
        overlay[predicted_teeth & ~target_teeth] = (60, 146, 245)
        image = Image.fromarray(overlay, "RGB")
        image.save(stage_dir / f"{name}_overlay.png")
        overlay_panels.append((name, image))
    panel_image(semantic_panels, stage_dir / "semantic_sheet.png")
    panel_image(overlay_panels, stage_dir / "overlay_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pose_only.json")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = dt.datetime.now(dt.timezone.utc).strftime("pose_only_%Y%m%dT%H%M%SZ_") + sha256(config_path)[:8]
    run_root = PROJECT_ROOT / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        log(run_root, "START pose-only fit")
        fit_root = Path(config["fit_input"]).resolve(strict=True)
        shape_root = Path(config["fixed_meshes"]).resolve(strict=True)
        manifest_path = fit_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "KNOWN_CAMERA_SYNTHETIC_FIT_INPUT_READY":
            raise ValueError("fit input status is not ready")
        camera_path = fit_root / manifest["camera_file"]
        if sha256(camera_path) != manifest["camera_sha256"]:
            raise ValueError("camera file hash mismatch")
        cameras = json.loads(camera_path.read_text(encoding="utf-8"))
        if len(cameras) != len(manifest["masks"]):
            raise ValueError("camera and mask counts differ")
        labels = [int(label) for label in manifest["visible_fdi_ids"]]
        targets = {}
        for entry in manifest["masks"]:
            mask_path = fit_root / entry["path"]
            if sha256(mask_path) != entry["sha256"]:
                raise ValueError(f"mask hash mismatch: {mask_path}")
            targets[entry["camera"]] = np.asarray(Image.open(mask_path).convert("L"))
        if set(targets) != {camera["name"] for camera in cameras}:
            raise ValueError("mask/camera names differ")
        meshes = {}
        mesh_hashes = {}
        for label in labels:
            mesh_path = shape_root / f"tooth{label}.ply"
            meshes[label] = load_ply(mesh_path)
            mesh_hashes[str(label)] = sha256(mesh_path)
        mesh_centers = {label: xyz.mean(axis=0).astype(float) for label, (xyz, _) in meshes.items()}
        features = target_features(targets, cameras, labels)
        if len(features) < 3 * len(labels):
            raise ValueError("at least one FDI is missing from a view")
        initial = np.r_[np.deg2rad(config["initial_rotation_degrees_xyz"]),
                        np.asarray(config["initial_translation_dmm"], dtype=float)]
        if initial.shape != (6,) or not np.isfinite(initial).all():
            raise ValueError("initial pose must be six finite values")
        provenance = {
            "run_id": run_id, "config_sha256": sha256(config_path),
            "fit_manifest_sha256": sha256(manifest_path),
            "camera_sha256": sha256(camera_path),
            "mask_sha256": {entry["camera"]: entry["sha256"] for entry in manifest["masks"]},
            "fixed_mesh_sha256": mesh_hashes,
            "source_run_id": manifest["source_run_id"],
            "labels": labels, "views": [camera["name"] for camera in cameras],
            "objective_inputs_only": ["fit_input", "fixed_meshes"],
        }
        (run_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        (run_root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        log(run_root, f"INPUT_VERIFIED views={len(cameras)} labels={len(labels)}")

        stages = {}
        initial_render, _ = render_masks(meshes, cameras, initial)
        initial_iou, initial_per_view, initial_micro = mean_iou(initial_render, targets, labels)
        save_render_comparison(run_root, "initial", initial_render, targets, cameras, labels)
        stages["initial"] = {"pose": initial.tolist(), "mean_tooth_iou": initial_iou,
                             "micro_tooth_iou": initial_micro, "per_view": initial_per_view}
        log(run_root, f"INITIAL_IOU {initial_iou:.6f}")

        centroid_args = (cameras, labels, mesh_centers, features)
        centroid_fit = least_squares(centroid_residual, initial, args=centroid_args,
                                    max_nfev=100, loss="soft_l1", f_scale=5.0)
        centroid_pose = centroid_fit.x
        centroid_render, _ = render_masks(meshes, cameras, centroid_pose)
        centroid_iou, centroid_per_view, centroid_micro = mean_iou(centroid_render, targets, labels)
        save_render_comparison(run_root, "centroid", centroid_render, targets, cameras, labels)
        stages["centroid"] = {"pose": centroid_pose.tolist(), "mean_tooth_iou": centroid_iou,
                              "micro_tooth_iou": centroid_micro, "per_view": centroid_per_view,
                              "solver_success": bool(centroid_fit.success),
                              "solver_evaluations": int(centroid_fit.nfev)}
        log(run_root, f"CENTROID_IOU {centroid_iou:.6f} nfev={centroid_fit.nfev}")

        best_pose = centroid_pose.copy()
        best_iou = centroid_iou
        current_pose = centroid_pose.copy()
        iterations = []
        for outer in range(int(config["contour_outer_iterations"])):
            _, points, pixels, camera_indices, pair_info = contour_pairs(
                current_pose, meshes, cameras, targets, labels,
                int(config["max_boundary_samples_per_tooth_per_view"]),
                float(config["max_contour_pair_distance_pixels"]),
            )
            solution = least_squares(
                contour_residual, current_pose,
                args=(points, pixels, camera_indices, cameras, centroid_args,
                      float(config["centroid_regularizer_weight"])),
                max_nfev=50, loss="soft_l1", f_scale=float(config["robust_loss_scale_pixels"]),
            )
            candidate_pose = solution.x
            candidate_render, _ = render_masks(meshes, cameras, candidate_pose)
            candidate_iou, _, candidate_micro = mean_iou(candidate_render, targets, labels)
            info = {"outer_iteration": outer + 1, **pair_info,
                    "solver_evaluations": int(solution.nfev),
                    "pose": candidate_pose.tolist(),
                    "mean_tooth_iou": candidate_iou, "micro_tooth_iou": candidate_micro}
            iterations.append(info)
            log(run_root, f"CONTOUR_ITER {outer + 1} iou={candidate_iou:.6f} "
                f"pairs={pair_info['pair_count']} median_pair_px={pair_info['pair_distance_median_px']:.3f}", info)
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_pose = candidate_pose.copy()
            current_pose = candidate_pose
            if outer >= 1 and abs(candidate_iou - iterations[-2]["mean_tooth_iou"]) < 1e-4:
                break

        final_render, _ = render_masks(meshes, cameras, best_pose)
        final_iou, final_per_view, final_micro = mean_iou(final_render, targets, labels)
        save_render_comparison(run_root, "final", final_render, targets, cameras, labels)
        R = Rotation.from_rotvec(best_pose[:3]).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = best_pose[3:]
        pose_record = {"parameterization": "axis-angle rotation vector in radians; world translation in DMM units",
                       "rotation_vector_radians": best_pose[:3].tolist(),
                       "translation_dmm": best_pose[3:].tolist(),
                       "arch_to_world_4x4": transform.tolist(),
                       "applies_to": "all 14 fixed upper tooth meshes"}
        (run_root / "pose.json").write_text(json.dumps(pose_record, indent=2), encoding="utf-8")
        report = {"status": "POSE_ONLY_FIT_COMPLETED_EVAL_REQUIRED", "run_id": run_id,
                  "initial": stages["initial"], "centroid": stages["centroid"],
                  "contour_iterations": iterations,
                  "final": {"pose": best_pose.tolist(), "mean_tooth_iou": final_iou,
                            "micro_tooth_iou": final_micro, "per_view": final_per_view},
                  "elapsed_seconds": round(time.perf_counter() - started, 2)}
        (run_root / "fit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(run_root, f"POSE_ONLY_FIT_COMPLETED_EVAL_REQUIRED final_iou={final_iou:.6f}")
        print(json.dumps({"status": report["status"], "run_root": str(run_root),
                          "initial_iou": initial_iou, "final_iou": final_iou,
                          "elapsed_seconds": report["elapsed_seconds"]}))
    except Exception as exc:
        failure = {"status": "POSE_ONLY_FIT_FAILED", "type": type(exc).__name__, "error": str(exc)}
        (run_root / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        log(run_root, f"POSE_ONLY_FIT_FAILED {type(exc).__name__}: {exc}")
        print(json.dumps(failure))
        raise


if __name__ == "__main__":
    main()
