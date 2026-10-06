"""Decode RGB model probabilities into the stage-three interchange contract."""
from pathlib import Path
import colorsys
import re

import numpy as np
from PIL import Image

from . import CLASS_IDS, FDI_IDS, VERSION
from .data import PhotoScene, resolve, sha256, write_json


def decode(probabilities, acquisition_valid, confidence_threshold=0.6, margin_threshold=0.1,
           *, occlusion_probability=None, occlusion_threshold=.3):
    if not 0 <= confidence_threshold <= 1 or not 0 <= margin_threshold <= 1:
        raise ValueError("thresholds must lie in [0,1]")
    p = np.asarray(probabilities)
    if p.dtype != np.float32 or p.shape != (len(CLASS_IDS), *acquisition_valid.shape):
        raise ValueError("probabilities must be float32 C,H,W in declared FDI order")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(0), 1, atol=1e-5):
        raise ValueError("invalid probabilities")
    if acquisition_valid.ndim != 2 or not np.isin(acquisition_valid, [0, 1]).all():
        raise ValueError("invalid acquisition mask")
    indices = p.argmax(0)
    top = p.max(0)
    second = np.partition(p, -2, axis=0)[-2]
    valid = (acquisition_valid == 1) & (top >= confidence_threshold) & ((top - second) >= margin_threshold)
    if not 0 < occlusion_threshold < 1:
        raise ValueError("occlusion threshold must lie in (0,1)")
    if occlusion_probability is not None:
        occ = np.asarray(occlusion_probability)
        if occ.dtype != np.float32 or occ.shape != acquisition_valid.shape or not np.isfinite(occ).all() or not np.all((occ >= 0) & (occ <= 1)):
            raise ValueError("invalid RGB occlusion probability")
        # Unknown/likely occluding tissue must not become a DMM background constraint.
        valid &= occ < occlusion_threshold
    labels = np.array(CLASS_IDS, dtype=np.uint8)[indices]
    labels[~valid] = 255
    confidence = np.where(valid, top, 0).astype(np.float32)
    instances = []
    for fdi in FDI_IDS:
        mask = labels == fdi
        if not mask.any():
            continue
        y, x = np.nonzero(mask)
        distribution = p[1:, mask].mean(1, dtype=np.float64)
        distribution /= distribution.sum()
        instances.append({"fdi": fdi, "jaw": "upper" if fdi < 30 else "lower",
                          "visible_pixels": int(mask.sum()),
                          "bbox_xyxy_exclusive": [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)],
                          "confidence": float(confidence[mask].mean()), "fdi_uncertain": False,
                          "fdi_probabilities": {str(k): float(v) for k, v in zip(FDI_IDS, distribution)}})
    return labels, valid.astype(np.uint8), confidence, instances


def overlay(rgb, labels):
    colors = np.zeros((256, 3), dtype=np.uint8)
    for i, fdi in enumerate(FDI_IDS):
        colors[fdi] = np.array(colorsys.hsv_to_rgb(i * .61803398875 % 1, .8, 1)) * 255
    colors[255] = [255, 0, 255]
    output = rgb.copy()
    mask = labels != 0
    output[mask] = (0.5 * rgb[mask] + 0.5 * colors[labels[mask]]).astype(np.uint8)
    return output


def observe(input_manifest, output, predictor, confidence_threshold=.6, margin_threshold=.1, allow_fixture=False,
            *, occlusion_threshold=.3):
    scene = PhotoScene(input_manifest)
    if predictor.kind not in {"prediction", "fixture"} or (predictor.kind == "fixture" and not allow_fixture):
        raise ValueError("fixture predictor requires explicit allow_fixture")
    if predictor.metadata.get("class_ids") != list(CLASS_IDS):
        raise ValueError("predictor class mapping mismatch")
    if any(not isinstance(predictor.metadata.get(k), str) or not predictor.metadata[k]
           for k in ("backend", "version")):
        raise ValueError("predictor backend/version required")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", predictor.metadata.get("checkpoint_sha256") or ""):
        raise ValueError("checkpoint SHA256 required (fixture uses its generator source hash)")
    handling = predictor.metadata.get("occlusion_handling")
    if predictor.kind == "prediction" and handling not in {"rgb_learned_soft_tissue", "explicit_exposed_only_legacy"}:
        raise ValueError("production predictor must declare occlusion handling")
    output = Path(output).resolve()
    # Preserve the entire source case, including its supervision/truth siblings.
    source_case = scene.root.parent
    if output.is_relative_to(source_case) or source_case.is_relative_to(output):
        raise ValueError("output must be disjoint from source case")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("labels", "valid", "confidence", "instances", "review", "occlusion"):
        (output / folder).mkdir()
    views = []
    for i, (name, row) in enumerate(scene.rows.items()):
        rgb, valid, camera = scene.load(name)
        predicted = predictor(rgb)
        occ = predicted.get("occlusion_probability") if isinstance(predicted, dict) else None
        probabilities = predicted["fdi_probabilities"] if isinstance(predicted, dict) else predicted
        if handling == "rgb_learned_soft_tissue" and occ is None:
            raise ValueError("occlusion-capable predictor omitted its occlusion map")
        labels, fit_valid, confidence, instances = decode(probabilities, valid, confidence_threshold, margin_threshold,
                                                         occlusion_probability=occ, occlusion_threshold=occlusion_threshold)
        key = f"{i:04d}"
        refs = {"labels": f"labels/{key}.png", "valid_mask": f"valid/{key}.png",
                "confidence": f"confidence/{key}.npy", "instances": f"instances/{key}.json"}
        if occ is not None:
            refs["occlusion_probability"] = f"occlusion/{key}.npy"
            np.save(output / refs["occlusion_probability"], occ, allow_pickle=False)
        Image.fromarray(labels).save(output / refs["labels"])
        Image.fromarray(fit_valid).save(output / refs["valid_mask"])
        np.save(output / refs["confidence"], confidence, allow_pickle=False)
        write_json(output / refs["instances"], instances)
        Image.fromarray(overlay(rgb, labels)).save(output / "review" / f"{key}.png")
        views.append({"view_id": name, **refs, "width": camera["width"], "height": camera["height"],
                      "source_width": camera["width"], "source_height": camera["height"],
                      "K_source": camera["K"], "K": camera["K"], "A_fit_from_source_pixels": np.eye(3).tolist(),
                      "T_camera_from_world": camera["T_camera_from_world"],
                      "distortion_model": "none", "distortion_coefficients": [],
                      "source_sha256": {k: sha256(resolve(scene.root, row[k])) for k in ("rgb", "valid_mask")},
                      "quality": {"valid_pixels": int(fit_valid.sum()), "ignored_pixels": int((fit_valid == 0).sum()),
                                  "predicted_soft_tissue_ignored_pixels": int(((occ >= occlusion_threshold) & (valid == 1)).sum()) if occ is not None else 0,
                                  "observed_fdi": [v["fdi"] for v in instances]}})
    manifest = {"schema_id": "dental_tooth_observations", "schema_version": VERSION,
                "scene_id": scene.manifest["scene_id"], "length_unit": "mm",
                "pixel_convention": "edge_origin_centers_at_half",
                "world_axes": "X_patient_left_Y_posterior_Z_superior", "observation_kind": predictor.kind,
                "status": "OBSERVATIONS_READY_REVIEW_REQUIRED", "input_manifest": scene.path.as_posix(),
                "input_manifest_sha256": sha256(scene.path), "camera_file_sha256": sha256(scene.camera_path),
                "model": predictor.metadata,
                "config": {"confidence_threshold": confidence_threshold, "margin_threshold": margin_threshold,
                           "occlusion_threshold": occlusion_threshold,
                           "long_edge": getattr(predictor, "long_edge", None)}, "views": views}
    write_json(output / "manifest.json", manifest)
    return manifest
