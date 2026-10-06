"""User-selected Blender annotation observations; no RGB recognition or 3D truth."""
from pathlib import Path
import numpy as np
from PIL import Image

from . import CLASS_IDS, FDI_IDS
from .data import PhotoScene, read_json, resolve, sha256, supervision, write_json
from .pipeline import overlay


def package_oracle(input_manifest, annotations_manifest, output):
    scene = PhotoScene(input_manifest)
    ann_path = Path(annotations_manifest).resolve(strict=True)
    annotations = supervision(scene, ann_path, include_tissue=True)
    ann_rows = {v["view_id"]: v for v in read_json(ann_path)["views"]}
    output = Path(output).resolve()
    for root in (scene.root.parent, ann_path.parent):
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("oracle output must be disjoint from source")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("labels", "valid", "confidence", "instances", "review"):
        (output / folder).mkdir()
    rows = []
    for index, name in enumerate(scene.rows):
        rgb, acquisition_valid, camera = scene.load(name)
        labels = annotations[name]["fdi"].copy()
        soft = np.isin(annotations[name]["tissue"], [101, 102, 103])
        labels[soft | (acquisition_valid == 0)] = 255
        valid = (labels != 255).astype(np.uint8)
        confidence = valid.astype(np.float32)  # annotation weight, NOT learned confidence
        instances = []
        for fdi in FDI_IDS:
            yy, xx = np.where(labels == fdi)
            if not len(xx):
                continue
            instances.append({"fdi": fdi, "jaw": "upper" if fdi < 30 else "lower",
                              "visible_pixels": len(xx), "bbox_xyxy_exclusive": [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
                              "confidence": 1., "fdi_uncertain": False,
                              "fdi_probabilities": {str(k): float(k == fdi) for k in FDI_IDS}})
        key = f"{index:04d}"
        refs = {"labels": f"labels/{key}.png", "valid_mask": f"valid/{key}.png",
                "confidence": f"confidence/{key}.npy", "instances": f"instances/{key}.json"}
        Image.fromarray(labels).save(output / refs["labels"])
        Image.fromarray(valid).save(output / refs["valid_mask"])
        np.save(output / refs["confidence"], confidence, allow_pickle=False)
        write_json(output / refs["instances"], instances)
        Image.fromarray(overlay(rgb, labels)).save(output / "review" / f"{key}.png")
        rows.append({"view_id": name, **refs, "width": camera["width"], "height": camera["height"],
                     "source_width": camera["width"], "source_height": camera["height"],
                     "K": camera["K"], "K_source": camera["K"], "A_fit_from_source_pixels": np.eye(3).tolist(),
                     "T_camera_from_world": camera["T_camera_from_world"], "distortion_model": "none", "distortion_coefficients": [],
                     "source_sha256": {k: sha256(resolve(scene.root, scene.rows[name][k])) for k in ("rgb", "valid_mask")},
                     "annotation_sha256": {k: sha256(resolve(ann_path.parent, ann_rows[name][k])) for k in ("fdi", "tissue")},
                     "quality": {"valid_pixels": int(valid.sum()), "ignored_pixels": int((valid == 0).sum()),
                                 "oracle_soft_tissue_ignored_pixels": int((soft & (acquisition_valid == 1)).sum()),
                                 "observed_fdi": [v["fdi"] for v in instances]}})
    manifest = {"schema_id": "dental_tooth_observations", "schema_version": "1.1.0",
                "scene_id": scene.manifest["scene_id"], "length_unit": "mm",
                "pixel_convention": "edge_origin_centers_at_half", "world_axes": "X_patient_left_Y_posterior_Z_superior",
                "observation_kind": "oracle", "status": "OBSERVATIONS_READY_REVIEW_REQUIRED",
                "input_manifest": scene.path.as_posix(), "input_manifest_sha256": sha256(scene.path),
                "camera_file_sha256": sha256(scene.camera_path),
                "oracle": {"annotations_manifest": ann_path.as_posix(), "annotations_manifest_sha256": sha256(ann_path),
                           "confidence_semantics": "unit_annotation_weight_not_probability", "uses_3d_truth": False,
                           "presence_source": "not_inferred", "ignored_tissue_ids": [101, 102, 103]},
                "model": {"backend": "blender_annotation_oracle", "version": "1.0.0", "checkpoint_sha256": None, "class_ids": list(CLASS_IDS)},
                "config": {"confidence_threshold": 0., "margin_threshold": 0.}, "views": rows}
    write_json(output / "manifest.json", manifest)
    return manifest


def package_dataset(dataset, output, policy=None):
    from observation_fusion.contract import load_observations
    from observation_fusion.pipeline import prepare
    path = Path(dataset).resolve(strict=True)
    doc = read_json(path)
    if doc.get("schema_id") != "dental_multiview_photo" or doc.get("schema_version") != "1.0.0":
        raise ValueError("unsupported dataset")
    output = Path(output).resolve()
    if output.is_relative_to(path.parent) or path.parent.is_relative_to(output):
        raise ValueError("output must be disjoint from dataset")
    ids = [r["scene_id"] for r in doc["scenes"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty/duplicate scene IDs")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for i, row in enumerate(doc["scenes"]):
        input_path = resolve(path.parent, row["input_manifest"])
        if read_json(input_path)["scene_id"] != row["scene_id"]:
            raise ValueError("dataset scene mismatch")
        observation_dir = output / f"scene_{i:04d}" / "oracle_observations"
        obs = package_oracle(input_path, resolve(path.parent, row["annotations_manifest"]), observation_dir)
        loaded = load_observations(observation_dir / "manifest.json", allow_oracle=True)
        item = {k: row.get(k) for k in ("scene_id", "patient_id", "split", "accepted_for_training", "quality_status", "collision_status", "diagnostic_control")}
        item.update(oracle_manifest=(observation_dir / "manifest.json").relative_to(output).as_posix(),
                    views=len(loaded.views), foreground_pixels=sum(int(np.isin(v.labels, FDI_IDS).sum()) for v in loaded.views),
                    source_quality_accepted=False)
        if policy is not None:
            fusion_dir = output / f"scene_{i:04d}" / "fusion"
            report = prepare(observation_dir / "manifest.json", fusion_dir, policy, allow_oracle=True)
            item.update(fusion_report=(fusion_dir / "report.json").relative_to(output).as_posix(),
                        fit_ready=report["fit_ready"], pending=report["pending"],
                        arches=report["diagnostics"]["arches"])
        results.append(item)
    result = {"status": "ORACLE_OBSERVATIONS_PACKAGED", "observation_kind": "oracle",
              "dataset": path.as_posix(), "dataset_sha256": sha256(path), "source_status": doc.get("status"),
              "scenes": results, "view_count": sum(r["views"] for r in results),
              "segmentation_performed": False, "source_data_modified": False}
    write_json(output / "report.json", result)
    return result
