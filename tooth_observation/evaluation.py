"""Explicit supervision-only scoring. Ignored predictions still count as misses."""
from pathlib import Path
import numpy as np

from . import FDI_IDS
from .data import PhotoScene, resolve, supervision, write_json


def metrics(labels, valid, truth, tissue):
    eligible = truth != 255
    teeth = np.isin(truth, FDI_IDS) & eligible
    predicted_teeth = np.isin(labels, FDI_IDS) & eligible
    per_tooth = {}
    for fdi in FDI_IDS:
        a, b = (labels == fdi) & eligible, (truth == fdi) & eligible
        union = np.count_nonzero(a | b)
        if union:
            per_tooth[str(fdi)] = {"iou": float(np.count_nonzero(a & b) / union),
                                   "gt_pixels": int(b.sum()), "predicted_pixels": int(a.sum())}
    union = np.count_nonzero(teeth | predicted_teeth)
    soft = np.isin(tissue, [101, 102, 103]) & eligible
    return {"per_fdi": per_tooth,
            "macro_tooth_iou": float(np.mean([v["iou"] for v in per_tooth.values()])) if per_tooth else None,
            "binary_tooth_iou": float(np.count_nonzero(teeth & predicted_teeth) / union) if union else None,
            "retained_pixel_fraction": float(np.count_nonzero((valid == 1) & eligible) / eligible.sum()) if eligible.any() else None,
            "gt_tooth_ignored_fraction": float(np.count_nonzero(teeth & (valid == 0)) / teeth.sum()) if teeth.any() else None,
            "soft_tissue_ignored_fraction": float(np.count_nonzero(soft & (valid == 0)) / soft.sum()) if soft.any() else None,
            "numbering_error_pixels": int(np.count_nonzero(teeth & predicted_teeth & (labels != truth)))}


def evaluate_predictions(prediction_manifest, input_manifest, annotations_manifest, output, *, allow_fixture=False):
    from observation_fusion.contract import load_observations
    observations = load_observations(prediction_manifest, allow_fixture=allow_fixture)
    scene = PhotoScene(input_manifest)
    if observations.manifest["scene_id"] != scene.manifest["scene_id"]:
        raise ValueError("evaluation scene mismatch")
    from .data import sha256
    if observations.manifest["input_manifest_sha256"] != sha256(scene.path):
        raise ValueError("prediction was not produced from this input manifest")
    if {v.metadata["view_id"] for v in observations.views} != set(scene.rows):
        raise ValueError("evaluation views differ")
    annotations = supervision(scene, annotations_manifest, include_tissue=True)
    output = Path(output).resolve()
    for root in (scene.root.parent, Path(prediction_manifest).resolve().parent, Path(annotations_manifest).resolve().parent):
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("evaluation output must be disjoint from inputs")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for view in observations.views:
        name = view.metadata["view_id"]
        rgb, acquisition_valid, camera = scene.load(name)
        for key in ("rgb", "valid_mask"):
            digest = view.metadata.get("source_sha256", {}).get(key)
            if digest is not None and digest != sha256(resolve(scene.root, scene.rows[name][key])):
                raise ValueError("evaluation input resource changed since prediction")
        if (view.labels.shape != rgb.shape[:2]
            or not np.allclose(view.metadata["K"], camera["K"], rtol=0, atol=1e-8)
            or not np.allclose(view.metadata["T_camera_from_world"], camera["T_camera_from_world"], rtol=0, atol=1e-8)
            or not np.allclose(view.metadata["A_fit_from_source_pixels"], np.eye(3), rtol=0, atol=1e-8)):
            raise ValueError("evaluation currently requires original-size K/T/A")
        truth = annotations[name]["fdi"].copy()
        truth[acquisition_valid == 0] = 255
        rows.append({"view_id": name, **metrics(view.labels, view.valid, truth, annotations[name]["tissue"])})
    report = {"status": "FIXTURE_EVALUATION_ONLY" if observations.manifest["observation_kind"] == "fixture" else "SUPERVISED_SCENE_EVALUATION",
              "scene_id": scene.manifest["scene_id"], "views": rows,
              "prediction_sha256": observations.source_sha256, "annotations_manifest_sha256": sha256(annotations_manifest),
              "ignore_policy": "GT ignore and acquisition-invalid excluded; prediction ignore counts as a tooth miss",
              "real_photo_validated": False}
    write_json(output / "report.json", report)
    return report
