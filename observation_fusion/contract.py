"""Strict stage-two prediction consumer. Provenance strings are never opened."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

import numpy as np
from PIL import Image

from .geometry import rigid

FDI = tuple(q * 10 + i for q in range(1, 5) for i in range(1, 9))
TEETH = {"upper": tuple(x for x in FDI if x < 30 and x % 10 != 8),
         "lower": tuple(x for x in FDI if x > 30 and x % 10 != 8)}
THIRD_MOLARS = (18, 28, 38, 48)


class ContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ContractError(f"nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=invalid)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inference_path(path):
    path = Path(path).resolve()
    require(not {"annotations", "truth"}.intersection(x.lower() for x in path.parts),
            "annotations/truth forbidden for inference")
    return path


def resource(owner, reference):
    require(isinstance(reference, str) and bool(reference), "resource must be a relative path")
    p = PurePosixPath(reference)
    require("\\" not in reference and ":" not in reference and not p.is_absolute()
            and not PureWindowsPath(reference).drive and ".." not in p.parts,
            "unsafe resource reference")
    require(not {"annotations", "truth"}.intersection(x.lower() for x in p.parts),
            "annotations/truth forbidden for inference")
    root = Path(owner).resolve().parent
    target = inference_path(root / reference)
    require(target.is_relative_to(root) and target.is_file(), "resource missing or outside manifest directory")
    return target


def array(value, shape, name):
    a = np.asarray(value, dtype=float)
    require(a.shape == shape and np.isfinite(a).all(), f"invalid {name}")
    return a


def camera(row):
    for field in ("width", "height", "source_width", "source_height"):
        require(type(row[field]) is int and row[field] > 0, f"invalid {field}")
    k, source, a = (array(row[key], (3, 3), key) for key in ("K", "K_source", "A_fit_from_source_pixels"))
    for matrix in (k, source):
        require(matrix[0, 0] > 0 and matrix[1, 1] > 0 and abs(matrix[1, 0]) < 1e-8
                and np.allclose(matrix[2], [0, 0, 1], atol=1e-8, rtol=0), "invalid pinhole K")
    require(np.allclose(a[2], [0, 0, 1], atol=1e-8, rtol=0) and a[0, 0] > 0 and a[1, 1] > 0
            and abs(a[0, 1]) < 1e-8 and abs(a[1, 0]) < 1e-8, "invalid crop/resize A")
    require(np.allclose(k, a @ source, atol=1e-5, rtol=0), "K != A @ K_source")
    rigid(row["T_camera_from_world"])
    require(row["distortion_model"] == "none" and row["distortion_coefficients"] == [], "distortion unsupported")


def png(path, shape):
    with Image.open(path) as im:
        require(im.format == "PNG" and im.mode == "L", "expected PNG L")
        result = np.asarray(im).copy()
    require(result.shape == shape and result.dtype == np.uint8, "mask shape/dtype mismatch")
    return result


@dataclass
class View:
    metadata: dict
    labels: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    instances: list


@dataclass
class Observations:
    manifest: dict
    views: list
    source_sha256: str
    resource_sha256: dict


def load_observations(manifest_path, *, allow_fixture=False, allow_oracle=False):
    path = inference_path(manifest_path)
    data = read_json(path)
    require(data.get("schema_version") in ("1.0.0", "1.1.0"), "invalid schema_version")
    for key, value in {"schema_id": "dental_tooth_observations",
                       "length_unit": "mm", "pixel_convention": "edge_origin_centers_at_half",
                       "world_axes": "X_patient_left_Y_posterior_Z_superior"}.items():
        require(data.get(key) == value, f"invalid {key}")
    require(isinstance(data.get("scene_id"), str) and bool(data["scene_id"].strip()), "empty scene_id")
    require(data.get("observation_kind") in ("prediction", "fixture", "oracle"), "invalid observation_kind")
    oracle = data.get("observation_kind") == "oracle"
    require(not oracle or (allow_oracle and data["schema_version"] == "1.1.0"), "oracle requires explicit allow_oracle and v1.1")
    require(allow_fixture or data["observation_kind"] != "fixture", "fixture rejected; explicit debug opt-in required")
    require(data.get("status") == "OBSERVATIONS_READY_REVIEW_REQUIRED", "observations are not complete")
    require(isinstance(data["input_manifest"], str) and data["input_manifest"], "missing source provenance")
    if oracle:
        require(data["model"]["backend"] == "blender_annotation_oracle" and data["model"]["checkpoint_sha256"] is None, "invalid oracle backend")
        require(data["oracle"]["uses_3d_truth"] is False and data["oracle"]["confidence_semantics"] == "unit_annotation_weight_not_probability", "invalid oracle provenance")
    for value in (data["input_manifest_sha256"], data["oracle"]["annotations_manifest_sha256"] if oracle else data["model"]["checkpoint_sha256"]):
        require(isinstance(value, str) and re.fullmatch("[0-9a-fA-F]{64}", value), "invalid provenance SHA256")
    model = data["model"]
    require(all(isinstance(model.get(k), str) and model[k] for k in ("backend", "version")), "model identity required")
    require(isinstance(model["class_ids"], list) and all(type(x) is int for x in model["class_ids"])
            and len(model["class_ids"]) == 33 and set(model["class_ids"]) == {0, *FDI}, "invalid class_ids")
    for key in ("confidence_threshold", "margin_threshold"):
        value = data["config"][key]
        require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1, f"invalid {key}")
    require(isinstance(data["views"], list) and data["views"], "no views")
    views, ids, signatures, hashes = [], set(), set(), {}
    for row in data["views"]:
        identifier = row["view_id"]
        require(isinstance(identifier, str) and identifier.strip() and identifier not in ids, "empty/duplicate view_id")
        ids.add(identifier)
        camera(row)
        shape = (row["height"], row["width"])
        refs = {key: resource(path, row[key]) for key in ("labels", "valid_mask", "confidence", "instances")}
        hashes[identifier] = {key: sha256(value) for key, value in refs.items()}
        labels, valid = (png(refs[key], shape) for key in ("labels", "valid_mask"))
        require(set(np.unique(labels)) <= {0, 255, *FDI}, "invalid FDI labels")
        require(set(np.unique(valid)) <= {0, 1} and np.array_equal(valid, labels != 255), "valid/ignore mismatch")
        confidence = np.load(refs["confidence"], allow_pickle=False)
        require(isinstance(confidence, np.ndarray) and confidence.dtype == np.float32 and confidence.shape == shape,
                "confidence must be float32 HxW")
        require(np.isfinite(confidence).all() and np.all((confidence >= 0) & (confidence <= 1))
                and np.all(confidence[valid == 0] == 0), "invalid confidence values")
        instances = read_json(refs["instances"])
        observed = sorted(int(x) for x in np.unique(labels) if x in FDI)
        require(isinstance(instances, list) and len(instances) == len(observed), "instance count mismatch")
        found = set()
        for instance in instances:
            fdi = instance["fdi"]
            require(type(fdi) is int and fdi in observed and fdi not in found, "instance FDI mismatch/duplicate")
            found.add(fdi)
            yy, xx = np.where(labels == fdi)
            require(instance["jaw"] == ("upper" if fdi < 30 else "lower"), "instance jaw mismatch")
            require(type(instance["visible_pixels"]) is int and instance["visible_pixels"] == len(xx), "instance pixel count mismatch")
            require(instance["bbox_xyxy_exclusive"] == [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1], "instance bbox mismatch")
            require(np.isclose(instance["confidence"], confidence[labels == fdi].mean(), atol=1e-6, rtol=0), "instance confidence mismatch")
            require(type(instance["fdi_uncertain"]) is bool and not instance["fdi_uncertain"], "uncertain FDI must be ignored")
            probabilities = instance["fdi_probabilities"]
            require(isinstance(probabilities, dict) and set(probabilities) == {str(x) for x in FDI}, "need all 32 probability keys")
            probs = array(list(probabilities.values()), (32,), "FDI probabilities")
            require(np.all((probs >= 0) & (probs <= 1)) and np.isclose(probs.sum(), 1, atol=1e-5, rtol=0), "invalid FDI probabilities")
        quality = row["quality"]
        require(quality["valid_pixels"] == int(valid.sum()) and quality["ignored_pixels"] == int((valid == 0).sum())
                and quality["observed_fdi"] == observed, "quality inconsistent with labels")
        signature = hashlib.sha256(np.asarray(row["K"], float).tobytes() + np.asarray(row["T_camera_from_world"], float).tobytes()
                                   + np.asarray(shape).tobytes() + labels.tobytes() + valid.tobytes()).hexdigest()
        require(signature not in signatures, "duplicate camera/mask/valid observation")
        signatures.add(signature)
        views.append(View(row, labels, valid, confidence, instances))
    return Observations(data, views, sha256(path), hashes)
