"""Strict inference reader. Supervision is only reachable by an explicit call."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from . import FDI_IDS, VERSION


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(root, ref):
    """Only portable relative paths, with physical containment after resolution."""
    if not isinstance(ref, str) or not ref or "\\" in ref or ":" in ref:
        raise ValueError(f"invalid relative path: {ref!r}")
    p = PurePosixPath(ref)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe relative path: {ref}")
    root = Path(root).resolve(strict=True)
    result = (root / ref).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError(f"resource escapes root or is not a file: {ref}")
    return result


def _header(value, scene=None):
    if value.get("schema_id") != "dental_multiview_photo" or value.get("schema_version") != VERSION:
        raise ValueError("unsupported photo contract")
    if not isinstance(value.get("scene_id"), str) or not value["scene_id"]:
        raise ValueError("invalid scene_id")
    if scene is not None and value["scene_id"] != scene:
        raise ValueError("scene mismatch")


def view_map(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("views must be a nonempty list")
    output = {}
    for row in rows:
        name = row["view_id"]
        if not isinstance(name, str) or not name or name in output:
            raise ValueError("empty/duplicate view_id")
        output[name] = row
    return output


def camera_check(c):
    for key in ("width", "height"):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError("invalid camera dimensions")
    if c.get("length_unit") != "mm" or c.get("pixel_convention") != "edge_origin_centers_at_half":
        raise ValueError("unsupported unit/pixel convention")
    if c.get("distortion_model") != "none" or c.get("distortion_coefficients") != []:
        raise ValueError("only undistorted cameras supported")
    K, T = np.asarray(c["K"], dtype=float), np.asarray(c["T_camera_from_world"], dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("invalid K")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or abs(K[1, 0]) > 1e-8:
        raise ValueError("invalid pinhole intrinsics")
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("invalid camera transform")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError("camera rotation is not rigid")
    # Phase-one v1 has no crop. Do not silently accept a transformed input.
    if "A_fit_from_source_pixels" in c and not np.allclose(c["A_fit_from_source_pixels"], np.eye(3)):
        raise ValueError("source crop/resize requires a new input adapter")


def png(path, mode, shape):
    with Image.open(path) as im:
        if im.format != "PNG" or im.mode != mode:
            raise ValueError(f"expected PNG {mode}: {path}")
        array = np.array(im)
    if array.dtype != np.uint8 or array.shape != shape:
        raise ValueError(f"wrong image shape/dtype: {path}")
    return array


class PhotoScene:
    """Loads no directory index, annotations, geometry, or latent information."""
    def __init__(self, manifest):
        self.path = Path(manifest).resolve(strict=True)
        self.root = self.path.parent
        self.manifest = read_json(self.path)
        _header(self.manifest)
        allowed = {"schema_id", "schema_version", "scene_id", "length_unit", "camera_file", "views"}
        if set(self.manifest) != allowed or self.manifest["length_unit"] != "mm":
            raise ValueError("unexpected inference manifest fields/unit")
        self.rows = view_map(self.manifest["views"])
        self.camera_path = resolve(self.root, self.manifest["camera_file"])
        camera_doc = read_json(self.camera_path)
        _header(camera_doc, self.manifest["scene_id"])
        self.cameras = view_map(camera_doc["views"])
        if set(self.cameras) != set(self.rows):
            raise ValueError("camera/view IDs differ")
        for name, row in self.rows.items():
            if set(row) != {"view_id", "rgb", "valid_mask"}:
                raise ValueError("unexpected inference view fields")
            camera_check(self.cameras[name])
            resolve(self.root, row["rgb"])
            resolve(self.root, row["valid_mask"])

    def load(self, view_id):
        row, camera = self.rows[view_id], self.cameras[view_id]
        hw = (camera["height"], camera["width"])
        rgb = png(resolve(self.root, row["rgb"]), "RGB", (*hw, 3))
        valid = png(resolve(self.root, row["valid_mask"]), "L", hw)
        if not np.isin(valid, [0, 1]).all():
            raise ValueError("valid_mask must contain 0/1")
        return rgb, valid, camera


def supervision(scene, annotations_manifest, *, include_tissue=False):
    """Explicit supervised-only loader, never called by observation inference."""
    path = Path(annotations_manifest).resolve(strict=True)
    doc = read_json(path)
    _header(doc, scene.manifest["scene_id"])
    if doc.get("supervision_only") is not True:
        raise ValueError("supervision_only must be true")
    rows = view_map(doc["views"])
    if set(rows) != set(scene.rows):
        raise ValueError("supervision view IDs differ")
    output = {}
    for name, row in rows.items():
        c = scene.cameras[name]
        fdi = png(resolve(path.parent, row["fdi"]), "L", (c["height"], c["width"]))
        if not np.isin(fdi, [0, 255, *FDI_IDS]).all():
            raise ValueError("invalid supervision FDI")
        if include_tissue:
            tissue = png(resolve(path.parent, row["tissue"]), "L", fdi.shape)
            if not np.isin(tissue, [0, 100, 101, 102, 103, 104]).all():
                raise ValueError("invalid tissue supervision")
            if np.any((tissue != 0) & np.isin(fdi, FDI_IDS)):
                raise ValueError("visible tooth overlaps tissue supervision")
            output[name] = {"fdi": fdi, "tissue": tissue}
        else:
            output[name] = fdi
    return output
