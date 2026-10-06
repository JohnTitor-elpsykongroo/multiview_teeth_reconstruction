"""Read-only supervised acceptance. Never imported by the RGB inference path."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from . import FDI_IDS
from .data import PhotoScene, read_json, resolve, sha256, supervision, write_json, png
from .pipeline import overlay


def require(value, message):
    if not value:
        raise ValueError(message)


def audit_dataset(dataset, output, *, geometry=False):
    path = Path(dataset).resolve(strict=True)
    root = path.parent
    output = Path(output).resolve()
    require(not output.is_relative_to(root) and not root.is_relative_to(output), "audit output must be disjoint")
    output.mkdir(parents=True, exist_ok=False)
    index = read_json(path)
    require(index.get("schema_id") == "dental_multiview_photo" and index.get("schema_version") == "1.0.0", "index schema")
    patients, names, reports = {}, set(), []
    for row in index["scenes"]:
        require(row["scene_id"] not in names, "duplicate scene")
        names.add(row["scene_id"])
        require(patients.setdefault(row["patient_id"], row["split"]) == row["split"], "patient split leakage")
        scene = PhotoScene(resolve(root, row["input_manifest"]))
        require(scene.manifest["scene_id"] == row["scene_id"], "scene mismatch")
        ann_path = resolve(root, row["annotations_manifest"])
        annotations = supervision(scene, ann_path, include_tissue=True)
        ann_rows = {a["view_id"]: a for a in read_json(ann_path)["views"]}
        case = scene.root.parent
        checksum_path = case / "checksums.json"
        checksums = read_json(checksum_path)
        for ref, digest in checksums.items():
            require(sha256(resolve(case, ref)) == digest, f"artifact checksum mismatch: {ref}")
        truth = None
        meshes = {}
        if geometry:
            import trimesh
            truth_path = resolve(root, row["truth_manifest"])
            truth = read_json(truth_path)
            require(truth["evaluation_only"] is True and truth["length_unit"] == "mm", "truth contract")
            for record in truth["meshes"].values():
                if record["fdi"]:
                    meshes[record["fdi"]] = trimesh.load_mesh(resolve(truth_path.parent, record["path"]), process=False)
        views, tiles, errors, seen, all_present = [], [], [], set(), None
        rng = np.random.default_rng(51005)
        for name in scene.rows:
            rgb, valid, camera = scene.load(name)
            fdi, tissue = annotations[name]["fdi"], annotations[name]["tissue"]
            instances = read_json(resolve(ann_path.parent, ann_rows[name]["instances"]))
            present = {a["fdi"] for a in instances}
            require(len(present) == len(instances) and present <= set(FDI_IDS), "instances IDs")
            require(all_present is None or all_present == present, "static tooth inventory differs")
            all_present = present
            visible = sorted(int(k) for k in np.unique(fdi) if k in FDI_IDS)
            require(set(visible) <= present, "label has no instance")
            seen.update(visible)
            for a in instances:
                yy, xx = np.where(fdi == a["fdi"])
                bbox = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)] if len(xx) else None
                require(a["visible_pixels"] == len(xx) and a["bbox_xyxy_exclusive"] == bbox, "instance area/bbox")
                require(a["jaw"] == ("upper" if a["fdi"] < 30 else "lower"), "instance jaw")
            soft = np.isin(tissue, [101, 102, 103])
            blocked = np.zeros(fdi.shape, bool)
            if geometry:
                with np.load(resolve(truth_path.parent, truth["geometry"][name]), allow_pickle=False) as g:
                    depth, hit, normals = g["depth_mm"], g["valid"], g["normal_camera"]
                    require(depth.dtype == np.float32 and depth.shape == fdi.shape and np.isfinite(depth).all(), "depth format")
                    require(np.all(depth >= 0) and np.array_equal(hit, depth > 0), "depth hit mask")
                    require(normals.shape == (*fdi.shape, 3) and np.isfinite(normals).all(), "normal format")
                    require(np.all(hit[np.isin(fdi, FDI_IDS)]), "missing tooth depth")
                    K = np.asarray(camera["K"])
                    T = np.asarray(camera["T_camera_from_world"])
                    for label in visible:
                        yy, xx = np.where(fdi == label)
                        samples = rng.choice(len(xx), size=min(3, len(xx)), replace=False)
                        uv = np.column_stack([xx[samples] + .5, yy[samples] + .5, np.ones(len(samples))])
                        cam = (uv @ np.linalg.inv(K).T) * depth[yy[samples], xx[samples], None]
                        world = (cam - T[:3, 3]) @ T[:3, :3]
                        _, distances, _ = trimesh.proximity.closest_point_naive(meshes[label], world)
                        errors.extend(distances.tolist())
                if "amodal" in truth:
                    for label in present:
                        amodal = png(resolve(truth_path.parent, truth["amodal"][name] + f"/{label}.png"), "L", fdi.shape)
                        require(np.isin(amodal, [0, 1]).all() and np.all(amodal[fdi == label] == 1), "amodal subset")
                        blocked |= (amodal == 1) & soft
            views.append({"view_id": name, "visible_fdi": visible, "tooth_pixels": int(np.isin(fdi, FDI_IDS).sum()),
                          "soft_tissue_pixels": int(soft.sum()), "soft_occluded_amodal_pixels": int(blocked.sum()) if geometry else None,
                          "input_valid_pixels": int(valid.sum()), "annotation_ignore_pixels": int((fdi == 255).sum())})
            # Only labelled as supervision review. Never a prediction manifest.
            soft_preview = rgb.copy()
            soft_preview[soft] = (.4 * rgb[soft] + .6 * np.array([255, 0, 255])).astype(np.uint8)
            tile = Image.new("RGB", (rgb.shape[1]*3, rgb.shape[0]+24), "white")
            for j, arr in enumerate([rgb, overlay(rgb, fdi), soft_preview]):
                tile.paste(Image.fromarray(arr), (rgb.shape[1]*j, 24))
            ImageDraw.Draw(tile).text((4, 4), name + " | RGB / GT FDI / GT tissue (supervision review)", fill="black")
            tile.thumbnail((1200, 330))
            tiles.append(tile)
        if geometry:
            require(errors and max(errors) < .01, "backprojected tooth depth differs from mesh by >=0.01mm")
        montage = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles)), "white")
        y = 0
        for tile in tiles:
            montage.paste(tile, (0, y)); y += tile.height
        preview = f"review_{len(reports):03d}.jpg"
        montage.save(output / preview, quality=92)
        # Source manifests/reports are never rewritten. Recheck hashes after audit.
        for ref, digest in checksums.items():
            require(sha256(resolve(case, ref)) == digest, "source changed during audit")
        reports.append({"scene_id": row["scene_id"], "patient_id": row["patient_id"], "split": row["split"],
                        "source_accepted_for_training": row["accepted_for_training"], "artifact_hashes_verified": len(checksums),
                        "visible_fdi_union": sorted(seen), "never_visible_fdi": sorted(all_present - seen),
                        "views": views, "review": preview, "depth_mesh_samples": len(errors),
                        "depth_mesh_max_mm": max(errors) if errors else None,
                        "source_qc": read_json(resolve(root, row["qc_report"])),
                        "source_qc_note": "producer evidence; only depth_mesh_samples above independently recomputed"})
    report = {"status": "STAGE2_INPUT_AUDIT_PASS_VISUAL_REVIEW_REQUIRED", "dataset": str(path),
              "dataset_sha256": sha256(path), "scenes": reports, "source_unchanged": True,
              "geometry_sampled": geometry, "training_authorized": False,
              "limits": ["Sparse depth/mesh samples are not a full ray-occlusion or collision certificate",
                         "No prediction metrics or learned model acceptance", "Patient split and source training flags unchanged"]}
    write_json(output / "report.json", report)
    return report
