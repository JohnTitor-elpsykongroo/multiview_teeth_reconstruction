"""Diagnostics and candidate assembly. No GT or implicit presence inference."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

from .contract import (TEETH, THIRD_MOLARS, View, load_observations, require,
                       inference_path, read_json, resource, sha256)
from .geometry import ray, intersect_rays, project, rigid


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def adapt(views):
    result, excluded = [], {}
    for view in views:
        labels, valid, confidence = view.labels.copy(), view.valid.copy(), view.confidence.copy()
        mask = np.isin(labels, THIRD_MOLARS)
        excluded[view.metadata["view_id"]] = {str(x): int((labels == x).sum()) for x in THIRD_MOLARS}
        labels[mask], valid[mask], confidence[mask] = 255, 0, 0
        result.append(View(deepcopy(view.metadata), labels, valid, confidence,
                           [deepcopy(x) for x in view.instances if x["fdi"] not in THIRD_MOLARS]))
    return result, excluded


def presence_policy(policy, views):
    require(isinstance(policy, dict) and isinstance(policy.get("origin"), str) and policy["origin"].strip(),
            "external presence policy with origin required")
    require(policy.get("kind") in ("assume_all_present", "explicit_external"), "unsupported presence policy")
    if policy["kind"] == "assume_all_present":
        presence = {arch: {str(fdi): True for fdi in teeth} for arch, teeth in TEETH.items()}
    else:
        presence = deepcopy(policy["presence"])
        require(set(presence) == set(TEETH), "presence needs both arches")
        for arch, teeth in TEETH.items():
            require(set(presence[arch]) == {str(x) for x in teeth}
                    and all(type(v) is bool for v in presence[arch].values()), "presence needs 14 Boolean FDI per arch")
    visible = set().union(*(set(np.unique(v.labels)) - {0, 255} for v in views))
    absent = {int(k) for values in presence.values() for k, value in values.items() if not value}
    require(not visible.intersection(absent), "observed FDI conflicts with external absent policy")
    return presence


def diagnose(views, presence):
    teeth, risks = {}, []
    for fdi in (*TEETH["upper"], *TEETH["lower"]):
        entries, origins, directions, seen_cameras = [], [], [], set()
        for view in views:
            yy, xx = np.where(view.labels == fdi)
            if not len(xx):
                continue
            row = view.metadata
            uv = [float(xx.mean() + 0.5), float(yy.mean() + 0.5)]
            signature = (np.asarray(row["K"], float).tobytes(), np.asarray(row["T_camera_from_world"], float).tobytes())
            # Repeated camera evidence is not a new triangulation constraint.
            if signature not in seen_cameras:
                origin, direction = ray(uv, row["K"], row["T_camera_from_world"])
                origins.append(origin)
                directions.append(direction)
                seen_cameras.add(signature)
            instance = next(x for x in view.instances if x["fdi"] == fdi)
            probabilities = instance["fdi_probabilities"]
            own = probabilities[str(fdi)]
            alternative = max((k for k in probabilities if k != str(fdi)), key=probabilities.get)
            margin = own - probabilities[alternative]
            if margin < 0.1:
                risks.append(dict(kind="LOW_NUMBERING_MARGIN", view_id=row["view_id"], fdi=fdi,
                                  alternative_fdi=int(alternative), margin=margin))
            entries.append(dict(view_id=row["view_id"], pixels=len(xx), centroid_edge_pixels=uv,
                                mean_confidence=float(view.confidence[yy, xx].mean())))
        triangulation = intersect_rays(origins, directions)
        if triangulation["status"] == "CENTROID_HEURISTIC":
            point = np.asarray([triangulation["point_world_mm"]])
            errors = []
            for entry in entries:
                row = next(v.metadata for v in views if v.metadata["view_id"] == entry["view_id"])
                uv, _ = project(point, row["K"], row["T_camera_from_world"])
                errors.append(float(np.linalg.norm(uv[0] - entry["centroid_edge_pixels"])))
            triangulation["reprojection_rms_px"] = float(np.sqrt(np.mean(np.square(errors))))
            if triangulation["reprojection_rms_px"] > 5:
                risks.append(dict(kind="CROSS_VIEW_CENTROID_INCONSISTENCY", fdi=fdi,
                                  note="May reflect visibility/occlusion or numbering; not proof of misnumbering."))
        teeth[str(fdi)] = dict(views=entries, distinct_cameras=len(seen_cameras), triangulation=triangulation)
    arches = {}
    for arch, labels in TEETH.items():
        cameras = {(np.asarray(v.metadata["K"], float).tobytes(), np.asarray(v.metadata["T_camera_from_world"], float).tobytes())
                   for v in views if set(np.unique(v.labels)).intersection(labels)}
        arches[arch] = dict(distinct_visible_cameras=len(cameras), minimum_view_gate=len(cameras) >= 2,
                           unobserved_assumed_present=[fdi for fdi in labels if presence[arch][str(fdi)] and not teeth[str(fdi)]["views"]])
    return dict(arches=arches, teeth=teeth, numbering_risks=risks,
                valid_pixels=sum(int(v.valid.sum()) for v in views),
                foreground_pixels=sum(int(((v.labels != 0) & (v.labels != 255)).sum()) for v in views),
                caveat="Mask centroids are not homologous anatomical points; triangulations are diagnostics only.")


def _dmm():
    root = Path(__file__).resolve().parents[1] / "third_party" / "DMM"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def assemble_candidate(output, observations, views, presence, config):
    """Reuse DMM validators, bundle loader, and ArchState; never publish manifest.json.

    The current DMM prepare-scene entrypoint performs the final bundle/camera
    compatibility checks. Keep the candidate distinct until it has been run.
    """
    _dmm()
    from dmm.bundle import load_bundle
    from dmm.scene import ArchState, load_observations as load_dmm_observations
    from dmm.validation import stamped, reference
    from dmm.provenance import source_fingerprint

    before = source_fingerprint()
    require(set(config["model_bundle"]) == set(TEETH), "both model bundles required")
    poses = config["initialization"]
    require(poses.get("method") == "external_pose" and isinstance(poses.get("origin"), str) and poses["origin"].strip(),
            "independently supplied external_pose and origin required")
    require(set(poses["T_world_from_arch"]) == set(TEETH), "both arch poses required")
    parameters = stamped("scene_initialization", method="independent_estimate", origin=poses["origin"], arches={})
    bundle_paths = {}
    for arch in TEETH:
        source = inference_path(config["model_bundle"][arch])
        metadata = read_json(source)
        # Only these three explicit inference bundle files may be read/copied.
        for key in ("weights", "latent_statistics"):
            ref = metadata[key]
            require(set(ref) == {"path", "sha256"}, "invalid bundle reference")
            resolved = resource(source, ref["path"])
            require(sha256(resolved) == ref["sha256"], "bundle SHA256 mismatch")
        bundle = load_bundle(source, arch, "cpu", [source.parent])
        destination = output / "model_bundle" / arch
        destination.mkdir(parents=True)
        for key in ("weights", "latent_statistics"):
            target = destination / metadata[key]["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resource(source, metadata[key]["path"]), target)
        target = destination / "model.json"
        shutil.copyfile(source, target)
        bundle_paths[arch] = target
        initial = dict(T_world_from_arch=rigid(poses["T_world_from_arch"][arch]).tolist(),
                       q={fdi: [0.0] * bundle.model.dimensions[int(fdi)] for fdi, yes in presence[arch].items() if yes},
                       model_sha256=sha256(target), statistics_sha256=metadata["latent_statistics"]["sha256"])
        ArchState(bundle, {int(k): v for k, v in presence[arch].items()}, initial)
        parameters["arches"][arch] = initial
    fit = output / "fit_input"
    fit.mkdir()
    rows = []
    for index, view in enumerate(views):
        row = {key: deepcopy(view.metadata[key]) for key in (
            "view_id", "width", "height", "source_width", "source_height", "K", "K_source",
            "A_fit_from_source_pixels", "T_camera_from_world", "distortion_model", "distortion_coefficients")}
        for key, folder in (("mask", "labels"), ("valid_mask", "valid"), ("confidence", "confidence")):
            extension = "npy" if key == "confidence" else "png"
            row[key] = reference(output / "adapted" / folder / f"{index:04d}.{extension}", fit / "cameras.json")
        row["pixel_convention"] = "edge_origin_centers_at_half"
        rows.append(row)
    # Place masks under DMM's allowed fit_input tree (not an extra loader root).
    shutil.copytree(output / "adapted", fit / "observations")
    for row in rows:
        for key in ("mask", "valid_mask", "confidence"):
            row[key]["path"] = row[key]["path"].replace("../adapted/", "observations/")
    write_json(fit / "cameras.json", stamped("cameras", views=rows))
    ids = [v.metadata["view_id"] for v in views]
    load_dmm_observations(fit / "cameras.json", ids, {a: {int(k): v for k, v in p.items()} for a, p in presence.items()}, [fit])
    init = output / "initialization"
    init.mkdir()
    write_json(init / "parameters.json", parameters)
    candidate = stamped("fit_manifest", scene_id=observations.manifest["scene_id"], length_unit="mm", presence=presence,
                        views=ids, camera_file=reference(fit / "cameras.json", fit / "candidate_manifest.json"),
                        initialization=reference(init / "parameters.json", fit / "candidate_manifest.json"),
                        model_bundle={a: reference(p, fit / "candidate_manifest.json") for a, p in bundle_paths.items()},
                        observation_kind=observations.manifest["observation_kind"],
                        status="PENDING_DMM_PREPARE_SCENE_VALIDATION", fit_ready=False)
    write_json(fit / "candidate_manifest.json", candidate)
    require(source_fingerprint() == before, "DMM source fingerprint changed during assembly")
    return before


def prepare(manifest_path, output, policy, *, config=None, allow_fixture=False, allow_oracle=False):
    observations = load_observations(manifest_path, allow_fixture=allow_fixture, allow_oracle=allow_oracle)
    views, excluded = adapt(observations.views)
    presence = presence_policy(policy, views)
    diagnostics = diagnose(views, presence)
    output = Path(output).resolve()
    require(not output.exists(), "output must be a fresh directory")
    output.mkdir(parents=True)
    for name in ("labels", "valid", "confidence"):
        (output / "adapted" / name).mkdir(parents=True)
    adapted_rows = []
    for index, view in enumerate(views):
        Image.fromarray(view.labels).save(output / "adapted" / "labels" / f"{index:04d}.png")
        Image.fromarray(view.valid).save(output / "adapted" / "valid" / f"{index:04d}.png")
        np.save(output / "adapted" / "confidence" / f"{index:04d}.npy", view.confidence, allow_pickle=False)
        row = {key: deepcopy(view.metadata[key]) for key in (
            "view_id", "width", "height", "source_width", "source_height", "K", "K_source",
            "A_fit_from_source_pixels", "T_camera_from_world", "distortion_model", "distortion_coefficients")}
        row.update(labels=f"labels/{index:04d}.png", valid_mask=f"valid/{index:04d}.png",
                   confidence=f"confidence/{index:04d}.npy", instances=view.instances,
                   quality=dict(valid_pixels=int(view.valid.sum()), ignored_pixels=int((view.valid == 0).sum()),
                                observed_fdi=sorted(x["fdi"] for x in view.instances)))
        row["sha256"] = {key: sha256(output / "adapted" / row[key]) for key in ("labels", "valid_mask", "confidence")}
        adapted_rows.append(row)
    write_json(output / "adapted" / "manifest.json", dict(
        schema_id="dental_fused_observations", schema_version="1.0.0", scene_id=observations.manifest["scene_id"],
        observation_kind=observations.manifest["observation_kind"], pixel_convention="edge_origin_centers_at_half",
        length_unit="mm", world_axes=observations.manifest["world_axes"], views=adapted_rows))
    pending = ["DMM_PREPARE_SCENE_VALIDATION", "MODEL_MEAN_INITIALIZATION_NOT_RUN"]
    if not config or not config.get("model_bundle"):
        pending.append("MODEL_BUNDLES")
    if not config or not config.get("initialization"):
        pending.append("INDEPENDENT_ARCH_POSES")
    if not all(a["minimum_view_gate"] for a in diagnostics["arches"].values()) or diagnostics["foreground_pixels"] == 0:
        pending.append("INSUFFICIENT_OBSERVATIONS")
    report = dict(schema_id="dental_observation_fusion", schema_version="1.0.0",
                  scene_id=observations.manifest["scene_id"], status="PENDING", fit_ready=False,
                  observation_kind=observations.manifest["observation_kind"], debug_fixture=observations.manifest["observation_kind"] == "fixture",
                  source_manifest_sha256=observations.source_sha256, source_resource_sha256=observations.resource_sha256,
                  view_ids=[v.metadata["view_id"] for v in views], presence_policy=policy, presence=presence,
                  excluded_third_molar_pixels=excluded, diagnostics=diagnostics, pending=pending,
                  initialization={"status": "PENDING", "T_world_from_arch": None},
                  confidence_use="Preserved; existing DMM scene loader does not consume confidence weights.")
    if config and config.get("model_bundle") and config.get("initialization") and "INSUFFICIENT_OBSERVATIONS" not in pending:
        report["dmm_source_sha256"] = assemble_candidate(output, observations, views, presence, config)
        report["candidate_manifest"] = "fit_input/candidate_manifest.json"
        report["initialization"] = {"status": "EXTERNALLY_SUPPLIED_NOT_ESTIMATED", "parameters": "initialization/parameters.json"}
    # Completion marker for diagnostics, never a claim of completed fitting.
    write_json(output / "report.json", report)
    return report
