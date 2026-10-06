"""Audit completed synthetic results into a fresh directory without refitting.

Checks saved inputs and archived code, recomputes label-image IoU and pose
errors, and repeats one surface-distance failure with the recorded metric.
Historical runs are read-only. Incomplete campaigns remain incomplete.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
observed = {}
verified = []
warnings = []


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    observed[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def check(path, expected, kind):
    path = Path(path).resolve()
    actual = digest(path) if path.is_file() else None
    observed[str(path)] = actual
    verified.append({"path": str(path), "kind": kind,
                     "expected": expected, "actual": actual,
                     "matches": actual == expected})


def iou(pred, target, labels):
    values = []
    for label in labels:
        a, b = pred == label, target == label
        union = np.count_nonzero(a | b)
        if union:
            values.append(np.count_nonzero(a & b) / union)
    return values


def audit_one(group, evaluation_path):
    ev = read(evaluation_path)
    root = evaluation_path.parent
    if root.name.startswith("surface_review"):
        root = root.parent
        check(root / "evaluation.json", ev["original_evaluation_sha256"], "original_evaluation")
        for p, sha in ev.get("mesh_input_sha256", {}).items():
            check(p, sha, "reviewed_mesh")
    report = read(root / "fit_report.json")
    provenance = read(root / "provenance.json")
    config = read(root / "resolved_config.json")
    fit = Path(provenance.get("fit_input", config.get("fit_input", "")))
    check(fit / "manifest.json", provenance["fit_manifest_sha256"], "fit_manifest")
    manifest = read(fit / "manifest.json")
    check(fit / manifest["camera_file"], manifest["camera_sha256"], "camera")
    labels = manifest.get("active_labels", manifest.get("visible_fdi_ids", provenance.get("labels")))
    for item in manifest["masks"]:
        check(fit / item["path"], item["sha256"], "mask")
    for name, sha in manifest.get("extra_input_sha256", {}).items():
        check(fit / name, sha, "fit_extra")
    for label, sha in provenance.get("fixed_mesh_sha256", {}).items():
        check(Path(config["fixed_meshes"]) / f"tooth{label}.ply", sha, "fixed_shape")
    for name, sha in provenance.get("code_sha256", {}).items():
        archived = root / "code_snapshot" / name
        if archived.exists():
            check(archived, sha, "fit_code_snapshot")
        else:
            warnings.append({"root": str(root), "code_snapshot_missing": name})
    for name, sha in provenance.get("dmm_source_sha256", {}).items():
        check(name, sha, "reference_dmm_source")
    if "experiment" in config:
        exp = Path(config["experiment"])
        check(exp / "ModelParameters" / f'dmm_{config["checkpoint"]}.pth', manifest["model_sha256"], "model")
        check(exp / "specs.json", manifest["specs_sha256"], "specs")
    truth = fit.parent / "truth" / "manifest.json"
    check(truth, ev["truth_manifest_sha256"], "truth_manifest")
    tr = read(truth)
    for key in ["latent", "pose", "heldout_camera", "heldout_mask"]:
        if key + "_file" in tr:
            check(truth.parent / tr[key + "_file"], tr[key + "_sha256"], "truth_payload")
    for name, sha in tr.get("mesh_sha256", {}).items():
        check(truth.parent / "meshes" / name, sha, "truth_mesh")

    # Recompute from stored label pixels, independent of the report's means.
    values = []
    for item in manifest["masks"]:
        path = root / "renders" / "final" / f'{item["camera"]}_predicted_labels.png'
        if not path.exists():
            raise FileNotFoundError(path)
        observed[str(path)] = digest(path)
        values += iou(np.asarray(Image.open(path)), np.asarray(Image.open(fit / item["path"])), labels)
    image_iou = float(np.mean(values))
    reported_iou = report["final"].get("active_mean_tooth_iou", report["final"].get("mean_tooth_iou"))
    assertions = {"fitting_iou_from_pixels": bool(abs(image_iou - reported_iou) < 1e-12),
                  "limits_unchanged": ev["acceptance_limits"] == config["acceptance"],
                  "status_matches_saved_checks": ev["status"].endswith("PASS") == all(ev["checks"].values())}
    heldout = None
    if "heldout" in ev:
        cam = read(truth.parent / tr["heldout_camera_file"])
        p = root / "renders" / "heldout_final" / f'{cam["name"]}_predicted_labels.png'
        observed[str(p)] = digest(p)
        heldout = float(np.mean(iou(np.asarray(Image.open(p)), np.asarray(Image.open(truth.parent / tr["heldout_mask_file"])), labels)))
        assertions["heldout_iou_from_pixels"] = bool(abs(heldout - ev["heldout"]["final"]["active_mean_tooth_iou"]) < 1e-12)
    pose_errors = None
    if "pose_errors" in ev:
        pose = np.asarray(read(root / "final_pose.json")["arch_to_world"])
        gt = np.asarray(read(truth.parent / tr["pose_file"])["arch_to_world"])
        pose_errors = {"rotation_degrees": float(Rotation.from_matrix(pose[:3, :3] @ gt[:3, :3].T).magnitude() * 180 / np.pi),
                       "translation_dmm": float(np.linalg.norm(pose[:3, 3] - gt[:3, 3]))}
        assertions["pose_errors_from_transforms"] = all(abs(pose_errors[k] - ev["pose_errors"]["final"][k]) < 1e-12 for k in pose_errors)
    elif "final_rotation_error_degrees" in ev:
        pose = read(root / "pose.json")
        pose_errors = {"rotation_degrees": float(np.linalg.norm(pose["rotation_vector_radians"]) * 180 / np.pi),
                       "translation_dmm": float(np.linalg.norm(pose["translation_dmm"]))}
        assertions["pose_errors_from_transforms"] = abs(pose_errors["rotation_degrees"] - ev["final_rotation_error_degrees"]) < 1e-12 and abs(pose_errors["translation_dmm"] - ev["final_translation_error_dmm"]) < 1e-12
    else:
        original = dict(np.load(fit / manifest["initial_codes"]))
        initial = dict(np.load(root / "initial_codes.npz"))
        final = dict(np.load(root / "final_codes.npz"))
        assertions["initial_codes_match_input"] = all(np.array_equal(v, initial[k]) for k, v in original.items())
        assertions["frozen_codes_unchanged"] = all(np.array_equal(initial[f"label_{k}"], final[f"label_{k}"]) for k in manifest["frozen_labels"])
    if "gradient_max_relative_error" in ev:
        gc = read(root / "gradient_check.json")
        assertions["gradient_value_matches_record"] = abs(gc["max_relative_error"] - ev["gradient_max_relative_error"]) < 1e-12
    return {"group": group, "root": str(root), "evaluation": str(evaluation_path), "status": ev["status"],
            "active_labels": labels, "fitting_iou_recomputed": image_iou,
            "evaluated_clean_iou": ev.get("final_active_mean_tooth_iou", ev.get("final_mean_tooth_iou")),
            "heldout_iou_recomputed": heldout, "pose_errors_recomputed": pose_errors,
            "surface_gain": ev.get("surface_mean_improvement_fraction", ev.get("surface_aggregate", {}).get("canonical", {}).get("mean_improvement_fraction")),
            "failed_gates": [k for k, v in ev["checks"].items() if not v],
            "verification": assertions, "scope": ev.get("scope"),
            "oracle_conditions": manifest.get("diagnostic_oracle_conditions", ["known cameras", "fixed true shape"])}


def main():
    out = RUNS / dt.datetime.now(dt.timezone.utc).strftime("acceptance_%Y%m%dT%H%M%SZ")
    out.mkdir(exist_ok=False)
    print(str(out), flush=True)
    selected = []
    for group, batch in [("pose_random", "pose_random_20261004T043020Z_89047cb6"),
                         ("shape_expansion", "shape_expansion_20260930T020823Z_4695c457")]:
        state = read(RUNS / batch / "status.json")
        selected += [(group, Path(j["fit_root"]) / "evaluation.json") for j in state["jobs"].values()]
    cross = read(RUNS / "shape_crosscase_20260930T100642Z_b60557b7/review/summary.json")
    selected += [("shape_crosscase", Path(j["surface_review_evaluation"])) for j in cross["records"]]
    for name in ["pose_only_20260929T100436Z_413ea93f", "shape_only_20260930T014548Z_49b983d8", "shape_only_20260930T103215Z_603f9922", "joint_20260930T101555Z_ec4c7cfe", "joint_warmup_20261004T043110Z_ddd095ce"]:
        selected.append((name.split("_2026")[0], RUNS / name / "evaluation.json"))
    for pattern in ["joint_robustness_*", "joint_stabilization_*", "joint_population_*"]:
        for batch in sorted(RUNS.glob(pattern)):
            selected += [(batch.name.split("_2026")[0], p) for p in sorted(batch.glob("jobs/*/fit_attempt_*/evaluation.json"))]
    rows = []
    for group, path in selected:
        print(group, path.parent.name, flush=True)
        rows.append(audit_one(group, path))

    from evaluate_shape_only import surface_metrics
    from run_forward_check import load_ply
    failure_root = RUNS / "shape_expansion_20260930T104706Z_1b221430/jobs/full_14_seed101/fit_attempt_01"
    ev = read(failure_root / "surface_review_v1/evaluation.json")
    prov = read(failure_root / "provenance.json")
    tr = read(Path(prov["fit_input"]).parent / "truth/manifest.json")
    truth_mesh = Path(tr["source_run"]) / "meshes/training_case/tooth27.ply"
    surface = {}
    for stage in ["initial", "final"]:
        p = failure_root / "meshes" / stage / "tooth27.ply"
        surface[stage] = surface_metrics(load_ply(p), load_ply(truth_mesh))
        surface[stage]["matches_saved"] = all(abs(surface[stage][k] - ev["surface_per_tooth"]["27"][stage][k]) < 1e-12 for k in ["symmetric_sampled_surface_mean_dmm", "symmetric_sampled_surface_p95_dmm"])
    surface["mean_degradation_fraction"] = surface["final"]["symmetric_sampled_surface_mean_dmm"] / surface["initial"]["symmetric_sampled_surface_mean_dmm"] - 1

    upstream = ROOT / "third_party/DMM"
    cmd = ["git", "-c", f"safe.directory={upstream.as_posix()}", "-C", str(upstream)]
    commit = subprocess.check_output(cmd + ["rev-parse", "HEAD"], text=True).strip()
    source = {"origin": subprocess.check_output(cmd + ["remote", "get-url", "origin"], text=True).strip(),
              "commit": commit, "working_tree_clean": not subprocess.check_output(cmd + ["status", "--porcelain"], text=True).strip(), "files": {}}
    for name in ["networks/dmm_net.py", "networks/deform_net.py", "networks/loss.py", "utils/math.py", "utils/mesh.py", "data/data_with_labels.py", "train_dmm.py"]:
        official, reference = upstream / name, Path("D:/WorkSpace/Dental/DMM") / name
        source["files"][name] = {"upstream_sha256": digest(official), "reference_sha256": digest(reference),
                                 "same_text": official.read_text() == reference.read_text()}
    groups = {}
    for row in rows:
        g = groups.setdefault(row["group"], {"evaluated": 0, "passed": 0})
        g["evaluated"] += 1
        g["passed"] += int(row["status"].endswith("PASS"))
    mismatch = [r for r in verified if not r["matches"]]
    inconsistent = [r for r in rows if not all(r["verification"].values())]
    changed = [p for p, sha in observed.items() if sha is not None and digest(p) != sha]
    result = {"status": "AUDIT_VERIFIED_SCOPE_NOT_ACCEPTED" if not mismatch and not inconsistent and not changed and all(surface[s]["matches_saved"] for s in ["initial", "final"]) else "AUDIT_REQUIRES_REVIEW",
              "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "groups": groups, "records": rows,
              "hash_checks": verified, "hash_mismatches": mismatch, "inconsistent_records": inconsistent,
              "files_changed_during_audit": changed, "archive_warnings": warnings,
              "surface_failure_recomputed": surface, "upstream": source,
              "scope": "completed upper-only contour experiments; no new fit; no claim of bimaxillary or SemanticXY completion"}
    (out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out / "source_sha256.json").write_text(json.dumps(observed, indent=2), encoding="utf-8")
    (out / "audit_script.py").write_text(Path(__file__).read_text(), encoding="utf-8")
    print(json.dumps({"output": str(out), "status": result["status"], "groups": groups, "hash_checks": len(verified),
                      "hash_mismatches": len(mismatch), "inconsistent_records": len(inconsistent), "archive_warnings": len(warnings), "changed": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
