"""Evaluate a completed pose fit against synthetic truth after fitting has ended.

This script is deliberately separate from run_pose_only.py. Its truth input is
never passed to the optimizer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path, help="completed pose_only run directory")
    args = parser.parse_args()
    run_root = args.run_root.resolve(strict=True)
    output_path = run_root / "evaluation.json"
    if output_path.exists():
        raise FileExistsError(f"evaluation already exists: {output_path}")

    report_path = run_root / "fit_report.json"
    pose_path = run_root / "pose.json"
    provenance_path = run_root / "provenance.json"
    config_path = run_root / "resolved_config.json"
    report = read_json(report_path)
    pose = read_json(pose_path)
    provenance = read_json(provenance_path)
    config = read_json(config_path)
    if report["status"] != "POSE_ONLY_FIT_COMPLETED_EVAL_REQUIRED":
        raise ValueError("pose fitting has not completed")
    if report["run_id"] != run_root.name or provenance["run_id"] != run_root.name:
        raise ValueError("pose run IDs disagree")
    if provenance["objective_inputs_only"] != ["fit_input", "fixed_meshes"]:
        raise ValueError("unexpected fitter inputs")
    if report["final"]["pose"] != pose["rotation_vector_radians"] + pose["translation_dmm"]:
        raise ValueError("final report and pose file disagree")

    fit_root = Path(config["fit_input"]).resolve(strict=True)
    source_root = fit_root.parent
    if source_root.name != provenance["source_run_id"]:
        raise ValueError("source run ID does not match fit input")
    manifest_path = fit_root / "manifest.json"
    if sha256(manifest_path) != provenance["fit_manifest_sha256"]:
        raise ValueError("fit manifest changed since fitting")
    manifest = read_json(manifest_path)
    if manifest["source_run_id"] != provenance["source_run_id"]:
        raise ValueError("fit manifest has a different source run")
    camera_path = fit_root / manifest["camera_file"]
    if sha256(camera_path) != provenance["camera_sha256"]:
        raise ValueError("camera input changed since fitting")
    for item in manifest["masks"]:
        if sha256(fit_root / item["path"]) != provenance["mask_sha256"][item["camera"]]:
            raise ValueError(f"mask changed since fitting: {item['camera']}")
    shape_root = Path(config["fixed_meshes"]).resolve(strict=True)
    for label, expected_hash in provenance["fixed_mesh_sha256"].items():
        if sha256(shape_root / f"tooth{label}.ply") != expected_hash:
            raise ValueError(f"fixed mesh changed since fitting: {label}")

    # Truth is opened only here, after the completed result and all fit inputs
    # have been checked. The synthetic generator saved identity arch-to-world.
    truth_path = source_root / "truth" / "manifest.json"
    truth = read_json(truth_path)
    if truth["status"] != "SYNTHETIC_GENERATOR_TRUTH_ISOLATED":
        raise ValueError("unexpected synthetic truth status")
    if truth["ground_truth_arch_pose"] != "identity in the saved DMM world frame":
        raise ValueError("unsupported ground truth pose representation")
    initial = report["initial"]
    final = report["final"]
    rotation_error = math.degrees(norm(pose["rotation_vector_radians"]))
    translation_error = norm(pose["translation_dmm"])
    initial_rotation_error = math.degrees(norm(initial["pose"][:3]))
    initial_translation_error = norm(initial["pose"][3:])
    iou_gain = final["mean_tooth_iou"] - initial["mean_tooth_iou"]
    limits = config["acceptance"]
    checks = {
        "rotation_error": rotation_error <= limits["max_rotation_error_degrees"],
        "translation_error": translation_error <= limits["max_translation_error_dmm"],
        "mean_tooth_iou": final["mean_tooth_iou"] >= limits["min_final_mean_tooth_iou"],
        "mean_tooth_iou_gain": iou_gain >= limits["min_mean_tooth_iou_gain"],
    }
    evaluation = {
        "status": "POSE_ONLY_PASS" if all(checks.values()) else "POSE_ONLY_FAIL",
        "evaluated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": run_root.name,
        "truth_was_read_after_fit": True,
        "truth_manifest_sha256": sha256(truth_path),
        "source_case": truth["source_case"],
        "checkpoint_epoch": truth["checkpoint_epoch"],
        "ground_truth_arch_to_world": "identity",
        "initial_rotation_error_degrees": initial_rotation_error,
        "final_rotation_error_degrees": rotation_error,
        "initial_translation_error_dmm": initial_translation_error,
        "final_translation_error_dmm": translation_error,
        "initial_mean_tooth_iou": initial["mean_tooth_iou"],
        "final_mean_tooth_iou": final["mean_tooth_iou"],
        "mean_tooth_iou_gain": iou_gain,
        "final_micro_tooth_iou": final["micro_tooth_iou"],
        "acceptance_limits": limits,
        "checks": checks,
        "scope": "same-model noiseless synthetic masks, known cameras, fixed upper-tooth shape",
    }
    output_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": evaluation["status"], "evaluation_path": str(output_path),
                      "rotation_error_degrees": rotation_error,
                      "translation_error_dmm": translation_error,
                      "final_mean_tooth_iou": final["mean_tooth_iou"]}))


if __name__ == "__main__":
    main()
