"""Split forward-run synthetic observations from generator truth."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from run_forward_check import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    if report["status"] != "FORWARD_RENDERED_VISUAL_REVIEW_REQUIRED":
        raise ValueError(f"unexpected run status {report['status']}")
    fit_root = root / "fit_input"
    truth_root = root / "truth"
    if fit_root.exists() or truth_root.exists():
        raise FileExistsError("fit_input or truth already exists; refusing to overwrite")
    fit_root.mkdir()
    truth_root.mkdir()
    (fit_root / "masks").mkdir()

    camera_records = []
    mask_records = []
    for camera in report["cameras"]:
        name = camera["name"]
        source = root / "renders" / "training_case" / f"{name}_teeth_labels.png"
        destination = fit_root / "masks" / f"{name}.png"
        with Image.open(source) as image:
            array = np.asarray(image)
            if array.shape != (camera["image_height"], camera["image_width"]):
                raise ValueError(f"mask shape differs from camera: {name}")
            if not set(np.unique(array)).issubset(set(provenance["labels"]) | {0}):
                raise ValueError(f"unexpected semantic IDs in mask: {name}")
        shutil.copyfile(source, destination)
        camera_records.append({
            "name": name, "K": camera["K"],
            "R_world_to_camera": camera["R_world_to_camera"],
            "t_world_to_camera": camera["t_world_to_camera"],
            "image_width": camera["image_width"],
            "image_height": camera["image_height"],
            "coordinate_convention": camera["coordinate_convention"],
        })
        mask_records.append({"camera": name, "path": f"masks/{name}.png",
                             "sha256": sha256(destination)})
    (fit_root / "cameras.json").write_text(json.dumps(camera_records, indent=2), encoding="utf-8")
    fit_manifest = {
        "status": "KNOWN_CAMERA_SYNTHETIC_FIT_INPUT_READY",
        "source_run_id": provenance["run_id"],
        "observation_type": "per-tooth hard FDI labels, no gum or lip occluder",
        "background_id": 0,
        "visible_fdi_ids": [label for label in provenance["labels"] if label != 0],
        "all_pixels_valid": True,
        "camera_file": "cameras.json",
        "camera_sha256": sha256(fit_root / "cameras.json"),
        "masks": mask_records,
        "excludes": ["latent_codes", "generator_meshes", "depth_maps", "ground_truth_pose"],
    }
    (fit_root / "manifest.json").write_text(json.dumps(fit_manifest, indent=2), encoding="utf-8")

    latent_path = Path(config["experiment"]) / "LatentCodes" / f"latent_vecs_{config['checkpoint']}.pth"
    saved = torch.load(latent_path, map_location="cpu", weights_only=True)
    if int(saved["epoch"]) != int(provenance["epoch"]):
        raise ValueError("latent checkpoint changed since forward run")
    if sha256(latent_path) != provenance["input_sha256"][str(latent_path)]:
        raise ValueError("latent checkpoint hash changed since forward run")
    arrays = {}
    for key, matrix in saved["latent_codes"].items():
        label = int(key.split(".", 1)[0])
        arrays[f"label_{label}"] = matrix[int(provenance["case_row"])].cpu().numpy()
    if set(arrays) != {f"label_{label}" for label in provenance["labels"]}:
        raise ValueError("latent components changed since forward run")
    np.savez_compressed(truth_root / "latents.npz", **arrays)
    truth_manifest = {
        "status": "SYNTHETIC_GENERATOR_TRUTH_ISOLATED",
        "source_case": provenance["case"], "source_case_row": provenance["case_row"],
        "checkpoint_epoch": provenance["epoch"],
        "latent_file": "latents.npz", "latent_sha256": sha256(truth_root / "latents.npz"),
        "mesh_directory": "../meshes/training_case",
        "depth_directory": "../renders/training_case",
        "ground_truth_arch_pose": "identity in the saved DMM world frame",
        "warning": "Do not pass this directory to a reconstruction fitter.",
    }
    (truth_root / "manifest.json").write_text(json.dumps(truth_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": fit_manifest["status"], "fit_input": str(fit_root),
                      "truth": str(truth_root), "views": len(camera_records)}))


if __name__ == "__main__":
    main()
