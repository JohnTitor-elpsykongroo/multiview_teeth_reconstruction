"""Explicit dev-only numerical smoke on Blender data. Never exports a trained model."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F
from tooth_observation import CLASS_IDS
from tooth_observation.data import PhotoScene, read_json, resolve, supervision, sha256, write_json
from tooth_observation.model import OcclusionBaseline, ModelPredictor, prepare
from tooth_observation.pipeline import observe
from tooth_observation.evaluation import evaluate_predictions
from observation_fusion.contract import load_observations


def smoke(datasets, output):
    output = Path(output).resolve()
    for dataset in datasets:
        root = Path(dataset).resolve().parent
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("output must be disjoint")
    # Explicitly restricted to reserved, nonaccepted pilot data. No training gate override.
    scenes = []
    for dataset in datasets:
        path = Path(dataset).resolve(strict=True)
        for row in read_json(path)["scenes"]:
            if row["split"] != "development_reserved" or row["accepted_for_training"] is not False:
                raise ValueError("smoke only consumes development_reserved, nonaccepted scenes")
            scenes.append((PhotoScene(resolve(path.parent, row["input_manifest"])), resolve(path.parent, row["annotations_manifest"])))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(51005)
    model = OcclusionBaseline()
    reports = []
    for i, (scene, ann_path) in enumerate(scenes):
        annotations = supervision(scene, ann_path, include_tissue=True)
        # One backward pass per scene, no optimizer and no parameter update.
        name = next(iter(scene.rows))
        rgb, valid, _ = scene.load(name)
        x = prepare(rgb, 128, "cpu")
        labels = annotations[name]["fdi"].copy()
        labels[valid == 0] = 255
        lut = np.full(256, 255, dtype=np.int64)
        lut[list(CLASS_IDS)] = np.arange(len(CLASS_IDS))
        y = torch.from_numpy(lut[labels])[None, None].float()
        y = F.interpolate(y, size=x.shape[-2:], mode="nearest-exact")[:, 0].long()
        occ = np.isin(annotations[name]["tissue"], [101, 102, 103]).astype(np.float32)
        occ = F.interpolate(torch.from_numpy(occ)[None, None], size=x.shape[-2:], mode="nearest-exact")[:, 0]
        model.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = F.cross_entropy(prediction["fdi_logits"], y, ignore_index=255)
        occ_loss = F.binary_cross_entropy_with_logits(prediction["occlusion_logits"][:, 0], occ, reduction="none")
        loss = loss + occ_loss[y != 255].mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError("nonfinite real-RGB gradient smoke")
        # Fixture provenance is explicit; zero optimizer steps must remain visible.
        checkpoint = output / f"fixture_{i}.pth"
        torch.save({"format": "stage2_occlusion_baseline_v2", "class_ids": list(CLASS_IDS),
                    "occlusion_tissue_ids": [101, 102, 103], "purpose": "fixture", "training_steps": 0,
                    "state_dict": model.state_dict(), "dataset_sha256": sha256(scene.path)}, checkpoint)
        predictor = ModelPredictor(checkpoint, long_edge=128)
        pred_dir = output / f"prediction_{i}"
        obs = observe(scene.path, pred_dir, predictor, allow_fixture=True)
        loaded = load_observations(pred_dir / "manifest.json", allow_fixture=True)
        evaluation = evaluate_predictions(pred_dir / "manifest.json", scene.path, ann_path, output / f"evaluation_{i}", allow_fixture=True)
        reports.append({"scene_id": scene.manifest["scene_id"], "views": len(loaded.views),
                        "finite_loss": float(loss.detach()), "gradient_l2": float(torch.sqrt(sum((g*g).sum() for g in grads))),
                        "valid_pixels": sum(v["quality"]["valid_pixels"] for v in obs["views"]),
                        "prediction_kind": "fixture", "evaluation_status": evaluation["status"]})
    result = {"status": "REAL_RGB_NUMERICAL_AND_CONTRACT_SMOKE_ONLY", "scenes": reports,
              "optimizer_steps": 0, "trained_model": False, "segmentation_accuracy_validated": False}
    write_json(output / "report.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("datasets", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(smoke(a.datasets, a.output))
