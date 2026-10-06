"""Explicit supervised baseline training with patient-isolated validation."""
from pathlib import Path
import random

import numpy as np

from . import CLASS_IDS
from .data import PhotoScene, read_json, resolve, sha256, supervision, write_json


def preflight_dataset(dataset):
    path = Path(dataset).resolve(strict=True)
    doc = read_json(path)
    if doc.get("schema_id") != "dental_multiview_photo" or doc.get("schema_version") != "1.0.0":
        raise ValueError("unsupported dataset index")
    rows = doc["scenes"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty scene index")
    patients, scene_ids, input_paths = {}, set(), set()
    selected = {"train": [], "val": []}
    tissue_coverage = {split: {"positive": 0, "negative": 0} for split in selected}
    for row in rows:
        patient, split, name = row["patient_id"], row["split"], row["scene_id"]
        if not isinstance(patient, str) or not patient or not isinstance(name, str) or not name:
            raise ValueError("invalid patient/scene id")
        if split not in {"train", "val", "test", "development_reserved"}:
            raise ValueError("unknown split; map source splits explicitly before training")
        if name in scene_ids:
            raise ValueError("duplicate scene_id")
        scene_ids.add(name)
        if patient in patients and patients[patient] != split:
            raise ValueError("patient leakage across splits")
        patients[patient] = split
        if split not in selected:
            continue  # test/reserved annotations and truth are never opened
        if row.get("accepted_for_training") is not True:
            raise ValueError("train/val scene is not accepted_for_training")
        if row.get("dmm_split", split) != split:
            raise ValueError("DMM split mismatch")
        input_path = resolve(path.parent, row["input_manifest"])
        if input_path in input_paths:
            raise ValueError("duplicate input manifest")
        input_paths.add(input_path)
        annotation_path = resolve(path.parent, row["annotations_manifest"])
        scene = PhotoScene(input_path)
        if scene.manifest["scene_id"] != name:
            raise ValueError("index/input scene mismatch")
        annotations = supervision(scene, annotation_path, include_tissue=True)
        tooth_pixels = 0
        for view_id in scene.rows:
            _, valid, _ = scene.load(view_id)
            fdi = annotations[view_id]["fdi"]
            tooth_pixels += int(np.count_nonzero((valid == 1) & (fdi != 0) & (fdi != 255)))
            eligible = (valid == 1) & (fdi != 255)
            soft = np.isin(annotations[view_id]["tissue"], [101, 102, 103])
            tissue_coverage[split]["positive"] += int(np.count_nonzero(eligible & soft))
            tissue_coverage[split]["negative"] += int(np.count_nonzero(eligible & ~soft))
        if not tooth_pixels:
            raise ValueError("scene has no supervised valid tooth pixels")
        selected[split].append((scene, annotation_path, row))
    if not selected["train"] or not selected["val"]:
        raise ValueError("need accepted, patient-disjoint train and val scenes")
    if any(min(counts.values()) == 0 for counts in tissue_coverage.values()):
        raise ValueError("occlusion v2 requires positive and negative tissue supervision in train and val")
    return selected


def _samples(rows):
    for scene, annotation_path, _ in rows:
        annotations = supervision(scene, annotation_path, include_tissue=True)
        for view_id in scene.rows:
            rgb, valid, _ = scene.load(view_id)
            fdi = annotations[view_id]["fdi"].copy()
            fdi[valid == 0] = 255
            lut = np.full(256, 255, dtype=np.int64)
            lut[list(CLASS_IDS)] = np.arange(len(CLASS_IDS))
            occlusion = np.isin(annotations[view_id]["tissue"], [101, 102, 103]).astype(np.int64)
            occlusion[(valid == 0) | (fdi == 255)] = 255
            yield rgb, lut[fdi], occlusion


def train_baseline(dataset, output, epochs=10, device="cpu", long_edge=512, seed=0):
    import torch
    from torch.nn import functional as F
    from .model import OcclusionBaseline, prepare
    if type(epochs) is not int or epochs < 1 or type(long_edge) is not int or long_edge < 16:
        raise ValueError("invalid epochs/long_edge")
    splits = preflight_dataset(dataset)
    output = Path(output).resolve()
    source_root = Path(dataset).resolve().parent
    if output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("training output must be disjoint from dataset")
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = OcclusionBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    write_json(output / "config.json", {"dataset": str(Path(dataset).resolve()), "dataset_sha256": sha256(dataset),
               "epochs": epochs, "long_edge": long_edge, "seed": seed, "device": device,
               "class_ids": list(CLASS_IDS), "learning_rate": .001,
               "selection": "validation_macro_tooth_iou", "rgb_normalization": "sRGB uint8 / 255",
               "occlusion_tissue_ids": [101, 102, 103], "occlusion_loss_weight": 1.0})
    resources = {}
    for rows in splits.values():
        for scene, ann_path, _ in rows:
            for path in [scene.path, scene.camera_path, ann_path]:
                resources[str(path)] = sha256(path)
            ann = read_json(ann_path)
            for row in ann["views"]:
                for key in ("fdi", "tissue"):
                    path = resolve(ann_path.parent, row[key])
                    resources[str(path)] = sha256(path)
            for row in scene.rows.values():
                for key in ("rgb", "valid_mask"):
                    path = resolve(scene.root, row[key])
                    resources[str(path)] = sha256(path)
    write_json(output / "input_sha256.json", resources)
    best, steps, history = -1., 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        rows = list(splits["train"])
        random.shuffle(rows)
        for rgb, target, occlusion in _samples(rows):
            x = prepare(rgb, long_edge, device)
            y = torch.from_numpy(target)[None, None].to(device=device, dtype=torch.float32)
            y = F.interpolate(y, size=x.shape[-2:], mode="nearest-exact")[:, 0].long()
            if not (y != 255).any():
                continue
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            logits = outputs["fdi_logits"]
            # Equal weight for each observed semantic class within the view.
            losses = F.cross_entropy(logits, y, ignore_index=255, reduction="none")
            loss = torch.stack([losses[y == c].mean() for c in y.unique().tolist() if c != 255]).mean()
            occ = torch.from_numpy(occlusion)[None, None].to(device=device, dtype=torch.float32)
            occ = F.interpolate(occ, size=x.shape[-2:], mode="nearest-exact")[:, 0].long()
            occ_loss = F.binary_cross_entropy_with_logits(outputs["occlusion_logits"][:, 0], (occ == 1).float(), reduction="none")
            loss = loss + torch.stack([occ_loss[occ == c].mean() for c in occ.unique().tolist() if c != 255]).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            steps += 1
            train_losses.append(float(loss.detach()))
        if not train_losses:
            raise ValueError("no supervised training steps")
        model.eval()
        confusion = np.zeros((len(CLASS_IDS), len(CLASS_IDS)), dtype=np.int64)
        occ_counts = np.zeros(3, dtype=np.int64)  # intersection, prediction, target
        with torch.inference_mode():
            for rgb, y, occ_target in _samples(splits["val"]):
                outputs = model(prepare(rgb, long_edge, device))
                logits = outputs["fdi_logits"]
                pred = F.interpolate(logits, size=y.shape, mode="bilinear", align_corners=False)[0].argmax(0).cpu().numpy()
                valid = y != 255
                confusion += np.bincount((y[valid] * len(CLASS_IDS) + pred[valid]).ravel(),
                                         minlength=len(CLASS_IDS) ** 2).reshape(confusion.shape)
                occ_pred = F.interpolate(outputs["occlusion_logits"], size=y.shape, mode="bilinear", align_corners=False)[0, 0].sigmoid().cpu().numpy() >= .3
                occ_valid = occ_target != 255
                positive = occ_target == 1
                occ_counts += [np.count_nonzero(occ_pred & positive & occ_valid), np.count_nonzero(occ_pred & occ_valid), np.count_nonzero(positive & occ_valid)]
        intersection = confusion.diagonal()
        union = confusion.sum(0) + confusion.sum(1) - intersection
        ids = np.flatnonzero(union[1:] > 0) + 1
        score = float((intersection[ids] / union[ids]).mean())
        report = {"epoch": epoch, "training_steps": steps, "train_loss": float(np.mean(train_losses)),
                  "validation_macro_tooth_iou": score,
                  "validation_occlusion_iou_at_0_3": float(occ_counts[0] / (occ_counts[1] + occ_counts[2] - occ_counts[0])) if occ_counts[1] + occ_counts[2] else None,
                  "validation_fdi_iou": {str(CLASS_IDS[i]): float(intersection[i] / union[i]) for i in ids}}
        history.append(report)
        payload = {"format": "stage2_occlusion_baseline_v2", "class_ids": list(CLASS_IDS), "occlusion_tissue_ids": [101, 102, 103],
                   "purpose": "supervised_training", "training_steps": steps, "epoch": epoch,
                   "dataset_sha256": sha256(dataset), "input_sha256": resources,
                   "state_dict": model.state_dict(), "validation": report}
        torch.save(payload, output / f"epoch_{epoch:04d}.pth")
        if score > best:
            best = score
            torch.save(payload, output / "best.pth")
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            import json
            stream.write(json.dumps(report, allow_nan=False) + "\n")
        print(f"stage2 epoch {epoch}/{epochs}: validation tooth IoU={score:.6f}", flush=True)
    result = {"status": "BASELINE_TRAINED_VALIDATION_ONLY", "best_validation_macro_tooth_iou": best,
              "training_steps": steps, "epochs": epochs, "checkpoint": str(output / "best.pth"),
              "history": history, "real_photo_validated": False}
    write_json(output / "result.json", result)
    return result
