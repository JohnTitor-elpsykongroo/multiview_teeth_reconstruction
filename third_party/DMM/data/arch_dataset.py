"""Unpaired, independent single-arch training data with explicit presence."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dmm import MODEL_UNIT_MM, TEETH
from dmm.validation import (array, domain, header, nonempty, presence_map, read_json,
                         require, resolve_ref, rigid, sha256)


@dataclass
class TrainingCase:
    metadata: dict
    presence: dict
    centers: dict
    sample_path: Path

    @property
    def components(self):
        return (0,) + tuple(label for label, present in self.presence.items() if present)


def validate_patient_splits(*datasets):
    """Call with both manifests when available; one arch never requires its partner."""
    splits = {}
    for dataset in datasets:
        for case in dataset.cases:
            row = case.metadata
            patient = row["patient_id"]
            require(patient not in splits or splits[patient] == row["split"],
                    f"patient leaks across splits: {patient}")
            splits[patient] = row["split"]
    return splits


class ArchDataset(Dataset):
    """NPZs are verified sequentially, then loaded on demand (no full-RAM dataset).

    Each present component receives points_per_component samples, with replacement
    only for undersized pools. Missing components receive no samples or loss.
    """

    def __init__(self, manifest, arch, split="train", points_per_component=256,
                 offsurface_points=2048, seed=0, gum_points=None):
        require(arch in TEETH, "arch must be upper or lower")
        require(split in ("train", "val", "test", None), "unsupported split")
        require(type(points_per_component) is int and points_per_component > 0, "positive per-component quota required")
        require(type(offsurface_points) is int and offsurface_points > 0, "positive offsurface quota required")
        self.manifest_path = Path(manifest).resolve()
        self.manifest = read_json(manifest)
        header(self.manifest, "arch_training_manifest")
        require(self.manifest["arch"] == arch, "training manifest arch mismatch")
        self.arch, self.seed, self.epoch = arch, int(seed), 0
        self.points_per_component, self.offsurface_points = points_per_component, offsurface_points
        self.gum_points = points_per_component if gum_points is None else gum_points
        require(type(self.gum_points) is int and self.gum_points > 0, "positive gum quota required")
        rows = self.manifest["cases"]
        require(isinstance(rows, list) and len(rows) > 0, "empty training manifest")
        ids = [nonempty(row["case_id"], "case_id") for row in rows]
        require(ids == sorted(ids) and len(set(ids)) == len(ids), "case_id must be unique and lexicographically ordered")
        require([row["embedding_row"] for row in rows] == list(range(len(rows)))
                and all(type(row["embedding_row"]) is int for row in rows), "embedding_row must be continuous manifest row index")
        self.cases = []
        for index, row in enumerate(rows):
            self.cases.append(self._load(row))
            if len(rows) >= 100 and ((index + 1) % 100 == 0 or index + 1 == len(rows)):
                import sys
                print(f"[{arch}] verified {index + 1}/{len(rows)} records", file=sys.stderr, flush=True)
        self.canonical_reference_id = self.cases[0].metadata["canonical_reference_id"]
        require(all(c.metadata["canonical_reference_id"] == self.canonical_reference_id for c in self.cases), "mixed canonical references")
        first_domain = self.cases[0].metadata["sampling_domain_model"]
        require(all(c.metadata["sampling_domain_model"] == first_domain for c in self.cases), "manifest needs one shared sampling domain")
        self.sampling_domain = domain(first_domain)
        by_id = {c.metadata["case_id"]: c for c in self.cases}
        for case in self.cases:
            row = case.metadata
            parent_id = row["augmentation_parent_id"]
            if parent_id is not None:
                require(parent_id in by_id and parent_id != row["case_id"], "augmentation parent missing")
                parent = by_id[parent_id].metadata
                require(parent["augmentation_parent_id"] is None and parent["patient_id"] == row["patient_id"]
                        and parent["split"] == row["split"], "augmentation lineage/split mismatch")
                remap = row["augmentation_transform"]["label_map"]
                parent_presence = by_id[parent_id].presence
                require(all(case.presence[int(remap[str(k)])] == v for k, v in parent_presence.items()), "mirrored presence mismatch")
        validate_patient_splits(self)
        self.selected = [c for c in self.cases if split is None or c.metadata["split"] == split]
        require(len(self.selected) > 0, f"no cases in split {split}")

    def _load(self, row):
        require(row["arch"] == self.arch and row["split"] in ("train", "val", "test"), "invalid arch or split")
        nonempty(row["patient_id"], "patient_id")
        nonempty(row["canonical_reference_id"], "canonical_reference_id")
        nonempty(row["surface_definition"], "surface_definition")
        nonempty(row["unit_evidence"], "unit_evidence")
        units = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "um": .001}
        require(row["source_unit"] in units and row["source_unit_to_mm"] == units[row["source_unit"]], "unverified/unsupported source units")
        require(row["model_unit_mm"] == MODEL_UNIT_MM, "model_unit_mm must be 50")
        rigid(row["T_arch_mm_from_source_mm"], "T_arch_mm_from_source_mm")
        box = domain(row["sampling_domain_model"])
        presence = presence_map(row["presence"], self.arch)
        for field in ("source_geometry", "source_annotation"):
            resolve_ref(self.manifest_path, row[field])
        augmentation = row["augmentation_transform"]
        if row["augmentation_parent_id"] is None:
            require(augmentation is None, "original case must have null augmentation_transform")
        else:
            require(isinstance(augmentation, dict) and augmentation.get("kind") == "mirror_x", "only declared mirror_x augmentation is supported")
            expected = np.diag([-1., 1., 1., 1.])
            require(np.allclose(array(augmentation["matrix_model"], (4, 4), "mirror"), expected, atol=1e-8, rtol=0), "mirror must reflect model X")
            mapping = {str(k): (k + 10 if (k // 10) % 2 else k - 10) for k in TEETH[self.arch]}
            require(augmentation["label_map"] == mapping, "mirror needs explicit left/right FDI remap")
        samples = resolve_ref(self.manifest_path, row["samples"])
        with np.load(samples, allow_pickle=False) as source:
            require(set(source.files) == {"surface_points", "surface_normals", "surface_labels", "offsurface_points"}, "unexpected NPZ sample schema")
            points = np.asarray(source["surface_points"], dtype=np.float32)
            normals = np.asarray(source["surface_normals"], dtype=np.float32)
            labels = source["surface_labels"].copy()
            off = np.asarray(source["offsurface_points"], dtype=np.float32)
        require(points.ndim == 2 and points.shape[1] == 3 and len(points) > 0, "surface_points must be nonempty Nx3")
        require(normals.shape == points.shape and labels.shape == (len(points),), "surface arrays are not aligned")
        require(np.issubdtype(labels.dtype, np.integer), "surface_labels must be integer FDI")
        require(off.ndim == 2 and off.shape[1] == 3 and len(off) > 0, "offsurface_points must be nonempty Nx3")
        require(np.isfinite(points).all() and np.isfinite(off).all() and np.isfinite(normals).all(), "nonfinite samples")
        require(np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-3, rtol=0), "normals must be unit length")
        for coords in (points, off):
            require(np.all(coords >= box[0]) and np.all(coords <= box[1]), "samples outside declared model domain")
        present = {0} | {k for k, v in presence.items() if v}
        require(set(np.unique(labels)) == present, "sample labels must cover exactly gum and declared present teeth")
        center_path = resolve_ref(self.manifest_path, row["centers"])
        center_json = read_json(center_path)
        require(set(center_json) == {str(k) for k in present if k != 0}, "centers must cover present teeth only")
        centers = {int(k): array(v, (3,), f"center {k}").astype(np.float32) for k, v in center_json.items()}
        require(all(np.all(v >= box[0]) and np.all(v <= box[1]) for v in centers.values()), "centers outside domain")
        return TrainingCase(row, presence, centers, samples)

    def set_epoch(self, epoch):
        require(type(epoch) is int and epoch >= 0, "epoch must be nonnegative integer")
        self.epoch = epoch

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, index):
        case = self.selected[index]
        require(sha256(case.sample_path) == case.metadata["samples"]["sha256"].lower(), "training samples changed after validation")
        with np.load(case.sample_path, allow_pickle=False) as source:
            points = source["surface_points"].astype(np.float32)
            normals = source["surface_normals"].astype(np.float32)
            labels = source["surface_labels"].astype(np.int64)
            offsurface = source["offsurface_points"].astype(np.float32)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        chosen = []
        for label in case.components:
            pool = np.flatnonzero(labels == label)
            quota = self.gum_points if label == 0 else self.points_per_component
            chosen.extend(rng.choice(pool, quota, replace=len(pool) < quota))
        chosen = np.asarray(chosen)
        off_ids = rng.choice(len(offsurface), self.offsurface_points, replace=len(offsurface) < self.offsurface_points)
        return dict(case_id=case.metadata["case_id"], arch=self.arch,
                    embedding_row=case.metadata["embedding_row"], components=case.components,
                    presence=dict(case.presence), centers={k: torch.from_numpy(v.copy()) for k, v in case.centers.items()},
                    points=torch.from_numpy(points[chosen]), normals=torch.from_numpy(normals[chosen]),
                    labels=torch.from_numpy(labels[chosen]), offsurface=torch.from_numpy(offsurface[off_ids]))


def collate_cases(items):
    """Keep variable component counts; trainer averages cases rather than padding losses."""
    return items
