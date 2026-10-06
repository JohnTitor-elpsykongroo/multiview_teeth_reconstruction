import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tooth_observation import CLASS_IDS
from tooth_observation.data import PhotoScene, resolve, sha256, supervision, write_json
from tooth_observation.pipeline import decode, observe
from tooth_observation.training import preflight_dataset


def make_scene(root, name="scene", count=3):
    case = root / "cases" / name
    inp = case / "input"
    ann = case / "annotations"
    inp.mkdir(parents=True)
    ann.mkdir()
    cameras, views, labels = [], [], []
    for i in range(count):
        vid = f"view_{i}"
        camera = {"view_id": vid, "width": 24, "height": 16,
                  "K": [[30., 0, 12.], [0, 30., 8.], [0, 0, 1]],
                  "T_camera_from_world": [[1, 0, 0, i * 3], [0, 1, 0, 0], [0, 0, 1, 100], [0, 0, 0, 1]],
                  "distortion_model": "none", "distortion_coefficients": [],
                  "length_unit": "mm", "pixel_convention": "edge_origin_centers_at_half"}
        cameras.append(camera)
        rgb = np.zeros((16, 24, 3), dtype=np.uint8)
        rgb[:, 4:12, 0] = 255
        rgb[:, 12:20, 1] = 255
        valid = np.ones((16, 24), dtype=np.uint8)
        valid[0] = 0
        fdi = np.zeros((16, 24), dtype=np.uint8)
        fdi[:, 4:12] = 11
        fdi[:, 12:20] = 48
        tissue = np.zeros((16, 24), dtype=np.uint8)
        tissue[:, :2] = 101
        Image.fromarray(rgb).save(inp / f"{vid}.png")
        Image.fromarray(valid).save(inp / f"{vid}_valid.png")
        Image.fromarray(fdi).save(ann / f"{vid}.png")
        Image.fromarray(tissue).save(ann / f"{vid}_tissue.png")
        views.append({"view_id": vid, "rgb": f"{vid}.png", "valid_mask": f"{vid}_valid.png"})
        labels.append({"view_id": vid, "fdi": f"{vid}.png", "tissue": f"{vid}_tissue.png", "instances": "unused.json"})
    header = {"schema_id": "dental_multiview_photo", "schema_version": "1.0.0", "scene_id": name}
    write_json(inp / "cameras.json", {**header, "views": cameras})
    write_json(inp / "manifest.json", {**header, "length_unit": "mm", "camera_file": "cameras.json", "views": views})
    write_json(ann / "manifest.json", {**header, "supervision_only": True, "views": labels})
    return inp / "manifest.json", ann / "manifest.json"


class FixturePredictor:
    kind = "fixture"
    metadata = {"class_ids": list(CLASS_IDS), "backend": "fixture_rgb_rules", "version": "test", "checkpoint_sha256": sha256(__file__)}

    def __call__(self, rgb):
        p = np.full((len(CLASS_IDS), *rgb.shape[:2]), .1 / (len(CLASS_IDS) - 1), dtype=np.float32)
        ids = np.zeros(rgb.shape[:2], dtype=np.int64)
        ids[rgb[:, :, 0] > 0] = CLASS_IDS.index(11)
        ids[rgb[:, :, 1] > 0] = CLASS_IDS.index(48)
        np.put_along_axis(p, ids[None], .9, axis=0)
        return p


class StageTwoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input, self.ann = make_scene(self.root / "dataset")

    def edit_json(self, path, edit):
        doc = json.loads(path.read_text())
        edit(doc)
        path.write_text(json.dumps(doc))

    def test_inference_does_not_open_supervision(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if any(p in {"annotations", "truth"} for p in path.parts):
                raise AssertionError("supervision accessed by inference")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            scene = PhotoScene(self.input)
            self.assertEqual(len(scene.rows), 3)
            for name in scene.rows:
                scene.load(name)
            observe(self.input, self.root / "prediction", FixturePredictor(), allow_fixture=True)

    def test_prediction_contract_and_camera_preservation(self):
        result = observe(self.input, self.root / "prediction", FixturePredictor(), allow_fixture=True)
        self.assertEqual(result["observation_kind"], "fixture")
        self.assertEqual(len(result["views"]), 3)
        scene = PhotoScene(self.input)
        for v in result["views"]:
            base = self.root / "prediction"
            labels = np.array(Image.open(base / v["labels"]))
            valid = np.array(Image.open(base / v["valid_mask"]))
            confidence = np.load(base / v["confidence"], allow_pickle=False)
            instances = json.loads((base / v["instances"]).read_text())
            self.assertEqual(set(np.unique(labels)), {0, 11, 48, 255})
            np.testing.assert_array_equal(valid, labels != 255)
            self.assertTrue(np.all(confidence[labels == 255] == 0))
            self.assertEqual(confidence.dtype, np.float32)
            self.assertEqual([x["fdi"] for x in instances], [11, 48])
            self.assertEqual(instances[0]["bbox_xyxy_exclusive"], [4, 1, 12, 16])
            self.assertEqual(instances[0]["visible_pixels"], 120)
            self.assertEqual(len(instances[0]["fdi_probabilities"]), 32)
            self.assertEqual(v["K"], scene.cameras[v["view_id"]]["K"])
            np.testing.assert_array_equal(v["A_fit_from_source_pixels"], np.eye(3))

    def test_fixture_requires_opt_in(self):
        with self.assertRaisesRegex(ValueError, "fixture"):
            observe(self.input, self.root / "prediction", FixturePredictor())

    def test_stage_three_reads_actual_stage_two_output(self):
        from observation_fusion.contract import load_observations
        from observation_fusion.pipeline import adapt
        out = self.root / "prediction"
        observe(self.input, out, FixturePredictor(), allow_fixture=True)
        observations = load_observations(out / "manifest.json", allow_fixture=True)
        self.assertEqual(len(observations.views), 3)
        adapted, excluded = adapt(observations.views)
        for v in adapted:
            self.assertNotIn(48, np.unique(v.labels))
            np.testing.assert_array_equal(v.valid, v.labels != 255)
            self.assertTrue(np.all(v.confidence[v.valid == 0] == 0))
        self.assertEqual(excluded["view_0"]["48"], 120)

    def test_refuses_overwrite(self):
        output = self.root / "prediction"
        observe(self.input, output, FixturePredictor(), allow_fixture=True)
        with self.assertRaises(FileExistsError):
            observe(self.input, output, FixturePredictor(), allow_fixture=True)

    def test_preserves_source_case(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            observe(self.input, self.input.parent.parent / "predictions", FixturePredictor(), allow_fixture=True)

    def test_low_confidence_and_ambiguous_fdi_are_ignore(self):
        p = np.zeros((33, 1, 3), np.float32)
        p[0, 0, 0] = .9
        p[1, 0, 0] = .1
        p[1, 0, 1] = .51
        p[2, 0, 1] = .49
        p[:, 0, 2] = 1 / 33
        labels, valid, conf, instances = decode(p, np.ones((1, 3), np.uint8), .5, .1)
        np.testing.assert_array_equal(labels, [[0, 255, 255]])
        self.assertEqual(instances, [])
        self.assertEqual(float(conf[0, 1]), 0)

    def test_rgb_occlusion_is_ignore_even_for_confident_background(self):
        p = np.zeros((33, 1, 4), np.float32)
        p[0] = 1
        p[:, 0, 3] = 0
        p[1, 0, 3] = 1
        labels, valid, conf, instances = decode(p, np.ones((1, 4), np.uint8),
                    occlusion_probability=np.array([[0, .3, .9, .9]], np.float32))
        np.testing.assert_array_equal(labels, [[0, 255, 255, 255]])
        np.testing.assert_array_equal(valid, [[1, 0, 0, 0]])
        self.assertEqual(instances, [])
        self.assertTrue(np.all(conf[valid == 0] == 0))

    def test_tissue_supervision_keeps_visible_fdi_background_definition(self):
        scene = PhotoScene(self.input)
        data = supervision(scene, self.ann, include_tissue=True)["view_0"]
        self.assertTrue(np.all(data["fdi"][data["tissue"] == 101] == 0))
        tissue = np.array(Image.open(self.ann.parent / "view_0_tissue.png"))
        tissue[:, 6] = 101
        Image.fromarray(tissue).save(self.ann.parent / "view_0_tissue.png")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            supervision(scene, self.ann, include_tissue=True)

    def test_occlusion_predictor_output_consumed_by_stage_three(self):
        from observation_fusion.contract import load_observations
        class OcclusionFixture(FixturePredictor):
            metadata = {**FixturePredictor.metadata, "occlusion_handling": "rgb_learned_soft_tissue"}
            def __call__(self, rgb):
                occ = np.zeros(rgb.shape[:2], np.float32)
                occ[:, :8] = .9
                return {"fdi_probabilities": super().__call__(rgb), "occlusion_probability": occ}
        output = self.root / "prediction"
        result = observe(self.input, output, OcclusionFixture(), allow_fixture=True)
        loaded = load_observations(output / "manifest.json", allow_fixture=True)
        self.assertEqual(len(loaded.views), 3)
        self.assertTrue(np.all(loaded.views[0].labels[:, :8] == 255))
        self.assertEqual(result["views"][0]["quality"]["predicted_soft_tissue_ignored_pixels"], 120)

    def test_all_ignored_predictions_cannot_hide_tooth_error(self):
        from tooth_observation.evaluation import metrics
        labels = np.full((2, 2), 255, np.uint8)
        gt = np.array([[11, 11], [0, 0]], np.uint8)
        report = metrics(labels, np.zeros_like(labels), gt, np.zeros_like(labels))
        self.assertEqual(report["macro_tooth_iou"], 0)
        self.assertEqual(report["gt_tooth_ignored_fraction"], 1)
        self.assertEqual(report["retained_pixel_fraction"], 0)

    def test_supervised_evaluation_of_export_is_separate(self):
        from tooth_observation.evaluation import evaluate_predictions
        out = self.root / "prediction"
        observe(self.input, out, FixturePredictor(), allow_fixture=True)
        result = evaluate_predictions(out / "manifest.json", self.input, self.ann, self.root / "eval", allow_fixture=True)
        self.assertEqual(result["status"], "FIXTURE_EVALUATION_ONLY")
        self.assertEqual(result["views"][0]["macro_tooth_iou"], 1)

    def test_legacy_checkpoint_requires_exposed_only_opt_in(self):
        import torch
        from tooth_observation.model import SemanticBaseline, ModelPredictor
        path = self.root / "old.pth"
        torch.save({"format": "stage2_semantic_baseline_v1", "class_ids": list(CLASS_IDS),
                    "purpose": "fixture", "training_steps": 0, "state_dict": SemanticBaseline().state_dict()}, path)
        with self.assertRaisesRegex(ValueError, "no occlusion head"):
            ModelPredictor(path)
        predictor = ModelPredictor(path, allow_no_occlusion_head=True)
        self.assertEqual(predictor.metadata["occlusion_handling"], "explicit_exposed_only_legacy")

    def test_invalid_probability_rejected(self):
        p = np.zeros((33, 2, 2), np.float32)
        with self.assertRaises(ValueError):
            decode(p, np.ones((2, 2), np.uint8))
        p[0] = np.nan
        with self.assertRaises(ValueError):
            decode(p, np.ones((2, 2), np.uint8))

    def test_failed_predictor_has_no_completion_manifest(self):
        predictor = FixturePredictor()
        with patch.object(FixturePredictor, "__call__", side_effect=RuntimeError("model failed")):
            with self.assertRaises(RuntimeError):
                observe(self.input, self.root / "failed", predictor, allow_fixture=True)
        self.assertFalse((self.root / "failed" / "manifest.json").exists())

    def test_unsafe_paths_rejected(self):
        for ref in ["../annotations/manifest.json", "D:/file", "/absolute", "rgb\\a.png"]:
            with self.assertRaises(ValueError):
                resolve(self.input.parent, ref)

    def test_duplicate_views_rejected(self):
        self.edit_json(self.input, lambda x: x["views"].append(copy.deepcopy(x["views"][0])))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            PhotoScene(self.input)

    def test_wrong_pixel_convention_rejected(self):
        self.edit_json(self.input.parent / "cameras.json", lambda x: x["views"][0].update(pixel_convention="integer_centers"))
        with self.assertRaisesRegex(ValueError, "pixel"):
            PhotoScene(self.input)

    def test_nonrigid_camera_rejected(self):
        self.edit_json(self.input.parent / "cameras.json", lambda x: x["views"][0]["T_camera_from_world"][0].__setitem__(0, 2))
        with self.assertRaisesRegex(ValueError, "rigid"):
            PhotoScene(self.input)

    def test_supervision_explicit_and_scene_checked(self):
        scene = PhotoScene(self.input)
        self.assertEqual(set(supervision(scene, self.ann)), set(scene.rows))
        self.edit_json(self.ann, lambda x: x.update(scene_id="different"))
        with self.assertRaisesRegex(ValueError, "scene"):
            supervision(scene, self.ann)

    def make_index(self):
        make_scene(self.root / "dataset", "validation", count=1)
        rows = [{"scene_id": name, "patient_id": patient, "split": split, "dmm_split": split,
                 "accepted_for_training": True, "input_manifest": f"cases/{name}/input/manifest.json",
                 "annotations_manifest": f"cases/{name}/annotations/manifest.json"}
                for name, patient, split in [("scene", "patient1", "train"), ("validation", "patient2", "val")]]
        index = self.root / "dataset" / "dataset.json"
        write_json(index, {"schema_id": "dental_multiview_photo", "schema_version": "1.0.0", "scenes": rows})
        return index

    def test_training_patient_isolation(self):
        index = self.make_index()
        self.assertEqual(len(preflight_dataset(index)["val"]), 1)
        self.edit_json(index, lambda x: x["scenes"][1].update(patient_id="patient1"))
        with self.assertRaisesRegex(ValueError, "leakage"):
            preflight_dataset(index)

    def test_training_rejects_unaccepted_data(self):
        index = self.make_index()
        self.edit_json(index, lambda x: x["scenes"][0].update(accepted_for_training=False))
        with self.assertRaisesRegex(ValueError, "accepted"):
            preflight_dataset(index)

    def test_reserved_never_used_for_training(self):
        index = self.make_index()
        self.edit_json(index, lambda x: x["scenes"][0].update(split="development_reserved"))
        with self.assertRaisesRegex(ValueError, "need accepted"):
            preflight_dataset(index)

    def test_occlusion_training_requires_positive_tissue_examples(self):
        index = self.make_index()
        for path in (self.root / "dataset").glob("cases/*/annotations/*_tissue.png"):
            Image.fromarray(np.zeros((16, 24), np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            preflight_dataset(index)

    def test_real_rgb_model_training_checkpoint_inference_path(self):
        import torch
        from tooth_observation.model import ModelPredictor
        from tooth_observation.training import train_baseline
        torch.set_num_threads(2)
        index = self.make_index()
        result = train_baseline(index, self.root / "model", epochs=1, long_edge=32)
        self.assertEqual(result["training_steps"], 3)
        predictor = ModelPredictor(result["checkpoint"], long_edge=32)
        rgb, _, _ = PhotoScene(self.input).load("view_0")
        prediction = predictor(rgb)
        p = prediction["fdi_probabilities"]
        self.assertEqual(prediction["occlusion_probability"].shape, rgb.shape[:2])
        self.assertEqual(p.shape, (33, 16, 24))
        np.testing.assert_allclose(p.sum(0), 1, atol=1e-5)
        # Artificial training is never delivered as a usable model.
        payload = torch.load(result["checkpoint"], weights_only=True)
        payload["purpose"] = "fixture"
        torch.save(payload, self.root / "fixture.pth")
        predictor = ModelPredictor(self.root / "fixture.pth", long_edge=32)
        with self.assertRaisesRegex(ValueError, "fixture"):
            observe(self.input, self.root / "prediction", predictor)


if __name__ == "__main__":
    unittest.main()
