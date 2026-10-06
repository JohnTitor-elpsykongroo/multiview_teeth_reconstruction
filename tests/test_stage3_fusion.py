"""Independent analytical fixtures; no Blender, truth, trained weights or old runs."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observation_fusion.contract import ContractError, FDI, TEETH, load_observations, resource, read_json
from observation_fusion.geometry import project, ray, intersect_rays, estimate_rigid
from observation_fusion.pipeline import prepare, adapt, presence_policy, write_json, _dmm


POLICY = {"kind": "assume_all_present", "origin": "unit test assumption; not measured existence"}


def fixture(root, count=3, empty=False):
    root.mkdir()
    rows = []
    for index in range(count):
        labels = np.zeros((4, 6), np.uint8)
        labels[-1, -1] = 255
        if not empty:
            labels[0, 0:2], labels[2, 1:3], labels[1, 4] = 11, 31, 18
        valid = (labels != 255).astype(np.uint8)
        confidence = valid.astype(np.float32) * .8
        Image.fromarray(labels).save(root / f"labels{index}.png")
        Image.fromarray(valid).save(root / f"valid{index}.png")
        np.save(root / f"confidence{index}.npy", confidence)
        instances = []
        for fdi in sorted(set(labels.ravel()) - {0, 255}):
            y, x = np.where(labels == fdi)
            instances.append(dict(fdi=int(fdi), jaw="upper" if fdi < 30 else "lower", visible_pixels=len(x),
                                  bbox_xyxy_exclusive=[int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
                                  confidence=float(confidence[y, x].mean()), fdi_uncertain=False,
                                  fdi_probabilities={str(k): float(k == fdi) for k in FDI}))
        write_json(root / f"instances{index}.json", instances)
        t = np.eye(4)
        t[0, 3], t[2, 3] = index * 2, 100
        k = [[50, 0, 3], [0, 50, 2], [0, 0, 1]]
        rows.append(dict(view_id=f"camera{index}", width=6, height=4, source_width=6, source_height=4,
                         labels=f"labels{index}.png", valid_mask=f"valid{index}.png", confidence=f"confidence{index}.npy",
                         instances=f"instances{index}.json", K=k, K_source=k, A_fit_from_source_pixels=np.eye(3).tolist(),
                         T_camera_from_world=t.tolist(), distortion_model="none", distortion_coefficients=[],
                         quality=dict(valid_pixels=23, ignored_pixels=1, observed_fdi=[x["fdi"] for x in instances])))
    data = dict(schema_id="dental_tooth_observations", schema_version="1.0.0", scene_id="analytical_fixture",
                length_unit="mm", pixel_convention="edge_origin_centers_at_half", world_axes="X_patient_left_Y_posterior_Z_superior",
                observation_kind="fixture", status="OBSERVATIONS_READY_REVIEW_REQUIRED",
                input_manifest="Z:/does-not-exist/truth/manifest.json", input_manifest_sha256="0" * 64,
                model=dict(backend="fixture", version="unit_test", checkpoint_sha256="0" * 64, class_ids=[0, *FDI]),
                config=dict(confidence_threshold=.6, margin_threshold=.1), views=rows)
    write_json(root / "manifest.json", data)
    return root / "manifest.json"


class GeometryTests(unittest.TestCase):
    def test_half_pixel_roundtrip_rotated_camera(self):
        k = np.array([[120., 0, 3], [0, 100, 2], [0, 0, 1]])
        t = np.array([[0., -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 20], [0, 0, 0, 1]])
        uv = [.5, 2.5]
        origin, direction = ray(uv, k, t)
        actual, depth = project(np.array([origin + direction * 50]), k, t)
        np.testing.assert_allclose(actual[0], uv, atol=1e-12)
        self.assertGreater(depth[0], 0)

    def test_analytical_intersection(self):
        origins = np.array([[-20., 0, 0], [20., 0, 0], [0., 20, 0]])
        target = np.array([3., -2, 120])
        result = intersect_rays(origins, target - origins)
        np.testing.assert_allclose(result["point_world_mm"], target, atol=1e-10)
        self.assertLess(result["rms_ray_distance_mm"], 1e-10)

    def test_degenerate_and_behind(self):
        self.assertEqual(intersect_rays([], [])["status"], "INSUFFICIENT_RAYS")
        self.assertEqual(intersect_rays([[0, 0, 0], [1, 0, 0]], [[0, 0, 1]] * 2)["status"], "DEGENERATE_RAYS")
        self.assertEqual(intersect_rays([[0, 0, 0]] * 2, [[0, 0, 1], [1, 0, 1]])["status"], "DEGENERATE_RAYS")
        self.assertEqual(intersect_rays([[-1, 0, 0], [1, 0, 0]], [[-1, 0, 1], [1, 0, 1]])["status"], "BEHIND_CAMERA")

    def test_rigid_recovery_and_collinear_rejection(self):
        a = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
        r = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        t, rms = estimate_rigid(a, a @ r.T + [2, 3, 4])
        np.testing.assert_allclose(t[:3, :3], r, atol=1e-12)
        np.testing.assert_allclose(t[:3, 3], [2, 3, 4], atol=1e-12)
        self.assertLess(rms, 1e-12)
        with self.assertRaises(ValueError):
            estimate_rigid([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])

    def test_crop_resize_no_offset(self):
        k = np.array([[100., 0, 50], [0, 100, 40], [0, 0, 1]])
        a = np.array([[2., 0, -20], [0, 3, -15], [0, 0, 1]])
        p = np.array([[2., 1, 20]])
        uv, _ = project(p, k, np.eye(4))
        resized, _ = project(p, a @ k, np.eye(4))
        np.testing.assert_allclose(resized[0], (a @ [*uv[0], 1])[:2])


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = fixture(self.root / "prediction")

    def load(self):
        return load_observations(self.path, allow_fixture=True)

    def mutate(self, change):
        data = read_json(self.path)
        change(data)
        write_json(self.path, data)

    def test_fixture_opt_in(self):
        with self.assertRaisesRegex(ContractError, "fixture rejected"):
            load_observations(self.path)
        self.assertEqual(len(self.load().views), 3)

    def test_dynamic_views_and_no_provenance_dereference(self):
        self.path = fixture(self.root / "seven", count=7)
        self.assertEqual(len(self.load().views), 7)

    def test_third_molar_adaptation_preserves_original(self):
        original = self.load()
        views, counts = adapt(original.views)
        self.assertEqual(counts["camera0"]["18"], 1)
        self.assertEqual(original.views[0].labels[1, 4], 18)
        self.assertEqual(views[0].labels[1, 4], 255)
        self.assertEqual(views[0].valid[1, 4], 0)
        self.assertEqual(views[0].confidence[1, 4], 0)

    def test_path_rejection(self):
        for reference in ("../truth/a.png", "truth/a.png", "annotations/a.png", "C:/a.png", "/a.png", "labels\\a.png", "labels0.png:stream"):
            with self.subTest(reference=reference), self.assertRaises(ContractError):
                resource(self.path, reference)

    def test_symlink_escape(self):
        outside = self.root / "outside.json"
        outside.write_text("{}")
        link = self.path.parent / "link.json"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("host does not permit symlink creation")
        with self.assertRaises(ContractError):
            resource(self.path, "link.json")

    def test_invalid_labels_valid_confidence(self):
        for kind in ("labels", "valid", "confidence"):
            with self.subTest(kind=kind):
                path = fixture(self.root / kind)
                if kind == "confidence":
                    a = np.load(path.parent / "confidence0.npy")
                    a[-1, -1] = .7
                    np.save(path.parent / "confidence0.npy", a)
                else:
                    target = path.parent / f"{kind}0.png"
                    with Image.open(target) as im:
                        a = np.array(im)
                    a[0, 0] = 99 if kind == "labels" else 0
                    Image.fromarray(a).save(target)
                with self.assertRaises(ContractError):
                    load_observations(path, allow_fixture=True)

    def test_duplicate_camera_and_id(self):
        original = read_json(self.path)
        for same_id in (True, False):
            data = deepcopy(original)
            data["views"][1] = deepcopy(data["views"][0])
            if not same_id:
                data["views"][1]["view_id"] = "other"
            write_json(self.path, data)
            with self.assertRaisesRegex(ContractError, "duplicate"):
                self.load()

    def test_camera_and_intrinsic_rejection(self):
        original = read_json(self.path)
        for field, value in (("K", [[50, 0, 3.5], [0, 50, 2], [0, 0, 1]]),
                             ("T_camera_from_world", np.diag([-1, 1, 1, 1]).tolist()),
                             ("distortion_coefficients", [.1])):
            data = deepcopy(original)
            data["views"][0][field] = value
            write_json(self.path, data)
            with self.assertRaises(ValueError):
                self.load()

    def test_instance_and_quality_consistency(self):
        original = read_json(self.path.parent / "instances0.json")
        for field, value in (("visible_pixels", 99), ("bbox_xyxy_exclusive", [0, 0, 1, 1]),
                             ("confidence", .1), ("fdi_uncertain", True), ("jaw", "lower"),
                             ("fdi_probabilities", {"11": 1.0})):
            rows = deepcopy(original)
            rows[0][field] = value
            write_json(self.path.parent / "instances0.json", rows)
            with self.subTest(field=field), self.assertRaises(ContractError):
                self.load()
        write_json(self.path.parent / "instances0.json", original)
        self.mutate(lambda data: data["views"][0]["quality"].update(valid_pixels=24))
        with self.assertRaises(ContractError):
            self.load()

    def test_external_presence_conflict_and_unknown(self):
        views, _ = adapt(self.load().views)
        presence = presence_policy(POLICY, views)
        self.assertTrue(presence["upper"]["17"])
        presence["upper"]["11"] = False
        with self.assertRaisesRegex(ContractError, "conflicts"):
            presence_policy(dict(kind="explicit_external", origin="test", presence=presence), views)
        with self.assertRaises(ContractError):
            presence_policy({}, views)

    def test_no_model_pending_no_fake_fit_input(self):
        output = self.root / "out"
        report = prepare(self.path, output, POLICY, allow_fixture=True)
        self.assertFalse(report["fit_ready"])
        self.assertIn("MODEL_BUNDLES", report["pending"])
        self.assertIn(17, report["diagnostics"]["arches"]["upper"]["unobserved_assumed_present"])
        self.assertEqual(report["diagnostics"]["arches"]["upper"]["distinct_visible_cameras"], 3)
        self.assertFalse((output / "fit_input" / "manifest.json").exists())
        with self.assertRaisesRegex(ContractError, "fresh"):
            prepare(self.path, output, POLICY, allow_fixture=True)

    def test_empty_views_preserved(self):
        path = fixture(self.root / "empty", count=4, empty=True)
        report = prepare(path, self.root / "out", POLICY, allow_fixture=True)
        self.assertEqual(len(report["view_ids"]), 4)
        self.assertIn("INSUFFICIENT_OBSERVATIONS", report["pending"])

    def test_all_ignore(self):
        for index in range(3):
            Image.fromarray(np.full((4, 6), 255, np.uint8)).save(self.path.parent / f"labels{index}.png")
            Image.fromarray(np.zeros((4, 6), np.uint8)).save(self.path.parent / f"valid{index}.png")
            np.save(self.path.parent / f"confidence{index}.npy", np.zeros((4, 6), np.float32))
            write_json(self.path.parent / f"instances{index}.json", [])
        self.mutate(lambda data: [row.update(quality=dict(valid_pixels=0, ignored_pixels=24, observed_fdi=[])) for row in data["views"]])
        report = prepare(self.path, self.root / "out", POLICY, allow_fixture=True)
        self.assertEqual(report["diagnostics"]["valid_pixels"], 0)
        self.assertIn("INSUFFICIENT_OBSERVATIONS", report["pending"])

    def test_numbering_risk_report(self):
        rows = read_json(self.path.parent / "instances0.json")
        rows[0]["fdi_probabilities"].update({"11": .51, "12": .49})
        write_json(self.path.parent / "instances0.json", rows)
        report = prepare(self.path, self.root / "out", POLICY, allow_fixture=True)
        self.assertTrue(any(x["kind"] == "LOW_NUMBERING_MARGIN" for x in report["diagnostics"]["numbering_risks"]))

    def test_json_duplicate_keys(self):
        self.path.write_text('{"a": 1, "a": 2}')
        with self.assertRaisesRegex(ContractError, "duplicate JSON"):
            self.load()


class AssemblyTests(unittest.TestCase):
    def test_real_dmm_loader_candidate_and_fingerprint(self):
        _dmm()
        import torch
        from networks.dmm_net import DMM
        from dmm.provenance import source_fingerprint
        from dmm.validation import stamped, reference
        from dmm.scene import load_scene
        before = source_fingerprint()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {}
            for arch, teeth in TEETH.items():
                folder = root / arch
                folder.mkdir()
                gum = dict(latent_dim=2, model_type="sine", hyper_hidden_layers=0, hyper_hidden_features=4,
                           mlp_input_dim=3, mlp_output_dim=5, mlp_num_hidden_layers=0, mlp_hidden_features=4)
                specs = dict(labels=[0, *teeth], NetworkArchRef="mlp", GumDeformNetworkSpecs=gum,
                             TeethDeformNetworkSpecs=dict(gum, mlp_output_dim=8),
                             NetworkSpecsRef=dict(init_dims=[4], output_dims=1, activation="sine"))
                model = DMM(specs, arch=arch)
                torch.save(model.state_dict(), folder / "weights.pth")
                stats = {}
                for label in model.labels:
                    stats.update({f"mu_{label}": np.zeros(2), f"cov_{label}": np.eye(2),
                                  f"L_{label}": np.eye(2), f"count_{label}": np.array(3, dtype=np.int64)})
                np.savez(folder / "latent_statistics.npz", **stats)
                path = folder / "model.json"
                weights = reference(folder / "weights.pth", path)
                metadata = stamped("arch_model_bundle", arch=arch, implementation_version="independent_arch_v1",
                                   model_unit_mm=50, gum_policy="fixed_training_mean", model_id="untrained-fixture",
                                   canonical_reference_id="fixture", upstream_commit="fixture", local_source_sha256=before,
                                   training_manifest_sha256="fixture", training_split_sha256="fixture",
                                   sampling_domain_model=[[-1] * 3, [1] * 3], specs=specs,
                                   component_dimensions={str(k): v for k, v in model.dimensions.items()}, weights=weights,
                                   latent_statistics=reference(folder / "latent_statistics.npz", path),
                                   statistics_binding=dict(weights_sha256=weights["sha256"], training_manifest_sha256="fixture",
                                                           training_split_sha256="fixture", original_only=True))
                write_json(path, metadata)
                paths[arch] = str(path)
            observation = fixture(root / "prediction")
            config = dict(model_bundle=paths, initialization=dict(method="external_pose", origin="unit test only",
                           T_world_from_arch={a: np.eye(4).tolist() for a in TEETH}))
            report = prepare(observation, root / "output", POLICY, config=config, allow_fixture=True)
            candidate = root / "output" / report["candidate_manifest"]
            self.assertTrue(candidate.exists())
            self.assertFalse((candidate.parent / "manifest.json").exists())
            self.assertFalse(report["fit_ready"])
            with self.assertRaises(ValueError):
                load_scene(candidate)
            cameras = read_json(candidate.parent / "cameras.json")
            self.assertEqual(cameras["views"][0]["K"], read_json(observation)["views"][0]["K"])
            initial = read_json(root / "output" / "initialization" / "parameters.json")
            self.assertEqual(len(initial["arches"]["lower"]["q"]), 14)
        self.assertEqual(source_fingerprint(), before)


if __name__ == "__main__":
    unittest.main()
