"""Small synthetic interface fixtures, unrelated to accepted historical sanity runs."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "DMM"))

from dmm import TEETH, CHANNEL_FDI
from dmm.bundle import load_bundle
from data.arch_dataset import ArchDataset, validate_patient_splits
from networks.dmm_net import DMM
from utils.math import transform_screw
from dmm.scene import load_scene, compose_depth_layers
from training.arch_training import ArchTrainingSystem, train_arch, export_bundle
from dmm.validation import (ContractError, read_json, reference, resolve_ref,
                                        rigid, sha256, stamped, write_json)


def specs_for(arch):
    gum = dict(latent_dim=2, model_type="sine", hyper_hidden_layers=0, hyper_hidden_features=4,
               mlp_input_dim=3, mlp_output_dim=5, mlp_num_hidden_layers=0, mlp_hidden_features=4)
    teeth = dict(gum, mlp_output_dim=8)
    return dict(labels=[0, *TEETH[arch]], NetworkArchRef="mlp", GumDeformNetworkSpecs=gum,
                TeethDeformNetworkSpecs=teeth, NetworkSpecsRef=dict(init_dims=[4], output_dims=1, activation="sine"))


def training_fixture(root, arch, missing=False):
    directory = root / arch
    directory.mkdir()
    manifest = directory / "manifest.json"
    source = directory / "source.txt"
    source.write_text("synthetic unit-test geometry: coordinates in mm", encoding="utf-8")
    cases = []
    rng = np.random.default_rng(3)
    for index in range(3):
        present = {str(k): not (missing and index == 0 and k == TEETH[arch][-1]) for k in TEETH[arch]}
        labels = np.repeat([0, *(k for k in TEETH[arch] if present[str(k)])], 3)
        points = rng.uniform(-.2, .2, (len(labels), 3)).astype(np.float32)
        normals = np.tile([0, 0, 1.], (len(points), 1)).astype(np.float32)
        sample = directory / f"case{index}.npz"
        np.savez(sample, surface_points=points, surface_normals=normals, surface_labels=labels,
                 offsurface_points=rng.uniform(-.8, .8, (10, 3)).astype(np.float32))
        centers = directory / f"centers{index}.json"
        write_json(centers, {k: [0, 0, 0] for k, v in present.items() if v})
        cases.append(dict(case_id=f"case{index}", patient_id=f"patient{index}", arch=arch, split="train", embedding_row=index,
                          source_geometry=reference(source, manifest), source_annotation=reference(source, manifest),
                          source_unit="mm", source_unit_to_mm=1., unit_evidence="synthetic fixture specified in mm",
                          canonical_reference_id=f"fixture_{arch}", T_arch_mm_from_source_mm=np.eye(4).tolist(),
                          model_unit_mm=50, sampling_domain_model=[[-1, -1, -1], [1, 1, 1]], presence=present,
                          samples=reference(sample, manifest), centers=reference(centers, manifest),
                          surface_definition="unit test points only; not anatomical training data",
                          augmentation_parent_id=None, augmentation_transform=None))
    write_json(manifest, stamped("arch_training_manifest", arch=arch, cases=cases))
    canonical = directory / "canonical.json"
    write_json(canonical, stamped("canonical_reference", arch=arch, reference_id=f"fixture_{arch}", model_unit_mm=50,
                                 axes="LPS", centers_model={str(k): [0, 0, 0] for k in TEETH[arch]},
                                 training_case_ids=[f"case{i}" for i in range(3)], direction_anchor_evidence="synthetic fixture axes"))
    specs = directory / "specs.json"
    write_json(specs, specs_for(arch))
    config = directory / "config.json"
    write_json(config, stamped("arch_training_config", arch=arch, manifest=reference(manifest, config),
                              canonical_reference=reference(canonical, config), specs=reference(specs, config),
                              epochs=1, points_per_component=1, offsurface_points=2, seed=2,
                              learning_rate=.0001, checkpoint_every=1))
    return config, manifest


def scene_fixture(root, bundles):
    root.mkdir()
    fit, init = root / "fit_input", root / "initialization"
    fit.mkdir()
    init.mkdir()
    (root / "model_bundle").mkdir()
    import shutil
    model_paths = {}
    for arch in TEETH:
        shutil.copytree(bundles[arch].parent, root / "model_bundle" / arch)
        model_paths[arch] = root / "model_bundle" / arch / "model.json"
    camera = fit / "cameras.json"
    rows = []
    for i in range(4):
        # The last valid empty view is retained as negative evidence.
        labels = np.zeros((3, 4), dtype=np.uint8)
        if i < 3:
            labels[0, 0], labels[1, 1] = 11, 31
        labels[-1, -1] = 255
        valid = (labels != 255).astype(np.uint8)
        mask_path, valid_path = fit / f"mask{i}.png", fit / f"valid{i}.png"
        Image.fromarray(labels).save(mask_path)
        Image.fromarray(valid).save(valid_path)
        pose = np.eye(4)
        pose[0, 3], pose[2, 3] = i * 2, 100
        k = [[10, 0, 1.5], [0, 10, 1], [0, 0, 1]]
        rows.append(dict(view_id=str(i), width=4, height=3, source_width=4, source_height=3,
                         K=k, K_source=k, A_fit_from_source_pixels=np.eye(3).tolist(),
                         T_camera_from_world=pose.tolist(), distortion_model="none",
                         mask=reference(mask_path, camera), valid_mask=reference(valid_path, camera)))
    write_json(camera, stamped("cameras", views=rows))
    initial = init / "parameters.json"
    states = {}
    for arch in TEETH:
        meta = read_json(model_paths[arch])
        pose = np.eye(4)
        pose[2, 3] = 5 if arch == "upper" else -5
        states[arch] = dict(T_world_from_arch=pose.tolist(), q={str(k): [0., 0.] for k in TEETH[arch]},
                            model_sha256=sha256(model_paths[arch]), statistics_sha256=meta["latent_statistics"]["sha256"])
    write_json(initial, stamped("scene_initialization", method="training_mean", origin="test independent initial pose", arches=states))
    manifest = fit / "manifest.json"
    write_json(manifest, stamped("fit_manifest", scene_id="fixture", length_unit="mm",
                                presence={arch: {str(k): True for k in TEETH[arch]} for arch in TEETH},
                                camera_file=reference(camera, manifest), views=[r["view_id"] for r in rows],
                                model_bundle={arch: reference(p, manifest) for arch, p in model_paths.items()},
                                initialization=reference(initial, manifest)))
    return manifest


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_independent_lower_and_explicit_missing(self):
        _, manifest = training_fixture(self.root, "lower", missing=True)
        dataset = ArchDataset(manifest, "lower", points_per_component=1, offsurface_points=2)
        sample = dataset[0]
        self.assertNotIn(47, sample["components"])
        self.assertFalse(sample["presence"][47])
        self.assertEqual(set(sample["labels"].tolist()), set(sample["components"]))
        self.assertTrue(torch.equal(sample["points"], dataset[0]["points"]))
        model = DMM(specs_for("lower"), arch="lower")
        system = ArchTrainingSystem(model, dataset, {k: [0, 0, 0] for k in TEETH["lower"]})
        before = system.latents["47"].weight.detach().clone()
        optimizer = torch.optim.Adam(system.parameters(), lr=.0001)
        loss, _ = system.loss(sample)
        loss.backward()
        optimizer.step()
        self.assertIsNone(system.latents["47"].weight.grad)
        self.assertTrue(torch.equal(system.latents["47"].weight, before))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(system.latents["31"].weight.grad)
        with self.assertRaises(ContractError):
            ArchDataset(manifest, "upper")

    def test_corrupt_hash_and_patient_split(self):
        _, upper = training_fixture(self.root, "upper")
        _, lower = training_fixture(self.root, "lower")
        upper_data = ArchDataset(upper, "upper")
        data = read_json(lower)
        data["cases"][0]["split"] = "test"
        write_json(lower, data)
        lower_data = ArchDataset(lower, "lower")
        with self.assertRaisesRegex(ContractError, "patient leaks"):
            validate_patient_splits(upper_data, lower_data)
        sample_path = lower.parent / "case1.npz"
        sample_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ContractError, "SHA256"):
            ArchDataset(lower, "lower")

    def test_path_allowlist_and_rigid(self):
        fit = self.root / "fit_input"
        truth = self.root / "truth"
        fit.mkdir()
        truth.mkdir()
        hidden = truth / "secret.json"
        hidden.write_text("{}")
        with self.assertRaisesRegex(ContractError, "allowlist"):
            resolve_ref(fit / "manifest.json", reference(hidden, fit / "manifest.json"), [fit])
        pose = np.eye(4)
        pose[0, 0] = -1
        with self.assertRaises(ContractError):
            rigid(pose, "test")

    def test_zero_rotation_first_and_second_derivative(self):
        points = torch.tensor([[.1, .2, .3]], dtype=torch.float64)
        screw = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda s: transform_screw(points, s), (screw,)))
        self.assertTrue(torch.autograd.gradgradcheck(lambda s: transform_screw(points, s), (screw,)))
        self.assertTrue(torch.equal(transform_screw(points, screw), points))

    def test_query_matches_native_inference_and_latent_finite_difference(self):
        model = DMM(specs_for("lower"), arch="lower").double()
        points = torch.tensor([[.02, -.01, .03], [.05, .06, -.03]], dtype=torch.float64)
        gum = torch.zeros(2, dtype=torch.float64)
        tooth = torch.tensor([.001, -.002], dtype=torch.float64, requires_grad=True)
        codes = {0: gum, 31: tooth}
        native = model.inference({k: v[None] for k, v in codes.items()}, points[None]).flatten()
        result = model.query(points, codes)
        torch.testing.assert_close(result["sdf"], native)
        self.assertTrue(torch.autograd.gradcheck(lambda z: model.query(points, {0: gum, 31: z})["sdf"], (tooth,)))

    def test_presence_never_inferred_from_incomplete_sample_pool(self):
        _, manifest = training_fixture(self.root, "upper")
        data = read_json(manifest)
        source = manifest.parent / "case0.npz"
        with np.load(source) as payload:
            arrays = {k: payload[k] for k in payload.files}
        keep = arrays["surface_labels"] != 17
        for name in ("surface_points", "surface_normals", "surface_labels"):
            arrays[name] = arrays[name][keep]
        np.savez(source, **arrays)
        data["cases"][0]["samples"] = reference(source, manifest)
        write_json(manifest, data)
        with self.assertRaisesRegex(ContractError, "declared present"):
            ArchDataset(manifest, "upper")


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.bundles, cls.configs = {}, {}
        for arch in TEETH:
            config, _ = training_fixture(cls.root, arch)
            cls.configs[arch] = config
            result = train_arch(config, cls.root / f"run_{arch}")
            assert result["epochs"] == 1
            export_bundle(cls.root / f"run_{arch}" / "final.pth", cls.root / f"bundle_{arch}", f"fixture_{arch}")
            cls.bundles[arch] = cls.root / f"bundle_{arch}" / "model.json"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def scene(self, suffix):
        return scene_fixture(self.root / suffix, self.bundles)

    def test_independent_train_export_and_scene_gradients(self):
        for arch in TEETH:
            checkpoint = torch.load(self.root / f"run_{arch}" / "final.pth", weights_only=True)
            self.assertEqual(checkpoint["arch"], arch)
            self.assertTrue((self.root / f"run_{arch}" / "best_train.pth").exists())
            bundle = load_bundle(self.bundles[arch], arch)
            self.assertEqual(bundle.model.labels, (0, *TEETH[arch]))
            self.assertFalse(any(p.requires_grad for p in bundle.model.parameters()))
        scene = load_scene(self.scene("gradients"))
        self.assertEqual(len(scene.observations), 4)
        self.assertEqual(scene.summary()["trainable_parameters"], 12 + 28 * 2)
        self.assertIn(12, scene.summary()["arches"]["upper"]["unobserved"])
        self.assertAlmostEqual(float(scene.T_upper_from_lower[2, 3].detach()), -10)
        result = scene.query_world(torch.tensor([[0., 0., 0.], [1., 2., 1.]]))
        for arch in TEETH:
            self.assertEqual(result[arch]["semantics"].shape, (2, 29))
            torch.testing.assert_close(result[arch]["semantics"].sum(-1), torch.ones(2))
        result["upper"]["sdf_mm"].sum().backward()
        self.assertIsNotNone(scene.arches["upper"].q["11"].grad)
        self.assertTrue(torch.isfinite(scene.arches["upper"].pose_delta.grad).all())
        self.assertIsNone(scene.arches["lower"].q["31"].grad)
        self.assertTrue(all(p.grad is None for s in scene.arches.values() for p in s.decoder.parameters()))

    def test_resume_matches_uninterrupted_and_refuses_overwrite(self):
        config = self.configs["upper"]
        data = read_json(config)
        data["epochs"] = 2
        write_json(config, data)
        train_arch(config, self.root / "resumed", resume=self.root / "run_upper" / "final.pth")
        train_arch(config, self.root / "uninterrupted")
        resumed = torch.load(self.root / "resumed" / "final.pth", weights_only=True)
        direct = torch.load(self.root / "uninterrupted" / "final.pth", weights_only=True)
        for key in resumed["system_state"]:
            torch.testing.assert_close(resumed["system_state"][key], direct["system_state"][key], rtol=0, atol=0)
        with self.assertRaises(ContractError):
            train_arch(config, self.root / "uninterrupted")

    def test_prepare_edge_candidate_creates_fresh_verified_package(self):
        from dmm.handoff import prepare_candidate
        manifest = self.scene("edge_handoff")
        cameras = manifest.parent / "cameras.json"
        data = read_json(cameras)
        for row in data["views"]: row["pixel_convention"] = "edge_origin_centers_at_half"
        write_json(cameras, data)
        candidate = manifest.parent / "candidate_manifest.json"
        meta = read_json(manifest); meta["camera_file"] = reference(cameras, candidate)
        meta.update(status="PENDING_RENDERER_PIXEL_CONVENTION", fit_ready=False)
        write_json(candidate, meta)
        result = prepare_candidate(candidate, self.root / "edge_prepared")
        loaded = load_scene(result["manifest"])
        self.assertTrue(loaded.manifest["fit_ready"])
        self.assertTrue(all(x.metadata["pixel_convention"] == "edge_origin_centers_at_half" for x in loaded.observations))
        self.assertFalse(read_json(candidate)["fit_ready"])
        self.assertFalse((self.root / "edge_prepared/truth").exists())

    def test_invalid_camera_duplicate_and_valid_mask(self):
        for problem in ("intrinsics", "duplicate", "valid", "coverage", "absent"):
            path = self.scene(problem)
            manifest = read_json(path)
            camera_path = path.parent / "cameras.json"
            cameras = read_json(camera_path)
            if problem == "intrinsics":
                cameras["views"][0]["K"][0][0] *= 2
            elif problem == "duplicate":
                cameras["views"][1]["T_camera_from_world"] = cameras["views"][0]["T_camera_from_world"]
            elif problem == "valid":
                p = path.parent / "valid0.png"
                Image.fromarray(np.ones((3, 4), np.uint8)).save(p)
                cameras["views"][0]["valid_mask"] = reference(p, camera_path)
            elif problem == "coverage":
                cameras["views"] = cameras["views"][:1]
                manifest["views"] = ["0"]
            else:
                manifest["presence"]["upper"]["11"] = False
            write_json(camera_path, cameras)
            manifest["camera_file"] = reference(camera_path, path)
            write_json(path, manifest)
            with self.assertRaises(ContractError, msg=problem):
                load_scene(path)

    def test_truth_is_not_read(self):
        path = self.scene("truth_isolation")
        truth = path.parent.parent / "truth"
        truth.mkdir()
        (truth / "manifest.json").write_text("not valid json")
        scene = load_scene(path)
        self.assertEqual(scene.manifest["scene_id"], "fixture")
        data = read_json(path)
        data["initialization"] = reference(truth / "manifest.json", path)
        write_json(path, data)
        with self.assertRaisesRegex(ContractError, "allowlist"):
            load_scene(path)

    def test_mesh_assembly_two_blended_surfaces(self):
        scene = load_scene(self.scene("mesh"))
        # Analytic test fields validate extraction/units/assembly, not trained geometry quality.
        def sphere(state, points):
            sdf = points.norm(dim=-1) - .2
            semantic = torch.zeros(len(points), 29)
            label = 11 if state.decoder.arch == "upper" else 31
            semantic[:, 1 + CHANNEL_FDI.index(label)] = .8
            semantic[:, 0] = .2
            return {"sdf": sdf, "semantics": semantic}
        with patch("dmm.scene.ArchState.query_model", sphere):
            mesh = scene.extract_mesh(16, chunk_size=1000)
        self.assertEqual(set(mesh.face_arch), {0, 1})
        self.assertTrue(np.allclose(mesh.semantics.sum(-1), 1))
        for arch_id, center_z in ((0, 5), (1, -5)):
            vertices = mesh.vertices_world_mm[np.unique(mesh.faces[mesh.face_arch == arch_id])]
            self.assertAlmostEqual(float(vertices.mean(0)[2]), center_z, places=4)
            self.assertLess(abs(np.linalg.norm(vertices - [0, 0, center_z], axis=1).mean() - 10), 1)

    def test_gum_occludes_and_background_no_hit(self):
        depth = torch.tensor([[[5., float("inf"), 8.]], [[10., float("inf"), 3.]]])
        probabilities = torch.zeros(2, 1, 3, 29)
        probabilities[0, ..., 0] = 1  # upper gum is closest in pixel 0
        probabilities[1, ..., 15] = 1  # lower tooth
        final_depth, final_prob = compose_depth_layers(depth, probabilities)
        self.assertEqual(final_prob.argmax(-1).tolist(), [[0, 0, 15]])
        self.assertEqual(float(final_depth[0, 0]), 5)

    def test_export_rejects_insufficient_original_support(self):
        source = self.root / "run_lower" / "final.pth"
        saved = torch.load(source, weights_only=True)
        saved["case_rows"][0]["presence"]["47"] = False
        path = self.root / "insufficient.pth"
        torch.save(saved, path)
        with self.assertRaisesRegex(ContractError, "component 47"):
            export_bundle(path, self.root / "bad_export", "insufficient")
        self.assertFalse((self.root / "bad_export").exists())

    def test_cli_train_entry_runs_without_legacy_optional_dependencies(self):
        import subprocess
        runtime = sys.executable
        entry = Path(__file__).resolve().parents[1] / "third_party" / "DMM" / "train_dmm.py"
        run = subprocess.run([runtime, "-B", str(entry), "--config", str(self.configs["lower"]),
                              "--output", str(self.root / "cli_lower")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('"arch": "lower"', run.stdout)
        self.assertTrue((self.root / "cli_lower" / "final.pth").exists())


if __name__ == "__main__":
    unittest.main()
