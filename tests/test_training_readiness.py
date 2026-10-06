"""Training/coordinate regression gates; synthetic fixtures, not dental accuracy."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from test_dual_arch import training_fixture
from rendering_fixtures import observation, fixture_scene
from test_rendering import triangle
from data.arch_dataset import ArchDataset
from networks.dmm_net import DMM
from training.arch_training import load_training_config, ArchTrainingSystem, train_arch, export_bundle, evaluate_checkpoint
from training.recipe import resolve_recipe, resolve_validation
from training.validation import validation_samples, validate_arch, mesh_metrics
from dmm.validation import read_json, write_json, reference, ContractError
from dmm.pixels import EDGE
from dmm.rendering import render_mesh, RenderConfig, clip_coordinates
from dmm.semanticxy import SemanticXYMatcher, SemanticXYConfig, normalize_xy
from dmm.provenance import decoder_contract, validate_decoder_binding
from dmm.initialization import cube_rotations, pose_proposals

torch.set_num_threads(1)


class PixelTests(unittest.TestCase):
    def pair(self):
        integer = observation()
        edge = copy.deepcopy(integer)
        edge.K[:2, 2] += .5
        edge.metadata["pixel_convention"] = EDGE
        return integer, edge

    def test_cpu_equivalent_cameras_and_semanticxy(self):
        integer, edge = self.pair()
        mesh = triangle()
        cfg = RenderConfig(backend="torch_reference")
        a, b = render_mesh(mesh, integer, cfg), render_mesh(mesh, edge, cfg)
        torch.testing.assert_close(a.semantics, b.semantics)
        hit = a.coverage > 0
        torch.testing.assert_close(a.xy_pixels[hit]+.5, b.xy_pixels[hit])
        torch.testing.assert_close(normalize_xy(a.xy_pixels[hit], 40, 40), normalize_xy(b.xy_pixels[hit], 40, 40, edge.metadata))
        matcher = SemanticXYMatcher(SemanticXYConfig(samples_per_tooth=4))
        for obs, result in ((integer, a), (edge, b)):
            obs.labels = np.where(result.coverage.detach().numpy()>0, 11, 0).astype(np.uint8)
        first = matcher.match_view(a, integer, mode="evaluate")
        second = matcher.match_view(b, edge, mode="evaluate")
        torch.testing.assert_close(first.loss, second.loss)
        with self.assertRaisesRegex(ContractError, "context"):
            render_mesh(mesh, replace(integer, metadata={"pixel_convention": EDGE}), cfg, a.context)

    def test_explicit_edge_clip_and_crop(self):
        integer, edge = self.pair()
        points = torch.tensor([[0., 0., 80.], [1., -2., 80.]], dtype=torch.float64)
        a = clip_coordinates(points, torch.tensor(integer.K), 40, 40, RenderConfig())
        b = clip_coordinates(points, torch.tensor(edge.K), 40, 40, RenderConfig(), edge.metadata)
        torch.testing.assert_close(a, b)
        # Crop/resize edge coordinates are affine without hidden half-pixel edits.
        affine = np.array([[2.,0.,-6.],[0.,2.,-4.],[0.,0.,1.]])
        uv = points.numpy() @ edge.K.T; uv = uv[:,:2]/uv[:,2:]
        projected = points.numpy() @ (affine@edge.K).T
        np.testing.assert_allclose(projected[:,:2]/projected[:,2:], uv*2-[6,4])
        with self.assertRaisesRegex(ContractError, "pixel_convention"):
            clip_coordinates(points, torch.tensor(edge.K), 40, 40, RenderConfig(), {"pixel_convention":"unknown"})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_equivalent_centers_and_backward(self):
        integer, edge = self.pair()
        mesh = triangle()
        mesh.vertices_world_mm = mesh.vertices_world_mm.detach().float().cuda().requires_grad_()
        mesh.faces = mesh.faces.cuda(); mesh.semantics = mesh.semantics.float().cuda(); mesh.face_arch = mesh.face_arch.cuda()
        a = render_mesh(mesh, integer, RenderConfig(supersample=2))
        b = render_mesh(mesh, edge, RenderConfig(supersample=2))
        torch.testing.assert_close(a.semantics, b.semantics, atol=1e-5, rtol=1e-5)
        hit = (a.coverage>.99)&(b.coverage>.99)
        torch.testing.assert_close(a.xy_pixels[hit]+.5, b.xy_pixels[hit], atol=1e-5, rtol=1e-5)
        b.semantics[...,1:].sum().backward()
        self.assertTrue(torch.isfinite(mesh.vertices_world_mm.grad).all())
        self.assertGreater(float(mesh.vertices_world_mm.grad.norm()), 0)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.config, manifest = training_fixture(self.root, "upper")
        data = read_json(manifest); row = copy.deepcopy(data["cases"][0])
        row.update(case_id="case3", patient_id="patient3", embedding_row=3, split="val")
        data["cases"].append(row); write_json(manifest, data)
        config = read_json(self.config); config["manifest"] = reference(manifest, self.config)
        config["validation"] = dict(enabled=True, fit_steps=2, points_per_component=1,
                                    evaluation_points_per_component=1, offsurface_points=1,
                                    mesh_resolution=8, mesh_samples_per_component=8)
        write_json(self.config, config)
        self.cfg, self.dataset, self.centers, self.specs = load_training_config(self.config)
        self.system = ArchTrainingSystem(DMM(self.specs, arch="upper"), self.dataset, self.centers, self.cfg["recipe"])

    def tearDown(self): self.tmp.cleanup()

    def test_disjoint_deterministic_original_val_only(self):
        case = self.dataset.cases[-1]
        a, b = validation_samples(case, "upper", self.cfg["validation"])
        self.assertFalse(set(a["surface_indices"])&set(b["surface_indices"]))
        self.assertFalse(set(a["offsurface_indices"])&set(b["offsurface_indices"]))
        again, _ = validation_samples(case, "upper", self.cfg["validation"])
        np.testing.assert_array_equal(a["surface_indices"], again["surface_indices"])
        with self.assertRaisesRegex(ContractError,"original val"):
            validation_samples(self.dataset.cases[0], "upper", self.cfg["validation"])

    def test_loss_reduction_changes_scale_explicitly(self):
        sample = self.dataset[0]
        self.system.recipe["reduction"] = "effective"
        _, effective = self.system.loss(sample)
        self.system.recipe["reduction"] = "all_points"
        _, full = self.system.loss(sample)
        self.assertAlmostEqual(effective["raw/surface"] / full["raw/surface"], 17., places=4)
        self.assertIn("raw/11/blend", full)
        self.assertIn("count/11/flip", full)

    def test_val_latents_do_not_update_training_and_failures_not_dropped(self):
        before = {k:v.clone() for k,v in self.system.state_dict().items()}
        flags = [p.requires_grad for p in self.system.parameters()]
        with patch("training.validation.mesh_metrics", return_value={"score_mm": 1.25}):
            report = validate_arch(self.system, self.dataset, self.cfg["validation"])
        self.assertEqual(report["score_mm"], 1.25)
        for k,v in self.system.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertEqual(flags,[p.requires_grad for p in self.system.parameters()])
        with patch("training.validation.mesh_metrics", side_effect=ContractError("controlled missing surface")):
            report = validate_arch(self.system, self.dataset, self.cfg["validation"])
        self.assertIsNone(report["score_mm"])
        self.assertEqual(report["failures"],["case3"])
        self.assertIn("heldout_sdf_abs_mean_model", report["cases"][0])
        self.assertIn("heldout_semantic_error", report["cases"][0])
        self.assertIn("evaluation_index_sha256", report["cases"][0])

    def test_best_val_scheduler_resume_export_compatibility(self):
        cfg = read_json(self.config); cfg["recipe"] = dict(lr_step_epochs=1, lr_gamma=.5); write_json(self.config,cfg)
        with patch("training.validation.mesh_metrics",return_value={"score_mm":1.}):
            train_arch(self.config,self.root/"first")
            self.assertTrue((self.root/"first/best_val.pth").is_file())
            cfg["epochs"]=2;write_json(self.config,cfg)
            train_arch(self.config,self.root/"resumed",resume=self.root/"first/final.pth")
            train_arch(self.config,self.root/"direct")
        resumed=torch.load(self.root/"resumed/final.pth",weights_only=True)
        direct=torch.load(self.root/"direct/final.pth",weights_only=True)
        for k,v in resumed["system_state"].items(): torch.testing.assert_close(v,direct["system_state"][k],rtol=0,atol=0)
        self.assertEqual(resumed["scheduler_state"],direct["scheduler_state"])
        self.assertEqual(resumed["optimizer_state"]["param_groups"][0]["lr"],.000025)
        export_bundle(self.root/"direct/best_val.pth",self.root/"bundle","test")
        metadata=read_json(self.root/"bundle/model.json")
        with patch("dmm.provenance.source_fingerprint",return_value="changed fitting code"):
            validate_decoder_binding(metadata)
        metadata["decoder_contract"]["decoder_sha256"]="incompatible"
        with self.assertRaisesRegex(ContractError,"decoder compatibility"):
            validate_decoder_binding(metadata)
        with patch("training.validation.mesh_metrics",return_value={"score_mm":1.}):
            report=evaluate_checkpoint(self.config,self.root/"direct/final.pth",self.root/"eval")
        self.assertEqual(report["score_mm"],1.)
        cfg["training_case_ids"]=["case0", "case1"];write_json(self.config,cfg)
        with self.assertRaisesRegex(ContractError,"training subset"):
            evaluate_checkpoint(self.config,self.root/"direct/final.pth",self.root/"wrong_eval")

    def test_config_rejects_test_validation_and_unknown_weights(self):
        cfg=read_json(self.config);cfg["validation"]["case_ids"]=["case0"];write_json(self.config,cfg)
        with self.assertRaisesRegex(ContractError,"original val"):
            load_training_config(self.config)
        with self.assertRaisesRegex(ContractError,"loss weights"):
            resolve_recipe(dict(learning_rate=.0001,recipe={"weights":{"made_up":1}}))


class GeometryTests(unittest.TestCase):
    def test_mean_initialization_full_render_and_failure_rollback(self):
        from test_fitting import observed_scene, SURFACE, MATCH
        from dmm.initialization import initialize_mean_scene
        scene, renderer = observed_scene(offset=4.)
        with torch.no_grad():
            for state in scene.arches.values():
                for q in state.q.values(): q.fill_(.2)
        # Keep the numerical integration bounded; the separate solver test checks
        # all 24 rotations. This uses the actual chart, renderer and matcher.
        with patch("dmm.initialization.cube_rotations", return_value=[np.eye(3)]):
            report = initialize_mean_scene(scene, SURFACE, renderer, MATCH)
        self.assertEqual(report["status"],"MEAN_POSE_PROPOSED_NOT_QUALITY_ACCEPTED")
        for arch, state in scene.arches.items():
            self.assertTrue(np.isfinite(report["arches"][arch]["selected_loss"]))
            for q in state.q.values(): self.assertEqual(float(q.detach().abs().sum()),0.)
        saved = {name:value.detach().clone() for name,value in scene.state_dict().items()}
        with patch("dmm.initialization.pose_proposals", side_effect=ContractError("controlled ray failure")):
            with self.assertRaisesRegex(ContractError,"controlled ray failure"):
                initialize_mean_scene(scene,SURFACE,renderer,MATCH)
        for name,value in scene.state_dict().items(): torch.testing.assert_close(value,saved[name],rtol=0,atol=0)

    def test_real_mesh_metric_and_pose_ray_solver(self):
        state=fixture_scene(torch.float32).arches["upper"]
        cfg=resolve_validation({"validation":{"mesh_resolution":18,"mesh_samples_per_component":512}})
        # Six exact octahedron vertices belong to the single present fixture tooth.
        points=np.concatenate((np.eye(3),-np.eye(3))).astype(np.float32)*.3
        sample=dict(points=torch.tensor(points),labels=torch.full((6,),11),components=(0,11))
        metrics=mesh_metrics(state.decoder,state.codes(),sample,[[-.6]*3,[.6]*3],cfg)
        self.assertTrue(np.isfinite(metrics["score_mm"]))
        self.assertEqual(metrics["boundary_edges"],0)
        self.assertEqual(len(cube_rotations()),24)
        centers={11:np.array([-4.,0.,0.]),12:np.array([4.,0.,0.]),13:np.array([0.,4.,0.])}
        observations=[]
        for shift in (-5.,5.):
            obs=observation(offset=shift);obs.K=np.array([[100.,0.,20.],[0.,100.,20.],[0.,0.,1.]])
            obs.labels[:]=0
            for label,center in centers.items():
                camera=center+[shift,0,100];uv=(obs.K@camera)[:2]/camera[2]
                obs.labels[int(uv[1]),int(uv[0])]=label
            observations.append(obs)
        proposals,report=pose_proposals(centers,observations)
        self.assertTrue(report["kabsch_available"])
        np.testing.assert_allclose(proposals[0][:3,3],[0,0,100],atol=1e-5)
        np.testing.assert_allclose(proposals[0][:3,:3],np.eye(3),atol=1e-5)


if __name__=="__main__": unittest.main()
