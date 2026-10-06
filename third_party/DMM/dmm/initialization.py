"""Mean-model pose proposals from mask centroids, ranked by full scene rendering.

Centroids are visibility-dependent heuristics, not anatomical correspondences.
No scene truth, training subject code or target 3D mesh is consumed.
"""
import itertools
import numpy as np
import torch
from dmm import MODEL_UNIT_MM
from dmm.pixels import pixel_offset
from dmm.surface import extract_chart
from dmm.semanticxy import SemanticXYMatcher
from dmm.validation import require, ContractError


def cube_rotations():
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1., 1.), repeat=3):
            rotation = np.eye(3)[list(permutation)] * np.asarray(signs)[:, None]
            if np.linalg.det(rotation) > .5: rotations.append(rotation)
    return sorted(rotations, key=lambda r: float(np.linalg.norm(r-np.eye(3))))


def pose_proposals(centers, observations):
    rows, targets = [], {}
    for obs in observations:
        R, t = obs.T_camera_from_world[:3, :3], obs.T_camera_from_world[:3, 3]
        origin = -R.T @ t
        for label in sorted(set(obs.visible_fdi) & set(centers)):
            yx = np.argwhere((obs.labels == label) & obs.valid).mean(0) + pixel_offset(obs.metadata)
            ray = np.linalg.solve(obs.K, [yx[1], yx[0], 1.])
            direction = R.T @ ray; direction /= np.linalg.norm(direction)
            projection = np.eye(3)-np.outer(direction, direction)
            rows.append((label, projection, origin, R, t))
            targets.setdefault(label, []).append((projection, origin))
    require(bool(rows), "no visible centroid constraints")
    A = np.concatenate([p for _, p, _, _, _ in rows])
    require(np.linalg.matrix_rank(A, tol=1e-8) == 3 and np.linalg.cond(A) < 1e6, "underconstrained initialization rays")
    source, target = [], []
    for label, rays in targets.items():
        if len(rays) < 2: continue
        matrix = np.concatenate([p for p, _ in rays])
        if np.linalg.matrix_rank(matrix, tol=1e-8) < 3 or np.linalg.cond(matrix) >= 1e6: continue
        source.append(centers[label]); target.append(np.linalg.lstsq(matrix, np.concatenate([p@o for p, o in rays]), rcond=None)[0])
    rotations = cube_rotations()
    kabsch_available = False
    if len(source) >= 3:
        x, y = np.asarray(source), np.asarray(target)
        if np.linalg.matrix_rank(x-x.mean(0), tol=1e-6) >= 2 and np.linalg.matrix_rank(y-y.mean(0), tol=1e-6) >= 2:
            U, _, Vt = np.linalg.svd((x-x.mean(0)).T @ (y-y.mean(0)))
            sign = np.diag([1., 1., np.linalg.det(Vt.T@U.T)])
            rotations.insert(0, Vt.T @ sign @ U.T)
            kabsch_available = True
    proposals = []
    for rotation in rotations:
        b = np.concatenate([p @ (origin-rotation@centers[label]) for label, p, origin, _, _ in rows])
        translation = np.linalg.lstsq(A, b, rcond=None)[0]
        if any((R@(rotation@centers[label]+translation)+t)[2] <= 0 for label, _, _, R, t in rows): continue
        pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = translation
        proposals.append(pose)
    require(bool(proposals), "no initialization proposal in front of observed cameras")
    return proposals, dict(centroid_constraints=len(rows), triangulated_centroids=len(source),
                          kabsch_available=kabsch_available, rotation_identifiable_from_centroids=kabsch_available,
                          caveat="visible mask centroids are not homologous anatomical points")


def initialize_mean_scene(scene, surface_config, render_config, matching_config):
    matcher = SemanticXYMatcher(matching_config)
    saved = {name: (s.base_pose.detach().clone(), s.pose_delta.detach().clone(),
                     {k: q.detach().clone() for k, q in s.q.items()}) for name, s in scene.arches.items()}
    reports = {}
    try:
        with torch.no_grad():
            for state in scene.arches.values():
                for q in state.q.values(): q.zero_()
                state.pose_delta.zero_()
        charts = {a: extract_chart(s, surface_config) for a, s in scene.arches.items()}
        proposals = {}
        for arch, state in scene.arches.items():
            chart = charts[arch]
            with torch.no_grad():
                weights = state.query_model(chart.anchors)["semantics"]
            from dmm import CHANNEL_FDI
            centers = {}
            triangle_points = chart.anchors[chart.faces]
            triangle_area = torch.linalg.cross(triangle_points[:, 1]-triangle_points[:, 0], triangle_points[:, 2]-triangle_points[:, 0]).norm(dim=-1)
            for label, present in state.presence.items():
                if not present: continue
                mass = weights[chart.faces, CHANNEL_FDI.index(label)+1].mean(-1)*triangle_area
                if float(mass.sum()) <= 1e-8: continue
                centers[label] = ((triangle_points.mean(1)*mass[:, None]).sum(0)/mass.sum()*MODEL_UNIT_MM).cpu().numpy()
            proposals[arch], reports[arch] = pose_proposals(centers, scene.observations)
            with torch.no_grad(): state.base_pose.copy_(state.base_pose.new_tensor(proposals[arch][0]))
        # Coordinate search: both arches share the depth buffer and all views.
        for _ in range(2):
            for arch, state in scene.arches.items():
                best, selected, records = float("inf"), None, []
                for i, pose in enumerate(proposals[arch]):
                    try:
                        with torch.no_grad():
                            state.base_pose.copy_(state.base_pose.new_tensor(pose))
                            images, _, _ = scene.render_views(surface_config, render_config, charts)
                            match = matcher.match_scene(images, scene.observations, mode="evaluate")
                            score = float(match["loss"])
                        records.append(dict(candidate=i, loss=score, missing=match["needs_visibility_recovery"]))
                        if score < best: best, selected = score, pose
                    except ContractError as exc:
                        records.append(dict(candidate=i, error=str(exc)))
                require(selected is not None, f"{arch}: all rendered initialization proposals failed")
                with torch.no_grad(): state.base_pose.copy_(state.base_pose.new_tensor(selected))
                reports[arch].update(candidates=records, selected_loss=best, T_world_from_arch=selected.tolist())
        return dict(status="MEAN_POSE_PROPOSED_NOT_QUALITY_ACCEPTED", arches=reports,
                    scope="mean_latent; centroid_heuristic; multi_rotation_full_render_ranking; no_truth")
    except BaseException:
        with torch.no_grad():
            for arch, state in scene.arches.items():
                pose, delta, qs = saved[arch]
                state.base_pose.copy_(pose); state.pose_delta.copy_(delta)
                for k, q in state.q.items(): q.copy_(qs[k])
        raise
