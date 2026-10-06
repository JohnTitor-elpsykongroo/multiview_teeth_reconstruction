"""Held-out auto-decoder validation, never updates decoder or training embeddings.

Only original val patients may enter. Fit/evaluation use disjoint point indices,
fixed seeds and equal latent budgets for every checkpoint. Distances are sampled
surface estimates, not exact Hausdorff distances or ground-truth roots/caps.
"""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from dmm import MODEL_UNIT_MM
from dmm.validation import require, sha256, ContractError


def validation_samples(case, arch, config):
    require(case.metadata["split"] == "val" and case.metadata["augmentation_parent_id"] is None,
            "validation accepts original val cases only")
    require(sha256(case.sample_path) == case.metadata["samples"]["sha256"], "validation sample hash mismatch")
    seed_id = int.from_bytes(hashlib.sha256(case.metadata["case_id"].encode()).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], seed_id]))
    with np.load(case.sample_path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    labels = arrays["surface_labels"]
    ids = [[], []]
    for label in case.components:
        pool = rng.permutation(np.flatnonzero(labels == label))
        require(len(pool) >= 2, f"validation component {label} needs disjoint fit/evaluation points")
        halves = (pool[:len(pool)//2], pool[len(pool)//2:])
        for side, quota in enumerate((config["points_per_component"], config["evaluation_points_per_component"])):
            ids[side].extend(halves[side][:min(quota, len(halves[side]))])
    off = rng.permutation(len(arrays["offsurface_points"]))
    require(len(off) >= 2, "validation needs disjoint offsurface points")
    samples = []
    for side in (0, 1):
        selected = np.asarray(ids[side], dtype=np.int64)
        off_ids = (off[:len(off)//2] if side == 0 else off[len(off)//2:])[:config["offsurface_points"]]
        samples.append(dict(case_id=case.metadata["case_id"], arch=arch, embedding_row=case.metadata["embedding_row"],
                            components=case.components, presence=case.presence,
                            # Compute center supervision from this partition only;
                            # cached whole-scan centers include held-out points.
                            centers={k: torch.from_numpy(arrays["surface_points"][selected[labels[selected] == k]].mean(0).astype(np.float32))
                                     for k in case.components if k != 0},
                            points=torch.from_numpy(arrays["surface_points"][selected].astype(np.float32)),
                            normals=torch.from_numpy(arrays["surface_normals"][selected].astype(np.float32)),
                            labels=torch.from_numpy(labels[selected].astype(np.int64)),
                            offsurface=torch.from_numpy(arrays["offsurface_points"][off_ids].astype(np.float32)),
                            surface_indices=selected, offsurface_indices=off_ids))
    return samples


def mesh_metrics(model, codes, sample, box, config):
    from skimage.measure import marching_cubes
    from scipy.spatial import cKDTree
    device = next(model.parameters()).device
    resolution = config["mesh_resolution"]
    box = np.asarray(box)
    axes = [np.linspace(box[0, i], box[1, i], resolution) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    def query(data, key):
        with torch.no_grad():
            return np.concatenate([model.query(torch.as_tensor(p, dtype=torch.float32, device=device), codes)[key].cpu().numpy()
                                   for p in np.array_split(data, max(1, int(np.ceil(len(data)/2048))))])
    volume = query(grid, "sdf").reshape((resolution,) * 3)
    require(np.isfinite(volume).all() and volume.min() < 0 < volume.max(), "no finite zero surface")
    boundary = np.concatenate([volume[0].ravel(), volume[-1].ravel(), volume[:, 0].ravel(), volume[:, -1].ravel(), volume[:, :, 0].ravel(), volume[:, :, -1].ravel()])
    require(bool((boundary > 0).all()), "non-positive domain boundary; truncated/invalid surface")
    vertices, faces, _, _ = marching_cubes(volume, 0, spacing=(box[1]-box[0])/(resolution-1))
    vertices += box[0]
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=-1)/2
    labels = tuple(k for k in model.labels if k in codes)
    face_labels = np.asarray(labels)[query(triangles.mean(1), "weights").argmax(-1)]
    rng = np.random.default_rng(config["seed"])
    truth = sample["points"].numpy()
    truth_labels = sample["labels"].numpy()
    per_tooth = {}
    for label in sample["components"]:
        if label == 0: continue
        pool = np.flatnonzero((face_labels == label) & (area > 0))
        require(len(pool) > 0, f"missing blended surface component {label}")
        chosen = rng.choice(pool, config["mesh_samples_per_component"], p=area[pool]/area[pool].sum())
        uv = rng.random((len(chosen), 2)); root = np.sqrt(uv[:, 0])
        bary = np.column_stack((1-root, root*(1-uv[:, 1]), root*uv[:, 1]))
        points = (triangles[chosen]*bary[:, :, None]).sum(1)*MODEL_UNIT_MM
        observed = truth[truth_labels == label]*MODEL_UNIT_MM
        forward = cKDTree(points).query(observed)[0]
        reverse = cKDTree(observed).query(points)[0]
        per_tooth[str(label)] = dict(observed_to_mesh_sample_mean_mm=float(forward.mean()),
                                    observed_to_mesh_sample_p95_mm=float(np.quantile(forward, .95)),
                                    generated_to_observed_sample_mean_mm=float(reverse.mean()),
                                    generated_to_observed_sample_p95_mm=float(np.quantile(reverse, .95)))
    require(bool(per_tooth), "no teeth available for geometry validation")
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices)))
    components = int(connected_components(graph, directed=False, return_labels=False))
    # Full generated implicit surface includes unsupported caps. Reverse distances
    # are diagnostics only, not truth about unobserved anatomy.
    return dict(score_mm=float(np.mean([v["observed_to_mesh_sample_mean_mm"] for v in per_tooth.values()])),
                per_tooth=per_tooth, vertices=len(vertices), faces=len(faces),
                connected_components=components,
                boundary_edges=int((edge_counts == 1).sum()), nonmanifold_edges=int((edge_counts > 2).sum()),
                metric="equal_tooth_observed_to_blended_mesh_samples_mm", resolution=resolution,
                samples_per_component=config["mesh_samples_per_component"],
                reverse_scope="diagnostic_only_includes_unobserved_generated_surface")


def validate_arch(system, dataset, config):
    cases = [c for c in dataset.cases if c.metadata["split"] == "val" and c.metadata["augmentation_parent_id"] is None]
    if config["case_ids"] is not None:
        cases = [c for c in cases if c.metadata["case_id"] in config["case_ids"]]
    require(bool(cases), "no validation cases")
    flags = [(p, p.requires_grad) for p in system.parameters()]
    was_training = system.training
    device = next(system.parameters()).device
    result = []
    try:
        system.eval().requires_grad_(False)
        for case in cases:
            record = dict(case_id=case.metadata["case_id"], patient_id=case.metadata["patient_id"], status="FAILED")
            try:
                fit, evaluation = validation_samples(case, dataset.arch, config)
                codes = {}
                for label in case.components:
                    rows = [c.metadata["embedding_row"] for c in dataset.selected if c.metadata["augmentation_parent_id"] is None and label in c.components]
                    require(bool(rows), f"no original train support for component {label}")
                    mean = system.latents[str(label)].weight[rows].detach().mean(0)
                    codes[label] = torch.nn.Parameter(mean.clone())
                optimizer = torch.optim.Adam(codes.values(), lr=config["learning_rate"])
                for _ in range(config["fit_steps"]):
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = system.loss(fit, codes)
                    require(bool(torch.isfinite(loss)), "nonfinite validation latent loss")
                    loss.backward()
                    require(all(c.grad is not None and bool(torch.isfinite(c.grad).all()) for c in codes.values()), "invalid validation latent gradient")
                    optimizer.step()
                require(all(bool(torch.isfinite(c).all()) for c in codes.values()), "invalid validation latent")
                points = evaluation["points"].to(device).requires_grad_(True)
                prediction = system.model.query(points, codes)
                grad = torch.autograd.grad(prediction["sdf"].sum(), points)[0]
                require(bool(torch.isfinite(prediction["sdf"]).all()) and bool(torch.isfinite(grad).all())
                        and bool(torch.isfinite(prediction["weights"]).all()), "nonfinite heldout field metrics")
                lookup = torch.tensor(prediction["labels"], device=device)
                predicted = lookup[prediction["weights"].argmax(-1)]
                # Retain field diagnostics even when the early decoder has no valid mesh.
                record.update(heldout_sdf_abs_mean_model=float(prediction["sdf"].detach().abs().mean()),
                              heldout_normal_cosine_error=float((1-F.cosine_similarity(grad, evaluation["normals"].to(device), dim=-1)).mean()),
                              heldout_semantic_error=float((predicted != evaluation["labels"].to(device)).float().mean()),
                              fit_points=len(fit["points"]), evaluation_points=len(evaluation["points"]),
                              fit_index_sha256=hashlib.sha256(fit["surface_indices"].tobytes()).hexdigest(),
                              evaluation_index_sha256=hashlib.sha256(evaluation["surface_indices"].tobytes()).hexdigest())
                metrics = mesh_metrics(system.model, codes, evaluation, dataset.sampling_domain, config)
                record.update(status="EVALUATED_NOT_QUALITY_ACCEPTED", geometry=metrics)
            except (ContractError, RuntimeError, ValueError) as exc:
                record["error"] = str(exc)
            result.append(record)
    finally:
        system.train(was_training)
        for parameter, enabled in flags: parameter.requires_grad_(enabled)
    failures = [r["case_id"] for r in result if r["status"] == "FAILED"]
    score = None if failures else float(np.mean([r["geometry"]["score_mm"] for r in result]))
    return dict(status="VALIDATION_FAILED" if failures else "VALIDATION_EVALUATED_NOT_QUALITY_ACCEPTED",
                score_mm=score, cases=result, failures=failures, config=config,
                scope="fixed_budget_val_latent_fitting; decoder_frozen; disjoint_points_and_centers; no_test_optimization_or_scoring; sampled_surface_metrics")
