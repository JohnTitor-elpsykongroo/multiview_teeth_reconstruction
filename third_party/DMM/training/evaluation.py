"""Explicit learned-train and held-out evaluation; no decoder or embedding updates.

Train reconstruction reuses learned codes and is not an unseen-point claim.
Val/test infer new codes on deterministic disjoint fit/evaluation partitions.
All failures stay in the denominator. Test is a separate explicitly frozen run.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dmm.provenance import source_fingerprint, training_runtime, validate_decoder_binding
from dmm.validation import require, read_json, sha256, write_json, ContractError
from training.validation import validation_samples, mesh_metrics
from training.recipe import resolve_recipe


def evaluation_cases(dataset, split, case_ids=None):
    require(split in ("train", "val", "test"), "unsupported evaluation split")
    source = dataset.selected if split == "train" else dataset.cases
    available = {c.metadata["case_id"]: c for c in source
                 if c.metadata["split"] == split and c.metadata["augmentation_parent_id"] is None}
    require(bool(available), f"no original {split} cases")
    if case_ids is not None:
        require(isinstance(case_ids, list) and bool(case_ids) and len(set(case_ids)) == len(case_ids)
                and all(isinstance(x, str) for x in case_ids), "invalid evaluation case_ids")
        require(set(case_ids).issubset(available), f"evaluation case_ids must be original {split} cases")
        return [available[k] for k in case_ids]
    return [available[k] for k in sorted(available)]


def budget_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_fit_recipe(path, arch):
    value = read_json(path)
    require(isinstance(value, dict), 'fit recipe must be a mapping')
    require(set(value) == {'schema', 'arch', 'recipe'} and value['schema'] == 'dmm_latent_fit_recipe_v1'
            and value['arch'] == arch, 'fit recipe schema/arch mismatch')
    resolved = resolve_recipe(dict(learning_rate=.0001, recipe=value['recipe']))
    require(resolved == value['recipe'], 'fit recipe must explicitly resolve every field and weight')
    return resolved


def evaluate_cases(system, dataset, config, split, case_ids=None, artifact_root=None, *, fit_recipe=None):
    cases = evaluation_cases(dataset, split, case_ids)
    require(split != 'train' or fit_recipe is None, 'learned train evaluation does not fit codes')
    original_recipe = system.recipe
    if fit_recipe is not None:
        resolved = resolve_recipe(dict(learning_rate=.0001, recipe=fit_recipe))
        require(resolved == fit_recipe, 'fit recipe must explicitly resolve every field and weight')
        system.recipe = resolved
    effective_fit_recipe = system.recipe if split != 'train' else None
    flags = [(p, p.requires_grad) for p in system.parameters()]
    was_training = system.training
    device = next(system.parameters()).device
    records = []
    try:
        system.eval().requires_grad_(False)
        for index, case in enumerate(cases):
            started = time.perf_counter()
            record = dict(case_id=case.metadata["case_id"], patient_id=case.metadata["patient_id"], split=split,
                          status="FAILED", code_mode="learned_train" if split == "train" else "temporary_fitted")
            try:
                fit, evaluation = validation_samples(case, dataset.arch, config, split=split)
                codes = {}
                for label in case.components:
                    if split == "train":
                        codes[label] = system.latents[str(label)].weight[case.metadata["embedding_row"]].detach().clone()
                    else:
                        rows = [c.metadata["embedding_row"] for c in dataset.selected
                                if c.metadata["augmentation_parent_id"] is None and label in c.components]
                        require(bool(rows), f"no original train support for component {label}")
                        codes[label] = torch.nn.Parameter(system.latents[str(label)].weight[rows].detach().mean(0).clone())
                if split != "train":
                    optimizer = torch.optim.Adam(codes.values(), lr=config["learning_rate"])
                    for _ in range(config["fit_steps"]):
                        optimizer.zero_grad(set_to_none=True)
                        loss, _ = system.loss(fit, codes)
                        require(bool(torch.isfinite(loss)), "nonfinite evaluation fit loss")
                        loss.backward()
                        require(all(c.grad is not None and bool(torch.isfinite(c.grad).all()) for c in codes.values()),
                                "invalid evaluation code gradient")
                        optimizer.step()
                require(all(bool(torch.isfinite(c).all()) for c in codes.values()), "nonfinite evaluation code")
                points = evaluation["points"].to(device).requires_grad_(True)
                prediction = system.model.query(points, codes)
                grad = torch.autograd.grad(prediction["sdf"].sum(), points, retain_graph=True)[0]
                require(all(bool(torch.isfinite(x).all()) for x in (prediction["sdf"], grad, prediction["weights"])),
                        "nonfinite evaluation field")
                predicted = torch.tensor(prediction["labels"], device=device)[prediction["weights"].argmax(-1)]
                truth = evaluation["labels"].to(device)
                record.update(fit_steps=0 if split == "train" else config["fit_steps"],
                    fit_points=0 if split == "train" else len(fit["points"]), evaluation_points=len(points),
                    fit_index_sha256=hashlib.sha256(fit["surface_indices"].tobytes()).hexdigest() if split != "train" else None,
                    evaluation_index_sha256=hashlib.sha256(evaluation["surface_indices"].tobytes()).hexdigest(),
                    sdf_abs_mean_model=float(prediction["sdf"].detach().abs().mean()),
                    normal_cosine_error=float((1-F.cosine_similarity(grad, evaluation["normals"].to(device), dim=-1)).mean()),
                    semantic_error=float((predicted != truth).float().mean()),
                    gradient_norm_quantiles=np.quantile(grad.detach().norm(dim=-1).cpu().numpy(), [0, .5, .95, 1]).tolist())
                components = {}
                for label in case.components:
                    own = truth == label
                    part = prediction["components"][label]
                    part_grad = torch.autograd.grad(part["sdf"].sum(), points, retain_graph=True)[0]
                    require(bool(torch.isfinite(part_grad).all()), "nonfinite component gradient")
                    components[str(label)] = dict(points=int(own.sum()),
                        own_sdf_abs_mean_model=float(part["sdf"][own].detach().abs().mean()),
                        own_blend_mean=float(prediction["weights"][own, prediction["labels"].index(label)].detach().mean()),
                        own_semantic_error=float((predicted[own] != label).float().mean()),
                        own_gradient_norm_median=float(part_grad[own].detach().norm(dim=-1).median()))
                record["components"] = components
                artifact = None if artifact_root is None else Path(artifact_root) / f"case_{index:04d}"
                if artifact is not None:
                    record["artifact_directory"] = artifact.name
                record["geometry"] = mesh_metrics(system.model, codes, evaluation, dataset.sampling_domain, config,
                                                   artifact_dir=artifact)
                record["status"] = "EVALUATED_NOT_QUALITY_ACCEPTED"
            except (ContractError, RuntimeError, ValueError) as exc:
                record["error"] = str(exc)
            record["seconds"] = time.perf_counter() - started
            records.append(record)
            if artifact_root is not None:
                # Preserve partial progress even if a later case/process fails.
                write_json(Path(artifact_root) / "cases_partial.json", records)
    finally:
        system.recipe = original_recipe
        system.train(was_training)
        for parameter, enabled in flags:
            parameter.requires_grad_(enabled)
    failures = [r["case_id"] for r in records if r["status"] == "FAILED"]
    scores = [r["geometry"]["score_mm"] for r in records if r["status"] != "FAILED"]
    descriptive = None
    if scores:
        descriptive = dict(denominator=len(scores), requested_denominator=len(cases),
                           case_score_p95_mm=float(np.quantile(scores, .95)), case_score_worst_mm=float(max(scores)),
                           scope="successful_cases_only_diagnostic_not_selection_score")
        per_fdi = {}
        for record in records:
            for label, metrics in record.get("geometry", {}).get("per_tooth", {}).items():
                per_fdi.setdefault(label, []).append(metrics)
        descriptive["per_fdi"] = {label: dict(successful_case_count=len(values),
            mean_mm=float(np.mean([v["observed_to_mesh_sample_mean_mm"] for v in values])),
            case_mean_p95_mm=float(np.quantile([v["observed_to_mesh_sample_mean_mm"] for v in values], .95)),
            case_mean_worst_mm=float(max(v["observed_to_mesh_sample_mean_mm"] for v in values)),
            observed_point_p95_worst_mm=float(max(v["observed_to_mesh_sample_p95_mm"] for v in values)))
            for label, values in per_fdi.items()}
    return dict(status="EVALUATION_FAILED" if failures else "EVALUATED_NOT_QUALITY_ACCEPTED", split=split,
                requested_case_count=len(cases), successful_case_count=len(scores), failed_case_count=len(failures),
                score_mm=None if failures else float(np.mean(scores)), failures=failures, cases=records, budget=config,
                fit_recipe=effective_fit_recipe,
                successful_only_diagnostics=descriptive,
                scope="learned_train_reconstruction_not_unseen_points" if split == "train" else
                      "decoder_frozen_temporary_codes_disjoint_fit_evaluation_points_and_centers")


def evaluate_checkpoint_cases(config_path, checkpoint, output_dir, *, split, case_ids=None, device="cpu", test_protocol=None,
                              fit_recipe_path=None):
    from networks.dmm_net import DMM
    from training.arch_training import load_training_config, ArchTrainingSystem
    output = Path(output_dir).resolve()
    require(not output.exists(), "evaluation output must be fresh")
    require(split != 'train' or fit_recipe_path is None, 'learned train evaluation does not fit codes')
    config, dataset, centers, specs = load_training_config(config_path)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    require(saved["arch"] == config["arch"] and saved["specs"] == specs
            and saved["manifest_sha256"] == sha256(dataset.manifest_path), "checkpoint/data/spec mismatch")
    require(saved["canonical_reference_id"] == dataset.canonical_reference_id, "checkpoint canonical mismatch")
    require(set(saved["training_case_ids"]) == {c.metadata["case_id"] for c in dataset.selected}, "checkpoint training subset mismatch")
    require(saved["resolved_config"]["recipe"] == config["recipe"], "checkpoint training recipe mismatch")
    binding = dict(local_source_sha256=saved["source_sha256"])
    if "decoder_contract" in saved:
        binding["decoder_contract"] = saved["decoder_contract"]
    validate_decoder_binding(binding)
    budget = config["validation"]
    fit_recipe = None if fit_recipe_path is None else load_fit_recipe(fit_recipe_path, config['arch'])
    fit_recipe_sha256 = None if fit_recipe_path is None else sha256(fit_recipe_path)
    if split == "val" and case_ids is None:
        case_ids = budget["case_ids"]
    cases = evaluation_cases(dataset, split, case_ids)
    protocol = None
    if split == "test":
        require(test_protocol is not None, "test requires a frozen final protocol")
        protocol = read_json(test_protocol)
        require(protocol.get("stage") == "R4_FINAL_FROZEN" and protocol.get("checkpoint_sha256") == sha256(checkpoint)
                and protocol.get("manifest_sha256") == sha256(dataset.manifest_path)
                and protocol.get("evaluation_budget_sha256") == budget_digest(budget)
                and protocol.get("case_ids") == [c.metadata["case_id"] for c in cases], "test frozen protocol mismatch")
        require(case_ids is None, "final test evaluates all original test cases")
        require(protocol.get('fit_recipe_sha256') == fit_recipe_sha256,
                'test frozen fit recipe file mismatch')
        if fit_recipe is not None:
            require(protocol.get('fit_recipe_resolved_sha256') == budget_digest(fit_recipe),
                    'test frozen resolved fit recipe mismatch')
    else:
        require(test_protocol is None, "test protocol only valid for test split")
    system = ArchTrainingSystem(DMM(specs, arch=config["arch"]), dataset, centers, config["recipe"]).to(device)
    system.load_state_dict(saved["system_state"], strict=True)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provenance.json", dict(split=split, checkpoint_sha256=sha256(checkpoint), training_epoch=saved["epoch"],
               training_source_sha256=saved["source_sha256"], evaluation_source_sha256=source_fingerprint(),
               config_sha256=sha256(config_path), runtime=training_runtime(device), protocol=protocol,
               manifest_sha256=sha256(dataset.manifest_path), canonical_reference_id=dataset.canonical_reference_id,
               training_recipe=config['recipe'], fit_recipe_sha256=fit_recipe_sha256,
               fit_recipe=fit_recipe if fit_recipe is not None else config['recipe'] if split != 'train' else None,
               test_protocol_sha256=None if test_protocol is None else sha256(test_protocol)))
    try:
        result = evaluate_cases(system, dataset, budget, split, case_ids, output, fit_recipe=fit_recipe)
        write_json(output / "evaluation.json", result)
        return result
    except BaseException as exc:
        write_json(output / "failure.json", dict(error=str(exc), split=split))
        raise
