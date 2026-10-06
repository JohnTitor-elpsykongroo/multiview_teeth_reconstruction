"""Independent single-arch training and inference-bundle export.

Uses the official component networks and loss families, with manifest presence
and explicitly versioned reductions. This is a new manifest training path, not a claim of
bitwise equivalence to upstream training or validated model quality.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dmm import MODEL_UNIT_MM, TEETH
from data.arch_dataset import ArchDataset
from networks.dmm_net import DMM
from dmm import IMPLEMENTATION_VERSION
from dmm.provenance import source_fingerprint, upstream_commit, decoder_contract, training_runtime, project_revision
from training.recipe import resolve_recipe, resolve_validation
from dmm.validation import (array, header, read_json, reference, require, resolve_ref,
                         sha256, stamped, write_json)


def gradient(output, points):
    return torch.autograd.grad(output.sum(), points, create_graph=True, retain_graph=True)[0]


def selected_mean(value, mask):
    selected = value[mask]
    return selected.mean() if selected.numel() else value.sum() * 0


class ArchTrainingSystem(nn.Module):
    def __init__(self, model, dataset, reference_centers, recipe=None):
        super().__init__()
        self.model = model
        self.recipe = recipe or resolve_recipe({"learning_rate": .0001})
        self.latents = nn.ModuleDict({str(k): nn.Embedding(len(dataset.cases), dim) for k, dim in model.dimensions.items()})
        for embedding in self.latents.values():
            nn.init.normal_(embedding.weight, std=self.recipe["latent_init_std"])
        self.center_index = {label: index for index, label in enumerate(TEETH[model.arch])}
        self.register_buffer("canonical_centers", torch.tensor(np.asarray([reference_centers[k] for k in TEETH[model.arch]]), dtype=torch.float32))

    def loss(self, sample, codes=None):
        require(sample["arch"] == self.model.arch, "training sample arch mismatch")
        device = next(self.parameters()).device
        coords = torch.cat((sample["points"], sample["offsurface"])).to(device).requires_grad_(True)
        ns = len(sample["points"])
        labels = sample["labels"].to(device)
        normals = sample["normals"].to(device)
        row = torch.tensor(sample["embedding_row"], dtype=torch.long, device=device)
        codes = {k: self.latents[str(k)](row) for k in sample["components"]} if codes is None else codes
        require(set(codes) == set(sample["components"]), "loss codes must match explicit presence")
        result = self.model.query(coords, codes)
        weights = self.recipe["weights"]
        normal_epsilon = self.recipe["normal_epsilon"]
        raw, counts = {}, {}
        def reduced(value, mask):
            if self.recipe["reduction"] == "effective":
                return selected_mean(value, mask)
            return value[mask].sum() / len(coords)
        def add(name, value):
            raw.setdefault(name, []).append(value)
        for label, prediction in result["components"].items():
            sdf = prediction["sdf"]
            own = labels == label
            other_inside = (~own) & (sdf[:ns] < -self.recipe["flip_threshold_model"])
            grad = gradient(sdf, coords)
            template_grad = gradient(prediction["template"], prediction["warped"])
            jacobian = torch.stack([gradient(prediction["deformation"][:, axis], coords) for axis in range(3)], -2)
            non_own = torch.cat((~own, torch.ones(len(coords) - ns, dtype=torch.bool, device=device)))
            add("surface", reduced(sdf[:ns].abs(), own))
            add("flip", reduced(sdf[:ns].abs(), other_inside))
            add("offsurface", reduced(torch.exp(-100 * sdf.abs()), non_own))
            add("normal", reduced(1 - F.cosine_similarity(grad[:ns], normals, dim=-1, eps=normal_epsilon), own))
            add("eikonal", (grad.norm(dim=-1) - 1).abs().mean())
            add("correction", prediction["correction"].abs().mean())
            add("smooth", jacobian.norm(dim=-1).mean())
            add("template_normal", reduced(1 - F.cosine_similarity(template_grad[:ns], normals, dim=-1, eps=normal_epsilon), own))
            add("blend", F.binary_cross_entropy_with_logits(prediction["logit"][:ns], own.float()))
            counts[f"count/{label}/own"] = float(own.sum())
            counts[f"count/{label}/flip"] = float(other_inside.sum())
            counts[f"count/{label}/non_own"] = float(non_own.sum())
            counts[f"raw/{label}/surface"] = float(raw["surface"][-1].detach())
            counts[f"raw/{label}/normal"] = float(raw["normal"][-1].detach())
            counts[f"raw/{label}/blend"] = float(raw["blend"][-1].detach())
            if label:
                center = sample["centers"][label].to(device)[None]
                warped = self.model.component(center, codes[label], label)["warped"][0]
                target = self.canonical_centers[self.center_index[label]]
                add("center", (warped - target).abs().sum())
            elif self.recipe["center_reduction"] == "all_components":
                add("center", sdf.sum() * 0)
        sdf = result["sdf"]
        grad = gradient(sdf, coords)
        on = torch.arange(len(coords), device=device) < ns
        add("global_surface", reduced(sdf.abs(), on))
        add("global_offsurface", reduced(torch.exp(-100 * sdf.abs()), ~on))
        add("global_normal", reduced(1 - F.cosine_similarity(grad[:ns], normals, dim=-1, eps=normal_epsilon), torch.ones(ns, dtype=torch.bool, device=device)))
        add("global_eikonal", (grad.norm(dim=-1) - 1).abs().mean())
        add("latent", torch.stack([code.square().mean() for code in codes.values()]).mean())
        if "center" not in raw: add("center", sdf.sum() * 0)
        raw = {key: torch.stack(values).mean() for key, values in raw.items()}
        weighted = {key: value * weights[key] for key, value in raw.items()}
        total = sum(weighted.values())
        log = {f"raw/{key}": float(value.detach()) for key, value in raw.items()}
        log.update({f"weighted/{key}": float(value.detach()) for key, value in weighted.items()})
        log.update(counts)
        return total, log


def load_training_config(path, config=None):
    path = Path(path).resolve()
    config = read_json(path) if config is None else config
    header(config, "arch_training_config")
    allowed = {"contract_id", "version", "artifact_kind", "representation_profile", "arch", "manifest", "canonical_reference", "specs",
               "epochs", "points_per_component", "gum_points", "offsurface_points", "learning_rate", "seed", "checkpoint_every",
               "recipe", "validation", "training_case_ids"}
    require(not set(config) - allowed, "unknown training config field")
    require(config["arch"] in TEETH, "invalid training arch")
    for key in ("epochs", "points_per_component", "offsurface_points", "checkpoint_every"):
        require(type(config[key]) is int and config[key] > 0, f"{key} must be a positive integer")
    require(type(config["seed"]) is int and config["seed"] >= 0, "seed must be nonnegative integer")
    require(isinstance(config["learning_rate"], (int, float)) and np.isfinite(config["learning_rate"]) and config["learning_rate"] > 0, "invalid learning_rate")
    manifest = resolve_ref(path, config["manifest"])
    canonical = resolve_ref(path, config["canonical_reference"])
    specs_path = resolve_ref(path, config["specs"])
    dataset = ArchDataset(manifest, config["arch"], "train", config["points_per_component"], config["offsurface_points"], config["seed"], config.get("gum_points"))
    ref = read_json(canonical)
    header(ref, "canonical_reference")
    require(ref["arch"] == config["arch"] and ref["reference_id"] == dataset.canonical_reference_id, "canonical reference mismatch")
    require(ref["model_unit_mm"] == MODEL_UNIT_MM and ref["axes"] == "LPS", "canonical units/axes mismatch")
    require(set(ref["centers_model"]) == {str(k) for k in TEETH[config["arch"]]}, "canonical reference needs 14 centers")
    centers = {int(k): array(v, (3,), f"canonical center {k}") for k, v in ref["centers_model"].items()}
    require(np.allclose(np.mean(list(centers.values()), 0), 0, atol=1e-5, rtol=0), "canonical center mean must be zero")
    labels = TEETH[config["arch"]]
    require(all(np.allclose(centers[k] * [-1, 1, 1], centers[k + 10], atol=1e-5, rtol=0) for k in labels[:7]), "canonical centers must be left/right symmetric")
    original_train = {c.metadata["case_id"] for c in dataset.selected if c.metadata["augmentation_parent_id"] is None}
    require(bool(ref["training_case_ids"]) and len(set(ref["training_case_ids"])) == len(ref["training_case_ids"])
            and set(ref["training_case_ids"]).issubset(original_train), "canonical reference must use original training cases")
    require(isinstance(ref["direction_anchor_evidence"], str) and bool(ref["direction_anchor_evidence"].strip()), "canonical direction evidence required")
    specs = read_json(specs_path)
    config = dict(config, recipe=resolve_recipe(config), validation=resolve_validation(config))
    subset = config.get("training_case_ids")
    if subset is not None:
        require(isinstance(subset, list) and bool(subset) and all(isinstance(x, str) for x in subset)
                and len(set(subset)) == len(subset), "invalid training_case_ids")
        by_id = {c.metadata["case_id"]: c for c in dataset.selected}
        require(set(subset).issubset(by_id), "training subset must use train cases only")
        require(all(by_id[k].metadata["augmentation_parent_id"] is None or by_id[k].metadata["augmentation_parent_id"] in subset for k in subset),
                "training subset must include augmentation parents")
        dataset.selected = [c for c in dataset.selected if c.metadata["case_id"] in subset]
    if config["validation"]["enabled"]:
        available = {c.metadata["case_id"] for c in dataset.cases if c.metadata["split"] == "val" and c.metadata["augmentation_parent_id"] is None}
        require(bool(available), "validation requires original val cases")
        requested = config["validation"]["case_ids"]
        require(requested is None or set(requested).issubset(available), "validation case_ids must be original val cases only")
    return config, dataset, centers, specs


def _save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_arch(config_path, output_dir, device="cpu", resume=None):
    output = Path(output_dir).resolve()
    existed = output.exists()
    try:
        return _train_arch(config_path, output_dir, device, resume)
    except BaseException as exc:
        if not existed and output.is_dir():
            write_json(output / "failure.json", dict(status="ARCH_TRAINING_FAILED", error=str(exc),
                       exception_type=type(exc).__name__, completed_checkpoint_scope="last fully saved epoch; do not overwrite this run"))
        raise


def _train_arch(config_path, output_dir, device="cpu", resume=None):
    output = Path(output_dir).resolve()
    require(not output.exists(), f"output already exists: {output}")
    config, dataset, centers, specs = load_training_config(config_path)
    torch.manual_seed(config["seed"])
    model = DMM(specs, arch=config["arch"])
    system = ArchTrainingSystem(model, dataset, centers, config["recipe"]).to(device)
    recipe = config["recipe"]
    optimizer = torch.optim.Adam([
        dict(params=model.deform_nets_dict.parameters(), lr=recipe["deformation_lr"], name="deformation"),
        dict(params=model.ref_nets_dict.parameters(), lr=recipe["reference_lr"], name="reference"),
        dict(params=system.latents.parameters(), lr=recipe["latent_lr"], name="latent")])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=recipe["lr_step_epochs"], gamma=recipe["lr_gamma"])
    # Adam weight decay stays zero: absent/held-out embedding rows must not move.
    fingerprint = source_fingerprint()
    commit = upstream_commit()
    project = project_revision()
    manifest_hash = sha256(dataset.manifest_path)
    runtime = training_runtime(device)
    signature_data = {k: v for k, v in config.items() if k not in ("epochs", "checkpoint_every")}
    signature = hashlib.sha256((json.dumps(signature_data, sort_keys=True) + fingerprint).encode()).hexdigest()
    start, best, best_val = 0, float("inf"), float("inf")
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=True)
        require(saved["training_signature"] == signature, "resume config/data/source mismatch")
        require(saved.get("runtime_environment") == runtime, "resume numerical runtime mismatch; use the recorded training environment")
        system.load_state_dict(saved["system_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        start, best = saved["epoch"], saved["best_train_loss"]
        best_val = saved.get("best_val_score", float("inf"))
        torch.set_rng_state(saved["rng_state"].cpu())
        if device.startswith("cuda") and "cuda_rng_state" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    require(config["epochs"] > start, "no remaining epochs")
    output.mkdir(parents=True, exist_ok=False)
    # Exact executable snapshot, including local untracked implementation files.
    import shutil
    from dmm import DMM_ROOT
    for source in DMM_ROOT.rglob("*.py"):
        if ".git" in source.parts or "__pycache__" in source.parts: continue
        target = output / "source_snapshot" / source.relative_to(DMM_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if (DMM_ROOT / "UPSTREAM.json").is_file():
        shutil.copy2(DMM_ROOT / "UPSTREAM.json", output / "source_snapshot" / "UPSTREAM.json")
    write_json(output / "config.json", config)
    write_json(output / "runtime.json", runtime)
    write_json(output / "provenance.json", dict(config_path=str(Path(config_path).resolve()), source_sha256=fingerprint,
               upstream_commit=commit, project_revision=project, manifest_path=str(dataset.manifest_path), manifest_sha256=manifest_hash, training_signature=signature,
               resume=None if resume is None else dict(path=str(Path(resume).resolve()), sha256=sha256(resume))))
    for epoch in range(start, config["epochs"]):
        dataset.set_epoch(epoch)
        order = np.random.default_rng(config["seed"] + epoch).permutation(len(dataset))
        totals, term_sums, term_counts = [], {}, {}
        rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss, terms = system.loss(dataset[int(index)])
            require(bool(torch.isfinite(loss)), "nonfinite training loss")
            loss.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in system.parameters()), "nonfinite training gradient")
            for group in optimizer.param_groups:
                terms["gradient/" + group["name"]] = float(torch.stack([p.grad.detach().square().sum() for p in group["params"] if p.grad is not None]).sum().sqrt())
            if recipe["gradient_clip_norm"]:
                torch.nn.utils.clip_grad_norm_(system.parameters(), recipe["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in system.parameters()), "nonfinite training parameters")
            totals.append(float(loss.detach()))
            for key, value in terms.items():
                term_sums[key] = term_sums.get(key, 0.) + value
                term_counts[key] = term_counts.get(key, 0) + 1
        mean_loss = float(np.mean(totals))
        improved = mean_loss < best
        best = min(best, mean_loss)
        validation, val_improved = None, False
        val_config = config["validation"]
        if val_config["enabled"] and ((epoch + 1) % val_config["every_epochs"] == 0 or epoch + 1 == config["epochs"]):
            from training.validation import validate_arch
            validation = validate_arch(system, dataset, val_config)
            write_json(output / f"validation_{epoch + 1:06d}.json", validation)
            if validation["score_mm"] is not None:
                val_improved = validation["score_mm"] < best_val
                best_val = min(best_val, validation["score_mm"])
        record = dict(epoch=epoch + 1, mean_train_loss=mean_loss, best_train_loss=best,
                      mean_terms={k: v / term_counts[k] for k, v in term_sums.items()}, term_case_counts=term_counts,
                      learning_rates=rates, validation=validation, best_val_score=None if best_val == float("inf") else best_val)
        scheduler.step()
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        import sys
        print(json.dumps({k: record[k] for k in ("epoch", "mean_train_loss", "best_train_loss", "best_val_score", "learning_rates")}), file=sys.stderr, flush=True)
        payload = dict(implementation_version=IMPLEMENTATION_VERSION, arch=config["arch"], specs=specs, epoch=epoch + 1,
                       best_train_loss=best, training_signature=signature, source_sha256=fingerprint,
                       manifest_sha256=manifest_hash, manifest_path=str(dataset.manifest_path), upstream_commit=commit,
                       canonical_reference_id=dataset.canonical_reference_id, sampling_domain_model=dataset.sampling_domain.tolist(),
                       case_rows=[c.metadata for c in dataset.cases], system_state=system.state_dict(),
                       optimizer_state=optimizer.state_dict(), rng_state=torch.get_rng_state())
        payload.update(scheduler_state=scheduler.state_dict(), best_val_score=best_val,
                       project_revision=project,
                       runtime_environment=runtime,
                       validation=validation, decoder_contract=decoder_contract(), resolved_config=config,
                       training_case_ids=[c.metadata["case_id"] for c in dataset.selected])
        if device.startswith("cuda"):
            payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        if improved:
            _save_checkpoint(output / "best_train.pth", payload)
        if val_improved:
            _save_checkpoint(output / "best_val.pth", payload)
        if (epoch + 1) % config["checkpoint_every"] == 0:
            _save_checkpoint(output / f"epoch_{epoch + 1:06d}.pth", payload)
        if epoch + 1 == config["epochs"]:
            _save_checkpoint(output / "final.pth", payload)
    result = dict(status="ARCH_TRAINING_COMPLETED_NOT_QUALITY_ACCEPTED", arch=config["arch"], epochs=config["epochs"],
                  cases=len(dataset), output_dir=str(output), best_train_loss=best,
                  best_val_score=None if best_val == float("inf") else best_val,
                  validation_enabled=config["validation"]["enabled"], formal_quality_accepted=False)
    write_json(output / "report.json", result)
    return result


def export_bundle(checkpoint, output_dir, model_id):
    """Only original present training latents enter the empirical prior."""
    require(isinstance(model_id, str) and bool(model_id.strip()), "nonempty model_id required")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved["implementation_version"] == IMPLEMENTATION_VERSION, "unsupported training checkpoint")
    require(saved["source_sha256"] == source_fingerprint(), "checkpoint/runtime source mismatch; export with the training source revision")
    model = DMM(saved["specs"], arch=saved["arch"])
    model_state = {k.removeprefix("model."): v for k, v in saved["system_state"].items() if k.startswith("model.")}
    model.load_state_dict(model_state, strict=True)
    require(all(bool(torch.isfinite(v).all()) for v in model_state.values()), "nonfinite checkpoint weights")
    rows = saved["case_rows"]
    trained = set(saved.get("training_case_ids", [r["case_id"] for r in rows if r["split"] == "train"]))
    original = [r for r in rows if r["case_id"] in trained and r["split"] == "train" and r["augmentation_parent_id"] is None]
    arrays = {}
    for label, dim in model.dimensions.items():
        indices = [r["embedding_row"] for r in original if label == 0 or r["presence"][str(label)]]
        require(len(indices) >= dim + 1, f"component {label}: needs at least {dim + 1} original present training cases")
        table = saved["system_state"][f"latents.{label}.weight"].numpy().astype(np.float64)
        z = table[indices]
        require(z.shape == (len(indices), dim) and np.isfinite(z).all(), "invalid embedding table")
        covariance = np.atleast_2d(np.cov(z, rowvar=False, ddof=1))
        variance = np.trace(covariance) / dim
        require(variance > 0 and np.all(np.diag(covariance) > 0), f"component {label}: zero latent variance")
        covariance = .95 * covariance + .05 * np.diag(np.diag(covariance))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.maximum(eigenvalues, 1e-6 * variance)) @ eigenvectors.T
        arrays.update({f"mu_{label}": z.mean(0), f"cov_{label}": covariance,
                       f"L_{label}": np.linalg.cholesky(covariance), f"count_{label}": np.asarray(len(indices), dtype=np.int64)})
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.save(model_state, output / "weights.pth")
    np.savez(output / "latent_statistics.npz", **arrays)
    path = output / "model.json"
    split_digest = hashlib.sha256(json.dumps([(r["case_id"], r["patient_id"], r["split"], r["augmentation_parent_id"], r["case_id"] in trained) for r in rows], separators=(",", ":")).encode()).hexdigest()
    metadata = stamped("arch_model_bundle", arch=saved["arch"], model_id=model_id, implementation_version=IMPLEMENTATION_VERSION,
                       specs=saved["specs"], component_dimensions={str(k): d for k, d in model.dimensions.items()},
                       model_unit_mm=MODEL_UNIT_MM, canonical_reference_id=saved["canonical_reference_id"],
                       sampling_domain_model=saved["sampling_domain_model"], gum_policy="fixed_training_mean",
                       upstream_commit=saved["upstream_commit"], local_source_sha256=saved["source_sha256"],
                       training_manifest_sha256=saved["manifest_sha256"], training_split_sha256=split_digest,
                       training_checkpoint_sha256=sha256(checkpoint), training_epoch=saved["epoch"],
                       training_case_ids=sorted(trained),
                       decoder_contract=saved.get("decoder_contract", decoder_contract()),
                       selection=dict(validation=saved.get("validation"), quality_accepted=False),
                       weights=reference(output / "weights.pth", path), latent_statistics=reference(output / "latent_statistics.npz", path),
                       statistics_binding=dict(weights_sha256=sha256(output / "weights.pth"), training_manifest_sha256=saved["manifest_sha256"],
                                               training_split_sha256=split_digest, original_only=True))
    write_json(path, metadata)
    return dict(status="ARCH_BUNDLE_EXPORTED_NOT_QUALITY_ACCEPTED", arch=saved["arch"], model=str(path))


def evaluate_checkpoint(config_path, checkpoint, output_dir, device="cpu"):
    from training.validation import validate_arch
    from dmm.provenance import validate_decoder_binding
    output = Path(output_dir).resolve()
    require(not output.exists(), "validation output already exists")
    config, dataset, centers, specs = load_training_config(config_path)
    require(config["validation"]["enabled"], "enable validation explicitly")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    require(saved["arch"] == config["arch"] and saved["specs"] == specs
            and saved["manifest_sha256"] == sha256(dataset.manifest_path), "checkpoint/data/spec mismatch")
    require(saved["canonical_reference_id"] == dataset.canonical_reference_id,
            "checkpoint canonical reference mismatch")
    selected = {c.metadata["case_id"] for c in dataset.selected}
    trained = set(saved.get("training_case_ids", [r["case_id"] for r in saved["case_rows"] if r["split"] == "train"]))
    require(selected == trained, "checkpoint training subset mismatch")
    if "resolved_config" in saved:
        require(config["recipe"] == saved["resolved_config"]["recipe"], "checkpoint training recipe mismatch")
    binding = dict(local_source_sha256=saved["source_sha256"])
    if "decoder_contract" in saved: binding["decoder_contract"] = saved["decoder_contract"]
    validate_decoder_binding(binding)
    system = ArchTrainingSystem(DMM(specs, arch=config["arch"]), dataset, centers, config["recipe"]).to(device)
    system.load_state_dict(saved["system_state"], strict=True)
    result = validate_arch(system, dataset, config["validation"])
    result.update(checkpoint_sha256=sha256(checkpoint), training_epoch=saved["epoch"],
                  training_source_sha256=saved["source_sha256"], evaluation_source_sha256=source_fingerprint(),
                  config_sha256=sha256(config_path), resolved_recipe=config["recipe"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"validation.json", result)
    return result
