"""Strict inference bundle loading, without subject training embeddings."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dmm import MODEL_UNIT_MM
from networks.dmm_net import DMM
from dmm import IMPLEMENTATION_VERSION
from dmm.provenance import validate_decoder_binding
from dmm.validation import array, domain, header, nonempty, read_json, require, resolve_ref, sha256


@dataclass
class ModelBundle:
    model: DMM
    metadata: dict
    path: Path
    means: dict
    cholesky: dict

    def codes(self, presence, q):
        labels = (0,) + tuple(k for k, v in presence.items() if v)
        require(set(q) == set(labels) - {0}, "q must contain exactly present teeth (gum is fixed)")
        result = {0: self.means[0]}
        for label in labels[1:]:
            vector = q[label]
            require(vector.shape == self.means[label].shape and bool(torch.isfinite(vector).all()), f"invalid q for {label}")
            result[label] = self.means[label] + self.cholesky[label] @ vector
        return result


def load_bundle(path, arch, device="cpu", allowed_roots=None):
    path = Path(path).resolve()
    data = read_json(path)
    header(data, "arch_model_bundle")
    require(data["arch"] == arch and data["implementation_version"] == IMPLEMENTATION_VERSION, "bundle arch/implementation mismatch")
    require(data["model_unit_mm"] == MODEL_UNIT_MM and data["gum_policy"] == "fixed_training_mean", "bundle scale/gum policy mismatch")
    for key in ("model_id", "canonical_reference_id", "upstream_commit", "local_source_sha256",
                "training_manifest_sha256", "training_split_sha256"):
        nonempty(data[key], key)
    validate_decoder_binding(data)
    domain(data["sampling_domain_model"])
    weights = resolve_ref(path, data["weights"], allowed_roots)
    statistics = resolve_ref(path, data["latent_statistics"], allowed_roots)
    model = DMM(data["specs"], arch=arch).to(device)
    require(data["component_dimensions"] == {str(k): d for k, d in model.dimensions.items()}, "bundle dimension metadata mismatch")
    state = torch.load(weights, map_location=device, weights_only=True)
    require(isinstance(state, dict) and all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all()) for v in state.values()), "weights must be finite model tensors only")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    means, cholesky = {}, {}
    with np.load(statistics, allow_pickle=False) as source:
        expected = {f"{prefix}_{k}" for k in model.labels for prefix in ("mu", "cov", "L", "count")}
        require(set(source.files) == expected, "unexpected latent statistics schema")
        for k, dim in model.dimensions.items():
            mean = array(source[f"mu_{k}"], (dim,), f"mu {k}")
            cov = array(source[f"cov_{k}"], (dim, dim), f"cov {k}")
            factor = array(source[f"L_{k}"], (dim, dim), f"L {k}")
            count = source[f"count_{k}"]
            require(count.shape == () and np.issubdtype(count.dtype, np.integer) and int(count) >= dim + 1, f"insufficient original training support for {k}")
            require(np.all(np.diag(factor) > 0) and np.allclose(factor, np.tril(factor), rtol=0, atol=1e-10)
                    and np.allclose(factor @ factor.T, cov, rtol=1e-5, atol=1e-10), f"invalid covariance factor {k}")
            means[k] = torch.tensor(mean, dtype=torch.float32, device=device)
            cholesky[k] = torch.tensor(factor, dtype=torch.float32, device=device)
            require(bool(torch.isfinite(means[k]).all()) and bool(torch.isfinite(cholesky[k]).all())
                    and bool((torch.diag(cholesky[k]) > 0).all()), "statistics cannot be represented in float32")
    binding = data["statistics_binding"]
    require(binding == dict(weights_sha256=sha256(weights), training_manifest_sha256=data["training_manifest_sha256"],
                            training_split_sha256=data["training_split_sha256"], original_only=True), "statistics provenance binding mismatch")
    return ModelBundle(model, data, path, means, cholesky)
