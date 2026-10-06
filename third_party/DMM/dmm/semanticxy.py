"""SemanticXY v1: area-preserving sampling, partial soft OT and frozen Warp loss.

This module consumes renderer outputs; it does not implement a renderer or the
implicit-surface gradient bridge. See docs/semanticxy_matching_v1.md in the host
project for the mathematical contract and differences from ICCV 2025 Eq. (7).
"""
from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np
import torch

from dmm import CHANNEL_FDI
from dmm.validation import header, read_json, require, stamped
from dmm.pixels import image_center, pixel_offset


@dataclass(frozen=True)
class SemanticXYConfig:
    semantic_weight: float = 4.0
    xy_weight: float = 1.0
    unmatched_cost: float = 0.75
    entropy_epsilon: float = 0.05
    epsilon_start: float = 0.4
    epsilon_scaling_steps: int = 4
    warmup_iterations: int = 100
    newton_steps: int = 50
    max_newton_variables: int = 512
    samples_per_tooth: int = 16
    sampling_seed: int = 42
    max_iterations: int = 2000
    marginal_tolerance: float = 1e-6
    check_every: int = 10
    max_pair_entries: int = 2_000_000
    warp_loss_weight: float = 1.0
    unmatched_loss_weight: float = 1.0
    dense_semantic_loss_weight: float = 1.0
    low_match_fraction: float = 0.01

    def __post_init__(self):
        for name in ("semantic_weight", "xy_weight", "unmatched_cost", "entropy_epsilon", "epsilon_start",
                     "marginal_tolerance", "warp_loss_weight", "unmatched_loss_weight", "dense_semantic_loss_weight"):
            value = getattr(self, name)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0, f"{name} must be finite and positive")
        for name in ("samples_per_tooth", "max_iterations", "check_every", "max_pair_entries", "epsilon_scaling_steps", "warmup_iterations", "newton_steps", "max_newton_variables"):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0, f"{name} must be a positive integer")
        require(type(self.sampling_seed) is int and self.sampling_seed >= 0, "sampling_seed must be a nonnegative integer")
        require(type(self.low_match_fraction) in (int, float) and math.isfinite(self.low_match_fraction)
                and 0 <= self.low_match_fraction < 1, "invalid low_match_fraction")
        require(self.check_every <= self.max_iterations, "check_every exceeds max_iterations")
        require(self.epsilon_start >= self.entropy_epsilon, "epsilon_start must be at least entropy_epsilon")

    def resolved(self):
        return stamped("semanticxy_config", algorithm="partial_soft_semanticxy_v1", **asdict(self))

    @classmethod
    def load(cls, path):
        data = read_json(path)
        header(data, "semanticxy_config")
        require(data.get("algorithm") == "partial_soft_semanticxy_v1", "unsupported SemanticXY algorithm")
        fields = {k: v for k, v in data.items() if k not in ("contract_id", "version", "artifact_kind", "representation_profile", "algorithm")}
        require(set(fields) == set(cls.__dataclass_fields__), "SemanticXY config must resolve every field and reject unknown fields")
        return cls(**fields)


def normalize_xy(xy_pixels, width, height, metadata=None):
    require(type(width) is int and type(height) is int and width > 0 and height > 0, "positive integer image dimensions required")
    require(xy_pixels.shape[-1] == 2, "XY must end in two coordinates")
    center = xy_pixels.new_tensor(image_center(width, height, metadata or {}))
    return (xy_pixels - center) / math.hypot(width, height)


@dataclass
class RenderedSemanticXY:
    semantics: torch.Tensor  # H x W x 29, background then fixed CHANNEL_FDI
    xy_pixels: torch.Tensor  # H x W x 2, unpremultiplied projected geometry attribute
    xy_source: str = "projected_geometry"


@dataclass
class PointMeasure:
    semantic: torch.Tensor  # N x 28, conditional tooth probabilities
    xy: torch.Tensor        # N x 2, normalized geometric XY
    mass: torch.Tensor      # N, foreground area / number of valid pixels
    pixel_indices: torch.Tensor
    represented_pixels: int


@dataclass
class TransportPlan:
    real: torch.Tensor
    source_unmatched: torch.Tensor
    target_unmatched: torch.Tensor
    iterations: int
    marginal_error: float
    cost: float


@dataclass
class SemanticXYResult:
    loss: torch.Tensor
    warp_loss: torch.Tensor
    unmatched_loss: torch.Tensor
    dense_semantic_loss: torch.Tensor
    diagnostics: dict
    source: PointMeasure
    target: PointMeasure
    plan: TransportPlan


def _offset(seed, view_id, side, channel):
    digest = hashlib.sha256(f"{seed}:{view_id}:{side}:{channel}".encode()).digest()
    # A fixed systematic-resampling offset, independent of iteration and RNG state.
    return (int.from_bytes(digest[:8], "big") + .5) / 2**64


def sample_measure(semantics, xy, valid, view_id, side, config, metadata=None):
    """Stratify by detached dominant FDI; keep soft features and true area masses.

    Small strata keep all pixels and their exact alpha. Larger strata use weighted
    systematic resampling; duplicated indices are permitted. Each of their n
    representatives carries the FULL stratum mass / n, never unit point mass.
    Discrete membership and sampling are frozen for this forward/backward step.
    """
    height, width = valid.shape
    valid_count = int(valid.sum())
    require(valid_count > 0, "view has no valid pixels")
    tooth = semantics[..., 1:].reshape(-1, 28)
    alpha = tooth.sum(-1)
    candidate = valid.reshape(-1) & (alpha.detach() > 0)
    bins = tooth.detach().argmax(-1)
    chosen, masses = [], []
    for channel in range(28):
        members = torch.nonzero(candidate & (bins == channel), as_tuple=True)[0]
        if not len(members):
            continue
        n = min(config.samples_per_tooth, len(members))
        if len(members) <= n:
            indices = members
            mass = alpha[indices] / valid_count
        else:
            importance = alpha[members].detach().double()
            quantiles = (torch.arange(n, dtype=torch.float64, device=alpha.device)
                         + _offset(config.sampling_seed, view_id, side, channel)) * (importance.sum() / n)
            selected = torch.searchsorted(importance.cumsum(0), quantiles).clamp_max(len(members) - 1)
            indices = members[selected]
            mass = (alpha[members].sum() / (n * valid_count)).expand(n)
        chosen.append(indices)
        masses.append(mass)
    indices = torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long, device=semantics.device)
    mass = torch.cat(masses) if masses else alpha[indices]
    total = alpha[indices, None]
    conditional = tooth[indices] / torch.where(total > 0, total, torch.ones_like(total))
    return PointMeasure(conditional, normalize_xy(xy.reshape(-1, 2)[indices], width, height, metadata),
                        mass, indices, int(candidate.sum()))


def _newton_columns(log_kernel, rows, cols, log_v, steps, tolerance):
    """Damped Newton on the column semi-dual, fixing the last potential to zero.

    Rows remain exact via softmax. This removes the slow mass-exchange mode of
    nearly disconnected Sinkhorn blocks without relaxing the requested residual.
    The Hessian has one variable per target sample (<=448 with default quotas).
    """
    potential = log_v - log_v[-1]
    used = 0
    for _ in range(steps):
        probability = torch.softmax(log_kernel + potential[None], dim=1)
        pi = rows[:, None] * probability
        residual = pi.sum(0) - cols
        if float(residual.abs().sum()) <= tolerance:
            break
        p = probability[:, :-1]
        hessian = torch.diag(pi.sum(0)[:-1]) - p.T @ (rows[:, None] * p)
        # Numerical damping only; convergence is checked against original marginals.
        hessian += torch.diag(cols[:-1] * 1e-10 + 1e-15)
        direction = torch.linalg.solve(hessian, residual[:-1])
        require(bool(torch.isfinite(direction).all()), "nonfinite OT Newton direction")
        direction *= min(1., 20. / max(float(direction.abs().max()), 1e-300))
        objective = (rows * torch.logsumexp(log_kernel + potential[None], dim=1)).sum() - (cols * potential).sum()
        slope = (residual[:-1] * direction).sum()
        step, accepted = 1., False
        for _ in range(30):
            candidate = torch.cat((potential[:-1] - step * direction, potential[-1:]))
            candidate_objective = (rows * torch.logsumexp(log_kernel + candidate[None], dim=1)).sum() - (cols * candidate).sum()
            if bool(candidate_objective <= objective - 1e-4 * step * slope + 1e-14):
                potential = candidate
                accepted = True
                break
            step *= .5
        used += 1
        if not accepted:
            break
    pi = rows[:, None] * torch.softmax(log_kernel + potential[None], dim=1)
    error = float(torch.maximum((pi.sum(1) - rows).abs().sum(), (pi.sum(0) - cols).abs().sum()))
    return pi, used, error


@torch.no_grad()
def partial_transport(source, target, config):
    """Log-domain balanced Sinkhorn on a partial-transport dustbin extension.

    Real capacities are a,b. A=sum(a), B=sum(b), reserve R=A+B. Augmented
    marginals are [a,B+R] and [b,A+R], with total 2(A+B). The reserve prevents
    the dustbin/dustbin entry from vanishing when all real mass is unmatched.
    """
    a, b = source.mass.detach().double(), target.mass.detach().double()
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            and bool((a > 0).all()) and bool((b > 0).all()), "OT masses must be finite and strictly positive")
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return TransportPlan(a.new_zeros(n, m), a, b, 0, 0., float(config.unmatched_cost * (a.sum() + b.sum())))
    require((n + 1) * (m + 1) <= config.max_pair_entries, "SemanticXY pair budget exceeded; lower sampling quota explicitly")
    costs = config.semantic_weight * .5 * torch.cdist(source.semantic.detach().double(), target.semantic.detach().double(), p=1)
    costs += config.xy_weight * torch.cdist(source.xy.detach().double(), target.xy.detach().double(), p=1)
    augmented = costs.new_full((n + 1, m + 1), config.unmatched_cost)
    augmented[:n, :m] = costs
    augmented[-1, -1] = 0
    reserve = a.sum() + b.sum()
    total = 2 * reserve
    rows = torch.cat((a, (b.sum() + reserve)[None])) / total
    cols = torch.cat((b, (a.sum() + reserve)[None])) / total
    log_u, log_v = torch.zeros_like(rows), torch.zeros_like(cols)
    error = float("inf")
    iteration = 0
    schedule = (np.geomspace(config.epsilon_start, config.entropy_epsilon, config.epsilon_scaling_steps).tolist()
                if config.epsilon_scaling_steps > 1 else [config.entropy_epsilon])
    previous_epsilon = schedule[0]
    for stage, epsilon in enumerate(schedule):
        # Warm-start cost-unit dual potentials, not raw log scaling factors.
        log_u *= previous_epsilon / epsilon
        log_v *= previous_epsilon / epsilon
        previous_epsilon = epsilon
        log_kernel = -augmented / epsilon
        budget = config.max_iterations - iteration
        budget = min(budget, config.warmup_iterations)
        for local_iteration in range(1, budget + 1):
            iteration += 1
            log_u = rows.log() - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
            log_v = cols.log() - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
            if local_iteration % config.check_every == 0 or local_iteration == budget:
                pi = torch.exp(log_kernel + log_u[:, None] + log_v[None, :])
                error = float(torch.maximum((pi.sum(1) - rows).abs().sum(), (pi.sum(0) - cols).abs().sum()))
                if error <= config.marginal_tolerance:
                    break
        if iteration == config.max_iterations:
            break
    if stage == len(schedule) - 1 and error > config.marginal_tolerance:
        require(m <= config.max_newton_variables, "OT Newton variable budget exceeded; lower sampling quota or explicitly increase budget")
        pi, newton_iterations, error = _newton_columns(log_kernel, rows, cols, log_v,
                                                      min(config.newton_steps, config.max_iterations - iteration),
                                                      config.marginal_tolerance)
        iteration += newton_iterations
    require(stage == len(schedule) - 1 and math.isfinite(error) and error <= config.marginal_tolerance,
            f"SemanticXY OT did not converge: iterations={iteration}, marginal_error={error:.3g}; do not skip this view")
    pi = pi * total
    return TransportPlan(pi[:n, :m], pi[:n, -1], pi[-1, :m], iteration, error,
                         float((pi * augmented).sum()))


def warp_loss_from_plan(source, target, plan, config):
    """Symmetric barycentric L1 Warp surrogate with a FROZEN transport plan.

    Target features are constants; source geometric XY/soft semantics keep their
    gradient. Returns warp and capacity-deficit losses. This is intentionally an
    alternating correspondence objective, not differentiation through Sinkhorn.
    """
    pi = plan.real.to(device=source.xy.device, dtype=source.xy.dtype).detach()
    require(pi.shape == (len(source.mass), len(target.mass)), "plan/measure shape mismatch")
    target_sem, target_xy = target.semantic.detach(), target.xy.detach()
    row, col = pi.sum(1), pi.sum(0)
    zero = source.xy.sum() * 0 + source.semantic.sum() * 0 + source.mass.sum() * 0
    if pi.numel():
        safe_row = torch.where(row > 0, row, torch.ones_like(row))
        safe_col = torch.where(col > 0, col, torch.ones_like(col))
        warped_target_sem = (pi @ target_sem) / safe_row[:, None]
        warped_target_xy = (pi @ target_xy) / safe_row[:, None]
        warped_source_sem = (pi.T @ source.semantic) / safe_col[:, None]
        warped_source_xy = (pi.T @ source.xy) / safe_col[:, None]
        forward = config.semantic_weight * .5 * (source.semantic - warped_target_sem).abs().sum(-1)
        forward += config.xy_weight * (source.xy - warped_target_xy).abs().sum(-1)
        reverse = config.semantic_weight * .5 * (warped_source_sem - target_sem).abs().sum(-1)
        reverse += config.xy_weight * (warped_source_xy - target_xy).abs().sum(-1)
        warp = .5 * ((row * forward).sum() + (col * reverse).sum())
    else:
        warp = zero
    # At the correspondence step this equals the dustbin cost, up to solver
    # tolerance. Live source capacities retain a foreground-opacity gradient.
    unmatched = config.unmatched_cost * ((source.mass - row).clamp_min(0).sum()
                                         + (target.mass.detach() - col).clamp_min(0).sum())
    return warp, unmatched + zero


class SemanticXYMatcher:
    def __init__(self, config=None):
        self.config = SemanticXYConfig() if config is None else config

    def match_view(self, rendered, observation, mode="fit"):
        require(mode in ("fit", "evaluate"), "mode must be fit or evaluate")
        pred, xy = rendered.semantics, rendered.xy_pixels
        require(pred.dtype in (torch.float32, torch.float64) and xy.dtype == pred.dtype
                and pred.device == xy.device, "renderer semantics/XY need matching float32/64 dtype and device")
        require(isinstance(observation.view_id, str) and bool(observation.view_id), "nonempty view_id required")
        labels_np = np.asarray(observation.labels)
        valid_np = np.asarray(observation.valid)
        require(labels_np.ndim == 2 and valid_np.shape == labels_np.shape
                and valid_np.dtype == np.bool_ and labels_np.dtype == np.uint8, "observation needs uint8 labels and Boolean valid")
        require(np.array_equal(labels_np == 255, ~valid_np), "valid/ignore mismatch")
        require(set(np.unique(labels_np)).issubset({0, 255, *CHANNEL_FDI}), "unknown observation FDI")
        require(pred.shape == (*labels_np.shape, 29) and xy.shape == (*labels_np.shape, 2), "renderer/observation shape mismatch")
        valid = torch.tensor(valid_np, dtype=torch.bool, device=pred.device)
        require(bool(valid.any()), "view has no valid pixels; fix input instead of silently dropping it")
        values = pred[valid]
        require(bool(torch.isfinite(values).all()) and bool((values >= 0).all())
                and bool((values <= 1).all()) and bool(torch.allclose(values.sum(-1), torch.ones_like(values[:, 0]), atol=1e-5, rtol=0)),
                "valid-pixel semantic probabilities must be finite, nonnegative and sum to one")
        foreground = valid & (pred[..., 1:].detach().sum(-1) > 0)
        require(bool(torch.isfinite(xy[foreground]).all()), "nonfinite projected foreground XY")
        if mode == "fit":
            require(rendered.xy_source == "projected_geometry" and xy.requires_grad,
                    "fit needs differentiable projected-geometry XY, not a constant pixel grid")
        else:
            require(rendered.xy_source in ("projected_geometry", "fixed_grid_diagnostic"), "unknown XY provenance")
        labels = torch.tensor(labels_np, device=pred.device)
        target_sem = torch.stack([(labels == k) & valid for k in (0, *CHANNEL_FDI)], -1).to(pred.dtype)
        height, width = labels_np.shape
        y, x = torch.meshgrid(torch.arange(height, device=pred.device), torch.arange(width, device=pred.device), indexing="ij")
        metadata = getattr(observation, "metadata", {})
        target_xy = torch.stack((x, y), -1).to(pred.dtype) + pixel_offset(metadata)
        source = sample_measure(pred, xy, valid, observation.view_id, "prediction", self.config, metadata)
        target = sample_measure(target_sem, target_xy, valid, observation.view_id, "target", self.config, metadata)
        plan = partial_transport(source, target, self.config)
        warp, unmatched = warp_loss_from_plan(source, target, plan, self.config)
        # Exact full-valid-image loss. No division by the number of channels.
        dense = .5 * (pred[valid] - target_sem[valid]).abs().sum() / valid.sum()
        loss = (self.config.warp_loss_weight * warp + self.config.unmatched_loss_weight * unmatched
                + self.config.dense_semantic_loss_weight * dense)
        source_mass, target_mass = float(source.mass.detach().sum()), float(target.mass.sum())
        matched = float(plan.real.sum())
        if not len(source.mass) and len(target.mass):
            status = "EMPTY_PREDICTION_REQUIRES_VISIBILITY_RECOVERY"
        elif len(source.mass) and not len(target.mass):
            status = "EMPTY_TARGET_NEGATIVE_EVIDENCE"
        elif not len(source.mass) and not len(target.mass):
            status = "BOTH_EMPTY_VALID_BACKGROUND"
        elif matched / max(min(source_mass, target_mass), 1e-300) < self.config.low_match_fraction:
            status = "LOW_MATCH_MASS"
        else:
            status = "MATCHED"
        diagnostics = dict(view_id=observation.view_id, status=status, mode=mode,
                           xy_source=rendered.xy_source, source_represented_pixels=source.represented_pixels,
                           target_represented_pixels=target.represented_pixels,
                           valid_pixels=int(valid.sum()), source_samples=len(source.mass), target_samples=len(target.mass),
                           source_unique_samples=len(torch.unique(source.pixel_indices)), target_unique_samples=len(torch.unique(target.pixel_indices)),
                           source_area_fraction=source_mass, target_area_fraction=target_mass, matched_mass=matched,
                           source_unmatched_mass=float(plan.source_unmatched.sum()), target_unmatched_mass=float(plan.target_unmatched.sum()),
                           source_unmatched_fraction=float(plan.source_unmatched.sum()) / source_mass if source_mass else 0.,
                           target_unmatched_fraction=float(plan.target_unmatched.sum()) / target_mass if target_mass else 0.,
                           ot_iterations=plan.iterations, ot_marginal_error=plan.marginal_error,
                           matching_cost=plan.cost, warp_loss=float(warp.detach()), unmatched_loss=float(unmatched.detach()),
                           dense_semantic_loss=float(dense.detach()), loss=float(loss.detach()),
                           needs_visibility_recovery=status == "EMPTY_PREDICTION_REQUIRES_VISIBILITY_RECOVERY")
        require(bool(torch.isfinite(loss)), "nonfinite SemanticXY loss")
        return SemanticXYResult(loss, warp, unmatched, dense, diagnostics, source, target, plan)

    def match_scene(self, rendered_by_view, observations, mode="fit"):
        """Use every supplied observation once; equal view mean, including negatives."""
        identifiers = [view.view_id for view in observations]
        require(bool(identifiers) and len(identifiers) == len(set(identifiers)), "nonempty unique view list required")
        require(set(rendered_by_view) == set(identifiers), "rendered views must match ALL observations exactly")
        results = {view.view_id: self.match_view(rendered_by_view[view.view_id], view, mode) for view in observations}
        return dict(loss=torch.stack([r.loss for r in results.values()]).mean(), views=results,
                    needs_visibility_recovery=[key for key, value in results.items() if value.diagnostics["needs_visibility_recovery"]],
                    resolved_config=self.config.resolved())
