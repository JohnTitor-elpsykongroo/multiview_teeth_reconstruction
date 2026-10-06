"""Native staged image fitting: fixed cameras, frozen independent arch decoders."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch

from dmm import CHANNEL_FDI
from dmm.provenance import source_fingerprint
from dmm.rendering import RenderConfig
from dmm.surface import SurfaceConfig
from dmm.semanticxy import SemanticXYConfig, SemanticXYMatcher
from dmm.recovery import RecoveryConfig, missing_regions, recovery_loss, bootstrap_translation
from dmm.collision import CollisionConfig, collision_loss
from dmm.render_io import write_render_artifacts
from dmm.validation import ContractError, header, read_json, require, stamped, write_json


@dataclass(frozen=True)
class FitConfig:
    pose_iterations: int = 40
    shape_iterations: int = 40
    joint_iterations: int = 80
    rotation_step_rad: float = .01
    translation_step_mm: float = 1.
    latent_step: float = .1
    backtracks: int = 8
    prior_weight: float = .001
    recovery_weight: float = .2
    max_q_rms: float = 4.
    loss_tolerance: float = 1e-5
    gradient_tolerance: float = 1e-6
    patience: int = 5
    min_tooth_iou: float = .85
    diagnostic_reference: bool = False
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    def __post_init__(self):
        for name in ('pose_iterations', 'shape_iterations', 'joint_iterations'):
            require(type(getattr(self, name)) is int and getattr(self, name) >= 0, f'invalid {name}')
        require(self.pose_iterations + self.shape_iterations + self.joint_iterations > 0, 'empty fitting schedule')
        for name in ('backtracks', 'patience'):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0, f'invalid {name}')
        for name in ('rotation_step_rad', 'translation_step_mm', 'latent_step', 'prior_weight', 'recovery_weight', 'max_q_rms', 'loss_tolerance', 'gradient_tolerance'):
            require(math.isfinite(getattr(self, name)) and getattr(self, name) > 0, f'invalid {name}')
        require(0 < self.min_tooth_iou <= 1 and type(self.diagnostic_reference) is bool, 'invalid fitting gate')

    def resolved(self): return stamped('fit_config', algorithm='staged_backtracking_v1', **asdict(self))

    @classmethod
    def load(cls, path):
        data = read_json(path)
        header(data, 'fit_config')
        require(data.pop('algorithm') == 'staged_backtracking_v1', 'unknown fitting algorithm')
        for key in ('contract_id', 'version', 'artifact_kind', 'representation_profile'): data.pop(key)
        require(set(data) == set(cls.__dataclass_fields__), 'fit config must be fully resolved')
        require(set(data['recovery']) == set(RecoveryConfig.__dataclass_fields__), 'recovery config must be fully resolved')
        data['recovery'] = RecoveryConfig(**data['recovery'])
        return cls(**data)


def parameter_state(scene):
    return {name: dict(base_pose=state.base_pose.detach().cpu().tolist(),
                       pose_delta=state.pose_delta.detach().cpu().tolist(),
                       T_world_from_arch=state.T_world_from_arch.detach().cpu().tolist(),
                       q={k: value.detach().cpu().tolist() for k, value in state.q.items()})
            for name, state in scene.arches.items()}


@torch.no_grad()
def restore_parameters(scene, saved):
    require(set(saved) == set(scene.arches), 'checkpoint arch mismatch')
    for name, state in scene.arches.items():
        row = saved[name]
        require(set(row['q']) == set(state.q), 'checkpoint presence mismatch')
        for parameter, value in [(state.base_pose, row['base_pose']), (state.pose_delta, row['pose_delta']),
                                 *[(p, row['q'][k]) for k, p in state.q.items()]]:
            array = parameter.new_tensor(value)
            require(array.shape == parameter.shape and bool(torch.isfinite(array).all()), 'invalid checkpoint parameters')
            parameter.copy_(array)


def _input_signature(scene):
    digest = hashlib.sha256(json.dumps(scene.manifest, sort_keys=True).encode())
    for obs in scene.observations:
        digest.update(obs.view_id.encode())
        from dmm.pixels import pixel_offset
        digest.update(str(pixel_offset(obs.metadata)).encode())
        for array in (obs.K, obs.T_camera_from_world, obs.labels, obs.valid): digest.update(np.asarray(array).tobytes())
    # Include actual decoder/statistical-prior tensors for in-memory fixtures too.
    for name, state in scene.arches.items():
        digest.update(name.encode())
        for key, tensor in state.state_dict().items():
            if key in ('base_pose', 'pose_delta') or key.startswith('q.'): continue
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _metrics(scene, rendered):
    rows = []
    fdi = np.array([0, *CHANNEL_FDI], np.uint8)
    for obs in scene.observations:
        pred = fdi[rendered[obs.view_id].semantics.detach().argmax(-1).cpu().numpy()]
        # Include false-positive tooth regions in valid background views.
        for label in sorted((set(np.unique(pred[obs.valid])) | set(obs.visible_fdi)) - {0}):
            a, b = (pred == label) & obs.valid, (obs.labels == label) & obs.valid
            rows.append(dict(view_id=obs.view_id, label=int(label), iou=float((a & b).sum() / (a | b).sum())))
    return dict(per_tooth_view=rows, min_tooth_iou=min((r['iou'] for r in rows), default=0.))


def _direction(name, parameter, config):
    grad = parameter.grad
    if grad is None: return torch.zeros_like(parameter)
    require(bool(torch.isfinite(grad).all()), f'nonfinite gradient: {name}')
    def unit(g, step): return -step * g / g.norm().clamp_min(config.gradient_tolerance)
    if name.endswith('pose_delta'):
        return torch.cat((unit(grad[:3], config.rotation_step_rad), unit(grad[3:], config.translation_step_mm)))
    return unit(grad, config.latent_step)


def fit_scene(scene, output, config=None, surface_config=None, render_config=None, matching_config=None, resume=None, collision_config=None):
    config = FitConfig() if config is None else config
    surface_config = SurfaceConfig() if surface_config is None else surface_config
    render_config = RenderConfig(supersample=4) if render_config is None else render_config
    matcher = SemanticXYMatcher(matching_config)
    collision_config = collision_config or CollisionConfig()
    require(render_config.backend == 'nvdiffrast' or config.diagnostic_reference, 'image fitting requires CUDA antialiasing; reference is diagnostic only')
    runtime = {'backend': render_config.backend}
    if render_config.backend == 'nvdiffrast':
        from dmm.gpu_validation import validate_cuda_backend
        runtime.update(validate_cuda_backend(next(iter(scene.arches.values())).base_pose.device))
    output = Path(output)
    require(not output.exists(), 'fit output must be a fresh directory')
    for state in scene.arches.values():
        state.decoder.eval().requires_grad_(False)
    visible = set().union(*(set(obs.visible_fdi) for obs in scene.observations))
    require(bool(visible), 'no observed teeth: cannot fit a prior from background only')
    frozen_unobserved = {name: [int(k) for k in state.q if int(k) not in visible] for name, state in scene.arches.items()}
    settings = dict(fit=config.resolved(), surface=surface_config.resolved(), render=render_config.resolved(), matching=matcher.config.resolved())
    settings['collision'] = collision_config.resolved()
    identity = dict(source_sha256=source_fingerprint(), input_sha256=_input_signature(scene), settings=settings,
                    runtime={k: v for k, v in runtime.items() if k not in ('gradient', 'finite_difference')})
    history, bootstrap_history, start_stage, start_iteration, stable = [], [], 0, 0, 0
    final_stage_converged = False
    if resume:
        checkpoint = read_json(resume)
        require(checkpoint['identity'] == identity, 'checkpoint source/input/config mismatch')
        require(checkpoint['status'] == 'RUNNING', 'only an unfinished fitting checkpoint can resume')
        restore_parameters(scene, checkpoint['parameters'])
        history, bootstrap_history = checkpoint['history'], checkpoint['bootstrap']
        start_stage, start_iteration, stable = checkpoint['next_stage'], checkpoint['next_iteration'], checkpoint['stable']
        final_stage_converged = checkpoint['converged']
        require(0 <= start_stage <= 3 and 0 <= start_iteration and 0 <= stable, 'invalid checkpoint cursor')
    output.mkdir(parents=True)
    write_json(output / 'settings.json', identity)
    write_json(output / 'starting_parameters.json', parameter_state(scene))

    def evaluate(charts=None, active=None, recovery_charts=None):
        rendered, mesh, charts = scene.render_views(surface_config, render_config, charts)
        match = matcher.match_scene(rendered, scene.observations, mode='evaluate')
        current = missing_regions(scene, rendered, config.recovery)
        recovery, recovery_charts = recovery_loss(scene, rendered, current if active is None else active,
                          surface_config, render_config, config.recovery, recovery_charts)
        prior = scene.latent_prior()
        collision, collision_report = collision_loss(scene, mesh, collision_config)
        loss = match['loss'] + config.prior_weight * prior + config.recovery_weight * recovery + collision_config.weight * collision
        require(bool(torch.isfinite(loss)), 'nonfinite fitting objective')
        return dict(loss=loss, image=match['loss'], recovery=recovery, prior=prior, missing=current,
                    rendered=rendered, mesh=mesh, charts=charts, recovery_charts=recovery_charts, match=match,
                    collision=collision, collision_report=collision_report)

    def checkpoint(stage, iteration, status='RUNNING'):
        data = dict(identity=identity, status=status, parameters=parameter_state(scene), history=history,
                    bootstrap=bootstrap_history, next_stage=stage, next_iteration=iteration, stable=stable, converged=final_stage_converged)
        temporary = output / 'checkpoint.tmp.json'
        write_json(temporary, data)
        temporary.replace(output / 'checkpoint.json')

    stages = [('pose', config.pose_iterations), ('shape', config.shape_iterations), ('joint', config.joint_iterations)]
    try:
        with torch.no_grad(): initial = evaluate()
        write_render_artifacts(output / 'initial_render', scene, initial['rendered'], initial['mesh'])
        initial_metrics = _metrics(scene, initial['rendered'])
        if not resume and config.pose_iterations:
            for arch in scene.arches:
                before = parameter_state(scene)
                pose, detail = bootstrap_translation(scene, arch, initial['missing'], surface_config, config.recovery)
                if pose is not None:
                    with torch.no_grad():
                        scene.arches[arch].base_pose.copy_(pose)
                        scene.arches[arch].pose_delta.zero_()
                        trial = evaluate(active=initial['missing'])
                    if float(trial['loss']) < float(initial['loss']):
                        detail['status'] = 'ACCEPTED_BY_FULL_RENDER'
                        with torch.no_grad(): initial = evaluate()
                    else:
                        restore_parameters(scene, before)
                        detail['status'] = 'REJECTED_BY_FULL_RENDER'
                bootstrap_history.append(detail)
        checkpoint(start_stage, start_iteration)
        for stage_index, (stage, budget) in enumerate(stages):
            if stage_index < start_stage: continue
            if stage_index != start_stage: stable = 0
            begin = start_iteration if stage_index == start_stage else 0
            for iteration in range(begin, budget):
                block = stage if stage != 'joint' else ('pose' if iteration % 2 == 0 else 'shape')
                for state in scene.arches.values():
                    state.pose_delta.requires_grad_(block == 'pose')
                    for key, value in state.q.items(): value.requires_grad_(block == 'shape' and int(key) in visible)
                scene.zero_grad(set_to_none=True)
                baseline = evaluate()
                params = [(name, value) for name, value in scene.named_parameters() if value.requires_grad]
                require(params and baseline['loss'].requires_grad, 'active fitting block has no gradient path')
                baseline['loss'].backward()
                directions = {name: _direction(name, value, config) for name, value in params}
                saved = {name: value.detach().clone() for name, value in params}
                old = float(baseline['loss'].detach())
                grad_norm = max((float(p.grad.norm()) for _, p in params if p.grad is not None), default=0.)
                accepted, errors, new = False, [], old
                scale = 0.
                for attempt in range(config.backtracks):
                    scale = .5**attempt
                    with torch.no_grad():
                        for name, value in params: value.copy_(saved[name] + scale * directions[name])
                    try:
                        require(all(float(q.detach().square().mean().sqrt()) <= config.max_q_rms for state in scene.arches.values() for q in state.q.values()), 'latent trust bound')
                        with torch.no_grad():
                            trial = evaluate(baseline['charts'], baseline['missing'], baseline['recovery_charts'])
                        new = float(trial['loss'])
                        if new < old and block == 'shape':
                            # A shape update is accepted only after fresh marching
                            # cubes and physical re-rasterization also improve it.
                            with torch.no_grad(): trial = evaluate(active=baseline['missing'])
                            new = float(trial['loss'])
                        if new < old:
                            accepted = True
                            break
                    except ContractError as exc:
                        errors.append(str(exc))
                if not accepted:
                    with torch.no_grad():
                        for name, value in params: value.copy_(saved[name])
                    new, scale = old, 0.
                improved = old - new
                # A failed line search with a substantial gradient is a stall,
                # never evidence of convergence.
                small = (accepted and improved <= config.loss_tolerance) or grad_norm <= config.gradient_tolerance
                stable = stable + 1 if small else 0
                row = dict(stage=stage, block=block, iteration=iteration, accepted=accepted, step_scale=scale,
                           before=old, after=new, image_loss=float(baseline['image'].detach()),
                           recovery_loss=float(baseline['recovery'].detach()), gradient_norm=grad_norm,
                           collision_loss=float(baseline['collision'].detach()), collision=baseline['collision_report'],
                           missing=baseline['missing'], rejected_errors=errors,
                           parameter_updates={name: float((value.detach()-saved[name]).norm()) for name, value in params})
                history.append(row)
                with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps(row) + '\n')
                # Re-extract topology and re-check the actual renderer on the next iteration.
                checkpoint(stage_index, iteration + 1)
                if stable >= config.patience:
                    final_stage_converged = stage_index == 2 or all(n == 0 for _, n in stages[stage_index + 1:])
                    break
            checkpoint(stage_index + 1, 0)
        with torch.no_grad(): final = evaluate()
        metrics = _metrics(scene, final['rendered'])
        if final['missing']:
            status = 'VISIBILITY_RECOVERY_UNRESOLVED'
        elif final['collision_report'].get('unresolved', False):
            status = 'COLLISION_GATE_NOT_MET'
        elif final_stage_converged and metrics['min_tooth_iou'] >= config.min_tooth_iou:
            status = 'IMAGE_FIT_CONVERGED'
        elif metrics['min_tooth_iou'] < config.min_tooth_iou:
            status = 'IMAGE_GATE_NOT_MET'
        else:
            status = 'BUDGET_EXHAUSTED_OR_STALLED'
        if render_config.backend != 'nvdiffrast': status = 'REFERENCE_DIAGNOSTIC_' + status
        write_render_artifacts(output / 'final_render', scene, final['rendered'], final['mesh'],
                               matching={key: value.diagnostics for key, value in final['match']['views'].items()})
        report = dict(status=status, formal_training=False, iterations=len(history),
                      identity=identity, gpu_preflight=runtime, initial_metrics=initial_metrics, final_metrics=metrics,
                      final_image_loss=float(final['image']), final_prior=float(final['prior']),
                      final_collision_loss=float(final['collision']), collision=final['collision_report'],
                      T_upper_from_lower=scene.T_upper_from_lower.detach().cpu().tolist(),
                      unresolved=final['missing'], frozen_unobserved=frozen_unobserved,
                      bootstrap=bootstrap_history, parameters=parameter_state(scene),
                      replay_scope='checkpoint parameters/cursor restored; CUDA trajectory determinism not guaranteed',
                      scope='image fitting of a frozen prior; not geometric or clinical acceptance')
        write_json(output / 'result.json', report)
        checkpoint(3, 0, status)
        return report
    except Exception as exc:
        write_json(output / 'failure.json', dict(status='FIT_FAILED', error=str(exc), completed_iterations=len(history)))
        raise
    finally:
        for state in scene.arches.values():
            state.pose_delta.requires_grad_(True)
            for value in state.q.values(): value.requires_grad_(True)
