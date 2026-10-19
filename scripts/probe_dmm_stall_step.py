"""Reproduce one stalled shape direction and separate discrete refreshes."""
import gc
import math
import traceback
import numpy as np
import torch
from dmm_stall_common import *
from dmm_stall_common import _direction
from dmm import CHANNEL_FDI
from dmm.semanticxy import PointMeasure, normalize_xy, warp_loss_from_plan
from dmm.rendering import camera_coordinates, project
from dmm.pixels import pixel_offset

scene = scene_at('final')
pack, unpack, memo, counts = install_execution_helpers(scene)
matcher = SemanticXYMatcher(MATCH)
params = [(n,p) for n,p in scene.named_parameters() if p.requires_grad]
saved = {n:p.detach().clone() for n,p in params}

def recovery_eval(rendered, active, charts=None, frozen=None):
    charts = {} if charts is None else charts
    observations = {o.view_id:o for o in scene.observations}
    terms, templates = [], []
    with memo():
        for i, row in enumerate(active):
            obs = observations[row['view_id']]
            points = recovery.component_points(scene, row['arch'], row['label'], SURFACE, charts)
            points = recovery._sample(points, FIT.recovery.source_samples)
            camera = camera_coordinates(points, obs)
            xy = project(torch.cat((camera[:,:2], camera[:,2:3].clamp_min(RENDER.near_mm)), -1), points.new_tensor(obs.K))
            yx = recovery._sample(torch.as_tensor(np.argwhere((obs.labels == row['label']) & obs.valid), device=points.device), FIT.recovery.target_samples)
            target = yx[:,[1,0]].to(points.dtype)+pixel_offset(obs.metadata)
            if frozen is None:
                depth = rendered[obs.view_id].depth_mm.detach()[yx[:,0],yx[:,1]]
                finite = torch.isfinite(depth)
                depth = torch.where(finite, depth, torch.zeros_like(depth))
            else:
                depth, finite = frozen[i]['depth'], frozen[i]['finite']
            distance = (target[:,None]-xy[None]).square().sum(-1)/math.hypot(*obs.labels.shape)**2
            barrier = torch.relu(camera[None,:,2]-depth[:,None]+FIT.recovery.depth_margin_mm)/FIT.recovery.depth_scale_mm
            barrier = torch.where(finite[:,None], barrier, torch.zeros_like(barrier))
            near = torch.relu(RENDER.near_mm-camera[:,2])/FIT.recovery.depth_scale_mm
            cost = distance+barrier.square()+near[None].square()
            nearest = cost.detach().argmin(-1) if frozen is None else frozen[i]['nearest']
            terms.append(cost[torch.arange(len(yx),device=points.device),nearest].mean())
            templates.append(dict(depth=depth,finite=finite,nearest=nearest))
    return torch.stack(terms).sum()/len(scene.observations), charts, templates

def frozen_image(rendered, templates):
    terms=[]
    for obs in scene.observations:
        old, groups = templates[obs.view_id]
        pred, xy = rendered[obs.view_id].semantics, rendered[obs.view_id].xy_pixels
        tooth = pred[...,1:].reshape(-1,28)
        alpha = tooth.sum(-1)
        indices = old.source.pixel_indices
        masses=[]
        for members, indices_group, n, small in groups:
            masses.append(alpha[indices_group]/int(obs.valid.sum()) if small else (alpha[members].sum()/(n*int(obs.valid.sum()))).expand(n))
        mass = torch.cat(masses) if masses else alpha[indices]
        total = alpha[indices,None]
        source=PointMeasure(tooth[indices]/torch.where(total>0,total,torch.ones_like(total)),
            normalize_xy(xy.reshape(-1,2)[indices],pred.shape[1],pred.shape[0],obs.metadata),mass,indices,old.source.represented_pixels)
        warp, unmatched=warp_loss_from_plan(source,old.target,old.plan,MATCH)
        target = torch.tensor(obs.semantic_target(),device=pred.device,dtype=pred.dtype)
        valid=torch.tensor(obs.valid,device=pred.device)
        dense=.5*(pred[valid]-target[valid]).abs().sum()/valid.sum()
        terms.append(MATCH.warp_loss_weight*warp+MATCH.unmatched_loss_weight*unmatched+MATCH.dense_semantic_loss_weight*dense)
    return torch.stack(terms).mean()

def templates_from(baseline):
    templates={}
    for obs in scene.observations:
        old=baseline['match']['views'][obs.view_id]
        pred=baseline['rendered'][obs.view_id].semantics.detach()
        tooth=pred[...,1:].reshape(-1,28); alpha=tooth.sum(-1)
        bins=tooth.argmax(-1); valid=torch.tensor(obs.valid,device=pred.device).reshape(-1)
        groups=[]; start=0
        for channel in range(28):
            members=torch.nonzero(valid & (alpha>0) & (bins==channel)).flatten()
            if not len(members):continue
            n=min(MATCH.samples_per_tooth,len(members))
            groups.append((members,old.source.pixel_indices[start:start+n],n,len(members)<=n)); start+=n
        assert start==len(old.source.mass)
        templates[obs.view_id]=(old,groups)
    return templates

def evaluate(charts=None, active=None, recovery_charts=None, templates=None, recovery_templates=None):
    rendered, mesh, charts=scene.render_views(SURFACE,RENDER,charts)
    match=matcher.match_scene(rendered,scene.observations,mode='evaluate')
    missing=recovery.missing_regions(scene,rendered,FIT.recovery)
    active=missing if active is None else active
    rec,recovery_charts,rec_templates=recovery_eval(rendered,active,recovery_charts)
    prior=scene.latent_prior()
    total=match['loss']+FIT.recovery_weight*rec+FIT.prior_weight*prior
    result=dict(loss=total,image=match['loss'],recovery=rec,prior=prior,rendered=rendered,charts=charts,
        recovery_charts=recovery_charts,recovery_templates=rec_templates,match=match,missing=missing)
    if templates is not None:
        frozen=frozen_image(rendered,templates)
        frec,_,_=recovery_eval(rendered,active,recovery_charts,recovery_templates)
        result.update(frozen_image=frozen,frozen_recovery=frec,frozen_total=frozen+FIT.recovery_weight*frec+FIT.prior_weight*prior)
        changes=0
        for obs in scene.observations:
            a=templates[obs.view_id][0].source.pixel_indices
            b=match['views'][obs.view_id].source.pixel_indices
            changes+=int(not torch.equal(a,b))
        result['resampled_views']=changes
    return result

def scalar(result):
    names=('loss','image','recovery','prior','frozen_image','frozen_recovery','frozen_total')
    return {k:float(result[k].detach()) for k in names if k in result} | dict(missing=len(result['missing']),resampled_views=result.get('resampled_views'))

rows=[]
FOLLOWUP='--followup' in sys.argv
RESULT_NAME='shape_backtrack_followup.json' if FOLLOWUP else 'shape_backtrack_probes.json'
STATE_NAME='step_followup_state.json' if FOLLOWUP else 'step_probe_state.json'
try:
    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
        event(stage='native_baseline_forward')
        if FOLLOWUP:
            with torch.no_grad():baseline=evaluate()
        else:
            baseline=evaluate()
        templates=templates_from(baseline)
        fimage=frozen_image(baseline['rendered'],templates)
        assert torch.allclose(fimage,baseline['image'],atol=1e-7,rtol=0)
        event(stage='native_baseline_backward',baseline=scalar(baseline))
        if FOLLOWUP:
            direction_data=json.loads((OUT/'native_shape_direction.json').read_text('utf-8'))
            directions={n:p.new_tensor(direction_data['directions'][n]) for n,p in params}
            slope=direction_data['slope']
        else:
            baseline['loss'].backward()
            directions={n:_direction(n,p,FIT) for n,p in params}
            slope=sum(float((p.grad*directions[n]).sum()) for n,p in params if p.grad is not None)
            gradients={n:p.grad.detach().cpu().tolist() if p.grad is not None else None for n,p in params}
            write('native_shape_direction.json',dict(slope=slope,gradient_norms={n:float(p.grad.norm()) for n,p in params if p.grad is not None},gradients=gradients,directions={n:d.cpu().tolist() for n,d in directions.items()},baseline=scalar(baseline)))
        # Dense/frozen-chart FD uses a fixed graph definition. Native scoring
        # recomputes correspondences, and fresh topology is an additional gate.
        scales=(1.,.5,.25,1/16,1/32) if FOLLOWUP else (0.,1/128,-1/128,1/1024,1/64,1/8,1.)
        for scale in scales:
            with torch.no_grad():
                for name,p in params:p.copy_(saved[name]+scale*directions[name])
            row=dict(scale=scale,expected_linear_change=scale*slope)
            event(stage='candidate',scale=scale)
            try:
                with torch.no_grad():
                    candidate=evaluate(baseline['charts'],baseline['missing'],baseline['recovery_charts'],templates,baseline['recovery_templates'])
                row['fixed_chart']=scalar(candidate)
                del candidate;gc.collect()
                if scale in (0.,1/128,1/1024) or (FOLLOWUP and row['fixed_chart']['loss']<float(baseline['loss'])):
                    event(stage='fresh_topology',scale=scale)
                    with torch.no_grad():fresh=evaluate(active=baseline['missing'])
                    row['fresh_topology']=scalar(fresh)
                    del fresh;gc.collect()
            except Exception as exc:
                row['error']=str(exc)
            rows.append(row);write(RESULT_NAME,dict(baseline=scalar(baseline),slope=slope,rows=rows,cache=counts))
            event(completed=row)
    write(STATE_NAME,dict(status='COMPLETE',cache=counts))
except Exception:
    write(STATE_NAME,dict(status='FAILED_RETAINED',traceback=traceback.format_exc(),rows=rows))
    raise
