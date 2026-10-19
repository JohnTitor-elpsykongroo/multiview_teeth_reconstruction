"""Separate resampling from OT-plan refresh at the same rendered candidate."""
import torch
import numpy as np
from dmm_stall_common import *
from dmm.semanticxy import PointMeasure,normalize_xy,warp_loss_from_plan,partial_transport
scene=scene_at('final')
FRESH='--fresh' in sys.argv
install_execution_helpers(scene)
matcher=SemanticXYMatcher(MATCH)
direction=json.loads((OUT/'native_shape_direction.json').read_text('utf-8'))['directions']
event(stage='matching_separation_render_baseline')
with torch.no_grad():
    baseline,_,charts=scene.render_views(SURFACE,RENDER)
    old={o.view_id:matcher.match_view(baseline[o.view_id],o,mode='evaluate') for o in scene.observations}
    for name,p in scene.named_parameters():
        if name in direction:p.add_(p.new_tensor(direction[name])/128)
    event(stage='matching_separation_render_candidate')
    candidate,_,_=scene.render_views(SURFACE,RENDER,None if FRESH else charts)
rows=[]
with torch.no_grad():
    for obs in scene.observations:
        previous=old[obs.view_id];base=baseline[obs.view_id];new=candidate[obs.view_id]
        tooth0=base.semantics[...,1:].reshape(-1,28);alpha0=tooth0.sum(-1)
        bins=tooth0.argmax(-1);valid=torch.tensor(obs.valid,device=tooth0.device);count=int(valid.sum())
        tooth=new.semantics[...,1:].reshape(-1,28);alpha=tooth.sum(-1)
        indices=previous.source.pixel_indices;start=0;masses=[]
        for channel in range(28):
            members=torch.nonzero(valid.reshape(-1)&(alpha0>0)&(bins==channel)).flatten()
            if not len(members):continue
            n=min(MATCH.samples_per_tooth,len(members));chosen=indices[start:start+n];start+=n
            masses.append(alpha[chosen]/count if len(members)<=n else (alpha[members].sum()/(n*count)).expand(n))
        source=PointMeasure(tooth[indices]/alpha[indices,None].clamp_min(1e-30),
            normalize_xy(new.xy_pixels.reshape(-1,2)[indices],new.semantics.shape[1],new.semantics.shape[0],obs.metadata),
            torch.cat(masses),indices,previous.source.represented_pixels)
        target=torch.tensor(obs.semantic_target(),device=tooth.device)
        dense=.5*(new.semantics[valid]-target[valid]).abs().sum()/count
        def loss(plan):
            warp,unmatched=warp_loss_from_plan(source,previous.target,plan,MATCH)
            return float(MATCH.warp_loss_weight*warp+MATCH.unmatched_loss_weight*unmatched+MATCH.dense_semantic_loss_weight*dense)
        frozen=loss(previous.plan)
        refreshed=loss(partial_transport(source,previous.target,MATCH))
        native=matcher.match_view(new,obs,mode='evaluate')
        rows.append(dict(view=obs.view_id,baseline=float(previous.loss),frozen_samples_frozen_plan=frozen,
            frozen_samples_new_plan=refreshed,new_samples_new_plan=float(native.loss),
            baseline_dense=float(previous.dense_semantic_loss),candidate_dense=float(native.dense_semantic_loss),
            baseline_warp=float(previous.warp_loss),candidate_warp=float(native.warp_loss),
            baseline_unmatched=float(previous.unmatched_loss),candidate_unmatched=float(native.unmatched_loss),
            resampled=not torch.equal(indices,native.source.pixel_indices)))
means={key:float(np.mean([r[key] for r in rows])) for key in ('baseline','frozen_samples_frozen_plan','frozen_samples_new_plan','new_samples_new_plan','baseline_dense','candidate_dense','baseline_warp','candidate_warp','baseline_unmatched','candidate_unmatched')}
write('matching_refresh_fresh_mesh.json' if FRESH else 'matching_refresh_separation.json',dict(scale=1/128,mesh='fresh' if FRESH else 'fixed_chart',scope='Same rendered candidate in every matching mode; target samples unchanged',means=means,per_view=rows))
event(matching_means=means)
