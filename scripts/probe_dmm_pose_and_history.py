"""CPU post-fit diagnosis: history and oracle rigid pose removal, no scale."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh

PROJECT=Path(__file__).resolve().parents[1]
OUT=Path((PROJECT/'tmp/dmm_diagnostic_current.txt').read_text('utf-8-sig'))
OLD=Path(json.loads((OUT/'plan.json').read_text('utf-8-sig'))['source_run'])
checkpoint=json.loads((OLD/'fit_pilot_008_resume_persistent/checkpoint.json').read_text('utf-8'))
history=checkpoint['history']
record={'groups':{},'final_latents':{},'rejected_error_counts':{}}
for block in ('pose','shape'):
    rows=[r for r in history if r['block']==block]
    record['groups'][block]=dict(steps=len(rows),accepted=sum(r['accepted'] for r in rows),
        gradient_min=min(r['gradient_norm'] for r in rows),gradient_max=max(r['gradient_norm'] for r in rows),
        errors=sum(len(r['rejected_errors']) for r in rows))
    for r in rows:
        for error in r['rejected_errors']:
            record['rejected_error_counts'][error]=record['rejected_error_counts'].get(error,0)+1
for arch,p in checkpoint['parameters'].items():
    q=np.concatenate(list(p['q'].values()))
    record['final_latents'][arch]=dict(rms=float(np.sqrt(np.mean(q*q))),max_component_l2=max(float(np.linalg.norm(v)) for v in p['q'].values()))
(OUT/'history_analysis.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')

# Reuse exactly the previous analytical point-to-triangle routine without its
# executable top-level evaluation workflow; retain its source hash.
helper=PROJECT/'scripts/evaluate_dmm_static_geometry.py'
tree=ast.parse(helper.read_text('utf-8'))
namespace={'np':np}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('sample','distances')],type_ignores=[]),str(helper),'exec'),namespace)
sample,distances=namespace['sample'],namespace['distances']
oldplan=json.loads((OLD/'plan.json').read_text('utf-8'))
data=Path(oldplan['source'])/'dataset/Patient_L428SD7J'
meta=json.loads((data/'source_scene_metadata.json').read_text('utf-8'))
config=json.loads((data/'source_resolved_config.json').read_text('utf-8'))
FDI=[*range(11,18),*range(21,28),*range(31,38),*range(41,48)]
truth={}
def centroid(t):
    area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    return np.average(t.mean(1),axis=0,weights=area)
for arch in ('upper','lower'):
    path=Path(config['source_root'])/arch/'L428SD7J'/('L428SD7J_'+arch+'.obj')
    for p in (path,path.with_suffix('.json')):
        assert hashlib.sha256(p.read_bytes()).hexdigest()==config['expected_source_files'][str(p)]
    mesh=trimesh.load(path,process=False,force='mesh',maintain_order=True)
    labels=np.array(json.loads(path.with_suffix('.json').read_text('utf-8'))['labels'])
    transform=np.array(meta['transforms'][arch]['source_to_world_mm'])
    vertices=np.asarray(mesh.vertices)@transform[:3,:3].T+transform[:3,3]
    faces=np.asarray(mesh.faces)
    for label in FDI:
        if (label<30)==(arch=='upper'):
            truth[label]=vertices[faces[(labels[faces]==label).sum(1)>=2]]

initial=json.loads((OLD/'fit_pilot_004/starting_parameters.json').read_text('utf-8'))
evaluation=dict(scope='DIAGNOSTIC ORACLE: 3D truth centroids used only to remove pose; not a valid image-only reconstruction',
    alignment='Proper rigid Kabsch on 14 corresponding tooth area centroids per arch, no scale; shape/size residual remains',
    helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),phases={},oracle_initial_poses={})
for phase,root in (('initial',OLD/'fit_pilot_004'),('final',OLD/'fit_pilot_008_resume_persistent')):
    with np.load(root/(phase+'_render/mesh.npz')) as mesh:
        vertices,faces=mesh['vertices_world_mm'].astype(float),mesh['faces']
        labels=np.array([0,*FDI])[mesh['semantics'][faces].mean(1).argmax(1)]
    result={}
    for arch in ('upper','lower'):
        selected=[label for label in FDI if (label<30)==(arch=='upper')]
        predicted={k:vertices[faces[labels==k]] for k in selected}
        a=np.stack([centroid(predicted[k]) for k in selected]);b=np.stack([centroid(truth[k]) for k in selected])
        ca,cb=a.mean(0),b.mean(0)
        u,s,vt=np.linalg.svd((a-ca).T@(b-cb));rotation=vt.T@u.T
        if np.linalg.det(rotation)<0:
            vt[-1]*=-1;rotation=vt.T@u.T
        translation=cb-rotation@ca
        delta=np.eye(4);delta[:3,:3]=rotation;delta[:3,3]=translation
        rows=[]
        for label in selected:
            tri=predicted[label]@rotation.T+translation
            p2g=distances(sample(tri,1000,42000+label),truth[label])
            g2p=distances(sample(truth[label],1000,84000+label),tri)
            rms=float(np.sqrt(np.mean(np.concatenate((p2g,g2p))**2)))
            rows.append(dict(label=label,rms_mm=rms))
        result[arch]=dict(correction_world=delta.tolist(),angle_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1)))),
            centroid_rms_before_mm=float(np.sqrt(np.mean(np.sum((a-b)**2,axis=1)))),
            centroid_rms_after_mm=float(np.sqrt(np.mean(np.sum((a@rotation.T+translation-b)**2,axis=1)))),
            surface_macro_rms_after_pose_removal_mm=float(np.mean([r['rms_mm'] for r in rows])),per_tooth=rows)
        if phase=='initial':
            evaluation['oracle_initial_poses'][arch]=(delta@np.array(initial[arch]['T_world_from_arch'])).tolist()
        print(phase,arch,{k:v for k,v in result[arch].items() if k not in ('per_tooth','correction_world')},flush=True)
    evaluation['phases'][phase]=result
    (OUT/'pose_vs_shape_3d.json').write_text(json.dumps(evaluation,indent=2)+'\n',encoding='utf-8')
print('COMPLETE',flush=True)
