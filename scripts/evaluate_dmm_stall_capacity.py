"""Measure actual MC component meshes, separately from the fitted field loss."""
import ast
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PROJECT=Path(__file__).resolve().parents[1]
OUT=Path((PROJECT/'tmp/dmm_diagnostic_current.txt').read_text('utf-8-sig'))
tree=ast.parse((PROJECT/'scripts/evaluate_dmm_static_geometry.py').read_text('utf-8'));ns={'np':np}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('sample','distances')],type_ignores=[]),'audited_geometry_functions','exec'),ns)
sample,distances=ns['sample'],ns['distances']
targets=np.load(OUT/'oracle_capacity_targets.npz')
capacity=json.loads((OUT/'component_capacity.json').read_text('utf-8'))
plot_only='--plot-only' in sys.argv
rows=json.loads((OUT/'component_geometry_evaluation.json').read_text('utf-8'))['rows'] if plot_only else []
geometry={}
for entry in capacity['results']:
    label=entry['label'];truth=targets[str(label)+'_triangles'];geometry[label]={'truth':truth}
    for phase in ('before','after'):
        path=OUT/f'capacity_{label}_{phase}.npz'
        if not path.exists():continue
        with np.load(path) as data:tri=data['vertices'][data['faces']]
        if plot_only:
            geometry[label][phase]=tri
            continue
        p=distances(sample(tri,1000,42000+label),truth)
        g=distances(sample(truth,1000,84000+label),tri)
        pair=np.concatenate((p,g))
        row=dict(label=label,phase=phase,rms_mm=float(np.sqrt(np.mean(pair*pair))),mean_mm=float(pair.mean()),p95_mm=float(np.percentile(pair,95)))
        rows.append(row);geometry[label][phase]=tri;print(row,flush=True)
(OUT/'component_geometry_evaluation.json').write_text(json.dumps(dict(scope='DIAGNOSTIC ORACLE: independent component MC64 surfaces, finite sampled exact triangle distance; not full blended-arch reconstruction',samples_per_direction=1000,rows=rows),indent=2)+'\n',encoding='utf-8')
figure,axes=plt.subplots(1,2,figsize=(11,4))
labels=[11,27,31,47];x=np.arange(4)
for phase,offset,color in (('before',-.18,'#8e9cab'),('after',.18,'#2479a8')):
    values=[next((r['rms_mm'] for r in rows if r['label']==label and r['phase']==phase),np.nan) for label in labels]
    axes[0].bar(x+offset,values,.36,label=phase,color=color)
    values=[entry[phase+'_heldout_first_order_abs_mm'] for entry in capacity['results']]
    axes[1].bar(x+offset,values,.36,label=phase,color=color)
for ax,title in zip(axes,('Actual component mesh: bidirectional surface RMS','Held-out field / spatial-gradient distance proxy')):
    ax.set_xticks(x,labels);ax.set_xlabel('FDI');ax.set_ylabel('mm');ax.set_title(title);ax.legend()
figure.suptitle('Frozen R4 weights | 80 direct-3D latent steps | diagnostic oracle pose and 3D targets')
figure.tight_layout();figure.savefig(OUT/'capacity_controls.png',dpi=150);plt.close(figure)

figure=plt.figure(figsize=(14,7.2))
for j,label in enumerate(labels):
    group=geometry[label]
    points=np.concatenate([t.reshape(-1,3) for t in group.values()])
    center=(points.min(0)+points.max(0))/2;radius=np.ptp(points,axis=0).max()*.6
    for i,phase in enumerate(('before','after')):
        ax=figure.add_subplot(2,4,i*4+j+1,projection='3d')
        for key,color,alpha in (('truth','#333333',.65),(phase,'#287fb0',.8)):
            if key not in group:continue
            p=sample(group[key],1000,19000+label)
            ax.scatter(*p.T,c=color,s=1,alpha=alpha,linewidths=0,rasterized=True)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        ax.set_box_aspect((1,1,1),zoom=1.5);ax.set_proj_type('ortho');ax.view_init(20,-105);ax.set_axis_off();ax.set_title(f'FDI {label} | {phase}')
figure.suptitle('Oracle component control only | grey = target, blue = model | same pose / bounds per tooth')
figure.tight_layout();figure.savefig(OUT/'capacity_geometry.png',dpi=150);plt.close(figure)
