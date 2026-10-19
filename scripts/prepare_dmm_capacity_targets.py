"""Export explicitly oracle 3D diagnostic samples for four preselected teeth."""
import ast
import json
from pathlib import Path
import numpy as np
import trimesh
PROJECT=Path(__file__).resolve().parents[1]
OUT=Path((PROJECT/'tmp/dmm_diagnostic_current.txt').read_text('utf-8-sig'))
OLD=Path(json.loads((OUT/'plan.json').read_text('utf-8-sig'))['source_run'])
source=Path(json.loads((OLD/'plan.json').read_text('utf-8'))['source'])/'dataset/Patient_L428SD7J'
config=json.loads((source/'source_resolved_config.json').read_text('utf-8'))
meta=json.loads((source/'source_scene_metadata.json').read_text('utf-8'))
namespace={'np':np};tree=ast.parse((PROJECT/'scripts/evaluate_dmm_static_geometry.py').read_text('utf-8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample'],type_ignores=[]),'original_sample','exec'),namespace)
result={}
for arch,chosen in (('upper',(11,27)),('lower',(31,47))):
    path=Path(config['source_root'])/arch/'L428SD7J'/('L428SD7J_'+arch+'.obj')
    mesh=trimesh.load(path,process=False,force='mesh',maintain_order=True)
    labels=np.array(json.loads(path.with_suffix('.json').read_text('utf-8'))['labels'])
    transform=np.array(meta['transforms'][arch]['source_to_world_mm'])
    vertices=np.asarray(mesh.vertices)@transform[:3,:3].T+transform[:3,3];faces=np.asarray(mesh.faces)
    for label in chosen:
        triangles=vertices[faces[(labels[faces]==label).sum(1)>=2]]
        result[str(label)+'_triangles']=triangles
        for split,seed in (('fit',70000),('heldout',90000)):
            result[str(label)+'_'+split]=namespace['sample'](triangles,512,seed+label)
np.savez_compressed(OUT/'oracle_capacity_targets.npz',**result)
print('DIAGNOSTIC TARGETS ONLY: upper 11/27, lower 31/47; 512 fit + 512 heldout area samples each')
