"""Re-extract a frozen final DMM field on 256^3, without optimization/upsampling."""
import hashlib
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from skimage.measure import marching_cubes

PROJECT=Path(__file__).resolve().parents[1]
OLD=Path((PROJECT/'tmp/dmm_static_trial_current.txt').read_text('utf-8-sig'))
PACKAGE=PROJECT/'.runtime/dmm_r4_best_val_20261015/package'
sys.path.insert(0,str(PACKAGE/'third_party/DMM'))
from dmm.scene import load_scene
from dmm.fitting import restore_parameters,parameter_state
from dmm.provenance import source_fingerprint
from dmm import MODEL_UNIT_MM

stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT=PROJECT/'runs'/('dmm_export256_'+stamp)
OUT.mkdir(exist_ok=False)
(PROJECT/'tmp/dmm_export256_current.txt').write_text(str(OUT),encoding='utf-8')
RESOLUTION=256
CHUNK=65536
CHECKPOINT=OLD/'fit_pilot_008_resume_persistent/checkpoint.json'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(name,data):
    path=OUT/name;temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');temp.replace(path)
def event(**data):
    data['utc']=datetime.now(timezone.utc).isoformat()
    with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(data)+'\n')
    print(json.dumps(data),flush=True)
checkpoint=json.loads(CHECKPOINT.read_text('utf-8'))
source_hash=digest(CHECKPOINT)
report=dict(status='EXPORT_RUNNING',source_run=str(OLD),checkpoint=str(CHECKPOINT),checkpoint_sha256=source_hash,
    original_fit_status=checkpoint['status'],resolution=RESOLUTION,chunk_size=CHUNK,
    sampling_points_per_arch=RESOLUTION**3,model_unit_mm=MODEL_UNIT_MM,
    algorithm='Native-compatible float32 mixed-field sampling and zero-level marching cubes; no normal-root projection; no mesh subdivision',
    source_sha256=source_fingerprint(),torch=str(torch.__version__),arches={},quality_accepted=False)
write('export_report.json',report)
started=time.monotonic()
try:
    scene=load_scene(OLD/'scenes/L428SD7J_pilot256/fit_input/manifest.json','cuda:0')
    restore_parameters(scene,checkpoint['parameters'])
    scene.eval().requires_grad_(False)
    before=parameter_state(scene)
    for arch in scene.arches:
        assert np.allclose(before[arch]['T_world_from_arch'],checkpoint['parameters'][arch]['T_world_from_arch'],atol=1e-5,rtol=0)
    write('frozen_parameters.json',before)
    report['gpu']=torch.cuda.get_device_name(0)
    all_vertices=[];all_faces=[];all_semantics=[];all_owners=[];offset=0
    with torch.no_grad():
        for arch_index,arch in enumerate(('upper','lower')):
            state=scene.arches[arch]
            event(stage='sampling',arch=arch,points=RESOLUTION**3)
            arch_start=time.monotonic()
            box=np.asarray(state.bundle.metadata['sampling_domain_model'],dtype=float)
            axes=[np.linspace(box[0,i],box[1,i],RESOLUTION) for i in range(3)]
            volume_path=OUT/(arch+'_sdf256.partial.npy')
            volume=np.lib.format.open_memmap(volume_path,mode='w+',dtype=np.float32,shape=(RESOLUTION,)*3)
            flat=volume.reshape(-1)
            for first in range(0,RESOLUTION**3,CHUNK):
                ids=np.arange(first,min(first+CHUNK,RESOLUTION**3))
                coordinates=np.column_stack((axes[0][ids//(RESOLUTION*RESOLUTION)],axes[1][ids//RESOLUTION%RESOLUTION],axes[2][ids%RESOLUTION])).astype(np.float32)
                points=torch.from_numpy(coordinates).to('cuda:0')
                values=state.query_model(points)['sdf']
                flat[first:first+len(ids)]=values.cpu().numpy()
                if first%(CHUNK*16)==0:
                    event(stage='sampling',arch=arch,completed=first+len(ids),total=RESOLUTION**3,
                        gpu_allocated_bytes=torch.cuda.memory_allocated(),gpu_reserved_bytes=torch.cuda.memory_reserved())
            assert np.isfinite(volume).all() and volume.min()<0<volume.max(),'No finite mixed zero surface'
            boundary=np.concatenate((volume[0].ravel(),volume[-1].ravel(),volume[:,0].ravel(),volume[:,-1].ravel(),volume[:,:,0].ravel(),volume[:,:,-1].ravel()))
            assert np.all(boundary>0),f'{arch}: nonpositive boundary; retain field for diagnosis'
            event(stage='marching_cubes',arch=arch,boundary_min=float(boundary.min()))
            vertices,faces,_,_=marching_cubes(volume,level=0.,spacing=(box[1]-box[0])/(RESOLUTION-1),allow_degenerate=False)
            vertices+=box[0]
            probabilities=[]
            for first in range(0,len(vertices),CHUNK):
                points=torch.tensor(vertices[first:first+CHUNK],dtype=torch.float32,device='cuda:0')
                probabilities.append(state.query_model(points)['semantics'].cpu().numpy())
            probability=np.concatenate(probabilities)
            pose=state.T_world_from_arch.cpu().numpy()
            world=(MODEL_UNIT_MM*vertices)@pose[:3,:3].T+pose[:3,3]
            assert np.isfinite(world).all() and np.isfinite(probability).all()
            assert np.allclose(probability.sum(-1),1,atol=1e-5,rtol=0)
            np.savez_compressed(OUT/(arch+'_mesh256.npz'),vertices_world_mm=world,faces=faces,semantics=probability,vertices_model=vertices)
            all_vertices.append(world);all_faces.append(faces+offset);all_semantics.append(probability)
            all_owners.append(np.full(len(faces),arch_index,np.uint8));offset+=len(vertices)
            report['arches'][arch]=dict(vertices=len(vertices),faces=len(faces),domain_model=box.tolist(),grid_spacing_mm=((box[1]-box[0])*MODEL_UNIT_MM/(RESOLUTION-1)).tolist(),
                boundary_min=float(boundary.min()),field_min=float(volume.min()),field_max=float(volume.max()),seconds=time.monotonic()-arch_start)
            del flat;volume.flush();del volume
            volume_path.replace(OUT/(arch+'_sdf256.npy'))
            write('export_report.json',report);event(stage='arch_complete',arch=arch,detail=report['arches'][arch])
    np.savez_compressed(OUT/'dual_arch_mesh256.npz',vertices_world_mm=np.concatenate(all_vertices),faces=np.concatenate(all_faces),
        semantics=np.concatenate(all_semantics),face_arch=np.concatenate(all_owners))
    assert parameter_state(scene)==before,'Parameters changed during frozen export'
    assert digest(CHECKPOINT)==source_hash,'Original checkpoint changed'
    shutil.copy2(Path(__file__),OUT/Path(__file__).name)
    report.update(status='FIELD_EXPORT_COMPLETE',parameters_unchanged=True,total_vertices=offset,total_faces=sum(len(f) for f in all_faces),
        seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    write('export_report.json',report);event(stage='complete',report=report)
except Exception:
    report.update(status='EXPORT_FAILED_RETAINED',error=traceback.format_exc(),seconds=time.monotonic()-started)
    write('export_report.json',report)
    raise
