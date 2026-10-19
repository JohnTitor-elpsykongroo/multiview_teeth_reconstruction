"""Stream a new 256^3 field grid from the unchanged final DMM parameters."""
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from skimage.measure import marching_cubes

PROJECT=Path(__file__).resolve().parents[1]
OUT=Path((PROJECT/'tmp/dmm_export256_current.txt').read_text('utf-8-sig'))
PLAN=json.loads((OUT/'plan.json').read_text('utf-8-sig'))
OLD=Path(PLAN['source_run']);FIT=OLD/PLAN['source_fit']
PACKAGE=PROJECT/'.runtime/dmm_r4_best_val_20261015/package'
sys.path.insert(0,str(PACKAGE/'third_party/DMM'))
from dmm.scene import load_scene
from dmm.fitting import restore_parameters,parameter_state
from dmm import MODEL_UNIT_MM
from dmm.provenance import source_fingerprint

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,value):
    path=OUT/name;temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');temporary.replace(path)
def event(**value):
    value['utc']=datetime.now(timezone.utc).isoformat()
    print(json.dumps(value),flush=True)
    with (OUT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')

start=time.monotonic()
assert not (OUT/'mesh_256.npz').exists()
result=json.loads((FIT/'result.json').read_text('utf-8'))
source_hash=digest(FIT/'result.json')
manifest=OLD/'scenes/L428SD7J_pilot256/fit_input/manifest.json'
scene=load_scene(manifest,'cuda:0')
restore_parameters(scene,result['parameters'])
scene.eval().requires_grad_(False)
current=parameter_state(scene)
for arch in scene.arches:
    for label,code in result['parameters'][arch]['q'].items():
        assert np.array_equal(current[arch]['q'][label],code)
    assert np.allclose(current[arch]['T_world_from_arch'],result['parameters'][arch]['T_world_from_arch'],atol=1e-6,rtol=0)
write('frozen_parameters.json',current)
report=dict(status='SAMPLING',case='L428SD7J',source_fit_status=result['status'],resolution=256,grid_points_per_arch=256**3,
    model_unit_mm=MODEL_UNIT_MM,length_unit='mm',source_result_sha256=source_hash,decoder_source_sha256=source_fingerprint(),
    source_fit_source_sha256=result['identity']['source_sha256'],chunk_size=PLAN['chunk_size'],
    extraction='Fresh coupled-field float32 grid; marching cubes level 0; no implicit-root attachment, optimization, smoothing, upsampling or component removal',
    torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0),arches={})
write('state.json',report)
vertices_all=[];faces_all=[];semantics_all=[];owners_all=[];vertex_offset=0
try:
    with torch.inference_mode():
        for arch_index,arch in enumerate(('upper','lower')):
            state=scene.arches[arch];arch_start=time.monotonic()
            box=np.asarray(state.bundle.metadata['sampling_domain_model'],dtype=float)
            size=256;chunk=PLAN['chunk_size'];device=state.base_pose.device
            axes=[torch.linspace(float(box[0,i]),float(box[1,i]),size,device=device,dtype=torch.float32) for i in range(3)]
            path=OUT/(arch+'_sdf_float32.npy')
            assert not path.exists()
            volume=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(size,size,size))
            flat=volume.reshape(-1)
            for first in range(0,size**3,chunk):
                ids=torch.arange(first,min(first+chunk,size**3),device=device)
                points=torch.stack((axes[0][ids//(size*size)],axes[1][ids//size%size],axes[2][ids%size]),-1)
                values=state.query_model(points)['sdf'].cpu().numpy()
                assert np.isfinite(values).all()
                flat[first:first+len(values)]=values
                if first//chunk%32==0 or first+chunk>=size**3:
                    volume.flush();done=min(first+chunk,size**3)
                    event(arch=arch,stage='sampling',completed=done,total=size**3,percent=round(100*done/size**3,1),seconds=round(time.monotonic()-arch_start,1),gpu_peak_mb=round(torch.cuda.max_memory_allocated()/1024**2))
                    report['progress']=dict(arch=arch,completed=done,total=size**3);write('state.json',report)
            assert volume.min()<0<volume.max()
            boundary=np.concatenate([volume[0].ravel(),volume[-1].ravel(),volume[:,0].ravel(),volume[:,-1].ravel(),volume[:,:,0].ravel(),volume[:,:,-1].ravel()])
            assert (boundary>0).all(),f'{arch}: surface intersects verified sampling-domain boundary'
            spacing=(box[1]-box[0])/(size-1)
            vertices,faces,_,_=marching_cubes(volume,0.,spacing=spacing,allow_degenerate=False)
            vertices+=box[0]
            semantics=[];residual=[]
            for first in range(0,len(vertices),chunk):
                query=state.query_model(torch.tensor(vertices[first:first+chunk],device=device,dtype=torch.float32))
                semantics.append(query['semantics'].cpu().numpy());residual.append(query['sdf'].cpu().numpy())
            semantics=np.concatenate(semantics);residual=np.concatenate(residual)
            pose=state.T_world_from_arch.cpu().numpy()
            world=(vertices*MODEL_UNIT_MM)@pose[:3,:3].T+pose[:3,3]
            assert np.isfinite(world).all() and np.isfinite(semantics).all()
            vertices_all.append(world);faces_all.append(faces+vertex_offset);semantics_all.append(semantics)
            owners_all.append(np.full(len(faces),arch_index,dtype=np.uint8));vertex_offset+=len(vertices)
            report['arches'][arch]=dict(vertices=len(vertices),faces=len(faces),grid_spacing_mm=(spacing*MODEL_UNIT_MM).tolist(),
                sampling_domain_model=box.tolist(),boundary_min=float(boundary.min()),field_residual_abs_max=float(np.max(np.abs(residual))),
                field_residual_abs_median=float(np.median(np.abs(residual))),seconds=time.monotonic()-arch_start,
                model_id=state.bundle.metadata['model_id'],weights_sha256=state.bundle.metadata['weights']['sha256'],statistics_sha256=state.bundle.metadata['latent_statistics']['sha256'])
            event(arch=arch,stage='extracted',vertices=len(vertices),faces=len(faces),seconds=round(time.monotonic()-arch_start,1))
            write('state.json',report)
            del volume,flat
    np.savez_compressed(OUT/'mesh_256.npz',vertices_world_mm=np.concatenate(vertices_all),faces=np.concatenate(faces_all),semantics=np.concatenate(semantics_all),face_arch=np.concatenate(owners_all))
    assert source_hash==digest(FIT/'result.json')
    assert current==parameter_state(scene)
    report.update(status='EXTRACTION_COMPLETE',seconds=time.monotonic()-start,gpu_peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,parameters_unchanged=True)
    write('extraction_report.json',report);write('state.json',report)
    event(status='EXTRACTION_COMPLETE',seconds=round(report['seconds'],1))
except Exception:
    report.update(status='FAILED_RETAINED',traceback=traceback.format_exc());write('state.json',report);raise
