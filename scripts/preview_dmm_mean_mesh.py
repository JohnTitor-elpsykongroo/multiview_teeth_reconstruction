"""Export and render the saved, unmodified q=0 DMM initial surfaces."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/dmm_static_trial_20261015T121206Z_4ccf8f'
OUT = ROOT / 'runs/dmm_initial_mesh_preview_20261016'
OUT.mkdir(exist_ok=True)
mesh_path = SOURCE / 'fit_pilot_004/initial_render/mesh.npz'
param_path = SOURCE / 'fit_pilot_004/starting_parameters.json'
mesh = np.load(mesh_path)
params = json.loads(param_path.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
data, provenance = {}, {'mesh_source': str(mesh_path), 'mesh_sha256': sha(mesh_path),
    'parameters_sha256': sha(param_path), 'geometry_modified': False,
    'coordinate_system': 'Per-arch canonical millimeters; inverse initial rigid transform',
    'resolution': 48, 'arches': {}}

def basis(eye):
    z = np.asarray(eye, dtype=float); z /= np.linalg.norm(z)
    up = np.array([0.,0.,1.]) if abs(z[2]) < .95 else np.array([0.,1.,0.])
    x = np.cross(up,z); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    return np.array([x,y,z]).T

def render(v, f, normal, eye, size=640):
    rot = basis(eye)
    p = (v-(v.min(0)+v.max(0))/2) @ rot
    n = normal @ rot
    p[:,:2] *= size/106
    p[:,0] += size/2; p[:,1] = size/2-p[:,1]
    depth = np.full((size,size), -np.inf)
    rgb = np.full((size,size,3), [232.,238.,242.])
    light=np.array([-.35,.45,.82]);light/=np.linalg.norm(light)
    for tri in f:
        t=p[tri]; lo=np.maximum(np.floor(t[:,:2].min(0)).astype(int),0)
        hi=np.minimum(np.ceil(t[:,:2].max(0)).astype(int),size-1)
        if np.any(hi<lo): continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
        a,b,c=t; den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-10: continue
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        w=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        k=1-u-w; z=u*a[2]+w*b[2]+k*c[2]
        sl=np.s_[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        mask=(u>=0)&(w>=0)&(k>=0)&(z>depth[sl])
        if not mask.any(): continue
        nn=u[...,None]*n[tri[0]]+w[...,None]*n[tri[1]]+k[...,None]*n[tri[2]]
        nn/=np.maximum(np.linalg.norm(nn,axis=2,keepdims=True),1e-8)
        nn*=np.where(nn[:,:,2:3]<0,-1,1)
        diffuse=np.maximum(nn@light,0)
        shade=.30+.65*diffuse+.05*np.maximum(nn[:,:,2],0)
        color=np.clip(shade[...,None]*[243,226,197],0,255)
        depth[sl][mask]=z[mask];rgb[sl][mask]=color[mask]
    return Image.fromarray(rgb.astype('uint8'))

canvas=Image.new('RGB',(1920,1500),'#e8eef2');draw=ImageDraw.Draw(canvas)
fontpath='C:/Windows/Fonts/msyh.ttc'
font=ImageFont.truetype(fontpath,28);small=ImageFont.truetype(fontpath,22)
draw.text((36,22),'DMM R4 · 初始显式网格 · q = 0（训练均值）',font=ImageFont.truetype(fontpath,38),fill='#172c3e')
draw.text((36,80),'原始 48³ 提取结果 | 含牙齿与牙龈 | 各颌独立坐标系 | 未平滑、未补洞',font=small,fill='#516878')
for idx,arch in enumerate(('upper','lower')):
    hashes={}
    for name in ('model.json','weights.pth','latent_statistics.npz'):
        a=SOURCE/'scenes/L428SD7J_pilot256/model_bundle'/arch/name
        b=ROOT/'.runtime/dmm_r4_best_val_20261015/package/model_bundle'/arch/name
        assert sha(a)==sha(b), name
        hashes[name]=sha(a)
    assert all(np.max(np.abs(q)) == 0 for q in params[arch]['q'].values())
    f0=mesh['faces'][mesh['face_arch']==idx]
    ids,reverse=np.unique(f0,return_inverse=True);f=reverse.reshape(-1,3)
    T=np.asarray(params[arch]['T_world_from_arch'])
    # Exact inverse, including the tiny numerical deviation from orthogonality.
    v=(mesh['vertices_world_mm'][ids]-T[:3,3]) @ np.linalg.inv(T[:3,:3]).T
    normal=np.zeros_like(v);cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    for j in range(3): np.add.at(normal,f[:,j],cross)
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    with (OUT/f'{arch}_mean.obj').open('w') as s:
        s.write('# DMM R4 q=0; canonical mm; exact saved triangles\n')
        for x in v:s.write('v %.9f %.9f %.9f\n'%tuple(x))
        for x in f+1:s.write('f %d %d %d\n'%tuple(x))
    data[arch]={'v':v.round(6).tolist(),'f':f.tolist(),'n':normal.round(6).tolist()}
    provenance['arches'][arch]={'vertices':len(v),'faces':len(f),'q_max_abs':0,'bundle_sha256':hashes,
        'world_reconstruction_max_error_mm':float(np.max(np.abs(v@T[:3,:3].T+T[:3,3]-mesh['vertices_world_mm'][ids])))}
    sign=-1 if arch=='upper' else 1
    for col,(label,eye) in enumerate([('咬合面',(0,0,sign)),('正面',(0,-1,0)),('斜视',(0.55,-.8,sign*.9))]):
        img=render(v,f,normal,eye)
        x=col*640;y=150+idx*670
        canvas.paste(img,(x,y))
        draw.text((x+26,y+8),('上颌' if idx==0 else '下颌')+' · '+label,font=font,fill='#172c3e')
    draw.text((26,150+idx*670+606),f'{len(v):,} vertices / {len(f):,} triangles',font=small,fill='#516878')
canvas.save(OUT/'preview.png')
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
template=(ROOT/'scripts/dmm_mesh_viewer_template.html').read_text(encoding='utf-8')
(OUT/'viewer.html').write_text(template.replace('__MESH_DATA__',json.dumps(data,separators=(',',':'))),encoding='utf-8')
print(json.dumps({'output':str(OUT),'arches':provenance['arches']},indent=2))
