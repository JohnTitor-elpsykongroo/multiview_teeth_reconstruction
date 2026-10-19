"""Save PLYs, verify their round trip and compare 48^3 versus 256^3."""
import colorsys
import hashlib
import html
import json
import shutil
import time
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT=Path(__file__).resolve().parents[1]
OUT=Path((PROJECT/'tmp/dmm_export256_current.txt').read_text('utf-8-sig'))
report=json.loads((OUT/'export_report.json').read_text('utf-8'))
assert report['status']=='FIELD_EXPORT_COMPLETE'
OLD=Path(report['source_run'])
palette=np.array([[.72,.73,.75],*[colorsys.hsv_to_rgb((i*.618)%1,.65,.95) for i in range(28)]])
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(path):
    with np.load(path) as data:return {k:data[k].copy() for k in data.files}
high=load(OUT/'dual_arch_mesh256.npz')
low=load(OLD/'fit_pilot_008_resume_persistent/final_render/mesh.npz')
meshes={'48 cubed':low,'256 cubed':high}
ply_records={}
for name,data in [('final_dual_arch_world_mm_256',high),*[(arch+'_world_mm_256',load(OUT/(arch+'_mesh256.npz'))) for arch in ('upper','lower')]]:
    vertices,faces,semantics=data['vertices_world_mm'],data['faces'],data['semantics']
    color=(palette[semantics.argmax(-1)]*255).astype(np.uint8)
    mesh=trimesh.Trimesh(vertices,faces,vertex_colors=color,process=False)
    path=OUT/(name+'.ply');mesh.export(path,file_type='ply',encoding='binary_little_endian')
    reread=trimesh.load(path,force='mesh',process=False)
    assert len(reread.vertices)==len(vertices) and np.array_equal(reread.faces,faces)
    error=float(np.max(np.abs(np.asarray(reread.vertices)-vertices)))
    assert np.isfinite(reread.vertices).all() and error<1e-4
    ply_records[path.name]=dict(vertices=len(vertices),faces=len(faces),roundtrip_max_coordinate_error_mm=error,
        watertight=bool(mesh.is_watertight),bytes=path.stat().st_size,sha256=digest(path))
    print(name,ply_records[path.name],flush=True)
report['ply']=ply_records

# Same camera, scale, semantic colors and plotting bounds in both rows.
all_points=np.concatenate([m['vertices_world_mm'] for m in meshes.values()])
center=(all_points.min(0)+all_points.max(0))/2
radius=np.ptp(all_points,axis=0).max()*.52
tooth_points=[]
for mesh in meshes.values():
    chosen=mesh['semantics'].argmax(-1)!=0
    tooth_points.append(mesh['vertices_world_mm'][chosen])
points=np.concatenate(tooth_points);tooth_center=(points.min(0)+points.max(0))/2
tooth_radius=np.ptp(points,axis=0).max()*.52
figure=plt.figure(figsize=(16,9))
for i,(name,mesh) in enumerate(meshes.items()):
    v,f,p=mesh['vertices_world_mm'],mesh['faces'],mesh['semantics']
    face_labels=p[f].mean(1).argmax(-1)
    vertex_colors=palette[p.argmax(-1)]
    face_colors=vertex_colors[f].mean(1)
    for j,(elev,azim,title,teeth_only) in enumerate(((0,-90,'Anterior; gum included',False),(25,-125,'Oblique; gum included',False),(25,-125,'Tooth-semantic faces only',True))):
        selected=face_labels!=0 if teeth_only else np.ones(len(f),bool)
        ax=figure.add_subplot(2,3,i*3+j+1,projection='3d')
        collection=Poly3DCollection(v[f[selected]],facecolors=face_colors[selected],edgecolor='none',linewidth=0,rasterized=True)
        ax.add_collection3d(collection)
        c,r=(tooth_center,tooth_radius) if teeth_only else (center,radius)
        ax.set(xlim=(c[0]-r,c[0]+r),ylim=(c[1]-r,c[1]+r),zlim=(c[2]-r,c[2]+r))
        ax.set_box_aspect((1,1,1),zoom=1.5);ax.set_proj_type('ortho');ax.view_init(elev,azim);ax.set_axis_off()
        ax.set_title(f'{name} | {title}\n{len(v):,} vertices / {len(f):,} faces',fontsize=10)
figure.suptitle('Same frozen final parameters | new mixed-field extraction, no fitting updates | world mm',fontsize=14)
figure.tight_layout();figure.savefig(OUT/'comparison_48_vs_256.png',dpi=150);plt.close(figure)

# Record the prior deliverables unchanged, including the original 48^3 PLY.
delivery=json.loads((OLD/'DELIVERY.json').read_text('utf-8'))
changed=[name for name,record in delivery['files'].items() if digest(OLD/name)!=record['sha256']]
assert not changed,changed
report['prior_delivery_checked']=len(delivery['files']);report['prior_delivery_changed']=changed
report['status']='COMPLETE'
report['low_resolution_mesh']=dict(vertices=len(low['vertices_world_mm']),faces=len(low['faces']))
(OUT/'export_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
text=f'''DMM 固定最终参数的256³重新导出
状态：完成；没有重新优化latent、位姿或网络权重。
原拟合状态保持：{report['original_fit_status']}。
每颌采样 {report['sampling_points_per_arch']:,} 点；分块 {report['chunk_size']:,} 点。
从连续混合隐式场重新提取零等值面，不是细分原PLY。
使用原生评估/导出路径的float32场和Marching Cubes；不做拟合用的法线IFT求根投影。
48³网格：{len(low['vertices_world_mm']):,} 顶点，{len(low['faces']):,} 三角面。
256³网格：{report['total_vertices']:,} 顶点，{report['total_faces']:,} 三角面。
场采样与网格生成耗时 {report['seconds']:.1f} 秒；峰值PyTorch已分配显存 {report['peak_gpu_allocated_bytes']/1024**3:.3f} GiB。
上颌网格间距(mm)：{report['arches']['upper']['grid_spacing_mm']}。
下颌网格间距(mm)：{report['arches']['lower']['grid_spacing_mm']}。
上下颌采样域边界均为正值，未进行裁剪补洞或删除网格组件。
PLY坐标单位mm、世界坐标；含牙龈，灰色为牙龈，彩色为牙位语义。
右列预览仅按软语义argmax隐藏牙龈面，不能作为逐牙解剖分割。
导出的PLY均独立读回核对顶点、三角面和坐标；原运行{len(delivery['files'])}个交付文件哈希未变。
更密的网格反映同一组拟合参数；不表示之前的拟合停滞或可见性问题已经修复。
'''
(OUT/'RESULT_ZH.txt').write_text(text,encoding='utf-8')
page='<!doctype html><meta charset="utf-8"><title>DMM 256 cubed export</title><style>body{font:16px/1.7 system-ui;max-width:1500px;margin:30px auto;padding:0 24px;color:#182936;background:#f5f7f9}pre{white-space:pre-wrap;background:white;padding:24px}img{width:100%}a{color:#176da3}</style><h1>DMM 256³ 固定参数导出</h1><pre>'+html.escape(text)+'</pre><img src="comparison_48_vs_256.png"><ul>'
for name in ply_records:page+=f'<li><a href="{name}">{name}</a></li>'
page+='<li><a href="export_report.json">导出与完整性记录</a></li></ul>'
(OUT/'index.html').write_text(page,encoding='utf-8')
shutil.copy2(Path(__file__),OUT/Path(__file__).name)
manifest={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file()}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(OUT/'index.html',flush=True)
