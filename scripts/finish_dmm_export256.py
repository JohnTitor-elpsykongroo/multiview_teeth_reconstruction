"""Save binary PLYs and matched 48/256 previews from completed sampling."""
import colorsys
import hashlib
import html
import json
import shutil
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT=Path(__file__).resolve().parents[1]
OUT=Path((PROJECT/'tmp/dmm_export256_current.txt').read_text('utf-8-sig'))
PLAN=json.loads((OUT/'plan.json').read_text('utf-8-sig'));OLD=Path(PLAN['source_run'])
report=json.loads((OUT/'extraction_report.json').read_text('utf-8'))
assert report['status']=='EXTRACTION_COMPLETE'
palette=np.array([[.62,.49,.49],*[colorsys.hsv_to_rgb((i*.618)%1,.55,.95) for i in range(28)]])
with np.load(OUT/'mesh_256.npz') as data:high={k:data[k].copy() for k in data.files}
with np.load(OLD/PLAN['source_fit']/'final_render/mesh.npz') as data:low={k:data[k].copy() for k in data.files}
v,f=high['vertices_world_mm'],high['faces']
color=(255*palette[high['semantics'].argmax(-1)]).astype(np.uint8)
mesh=trimesh.Trimesh(v,f,vertex_colors=color,process=False)
mesh.export(OUT/'final_dual_arch_world_mm_256.ply')
report['total_vertices']=len(v);report['total_faces']=len(f);report['watertight']=bool(mesh.is_watertight)
report['old_48_vertices']=len(low['vertices_world_mm']);report['old_48_faces']=len(low['faces'])
for index,arch in enumerate(('upper','lower')):
    part=trimesh.Trimesh(v,f[high['face_arch']==index],vertex_colors=color,process=False)
    part.remove_unreferenced_vertices();part.export(OUT/(arch+'_world_mm_256.ply'))
    report['arches'][arch]['watertight']=bool(part.is_watertight)
    report['arches'][arch]['connected_bodies']=int(part.body_count)
print(json.dumps(dict(status='PLY_SAVED',vertices=len(v),faces=len(f),watertight=report['watertight'])),flush=True)

def preview(teeth_only):
    phases=[('48 cubed',low),('256 cubed',high)]
    angles=[(0,-90,'Anterior'),(25,-125,'Oblique')]
    if teeth_only:angles.append((80,-90,'Occlusal'))
    bounds=[];selected=[]
    for name,data in phases:
        vertices,faces=data['vertices_world_mm'],data['faces']
        semantics=data['semantics'][faces].mean(1)
        if teeth_only:
            keep=semantics.argmax(1)!=0;faces=faces[keep];semantics=semantics[keep]
        selected.append((name,vertices,faces,semantics@palette))
        bounds.append(vertices[np.unique(faces)])
    vertices=np.concatenate(bounds)
    center=(vertices.min(0)+vertices.max(0))/2;radius=np.ptp(vertices,axis=0).max()*.55
    figure=plt.figure(figsize=(5.3*len(angles),8))
    for i,(name,v,f,colors) in enumerate(selected):
        for j,(elev,azim,title) in enumerate(angles):
            ax=figure.add_subplot(2,len(angles),i*len(angles)+j+1,projection='3d')
            ax.add_collection3d(Poly3DCollection(v[f],facecolors=colors,edgecolor='none',linewidths=0,rasterized=True,shade=True,lightsource=LightSource(azdeg=315,altdeg=45)))
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
            ax.set_box_aspect((1,1,1),zoom=1.4);ax.set_proj_type('ortho');ax.view_init(elev=elev,azim=azim);ax.set_axis_off()
            ax.set_title(name+' | '+title)
    figure.suptitle('Same final latent and pose | '+('tooth semantic regions only' if teeth_only else 'full blended surface, gum included')+' | no smoothing',fontsize=13)
    figure.tight_layout();figure.savefig(OUT/('teeth_48_vs_256.png' if teeth_only else 'comparison_48_vs_256.png'),dpi=140);plt.close(figure)
preview(False);print('FULL_PREVIEW_SAVED',flush=True)
preview(True);print('TEETH_PREVIEW_SAVED',flush=True)
text=f'''DMM 256³ 固定参数导出：L428SD7J
原拟合状态：{report['source_fit_status']}。本次只重提取，不重新优化。
双颌：{report['total_vertices']:,} 顶点，{report['total_faces']:,} 三角面。
原48³拟合网格：{report['old_48_vertices']:,} 顶点，{report['old_48_faces']:,} 三角面。
PyTorch显存峰值：{report['gpu_peak_allocated_mb']:.1f} MiB；采样/提取耗时：{report['seconds']:.1f} 秒。
每颌256³真实场采样，chunk={report['chunk_size']}；没有使用网格细分或平滑伪造细节。
上下颌位姿、latent与原结果完全一致；世界坐标单位mm。
上下颌分别正域边界通过；导出网格是否封闭：{report['watertight']}。
颜色为DMM软语义组件，牙龈用灰粉色；牙齿区域图仅为语义显示筛选，不能当作逐牙解剖分割。
这次使用原部署decoder的float32场和Marching Cubes；没有沿用48³拟合网格的法线求根附件。
与原网格的差别同时包含采样分辨率及表面重提取方式，不能解释为latent优化取得进步。
'''
(OUT/'RESULT_ZH.txt').write_text(text,encoding='utf-8')
page='<!doctype html><meta charset="utf-8"><title>DMM 256³导出</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:30px auto;padding:0 20px;background:#f6f7f9;color:#182734}pre{white-space:pre-wrap;background:white;padding:20px}img{width:100%}a{color:#176da3}</style><h1>DMM 48³ / 256³ 对照</h1><pre>'+html.escape(text)+'</pre>'
for name,title in (('comparison_48_vs_256.png','完整双颌（含牙龈）'),('teeth_48_vs_256.png','牙齿语义区域对照')):
    page+=f'<h2>{title}</h2><a href="{name}"><img src="{name}"></a>'
page+='<p><a href="final_dual_arch_world_mm_256.ply">双颌PLY</a> · <a href="upper_world_mm_256.ply">上颌PLY</a> · <a href="lower_world_mm_256.ply">下颌PLY</a></p>'
(OUT/'index.html').write_text(page,encoding='utf-8')
report['status']='PLY_AND_PREVIEWS_COMPLETE'
(OUT/'export_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
for name in ('export_dmm_frozen_256.py','finish_dmm_export256.py'):
    shutil.copy2(PROJECT/'scripts'/name,snapshot/name)
print(OUT,flush=True)
