"""Keep the already reconstructed Poisson main shell, without remeshing."""
from pathlib import Path
import sys,json,shutil
import numpy as np
from PIL import Image,ImageDraw
from diagnose_preprocess12 import read,write,sha,font
from audit_space_proxy12 import pd,contains,screen
import audit_make_solid_one as audit
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/dmm_poisson_main_only_00OMSZGW_20261017_01'
P=audit.P;RAW=audit.RAW;BASE=audit.BASE
REVIEW=ROOT/'runs/dmm_poisson_local_review_00OMSZGW_20261017_01'
SOLID=ROOT/'runs/dmm_make_solid_00OMSZGW_20261017_01'

def build():
    OUT.mkdir(exist_ok=False)
    inputs=[P/'poisson.npz',REVIEW/'component_ids.npz',RAW,SOLID/'evaluation.json']
    prior=read(REVIEW/'input_hashes.json');assert all(sha(x['path'])==x['sha256'] for x in prior)
    write(OUT/'input_hashes.json',[dict(path=str(x),sha256=sha(x)) for x in inputs])
    d=np.load(P/'poisson.npz');ids=np.load(REVIEW/'component_ids.npz');fl=ids['face_component'];main=int(ids['main_component']);mask=fl==main;faces=d['faces'][mask];used,inv=np.unique(faces,return_inverse=True);v=d['vertices'][used];f=inv.reshape(-1,3)
    assert np.array_equal(v[f],d['vertices'][faces])
    np.savez_compressed(OUT/'solid_main.npz',vertices=v,faces=f,source_vertex_ids=used,source_face_ids=np.flatnonzero(mask))
    pd(v,f).save(OUT/'poisson_main_PROXY_ONLY.ply')
    comps=[dict(id=int(i),faces=int((fl==i).sum()),kept=bool(i==main)) for i in np.unique(fl)];write(OUT/'components.json',comps)
    write(OUT/'operation.json',dict(method='Keep largest connected component of frozen Poisson proxy; discard other components; no vertex displacement or triangle addition',removed_components=len(comps)-1,removed_faces=int((~mask).sum()),retained_faces=len(f),retained_triangle_coordinates_exactly_equal=True,removed_absolute_volume_sum_mm3=read(REVIEW/'summary.json')['minor_abs_volume_sum_mm3'],training_export=False))
    print('Main component saved; no surface displacement',len(v),len(f),flush=True)

def evaluate():
    audit.OUT=OUT;oldwrite=audit.write
    def writer(p,x):
        if Path(p).name=='evaluation.json':x['method']='Frozen Poisson main connected component only; no remeshing or smoothing'
        oldwrite(p,x)
    audit.write=writer;audit.evaluate();audit.intersection_check()
    d=np.load(OUT/'solid_main.npz');rng=np.random.default_rng(2026101705);fixed=read(SOLID/'local_sampling.json');p=np.load(P/'poisson.npz');planar=np.load(BASE/'proxy_geometry.npz');records=[]
    for region in fixed:
        xyz=rng.uniform(-1.5,1.5,(10000,3))+region['center'];inside=contains(d['vertices'],d['faces'],xyz);pi=contains(p['vertices'],p['faces'],xyz);bi=contains(planar['vertices'],planar['faces'],xyz)
        records.append(dict(fdi=region['fdi'],center=region['center'],count=len(xyz),vs_full_poisson=int((inside!=pi).sum()),vs_planar=int((inside!=bi).sum()),inside=int(inside.sum())))
    write(OUT/'fixed_local_sampling.json',records)

def report():
    d=np.load(OUT/'solid_main.npz');po=np.load(P/'poisson.npz');raw=np.load(RAW);center=(d['vertices'].min(0)+d['vertices'].max(0))/2;im=Image.new('RGB',(1400,1490),'#edf1f5');dr=ImageDraw.Draw(im);dr.text((20,7),'00OMSZGW_lower · 泊松结果仅保留主体',font=font(29),fill='#18384a')
    for row,eye in enumerate([[.5,.7,-.8],[0,0,1]]):
        for col,(mesh,title) in enumerate([(po,'之前：主体 + 26 个小闭合片'),(d,'之后：主体保留，去除小片')]):im.paste(screen(mesh['vertices'],mesh['faces'],np.empty((0,3),int),raw['basis'],center,eye,90,title),(700*col,50+700*row))
    im.save(OUT/'before_after.png');r=read(OUT/'evaluation.json');ix=read(OUT/'intersection_check.json');op=read(OUT/'operation.json');old=read(SOLID/'evaluation.json')
    # Reuse fixed-section camera and report machinery; both source locations stay frozen.
    import audit_ms_tuning as tune
    import pyvista as pv
    dev=np.load(OUT/'deviation.npz');olddev=np.load(SOLID/'deviation.npz');solid=np.load(SOLID/'solid_main.npz')
    for region in read(SOLID/'local_sampling.json'):
        k=region['fdi'];c=np.asarray(region['center']);idx=np.argmin(np.linalg.norm(raw['vertices']-c,axis=1));view=olddev['closest_xyz'][idx]-c;view/=np.linalg.norm(view);up=raw['basis'][:,2]
        if abs(view@up)>.9:up=raw['basis'][:,1]
        canvas=Image.new('RGB',(1800,680),'#edf1f5')
        for col,(vv,ff,title) in enumerate([(raw['vertices'],raw['faces'],'原扫描'),(solid['vertices'],solid['faces'],'Make Solid 512/512'),(d['vertices'],d['faces'],'泊松主体（本次候选）')]):
            sel=np.linalg.norm(vv[ff].mean(1)-c,axis=1)<5;mesh=pd(vv,ff[sel]).clip(normal=view,origin=c+view*.08,invert=True);pl=pv.Plotter(off_screen=True,window_size=(600,600));pl.set_background('#edf1f5');pl.add_mesh(mesh,color='#d5c7af',ambient=.4);pl.add_mesh(pv.Sphere(radius=.04,center=c),color='#c52d50');pl.camera_position=[c+view*50,c,up];pl.enable_parallel_projection();pl.camera.parallel_scale=2.5;pic=Image.fromarray(pl.screenshot(return_img=True));pl.close();dd=ImageDraw.Draw(pic);dd.rectangle((0,0,600,43),fill='#edf1f5');dd.text((12,7),title,font=font(22),fill='#18384a');canvas.paste(pic,(600*col,0))
        ImageDraw.Draw(canvas).text((15,620),f'FDI {k} · 同位置、同剖切平面，5 mm 视野；红点为 Make Solid 基线最坏原顶点。',font=font(24),fill='#18384a');canvas.save(OUT/f'local_{k}.png')
    html='<!doctype html><meta charset="utf-8"><title>泊松主体代理验收</title><style>body{font:18px/1.7 system-ui;max-width:1500px;margin:25px auto;background:#edf1f5;color:#18384a}img{max-width:100%}td,th{border:1px solid #abc;padding:10px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>换个直接办法：泊松结果只保留主体</h1><p>停止 Make Solid 加密调参路线。已有泊松网格中，主体的顶点坐标和三角面原样保留；去除 26 个独立小闭合片（792 面，绝对体积合计约 0.1831 mm³）。小片包含内部小腔边界，去除意味着不再保留这些微小空腔；它不是原始扫描修复。</p><p>原始扫描、旧代理及所有试验均保留。这份模型仅用于空间点内外分类，不用于表面或法线训练。</p><table><tr><th>方法</th><th>37 原顶点 P95 / 最大 mm</th><th>47 原顶点 P95 / 最大 mm</th></tr>'
    for title,e in [('Make Solid 512/512',old),('本次：泊松主体',r)]:
        t=e['original_vertex_distances_by_fdi'];html+=f'<tr><td>{title}</td><td>{t["37"]["p95"]:.4f} / {t["37"]["max"]:.4f}</td><td>{t["47"]["p95"]:.4f} / {t["47"]["max"]:.4f}</td></tr>'
    html+='</table><p>沿用泊松本身的局部偏差，不能视为完全恢复原始扫描。先验收主体形状，再决定是否用于后续空间采样。</p><img src="before_after.png"><h2>37 固定位置剖切对照</h2><img src="local_37.png"><h2>47 固定位置剖切对照</h2><img src="local_47.png"><p>剖切开口是显示手段，不是网格漏洞。最近距离按原顶点和固定面积样本计算，不是全表面 Hausdorff 上界。</p><h2>数值检查</h2><pre>'+json.dumps(r,ensure_ascii=False,indent=2)+'</pre><pre>'+json.dumps(ix,ensure_ascii=False,indent=2)+'</pre><h2>固定局部空间点</h2><pre>'+json.dumps(read(OUT/'fixed_local_sampling.json'),ensure_ascii=False,indent=2)+'</pre><p>与其他代理的分类分歧是敏感性，不是真值错误率。MeshLab 自交检查为全网格数值筛查，不是精确数学证明。人工验收待定。</p><p><a href="poisson_main_PROXY_ONLY.ply">下载主体代理 PLY</a> · <a href="operation.json">处理记录</a></p>'
    (OUT/'index.html').write_text(html,encoding='utf-8');assert all(sha(x['path'])==x['sha256'] for x in read(OUT/'input_hashes.json'));assert all(sha(x['path'])==x['sha256'] for x in read(BASE/'input_hashes.json'));write(OUT/'source_hash_verification.json',dict(all_unchanged=True));shutil.copyfile(__file__,OUT/'implementation.py');write(OUT/'delivery_manifest.json',[dict(path=str(x.relative_to(OUT)),sha256=sha(x)) for x in OUT.iterdir() if x.is_file()]);print('report ready',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
