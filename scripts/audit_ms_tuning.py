"""Bounded parameter comparison, frozen raw data and matched diagnostic points."""
from pathlib import Path
import sys,json,shutil
import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image,ImageDraw
from diagnose_preprocess12 import read,write,sha,font
from audit_space_proxy12 import pd,contains,screen,edge_orientation_metrics
from localize_poisson_proxy_issues import locator,closest
import audit_make_solid_one as audit
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/dmm_ms_tune_00OMSZGW_20261017_01'
OLD=ROOT/'runs/dmm_make_solid_00OMSZGW_20261017_01'
RAW=audit.RAW;BASE=audit.BASE;P=audit.P

def quick(tag):
    dest=OUT/tag;params=read(dest/'parameters.json');assert params['status']=='EXPORTED'
    assert not (dest/'quick.json').exists()
    inputs=[RAW,BASE/'proxy_geometry.npz',OLD/'solid_main.npz',OLD/'deviation.npz',Path(params['export'])]
    write(dest/'input_hashes.json',[dict(path=str(x),sha256=sha(x)) for x in inputs])
    m=pv.read(params['export']).triangulate();v,inv=np.unique(np.asarray(m.points,dtype=np.float64),axis=0,return_inverse=True);f=inv[m.faces.reshape(-1,4)[:,1:]]
    n,lab=connected_components(coo_matrix((np.ones(f.size,np.int8),(f.ravel(),np.roll(f,-1,axis=1).ravel())),shape=(len(v),len(v))).tocsr(),directed=False);fl=lab[f[:,0]];counts=np.bincount(fl);main=np.argmax(counts);kept=f[fl==main];used,idx=np.unique(kept,return_inverse=True);v=v[used];f=idx.reshape(-1,3)
    write(dest/'components.json',[dict(id=i,faces=int(c),kept=bool(i==main)) for i,c in enumerate(counts)])
    np.savez_compressed(dest/'solid_main.npz',vertices=v,faces=f);pd(v,f).save(dest/'solid_main_PROXY_ONLY.ply')
    bc,nc,wc=edge_orientation_metrics(f);r=dict(tag=tag,parameters=params['actual'],vertices=len(v),faces=len(f),components_before=int(n),removed_faces=int(counts.sum()-counts[main]),boundary_edges=bc,nonmanifold_edges=nc,inconsistent_edges=wc)
    raw=np.load(RAW);dist,pos,near=closest(locator(v,f),raw['vertices']);np.savez_compressed(dest/'deviation.npz',distance_mm=dist,closest_xyz=pos,closest_face=near)
    r['by_fdi']={}
    for k in np.unique(raw['vertex_labels']):
        ids=np.flatnonzero(raw['vertex_labels']==k);ds=dist[ids];r['by_fdi'][str(int(k))]=dict(count=len(ds),p95=float(np.quantile(ds,.95)),max=float(ds.max()),over_0_1mm=int((ds>.1).sum()),max_vertex=int(ids[np.argmax(ds)]))
    probes=np.load(P/'geometry_deviation_samples.npz')['original_surface_points'];dd,_,_=closest(locator(v,f),probes);r['surface_20k']=dict(p95=float(np.quantile(dd,.95)),max=float(dd.max()))
    if not(bc or nc or wc):
        rng=np.random.default_rng(2026101705);fixed=read(OLD/'local_sampling.json');bd=np.load(BASE/'proxy_geometry.npz');po=np.load(P/'poisson.npz');od=np.load(OLD/'solid_main.npz');records=[]
        for region in fixed:
            xyz=rng.uniform(-1.5,1.5,(10000,3))+region['center'];inside=contains(v,f,xyz);previous=contains(od['vertices'],od['faces'],xyz);planar=contains(bd['vertices'],bd['faces'],xyz);poisson=contains(po['vertices'],po['faces'],xyz)
            records.append(dict(fdi=region['fdi'],center=region['center'],count=len(xyz),vs_baseline_512=int((inside!=previous).sum()),vs_planar=int((inside!=planar).sum()),vs_poisson=int((inside!=poisson).sum()),inside=int(inside.sum())))
        r['fixed_local_sampling']=records
    write(dest/'quick.json',r);print(json.dumps({k:r[k] for k in ['tag','parameters','faces','components_before','removed_faces','boundary_edges','nonmanifold_edges','inconsistent_edges','surface_20k']},indent=2),flush=True)
    print('37/47',json.dumps({k:r['by_fdi'][k] for k in ['37','47']}),flush=True)
    print('fixed local',r.get('fixed_local_sampling'),flush=True)

def full(tag):
    dest=OUT/tag;audit.OUT=dest
    # Reuse frozen audit with a truthful method label at serialization time.
    original_write=audit.write
    def writer(p,r):
        if Path(p).name=='evaluation.json':r['method']='Meshmixer Make Solid '+json.dumps(read(dest/'parameters.json')['actual'])+'; keep largest component'
        original_write(p,r)
    audit.write=writer
    if not (dest/'components.json').exists():write(dest/'components.json',[{'note':'See quick.json for original component count; only main is audited'}]*read(dest/'quick.json')['components_before'])
    audit.evaluate();audit.intersection_check()

def report(best):
    raw=np.load(RAW);old=np.load(OLD/'solid_main.npz');new=np.load(OUT/best/'solid_main.npz');basis=raw['basis'];center=(old['vertices'].min(0)+old['vertices'].max(0))/2
    im=Image.new('RGB',(1400,1490),'#edf1f5');dr=ImageDraw.Draw(im);dr.text((20,7),'00OMSZGW_lower · Make Solid 参数对照（原始 mm）',font=font(28),fill='#18384a')
    for row,eye in enumerate([[.5,.7,-.8],[0,0,1]]):
        for col,(d,title) in enumerate([(old,'基线 Accurate 512 / 512'),(new,'候选 '+best)]):im.paste(screen(d['vertices'],d['faces'],np.empty((0,3),int),basis,center,eye,90,title),(col*700,50+row*700))
    im.save(OUT/'before_after.png');details=[];base_dist=np.load(OLD/'deviation.npz');newdist=np.load(OUT/best/'deviation.npz');er=read(OUT/best/'evaluation.json');ix=read(OUT/best/'intersection_check.json')
    for region in read(OLD/'local_sampling.json'):
        k=region['fdi'];c=np.asarray(region['center']);nearest=base_dist['closest_xyz'][np.argmin(np.linalg.norm(raw['vertices']-c,axis=1))];view=nearest-c;view/=np.linalg.norm(view);up=basis[:,2]
        if abs(view@up)>.9:up=basis[:,1]
        canvas=Image.new('RGB',(1800,690),'#edf1f5')
        for j,(vv,ff,title) in enumerate([(raw['vertices'],raw['faces'],'原扫描'),(old['vertices'],old['faces'],'基线 512/512'),(new['vertices'],new['faces'],'调参 '+best)]):
            selected=np.linalg.norm(vv[ff].mean(1)-c,axis=1)<5;mesh=pd(vv,ff[selected]).clip(normal=view,origin=c+view*.08,invert=True);pl=pv.Plotter(off_screen=True,window_size=(600,600));pl.set_background('#edf1f5');pl.add_mesh(mesh,color='#d5c7af',ambient=.4);pl.add_mesh(pv.Sphere(radius=.04,center=c),color='#c52d50');pl.camera_position=[c+view*50,c,up];pl.enable_parallel_projection();pl.camera.parallel_scale=2.5;pic=Image.fromarray(pl.screenshot(return_img=True));pl.close();d=ImageDraw.Draw(pic);d.rectangle((0,0,600,43),fill='#edf1f5');d.text((12,7),title,font=font(22),fill='#18384a');canvas.paste(pic,(j*600,0))
        ImageDraw.Draw(canvas).text((15,620),f'FDI {k} · 同一位置、同一剖切平面、5 mm 视野；红点是基线最大偏差原顶点；剖切开口不是漏孔。',font=font(23),fill='#18384a');canvas.save(OUT/f'local_{k}.png')
    baseline=read(OLD/'evaluation.json');trials=[read(x) for x in sorted(OUT.glob('*/quick.json'))]
    html='<!doctype html><meta charset="utf-8"><title>Make Solid 参数验收</title><style>body{font:18px/1.7 system-ui;max-width:1500px;margin:25px auto;background:#edf1f5;color:#18384a}img{max-width:100%}td,th{border:1px solid #abc;padding:10px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>00OMSZGW_lower：Make Solid 参数对照</h1><p>每组都从同一份统一底平面代理重新开始；零偏移、零增厚；仅保留最大连通体。原始数据与此前试验不变，不跑全量、不导出训练数据。</p><table><tr><th>参数</th><th>原表面 20k 点 P95 mm</th><th>37 顶点 P95 / 最大 mm</th><th>47 顶点 P95 / 最大 mm</th><th>边界 / 非流形 / 朝向冲突</th></tr>'
    x=baseline['original_vertex_distances_by_fdi'];html+=f'<tr><td>Accurate 512/512 基线</td><td>{baseline["original_surface_samples_mm"]["p95"]:.5f}</td><td>{x["37"]["p95"]:.4f} / {x["37"]["max"]:.4f}</td><td>{x["47"]["p95"]:.4f} / {x["47"]["max"]:.4f}</td><td>0/0/0</td></tr>'
    for r in trials:
        x=r['by_fdi'];html+=f'<tr><td>{r["tag"]}</td><td>{r["surface_20k"]["p95"]:.5f}</td><td>{x["37"]["p95"]:.4f} / {x["37"]["max"]:.4f}</td><td>{x["47"]["p95"]:.4f} / {x["47"]["max"]:.4f}</td><td>{r["boundary_edges"]}/{r["nonmanifold_edges"]}/{r["inconsistent_edges"]}</td></tr>'
    html+='</table><h2>整体前后对照</h2><img src="before_after.png"><h2>37 局部固定剖切</h2><img src="local_37.png"><h2>47 局部固定剖切</h2><img src="local_47.png"><h2>同一局部空间点的分类敏感性</h2><p>沿用基线两处 3 mm 立方体，每处同一批 10,000 点。与旧代理/泊松的分歧不是错误率；未提供内部真值。最大原顶点距离也不是连续表面的 Hausdorff 上界。</p>'
    for r in trials:html+='<details><summary>'+r['tag']+'</summary><pre>'+json.dumps(r,ensure_ascii=False,indent=2)+'</pre></details>'
    html+='<h2>所选候选完整检查：'+best+'</h2><pre>'+json.dumps(er,ensure_ascii=False,indent=2)+'</pre><pre>'+json.dumps(ix,ensure_ascii=False,indent=2)+'</pre><p>MeshLab 全网格数值自交筛查不等于精确数学证明。人工验收待定。</p><a href="'+best+'/solid_main_PROXY_ONLY.ply">候选代理 PLY</a>'
    (OUT/'index.html').write_text(html,encoding='utf-8');shutil.copyfile(__file__,OUT/'audit_implementation.py');shutil.copyfile(ROOT/'scripts/mm_tune_solid.ps1',OUT/'api_implementation.ps1')
    assert all(sha(x['path'])==x['sha256'] for t in trials for x in read(OUT/t['tag']/'input_hashes.json'))
    assert all(sha(x['path'])==x['sha256'] for x in read(BASE/'input_hashes.json'))
    write(OUT/'source_hash_verification.json',dict(all_unchanged=True,best_candidate=best,human_review='PENDING',training_export=False))
    write(OUT/'delivery_manifest.json',[dict(path=str(x.relative_to(OUT)),sha256=sha(x)) for x in OUT.rglob('*') if x.is_file()]);print('report ready',flush=True)

if __name__=='__main__':globals()[sys.argv[1]](sys.argv[2])
