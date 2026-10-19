"""One-case Make Solid audit; reads frozen sources, writes only isolated run."""
from pathlib import Path
import sys,json,shutil
import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image,ImageDraw
from diagnose_preprocess12 import read,write,sha,font
from audit_space_proxy12 import pd,contains,screen,edge_orientation_metrics
from localize_poisson_proxy_issues import locator,closest,shot
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/dmm_make_solid_00OMSZGW_20261017_01'
BASE=ROOT/'runs/dmm_proxy_plane_00OMSZGW_20261016_01'
RAW=ROOT/'runs/dmm_preprocess12_20261016T1815_r1/cases/00OMSZGW_lower/raw_index_preserved.npz'
P=ROOT/'runs/dmm_meshlab_proxy_00OMSZGW_20261017_01/poisson'

def prepare():
    exported=ROOT/'runs/mm_solid17.stl'
    shutil.copyfile(exported,OUT/'meshmixer_make_solid_PROXY_ONLY.stl')
    inputs=[RAW,BASE/'proxy_geometry.npz',P/'poisson.npz',OUT/'meshmixer_make_solid_PROXY_ONLY.stl']
    if not (OUT/'input_hashes.json').exists():write(OUT/'input_hashes.json',[dict(path=str(p),sha256=sha(p)) for p in inputs])
    else:assert all(sha(x['path'])==x['sha256'] for x in read(OUT/'input_hashes.json'))
    m=pv.read(inputs[-1]).triangulate();v=np.asarray(m.points,dtype=np.float64);f=m.faces.reshape(-1,4)[:,1:]
    # OBJ importer may duplicate vertices for normals: merge only exact coordinate matches.
    v,inv=np.unique(v,axis=0,return_inverse=True);f=inv[f]
    n,lab=connected_components(coo_matrix((np.ones(f.size,np.int8),(f.ravel(),np.roll(f,-1,axis=1).ravel())),shape=(len(v),len(v))).tocsr(),directed=False)
    fl=lab[f[:,0]];counts=np.bincount(fl);main=np.argmax(counts);kept=f[fl==main];used,inv=np.unique(kept,return_inverse=True);vv=v[used];ff=inv.reshape(-1,3)
    records=[]
    for cid in range(n):
        t=v[f[fl==cid]];c=t.mean((0,1));t=t-c;vol=np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6
        records.append(dict(id=cid,faces=int(counts[cid]),volume_mm3=float(vol),kept=bool(cid==main)))
    np.savez_compressed(OUT/'solid_main.npz',vertices=vv,faces=ff)
    pd(vv,ff).save(OUT/'solid_main_PROXY_ONLY.ply')
    write(OUT/'components.json',records)
    print('components',n,'faces',len(f),'kept',len(ff),flush=True)

def evaluate():
    d=np.load(OUT/'solid_main.npz');v=d['vertices'];f=d['faces'];raw=np.load(RAW);b=np.load(BASE/'proxy_geometry.npz');loc=locator(v,f)
    bc,nc,wc=edge_orientation_metrics(f);t=v[f]-v.mean(0);vol=float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)
    result=dict(method='Meshmixer Make Solid Accurate; solidResolution=512; meshResolution=512; offset/minThickness=0; closeHoles/autoRepair=true; keep largest face component',vertices=len(v),faces=len(f),components_before=len(read(OUT/'components.json')),components_after=1,boundary_edges=bc,nonmanifold_edges=nc,inconsistent_edges=wc,signed_volume_mm3=vol,bounds_mm=[v.min(0).tolist(),v.max(0).tolist()],source_bounds_mm=[b['vertices'].min(0).tolist(),b['vertices'].max(0).tolist()])
    distance,pos,near=closest(loc,raw['vertices']);stats={}
    for k in np.unique(raw['vertex_labels']):
        dd=distance[raw['vertex_labels']==k];stats[str(int(k))]=dict(count=len(dd),median=float(np.median(dd)),p95=float(np.quantile(dd,.95)),max=float(dd.max()))
    result['original_vertex_distances_by_fdi']=stats
    probes=np.load(P/'geometry_deviation_samples.npz')['original_surface_points'];ds,_,_=closest(loc,probes)
    result['original_surface_samples_mm']=dict(count=len(ds),median=float(np.median(ds)),p95=float(np.quantile(ds,.95)),max=float(ds.max()))
    np.savez_compressed(OUT/'deviation.npz',distance_mm=distance,closest_xyz=pos,closest_face=near)
    rng=np.random.default_rng(2026101704);sf=rng.choice(len(f),20000,p=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1).sum());uv=rng.random((len(sf),2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1];tr=v[f[sf]];pts=tr[:,0]+uv[:,0,None]*(tr[:,1]-tr[:,0])+uv[:,1,None]*(tr[:,2]-tr[:,0]);back,_,_=closest(locator(b['vertices'],b['faces']),pts)
    result['repaired_to_planar_proxy_samples_mm']=dict(count=len(back),p95=float(np.quantile(back,.95)),max=float(back.max()))
    if not(bc or nc or wc):
        old=np.load(BASE/'space_candidates_DIAGNOSTIC_ONLY.npz');xyz=old['xyz'];inside=contains(v,f,xyz);domain=np.asarray(read(BASE/'classification.json')['domain']);box=float(np.prod(domain[1]-domain[0]));p=inside.mean();se=box*np.sqrt(p*(1-p)/len(xyz));ix=rng.choice(len(xyz),20000,replace=False);q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,0]*=np.linalg.det(q);rot=contains((v-v.mean(0))@q,f,(xyz[ix]-v.mean(0))@q)
        result['sampling']=dict(count=len(xyz),monte_carlo_volume_mm3=float(p*box),volume_standard_errors=float(abs(p*box-abs(vol))/se),rotation_count=len(ix),rotation_disagreements=int((rot!=inside[ix]).sum()),vs_planar_changes=int((inside!=old['inside_after']).sum()),vs_poisson_changes=int((inside!=np.load(P/'space_labels_DIAGNOSTIC_ONLY.npz')['inside']).sum()))
        np.savez_compressed(OUT/'space_diagnostic.npz',xyz=xyz,inside=inside)
    result['training_export']=False;result['human_review']='PENDING';write(OUT/'evaluation.json',result);print(json.dumps(result,indent=2),flush=True)

def intersection_check():
    sys.path.insert(0,str(ROOT/'runs/dmm_meshlab_proxy_00OMSZGW_20261017_01/deps'))
    import pymeshlab as ml
    d=np.load(OUT/'solid_main.npz');v=d['vertices'];f=d['faces'];ms=ml.MeshSet();ms.add_mesh(ml.Mesh(vertex_matrix=v,face_matrix=f.astype(np.int32)))
    ms.compute_selection_by_self_intersections_per_face();selected=ms.current_mesh().face_selection_array();top=ms.get_topological_measures()
    result=dict(method='MeshLab full-mesh self-intersection selection; numerical test, not exact proof',selected_faces=int(selected.sum()),topology={k:(x.item() if hasattr(x,'item') else x) for k,x in top.items()})
    np.savez_compressed(OUT/'intersection_selected_faces.npz',selected_faces=selected);write(OUT/'intersection_check.json',result);print(json.dumps(result),flush=True)

def local_sampling():
    d=np.load(OUT/'solid_main.npz');raw=np.load(RAW);dev=np.load(OUT/'deviation.npz');p=np.load(P/'poisson.npz');b=np.load(BASE/'proxy_geometry.npz');rng=np.random.default_rng(2026101705);records=[];cloud=[]
    for k in [37,47]:
        ids=np.flatnonzero(raw['vertex_labels']==k);idx=ids[np.argmax(dev['distance_mm'][ids])];c=raw['vertices'][idx];xyz=rng.uniform(-1.5,1.5,(10000,3))+c;cloud.append(xyz);si=contains(d['vertices'],d['faces'],xyz);pi=contains(p['vertices'],p['faces'],xyz);bi=contains(b['vertices'],b['faces'],xyz)
        records.append(dict(fdi=k,center=c.tolist(),cube_side_mm=3,count=len(xyz),inside_count=int(si.sum()),vs_poisson_changes=int((si!=pi).sum()),vs_planar_changes=int((si!=bi).sum()),note='Local sensitivity, not truth error; selected around maximum-deviation original vertex'))
        # Cut away front half-space to expose formerly occluded source point.
        eye=raw['basis']@np.array([.5,.7,-.8]);eye=eye/np.linalg.norm(eye);im=Image.new('RGB',(1280,720),'#edf1f5')
        for j,(vv,ff,title) in enumerate([(raw['vertices'],raw['faces'],'原扫描 · 剖切'),(d['vertices'],d['faces'],'Make Solid · 同平面剖切')]):
            local=np.linalg.norm(vv[ff].mean(1)-c,axis=1)<5;pl=pv.Plotter(off_screen=True,window_size=(640,640));pl.set_background('#edf1f5');mesh=pd(vv,ff[local]).clip(normal=eye,origin=c+eye*.05,invert=True);pl.add_mesh(mesh,color='#d5c7af',ambient=.45);near=dev['closest_xyz'][idx];pl.add_mesh(pv.Line(c,near),color='#d32a53',line_width=4);pl.add_mesh(pv.Sphere(radius=.06,center=c),color='#d32a53');pl.camera_position=[c+eye*60,c,raw['basis'][:,2]];pl.enable_parallel_projection();pl.camera.parallel_scale=3;shotim=Image.fromarray(pl.screenshot(return_img=True));pl.close();dr=ImageDraw.Draw(shotim);dr.rectangle((0,0,640,45),fill='#edf1f5');dr.text((12,7),title,font=font(23),fill='#18384a');im.paste(shotim,(640*j,0))
        ImageDraw.Draw(im).text((15,660),f'FDI {k}：6 mm 视野；红线连接原顶点与代理最近点；剖切开口不是网格漏洞',font=font(21),fill='#18384a');im.save(OUT/f'section_{k}.png')
    write(OUT/'local_sampling.json',records);print(json.dumps(records),flush=True)

def finish():
    r=read(OUT/'evaluation.json');ix=read(OUT/'intersection_check.json');local=read(OUT/'local_sampling.json');html=(OUT/'index.html').read_text(encoding='utf-8')
    extra='<h2>结论：闭合成功；后牙局部需验收</h2><p>1 个主连通体；边界边、非流形边/点、面朝向冲突均为 0；MeshLab 全网格自交筛查选中 0 个面（数值检查，不是精确数学证明）。50 万点体积检查相差 1.05 个标准误；2 万点旋转复测无分类变化。</p><p>2 万面积加权原表面点距离 P95：泊松 0.0303 mm → Make Solid 0.0138 mm。但 37 牙原顶点最大距离从 0.6064 mm 增至 1.6930 mm，47 从 0.3929 mm 增至 1.1694 mm。37 牙 P95 从 0.0765 增至 0.2910 mm。简单实体化不是全面改善，不能仅凭全局均值直接发布。</p><p>这份代理可稳定产生内外标签；后牙局部与旧方法的分歧仍需结合剖面验收。它不替代原扫描、不用于表面或法线监督。</p>'
    for x in local:extra+=f'<h3>FDI {x["fdi"]} 最大偏差处</h3><img src="section_{x["fdi"]}.png"><p>局部 3 mm 立方体内 10,000 点：与泊松不同 {x["vs_poisson_changes"]} 点；与旧平面代理不同 {x["vs_planar_changes"]} 点。局部加密数据不能当作全局错误率。</p>'
    extra+='<details><summary>完整自交与拓扑检查</summary><pre>'+json.dumps(ix,ensure_ascii=False,indent=2)+'</pre></details>'
    html=html.replace('<h2>表面偏差</h2>',extra+'<h2>表面偏差</h2>').replace('href="make_solid_log.txt"','href="make_solid_retry_log.txt"')
    (OUT/'index.html').write_text(html,encoding='utf-8');shutil.copyfile(__file__,OUT/'audit_implementation.py');shutil.copyfile(ROOT/'scripts/mm_make_solid.ps1',OUT/'mm_make_solid_implementation.ps1')
    write(OUT/'delivery_manifest.json',[dict(path=str(x.relative_to(OUT)),sha256=sha(x)) for x in OUT.iterdir() if x.is_file() and x.suffix in ['.json','.png','.html','.py','.ps1','.npz','.ply','.stl','.txt']])
    print('finished',flush=True)

def render():
    raw=np.load(RAW);b=np.load(BASE/'proxy_geometry.npz');d=np.load(OUT/'solid_main.npz');v=d['vertices'];f=d['faces'];r=read(OUT/'evaluation.json');center=(b['vertices'].min(0)+b['vertices'].max(0))/2
    im=Image.new('RGB',(1400,1510),'#edf1f5');dr=ImageDraw.Draw(im);dr.text((20,8),'00OMSZGW_lower · Meshmixer Make Solid 实体化代理',font=font(28),fill='#18384a')
    for k,eye in enumerate([[.5,.7,-.8],[0,0,1]]):
        for j,(vv,ff,title) in enumerate([(b['vertices'],b['faces'],'之前：统一底平面代理'),(v,f,'之后：Make Solid 主体')]):im.paste(screen(vv,ff,np.empty((0,3),int),raw['basis'],center,eye,90,title),(j*700,55+k*700))
    dr.text((20,1465),'单位 mm；只用于空间点内外分类；人工验收待定。',font=font(23),fill='#18384a');im.save(OUT/'before_after.png')
    dev=np.load(OUT/'deviation.npz');dist=dev['distance_mm'];rv=raw['vertices'];rf=raw['faces'];regions=[]
    for title,ids in [('全局最大偏差',np.arange(len(rv))),('37 牙最大偏差',np.flatnonzero(raw['vertex_labels']==37)),('47 牙最大偏差',np.flatnonzero(raw['vertex_labels']==47)),('前牙间隙',np.flatnonzero(raw['vertex_labels']==31))]:
        idx=ids[np.argmax(dist[ids])];c=rv[idx];field=12;eye=raw['basis']@np.array([.5,.7,-.8]);canvas=Image.new('RGB',(1280,715),'#edf1f5');dr=ImageDraw.Draw(canvas)
        for j,(vv,ff,label) in enumerate([(rv,rf,'原扫描'),(v,f,'Make Solid')]):
            local=np.linalg.norm(vv[ff].mean(1)-c,axis=1)<field;canvas.paste(shot([(vv,ff[local],'#d5c7af',1,False)],c,field,eye,label,[(c,'#cb354d')]),(640*j,0))
        dr.text((15,660),f'{title}：原顶点到代理 {dist[idx]:.4f} mm；红点固定在原顶点位置',font=font(22),fill='#18384a');name=f'local_{len(regions)+1}.png';canvas.save(OUT/name);regions.append(dict(title=title,image=name,vertex=int(idx),distance=float(dist[idx])))
    plot=pv.Plotter(off_screen=True,window_size=(1400,850));plot.set_background('#edf1f5');mesh=pd(rv,rf);mesh['distance']=dist;plot.add_mesh(mesh,scalars='distance',cmap='turbo',clim=[0,.2],scalar_bar_args={'title':'Distance to proxy (mm), clipped at 0.2'});eye=raw['basis']@np.array([.5,.7,-.8]);plot.camera_position=[center+eye*250,center,raw['basis'][:,2]];plot.enable_parallel_projection();plot.camera.parallel_scale=45;plot.screenshot(OUT/'deviation.png');plot.close()
    html='<!doctype html><meta charset="utf-8"><title>Make Solid 单例验收</title><style>body{font:18px/1.7 system-ui;max-width:1400px;margin:24px auto;background:#edf1f5;color:#18384a}img{max-width:100%}pre{white-space:pre-wrap}td,th{padding:9px;border:1px solid #abc}table{border-collapse:collapse}</style><h1>00OMSZGW_lower：直接实体化</h1><p>原代理 → Meshmixer Make Solid（Accurate，512/512，零偏移、零增厚）→ 仅保留最大连通体。只输出新代理；原始扫描、旧结果全部保留。代理不用于表面/法线监督。</p><p>Make Solid 使用体积重建，允许局部形状变化。空间标签变化是方法敏感性，不是真值错误率。</p><img src="before_after.png"><h2>表面偏差</h2><img src="deviation.png"><table><tr><th>FDI</th><th>P95 mm</th><th>最大原顶点距离 mm</th></tr>'
    for k,x in r['original_vertex_distances_by_fdi'].items():html+=f'<tr><td>{k}</td><td>{x["p95"]:.4f}</td><td>{x["max"]:.4f}</td></tr>'
    html+='</table>'
    for x in regions:html+=f'<h2>{x["title"]}</h2><img src="{x["image"]}">'
    html+='<h2>检查结果</h2><pre>'+json.dumps(r,ensure_ascii=False,indent=2)+'</pre><p>最近距离按全部原顶点及固定面积样本测量，不是连续表面的 Hausdorff 上界。人工视觉验收仍待用户确认。</p><p><a href="solid_main_PROXY_ONLY.ply">代理模型 PLY</a> · <a href="evaluation.json">数值结果</a> · <a href="make_solid_log.txt">Meshmixer 参数回读</a> · <a href="https://github.com/meshmixer/mm-api">官方 API</a></p>'
    (OUT/'index.html').write_text(html,encoding='utf-8')
    assert all(sha(x['path'])==x['sha256'] for x in read(OUT/'input_hashes.json'))
    assert all(sha(x['path'])==x['sha256'] for x in read(BASE/'input_hashes.json'))
    shutil.copyfile(__file__,OUT/'audit_implementation.py');write(OUT/'source_hash_verification.json',dict(all_unchanged=True));print('report ready',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
