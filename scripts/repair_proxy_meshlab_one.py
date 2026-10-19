"""Isolated MeshLab repair pilot for one proxy, not training surfaces."""
from pathlib import Path
import sys,json,time,shutil,importlib.metadata
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/dmm_meshlab_proxy_00OMSZGW_20261017_01'
BASE=ROOT/'runs/dmm_proxy_plane_00OMSZGW_20261016_01'
sys.path.insert(0,str(OUT/'deps'))
from diagnose_preprocess12 import read,write,sha,now,font
from audit_space_proxy12 import pd,contains,screen,edge_orientation_metrics

def repair():
    import pymeshlab as ml
    OUT.mkdir(exist_ok=True);d=np.load(BASE/'proxy_geometry.npz')
    if not (OUT/'provenance.json').exists():write(OUT/'provenance.json',dict(created_utc=now(),case_id='00OMSZGW_lower',input=str(BASE/'proxy_geometry.npz'),sha256=sha(BASE/'proxy_geometry.npz'),pymeshlab_version=importlib.metadata.version('pymeshlab'),scope='proxy only; original training surface not changed',human_review='PENDING'))
    assert sha(BASE/'proxy_geometry.npz')==read(OUT/'provenance.json')['sha256']
    ms=ml.MeshSet();ms.add_mesh(ml.Mesh(vertex_matrix=d['vertices'],face_matrix=d['faces'].astype(np.int32)),'proxy')
    logs=[]
    def stats(stage):
        ms.compute_selection_by_self_intersections_per_face();m=ms.current_mesh();top=ms.get_topological_measures();sel=m.face_selection_array()
        record=dict(stage=stage,vertices=m.vertex_number(),faces=m.face_number(),selected_intersecting_faces=int(sel.sum()),topology={k:(v.item() if hasattr(v,'item') else v) for k,v in top.items()});logs.append(record);print(json.dumps(record),flush=True)
        if not (OUT/(stage+'.npz')).exists():
            np.savez_compressed(OUT/(stage+'.npz'),vertices=m.vertex_matrix(),faces=m.face_matrix(),selected_faces=sel)
            ms.save_current_mesh(str(OUT/(stage+'_PROXY_ONLY.ply')))
        return record
    stats('baseline')
    for op in ['meshing_remove_duplicate_vertices','meshing_remove_duplicate_faces','meshing_remove_null_faces','meshing_remove_folded_faces','meshing_remove_unreferenced_vertices']:
        ms.apply_filter(op)
    for step in range(1,4):
        ms.compute_selection_by_self_intersections_per_face();count=int(ms.current_mesh().face_selection_array().sum())
        if count:ms.meshing_remove_selected_faces()
        ms.meshing_repair_non_manifold_edges(method=0);ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        ms.meshing_close_holes(maxholesize=1000,selected=False,newfaceselected=True,selfintersection=True,refinehole=False)
        ms.meshing_remove_unreferenced_vertices();ms.meshing_re_orient_faces_coherently()
        r=stats('local_pass_'+str(step))
        if r['selected_intersecting_faces']==0 and r['topology']['boundary_edges']==0:break
    write(OUT/'meshlab_repair_log.json',logs)
    write(OUT/'local_candidate.json',dict(stage=r['stage'],mesh_file=r['stage']+'.npz',method='MeshLab local cleanup, delete self-intersecting faces, split nonmanifold vertices, close holes up to 1000 edges with intersection heuristic; at most 3 passes'))

def meshfix():
    import pymeshfix
    dest=OUT/'meshfix';dest.mkdir(exist_ok=False);d=np.load(BASE/'proxy_geometry.npz');t=time.time()
    print('MeshFix start on original planar proxy',flush=True)
    v,f=pymeshfix.clean_from_arrays(np.ascontiguousarray(d['vertices'],dtype=np.float64),np.ascontiguousarray(d['faces'],dtype=np.int32),verbose=True,joincomp=False,remove_smallest_components=False)
    np.savez_compressed(dest/'meshfix.npz',vertices=v,faces=f);pd(v,f).save(dest/'meshfix_PROXY_ONLY.ply')
    write(dest/'local_candidate.json',dict(stage='meshfix',mesh_file='meshfix.npz',method='MeshFix clean_from_arrays from original planar proxy; joincomp=False, remove_smallest_components=False',version=importlib.metadata.version('pymeshfix'),elapsed_seconds=time.time()-t))
    shutil.copyfile(OUT/'provenance.json',dest/'provenance.json');shutil.copyfile(OUT/'meshlab_repair_log.json',dest/'meshlab_repair_log.json')
    print('MeshFix saved',len(v),len(f),flush=True)

def poisson():
    import pymeshlab as ml
    dest=OUT/'poisson';dest.mkdir(exist_ok=False);d=np.load(BASE/'proxy_geometry.npz');tri=d['vertices'][d['faces']];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(normal,axis=1);normal/=area[:,None]
    rng=np.random.default_rng(2026101702);ids=rng.choice(len(tri),250000,p=area/area.sum());uv=rng.random((len(ids),2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1];points=tri[ids,0]+uv[:,0,None]*(tri[ids,1]-tri[ids,0])+uv[:,1,None]*(tri[ids,2]-tri[ids,0])
    ms=ml.MeshSet();ms.add_mesh(ml.Mesh(vertex_matrix=points,v_normals_matrix=normal[ids]),'proxy oriented samples');t=time.time()
    params=dict(depth=9,fulldepth=5,scale=1.1,samplespernode=1.5,pointweight=4.,iters=8,confidence=False,preclean=False,threads=4)
    ms.generate_surface_reconstruction_screened_poisson(**params);m=ms.current_mesh();v=m.vertex_matrix();f=m.face_matrix()
    np.savez_compressed(dest/'poisson.npz',vertices=v,faces=f);ms.save_current_mesh(str(dest/'poisson_PROXY_ONLY.ply'))
    write(dest/'local_candidate.json',dict(stage='poisson',mesh_file='poisson.npz',method='MeshLab Screened Poisson from 250000 area-weighted oriented samples of original planar proxy',sampling_seed=2026101702,parameters=params,elapsed_seconds=time.time()-t))
    shutil.copyfile(OUT/'provenance.json',dest/'provenance.json');shutil.copyfile(OUT/'meshlab_repair_log.json',dest/'meshlab_repair_log.json');print('Poisson saved',len(v),len(f),flush=True)

def evaluate():
    import vtk
    from rebuild_single_proxy_plane import intersections
    choice=read(OUT/'local_candidate.json');d=np.load(OUT/choice['mesh_file']);v=d['vertices'];f=d['faces'];b=np.load(BASE/'proxy_geometry.npz')
    # Independent all-face scan with candidate verification using separating axes.
    ix=intersections(v,f);verified=[];adj=[]
    for name,target in [('nonadjacent_pairs',verified),('adjacent_overlap_candidates',adj)]:
        for i,j in ix[name]:
            a=v[f[i]].copy();c=v[f[j]].copy()
            if name.startswith('adjacent'):a+=(a.mean(0)-a)*1e-7;c+=(c.mean(0)-c)*1e-7
            ea=np.roll(a,-1,axis=0)-a;eb=np.roll(c,-1,axis=0)-c;na=np.cross(ea[0],ea[1]);nb=np.cross(eb[0],eb[1]);axes=np.concatenate([[na,nb],np.cross(ea[:,None],eb[None,:]).reshape(-1,3),np.cross(ea,na),np.cross(eb,nb)])
            norms=np.linalg.norm(axes,axis=1);axes=axes[norms>1e-15]/norms[norms>1e-15,None];pa=a@axes.T;pb=c@axes.T
            if not np.any((pa.max(0)<pb.min(0)-1e-10)|(pb.max(0)<pa.min(0)-1e-10)):target.append([i,j])
    write(OUT/'independent_intersections.json',dict(nonadjacent_pairs=verified,adjacent_overlap_candidates=adj,raw_candidate_counts=[len(ix['nonadjacent_pairs']),len(ix['adjacent_overlap_candidates'])],triangle_tests=ix['triangle_tests']))
    bc,nc,wc=edge_orientation_metrics(f);tri=v[f];vol=float(np.einsum('ij,ij->i',tri[:,0]-v.mean(0),np.cross(tri[:,1]-v.mean(0),tri[:,2]-v.mean(0))).sum()/6)
    result=dict(stage=choice['stage'],boundary_edges=bc,nonmanifold_edges=nc,inconsistent_edges=wc,independent_intersection_pairs=len(verified)+len(adj),signed_volume_mm3=vol)
    # Nearest-triangle distances, bidirectional area-weighted deterministic surface probes.
    rng=np.random.default_rng(2026101701)
    def sample(a,fs,n):
        t=a[fs];ar=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1);ids=rng.choice(len(fs),n,p=ar/ar.sum());uv=rng.random((n,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1];return t[ids,0]+uv[:,0,None]*(t[ids,1]-t[ids,0])+uv[:,1,None]*(t[ids,2]-t[ids,0])
    def distance(points,a,fs):
        loc=vtk.vtkStaticCellLocator();loc.SetDataSet(pd(a,fs));loc.BuildLocator();out=[]
        for x in points:
            closest=[0.,0.,0.];ci=vtk.reference(0);si=vtk.reference(0);d2=vtk.reference(0.);loc.FindClosestPoint(x,closest,ci,si,d2);out.append(np.sqrt(float(d2)))
        return np.asarray(out)
    probes=sample(b['vertices'],b['source_faces'],20000);dist=distance(probes,v,f)
    result['original_surface_to_repaired_mm']=dict(count=len(dist),median=float(np.median(dist)),p95=float(np.quantile(dist,.95)),p99=float(np.quantile(dist,.99)),max=float(dist.max()))
    back=sample(v,f,20000);bd=distance(back,b['vertices'],b['faces']);result['repaired_to_previous_proxy_mm']=dict(count=len(bd),median=float(np.median(bd)),p95=float(np.quantile(bd,.95)),p99=float(np.quantile(bd,.99)),max=float(bd.max()))
    np.savez_compressed(OUT/'geometry_deviation_samples.npz',original_surface_points=probes,original_to_repaired_distance_mm=dist,repaired_points=back,repaired_to_previous_distance_mm=bd)
    if bc==0 and nc==0 and wc==0:
        points=np.load(BASE/'space_candidates_DIAGNOSTIC_ONLY.npz')['xyz'];mask=contains(v,f,points);lo,hi=read(BASE/'classification.json')['domain'];box=float(np.prod(np.array(hi)-lo));p=float(mask.mean());se=box*np.sqrt(p*(1-p)/len(points));ids=rng.choice(len(points),20000,replace=False);q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,0]*=np.linalg.det(q);other=contains((v-v.mean(0))@q,f,(points[ids]-v.mean(0))@q)
        result['classification']=dict(count=len(points),monte_carlo_volume_mm3=box*p,standard_error_mm3=se,difference_standard_errors=abs(box*p-abs(vol))/se,rotation_check_count=len(ids),rotation_disagreements=int((other!=mask[ids]).sum()))
        np.savez_compressed(OUT/'space_labels_DIAGNOSTIC_ONLY.npz',xyz=points,inside=mask)
    else:result['classification']={'status':'SKIPPED_OPEN_OR_NONMANIFOLD'}
    result['status']='NUMERICAL_PASS_PENDING_VISUAL' if not(bc or nc or wc or len(verified) or len(adj)) and result['classification'].get('difference_standard_errors',999)<=5 and result['classification'].get('rotation_disagreements',1)==0 else 'HOLD'
    result['training_export']=False;write(OUT/'evaluation.json',result);print(json.dumps(result,indent=2),flush=True)

def render():
    from PIL import Image,ImageDraw
    import pyvista as pv
    choice=read(OUT/'local_candidate.json');d=np.load(OUT/choice['mesh_file']);b=np.load(BASE/'proxy_geometry.npz');raw=np.load(ROOT/'runs/dmm_preprocess12_20261016T1815_r1/cases/00OMSZGW_lower/raw_index_preserved.npz');center=(b['vertices'].min(0)+b['vertices'].max(0))/2
    im=Image.new('RGB',(1400,1520),'#edf1f5');dr=ImageDraw.Draw(im);dr.text((20,12),'00OMSZGW_lower · '+choice['stage']+' 代理修复前后',font=font(29),fill='#18384a')
    for k,eye in enumerate([[.5,.7,-.8],[0,0,1]]):
        for j,(v,f,title) in enumerate([(b['vertices'],b['faces'],'修复前：统一底平面代理'),(d['vertices'],d['faces'],'修复后：'+choice['stage'])]):im.paste(screen(v,f,np.empty((0,3),int),raw['basis'],center,eye,90,title),(j*700,60+k*700))
    dr.text((20,1470),'仅修改独立代理；原始扫描/表面采样不使用修复网格。',font=font(23),fill='#18384a');im.save(OUT/'before_after.png')
    result=read(OUT/'evaluation.json');assert sha(BASE/'proxy_geometry.npz')==read(OUT/'provenance.json')['sha256']
    assert all(sha(x['path'])==x['sha256'] for x in read(BASE/'input_hashes.json'))
    dev=np.load(OUT/'geometry_deviation_samples.npz');imdev=Image.new('RGB',(1400,860),'#edf1f5');dv=ImageDraw.Draw(imdev);dv.text((20,10),'原扫描表面到修复代理 · 最近三角面距离（面积加权抽样）',font=font(28),fill='#18384a')
    for j,eye in enumerate([[.5,.7,-.8],[0,0,-1]]):
        pl=pv.Plotter(off_screen=True,window_size=(700,700));pl.set_background('#edf1f5');pl.add_mesh(pd(b['vertices'],b['source_faces']),color='#d5c7af',ambient=.4)
        pts=pv.PolyData(dev['original_surface_points']);pts['distance_mm']=dev['original_to_repaired_distance_mm'];pl.add_mesh(pts,scalars='distance_mm',cmap='turbo',clim=[0,.2],point_size=4,render_points_as_spheres=True,scalar_bar_args={'title':'Distance (mm); colors clipped at 0.2'})
        e=raw['basis']@np.array(eye,float);e/=np.linalg.norm(e);up=raw['basis'][:,1] if j else raw['basis'][:,2];pl.camera_position=[center+e*250,center,up];pl.enable_parallel_projection();pl.camera.parallel_scale=45
        imdev.paste(Image.fromarray(pl.screenshot(return_img=True)),(j*700,65));pl.close()
    stats=result['original_surface_to_repaired_mm'];dv.text((20,785),f'20,000 点：P95 {stats["p95"]:.4f} mm / P99 {stats["p99"]:.4f} mm / 抽样最大 {stats["max"]:.4f} mm',font=font(23),fill='#18384a');imdev.save(OUT/'geometry_deviation.png')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MeshLab 单例代理修复</title><style>body{font:18px/1.7 system-ui;max-width:1450px;margin:30px auto;background:#edf1f5;color:#18384a}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>MeshLab 单例代理修复</h1><p>病例 00OMSZGW_lower。只修改代理副本，原始扫描与上一版代理不变。状态：'+result['status']+'</p><img src="before_after.png"><h2>独立复核</h2><pre>'+json.dumps(result,ensure_ascii=False,indent=2)+'</pre><h2>MeshLab 各步日志</h2><pre>'+json.dumps(read(OUT/'meshlab_repair_log.json'),ensure_ascii=False,indent=2)+'</pre><p>几何距离是固定随机种子、面积加权采样点到三角面的最近距离，不是全表面 Hausdorff 上界。人工验收 PENDING。</p><a href="'+choice['stage']+'_PROXY_ONLY.ply">修复后代理 PLY</a> · <a href="geometry_deviation_samples.npz">几何偏差样本</a>',encoding='utf-8')
    page=(OUT/'index.html').read_text(encoding='utf-8').replace('MeshLab 单例代理修复','开源工具单例代理修复').replace('<h2>MeshLab 各步日志</h2>','<h2>MeshLab 局部修复对照（未通过，保留孔洞）</h2>')
    page=page.replace('<h2>独立复核</h2>','<h2>几何变化</h2><img src="geometry_deviation.png"><h2>独立复核</h2>')
    if choice['stage']=='meshfix':page=page.replace('<h1>开源工具单例代理修复</h1>','<h1>MeshFix 代理修复；MeshLab 局部修复作对照</h1><p>MeshFix 从统一底平面的原代理重新开始，不使用 MeshLab 清理后的失败网格。joincomp=False，remove_smallest_components=False。原始扫描、旧代理与旧审核包相关输入哈希核对一致。</p>')
    if choice['stage']=='poisson':
        page=page.replace('<h1>开源工具单例代理修复</h1>','<h1>MeshLab Screened Poisson 代理重建</h1><p>从统一底平面的原代理独立开始。25 万面积加权表面点与面法线，depth=9；重建网格只作内外分类代理，不用于原始表面或法线监督。</p>')
        page+='<h2>连通分量与拓扑</h2><pre>'+json.dumps(read(OUT/'meshlab_topology.json'),ensure_ascii=False,indent=2)+'</pre><p>保留 26 个微小闭合片，总面积 4.667 mm²（约 0.055%）。尚未按位置判定其合理性，不自动删除。</p><a href="components.json">分量面积明细</a>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    shutil.copyfile(__file__,OUT/'implementation.py');write(OUT/'delivery_manifest.json',[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()]);print('report ready',flush=True)

def finish():
    a=read(OUT/'meshlab_repair_log.json')[-1];m=read(OUT/'meshfix/evaluation.json');p=read(OUT/'poisson/evaluation.json')
    html='<!doctype html><meta charset="utf-8"><title>00OMSZGW 开源工具代理修复验收</title><style>body{font:18px/1.7 system-ui;max-width:1450px;margin:30px auto;background:#edf1f5;color:#18384a}img{max-width:100%}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #aebbc7}</style><h1>00OMSZGW_lower：开源工具代理修复对照</h1><p>只处理本例代理。原始数据、统一底平面代理及全部试验版本均保留，输入哈希核对一致。没有训练导出。</p><table><tr><th>方法</th><th>边界边</th><th>独立复核相交对</th><th>原表面距离 P95 / 抽样最大 mm</th><th>状态</th></tr>'
    html+=f'<tr><td>MeshLab 局部清理 + 补洞（3轮）</td><td>{a["topology"]["boundary_edges"]}</td><td>已有孔洞，未追加全量独立扫描；MeshLab 仍选中 {a["selected_intersecting_faces"]} 个面</td><td>未测量</td><td>不采用</td></tr>'
    for name,r in [('MeshFix',m),('MeshLab Screened Poisson',p)]:
        ds=r['original_surface_to_repaired_mm'];html+=f'<tr><td>{name}</td><td>{r["boundary_edges"]}</td><td>{r["independent_intersection_pairs"]}</td><td>{ds["p95"]:.4f} / {ds["max"]:.4f}</td><td>{r["status"]}</td></tr>'
    html+='</table><p>孔洞关闭、体积一致并不能单独证明修复合格。相交检查采用 VTK 初筛与分离轴二次复核；共享顶点/边以微小内缩排除正常接触，属于浮点数值检测。距离来自 2 万个面积加权点到三角面的最近距离，不是全表面最坏误差上界。</p><h2>泊松重建候选，待人工验收</h2><a href="poisson/index.html">完整数值与模型下载</a><img src="poisson/before_after.png"><img src="poisson/geometry_deviation.png"><p>泊松结果保留了 26 个微小闭合片（总面积约 0.055%），需结合位置审核；尚未自动删除或发布。</p><h2>MeshFix 对照（不采用）</h2><a href="meshfix/index.html">失败结果与几何偏差</a><h2>实现及来源</h2><a href="https://pymeshlab.readthedocs.io/en/latest/filter_list.html">MeshLab 官方滤镜文档</a> · <a href="https://pymeshfix.pyvista.org/">MeshFix 官方文档</a> · <a href="meshlab_repair_log.json">MeshLab 三轮日志</a>'
    (OUT/'index.html').write_text(html,encoding='utf-8');shutil.copyfile(__file__,OUT/'implementation.py')
    assert all(sha(x['path'])==x['sha256'] for x in read(BASE/'input_hashes.json'))
    write(OUT/'delivery_manifest.json',[dict(path=str(q.relative_to(OUT)),sha256=sha(q)) for q in sorted(OUT.rglob('*')) if q.is_file() and 'deps' not in q.relative_to(OUT).parts]);print('master review ready')

if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[2] in ['meshfix','poisson']:OUT=OUT/sys.argv[2]
    globals()[sys.argv[1]]()
