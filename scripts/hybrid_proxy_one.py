"""User-directed hybrid: locally repaired dental mesh + Make Solid outer shell."""
from pathlib import Path
import sys,json,shutil
import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from diagnose_preprocess12 import read,write,sha
from audit_space_proxy12 import pd,edge_orientation_metrics
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/dmm_hybrid_proxy_00OMSZGW_20261017_01'
SPLIT=ROOT/'runs/dmm_ms_split_00OMSZGW_20261017_01'
BASE=ROOT/'runs/dmm_proxy_plane_00OMSZGW_20261016_01'
RAW=ROOT/'runs/dmm_preprocess12_20261016T1815_r1/cases/00OMSZGW_lower/raw_index_preserved.npz'
sys.path.insert(0,str(ROOT/'runs/dmm_meshlab_proxy_00OMSZGW_20261017_01/deps'))

def prepare():
    import pymeshlab as ml
    OUT.mkdir(exist_ok=False);b=np.load(BASE/'proxy_geometry.npz');raw=np.load(RAW);part=np.load(SPLIT/'partition.npz');source_ids=np.flatnonzero(part['face_is_dental']);fs=b['faces'][source_ids];used,idx=np.unique(fs,return_inverse=True);v=b['vertices'][used];f=idx.reshape(-1,3)
    n,lab=connected_components(coo_matrix((np.ones(f.size,np.int8),(f.ravel(),np.roll(f,-1,axis=1).ravel())),shape=(len(v),len(v))),directed=False);fl=lab[f[:,0]];main=np.argmax(np.bincount(fl));removed=source_ids[fl!=main];assert np.all(raw['vertex_labels'][raw['faces'][removed]]==0)
    source_ids=source_ids[fl==main];f=f[fl==main];ids,idx=np.unique(f,return_inverse=True);v=v[ids];f=idx.reshape(-1,3)
    np.savez_compressed(OUT/'dental_original.npz',vertices=v,faces=f,source_face_ids=source_ids)
    write(OUT/'input_hashes.json',[dict(path=str(x),sha256=sha(x)) for x in [RAW,BASE/'proxy_geometry.npz',SPLIT/'partition.npz',ROOT/'runs/mt_split_outer.stl']])
    ms=ml.MeshSet();ms.add_mesh(ml.Mesh(vertex_matrix=v,face_matrix=f.astype(np.int32)));logs=[]
    def save(stage):
        ms.compute_selection_by_self_intersections_per_face();m=ms.current_mesh();selection=m.face_selection_array();top=ms.get_topological_measures();r=dict(stage=stage,self_intersecting_faces=int(selection.sum()),topology={k:(x.item() if hasattr(x,'item') else x) for k,x in top.items()});logs.append(r);np.savez_compressed(OUT/(stage+'.npz'),vertices=m.vertex_matrix(),faces=m.face_matrix(),selected_faces=selection);print(json.dumps(r),flush=True);return r
    save('dental_before')
    ms.compute_selection_by_self_intersections_per_face();sel=ms.current_mesh().face_selection_array();src=source_ids[sel];write(OUT/'local_repair_selection.json',dict(removed_unlabeled_fragment_faces=len(removed),intersecting_faces=int(sel.sum()),intersecting_faces_touching_fdi=int((raw['vertex_labels'][raw['faces'][src]]>0).any(1).sum()),repair_scope='Delete detected self-intersecting faces, remove zero/duplicate faces, split nonmanifold boundary vertices without displacement, close holes; no Make Solid, no smoothing'))
    if sel.any():ms.meshing_remove_selected_faces()
    ms.meshing_remove_duplicate_faces();ms.meshing_remove_null_faces();ms.meshing_remove_unreferenced_vertices();ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    ms.meshing_close_holes(maxholesize=10000,selected=False,newfaceselected=True,selfintersection=True,refinehole=False)
    ms.meshing_re_orient_faces_coherently();r=save('dental_local_repaired');write(OUT/'local_repair_log.json',logs)
    pd(ms.current_mesh().vertex_matrix(),ms.current_mesh().face_matrix()).save(OUT/'dental_local_repaired.ply')

def meshfix():
    import pymeshfix
    d=np.load(OUT/'dental_local_repaired.npz');v,f=pymeshfix.clean_from_arrays(np.ascontiguousarray(d['vertices'],dtype=np.float64),np.ascontiguousarray(d['faces'],dtype=np.int32),verbose=True,joincomp=False,remove_smallest_components=False)
    np.savez_compressed(OUT/'dental_meshfix.npz',vertices=v,faces=f);pd(v,f).save(OUT/'dental_meshfix.ply')
    import pymeshlab as ml
    ms=ml.MeshSet();ms.add_mesh(ml.Mesh(vertex_matrix=v,face_matrix=f.astype(np.int32)));ms.compute_selection_by_self_intersections_per_face();top=ms.get_topological_measures();r=dict(self_intersecting_faces=int(ms.current_mesh().face_selection_array().sum()),topology={k:(x.item() if hasattr(x,'item') else x) for k,x in top.items()});write(OUT/'meshfix_check.json',r);print(json.dumps(r),flush=True)

def closeout():
    from localize_poisson_proxy_issues import locator,closest
    from audit_space_proxy12 import screen
    from scipy.spatial import cKDTree
    from PIL import Image,ImageDraw
    from diagnose_preprocess12 import font
    raw=np.load(RAW);orig=np.load(OUT/'dental_original.npz');b=np.load(BASE/'proxy_geometry.npz');center=(b['vertices'].min(0)+b['vertices'].max(0))/2;records=[]
    im=Image.new('RGB',(2100,800),'#edf1f5')
    for col,(name,title) in enumerate([('dental_original','分区牙体原始表面'),('dental_local_repaired','MeshLab 局部补洞：仍未通过'),('dental_meshfix','MeshFix 补洞修复：后牙损伤，不采用')]):
        d=np.load(OUT/(name+'.npz'));im.paste(screen(d['vertices'],d['faces'],np.empty((0,3),int),raw['basis'],center,[.5,.7,-.8],90,title),(700*col,0))
        if col:
            dist,near,face=closest(locator(d['vertices'],d['faces']),raw['vertices']);stats={}
            for k in np.unique(raw['vertex_labels']):
                if k==0:continue
                ds=dist[raw['vertex_labels']==k];stats[str(int(k))]=dict(count=len(ds),p95_mm=float(np.quantile(ds,.95)),max_mm=float(ds.max()),over_0_1mm=int((ds>.1).sum()))
            np.savez_compressed(OUT/(name+'_deviation.npz'),distance_mm=dist,closest_xyz=near)
            # Coordinates retained to a tolerance; absent source vertices are not described as moved.
            td=cKDTree(d['vertices']).query(orig['vertices'])[0];records.append(dict(stage=name,faces=len(d['faces']),vertices=len(d['vertices']),source_vertices_represented_within_0_00001mm=int((td<1e-5).sum()),source_vertex_count=len(td),by_fdi=stats))
    ImageDraw.Draw(im).text((20,730),'候选均保留，未替换原始数据；局部修复失败，不进行强行拼合或继续调参。',font=font(29),fill='#18384a');im.save(OUT/'closeout_comparison.png')
    # Zoom into the failing 36/37 and 47 areas with a shared camera for each triplet.
    for fdi in [37,47]:
        ids=np.flatnonzero(raw['vertex_labels']==fdi);c=raw['vertices'][ids].mean(0);eye=raw['basis']@np.array([.5,.7,-.8]);canvas=Image.new('RGB',(1800,680),'#edf1f5')
        for col,(name,title) in enumerate([('dental_original','原始牙体'),('dental_local_repaired','局部补洞'),('dental_meshfix','自动修复：不采用')]):
            d=np.load(OUT/(name+'.npz'));vv=d['vertices'];ff=d['faces'];sel=np.linalg.norm(vv[ff].mean(1)-c,axis=1)<10;pl=pv.Plotter(off_screen=True,window_size=(600,600));pl.set_background('#edf1f5');pl.add_mesh(pd(vv,ff[sel]),color='#d5c7af',ambient=.4);pl.camera_position=[c+eye*100,c,raw['basis'][:,2]];pl.enable_parallel_projection();pl.camera.parallel_scale=8;pic=Image.fromarray(pl.screenshot(return_img=True));pl.close();dr=ImageDraw.Draw(pic);dr.rectangle((0,0,600,43),fill='#edf1f5');dr.text((12,7),title,font=font(23),fill='#18384a');canvas.paste(pic,(600*col,0))
        ImageDraw.Draw(canvas).text((15,620),f'FDI {fdi} 邻域；同位置同视角，16 mm 视野；对照用于说明修复损伤。',font=font(25),fill='#18384a');canvas.save(OUT/f'closeout_{fdi}.png')
    r=dict(status='CLOSED_OUT_REJECTED_DENTAL_REPAIR',reason='MeshLab retained openings/self-intersection candidates. MeshFix closed the region but substantially removed posterior geometry; no combined proxy is accepted or exported.',dental_make_solid_used=False,outer_make_solid_completed=True,outer_result=str(ROOT/'runs/mt_split_outer.stl'),union_performed=False,training_export=False,full_cohort_run=False,source_unchanged=True,stages=records)
    write(OUT/'closeout.json',r)
    html='<!doctype html><meta charset="utf-8"><title>分区修复试验收口</title><style>body{font:18px/1.7 system-ui;max-width:1500px;margin:25px auto;background:#edf1f5;color:#18384a}img{max-width:100%}td,th{padding:10px;border:1px solid #abc}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>00OMSZGW_lower：分区修复试验收口</h1><p><b>结论：牙体局部修复未通过保真验收，停止这条试验，不强行拼合、不追加参数或方法。</b></p><p>按最新要求，牙体未采用 Make Solid。只尝试去除无牙号孤立小片、修复局部自交、补洞。外围牙龈/侧壁/底面的独立 Make Solid 已完成。所有中间结果保留，不替换原始扫描、现有数据或训练输入。</p><p>MeshLab 局部补洞后仍有 89 条边界边，数值自交筛查选中 655 个面；MeshFix 后闭合且筛查为 0，但从 191,172 个牙体区域原面降为 144,354 个面，后牙几何显著丢失。闭合并不代表修复合格。</p><table><tr><th>牙号</th><th>MeshFix 原顶点到修复面 P95 mm</th><th>最大 mm</th></tr>'
    for k in ['36','37','46','47']:
        s=records[-1]['by_fdi'][k];html+=f'<tr><td>{k}</td><td>{s["p95_mm"]:.3f}</td><td>{s["max_mm"]:.3f}</td></tr>'
    html+='</table><img src="closeout_comparison.png"><h2>后牙局部</h2><img src="closeout_37.png"><img src="closeout_47.png"><h2>试验边界</h2><p>未进行牙体/外围并集，未生成可采用的混合代理。MeshFix 运行还输出了 “could not fix everything”；独立 MeshLab 检查虽显示闭合，但几何保真已经足以否决该结果。以上距离来自原始顶点到三角面，不是全表面 Hausdorff 上界。</p><h2>回到 DMM 的主要问题</h2><p>当前结果证明自动修复会引入形状变化，尚未证明封底问题导致新版初始模板牙变差。下一步讨论应首先明确训练实际读取哪些空间点字段和监督项，再用同一病例比较新旧预处理输出的尺度、对齐、牙号、表面采样与法线。必要时做固定模型和参数的数据对照，避免同时更换数据、模型和设置。这里仅列出分析方向，没有启动新实验。</p><details><summary>详细数值与处理记录</summary><pre>'+json.dumps(r,ensure_ascii=False,indent=2)+'</pre></details><p><a href="closeout.json">收口记录</a> · <a href="local_repair_log.json">补洞日志</a> · <a href="meshfix_check.json">独立拓扑检查</a></p>'
    (OUT/'index.html').write_text(html,encoding='utf-8')
    assert all(sha(x['path'])==x['sha256'] for x in read(OUT/'input_hashes.json'));assert all(sha(x['path'])==x['sha256'] for x in read(BASE/'input_hashes.json'));write(OUT/'source_hash_verification.json',dict(all_unchanged=True));shutil.copyfile(__file__,OUT/'implementation.py');write(OUT/'delivery_manifest.json',[dict(path=str(x.relative_to(OUT)),sha256=sha(x)) for x in OUT.iterdir() if x.is_file()]);print('CLOSED OUT. No union or training export.',flush=True)

def clarify_closeout():
    note='本次实现把分区形成的开放接合边界也交给了大孔自动补洞（maxholesize=10000），产生跨越牙弓的大补片；随后自动去自交又删伤了后牙。该失败首先属于本次处理策略，不能据此判断原始牙体需要重建，也不能推断它导致新版 DMM 模板变差。真实扫描小孔、分区接口和封底边界必须区分。'
    r=read(OUT/'closeout.json');r['identified_workflow_error']=note
    (OUT/'closeout.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    html=(OUT/'index.html').read_text(encoding='utf-8');html=html.replace('<h2>试验边界</h2>','<h2>这次试验走偏的位置</h2><p>'+note+'</p><h2>试验边界</h2>');(OUT/'index.html').write_text(html,encoding='utf-8');shutil.copyfile(__file__,OUT/'implementation.py')
    manifest=[dict(path=str(x.relative_to(OUT)),sha256=sha(x)) for x in OUT.iterdir() if x.is_file() and x.name!='delivery_manifest.json'];(OUT/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');assert all(sha(OUT/x['path'])==x['sha256'] for x in manifest);assert all(sha(x['path'])==x['sha256'] for x in read(OUT/'input_hashes.json'));print('Closeout verified; workflow mistake documented; inputs unchanged.',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
