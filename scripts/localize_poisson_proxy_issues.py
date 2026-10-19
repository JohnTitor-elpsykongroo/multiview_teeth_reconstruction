"""Read-only localization and sampling sensitivity for one frozen Poisson proxy."""
from pathlib import Path
import sys,json,math,time,shutil
from fractions import Fraction as F
import numpy as np
import vtk,pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image,ImageDraw
from diagnose_preprocess12 import read,write,sha,font
from audit_space_proxy12 import contains,pd
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'runs/dmm_meshlab_proxy_00OMSZGW_20261017_01/poisson'
B=ROOT/'runs/dmm_proxy_plane_00OMSZGW_20261016_01'
RAW=ROOT/'runs/dmm_preprocess12_20261016T1815_r1/cases/00OMSZGW_lower'
OUT=ROOT/'runs/dmm_poisson_local_review_00OMSZGW_20261017_01'

def locator(v,f):
    loc=vtk.vtkStaticCellLocator();loc.SetDataSet(pd(v,f));loc.BuildLocator();return loc

def closest(loc,points):
    dist=[];pos=[];faces=[];c=[0.,0.,0.];ci=vtk.reference(0);si=vtk.reference(0);d2=vtk.reference(0.)
    for k,p in enumerate(points):
        loc.FindClosestPoint(p,c,ci,si,d2);dist.append(math.sqrt(max(0,float(d2))));pos.append(c.copy());faces.append(int(ci))
        if k and k%100000==0:print('nearest triangles',k,'/',len(points),flush=True)
    return np.asarray(dist),np.asarray(pos),np.asarray(faces)

def dot(a,b):return sum(x*y for x,y in zip(a,b))
def sub(a,b):return [x-y for x,y in zip(a,b)]
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]

def analyze():
    OUT.mkdir(exist_ok=False);(OUT/'pairs').mkdir();(OUT/'components').mkdir();(OUT/'deviation').mkdir()
    files=[P/'poisson.npz',P/'independent_intersections.json',P/'geometry_deviation_samples.npz',P/'space_labels_DIAGNOSTIC_ONLY.npz',B/'proxy_geometry.npz',B/'space_candidates_DIAGNOSTIC_ONLY.npz',RAW/'raw_index_preserved.npz']
    write(OUT/'input_hashes.json',[dict(path=str(p),sha256=sha(p)) for p in files])
    d=np.load(P/'poisson.npz');v=d['vertices'];f=d['faces'];raw=np.load(RAW/'raw_index_preserved.npz');source=locator(raw['vertices'],raw['faces']);loc=locator(v,f)
    pairs=[]
    for num,(i,j) in enumerate(read(P/'independent_intersections.json')['adjacent_overlap_candidates'],1):
        shared=np.intersect1d(f[i],f[j]);assert len(shared)==2
        ai=int(np.setdiff1d(f[i],shared)[0]);bi=int(np.setdiff1d(f[j],shared)[0]);a,b,c,e=[[F(float(z)) for z in v[k]] for k in [*shared,ai,bi]]
        edge=sub(b,a);na=cross(edge,sub(c,a));nb=cross(edge,sub(e,a));det=dot(na,sub(e,a));area_a=dot(na,na);area_b=dot(nb,nb)
        if not area_a or not area_b:status='EXACT_DEGENERATE'
        elif det:status='SHARED_EDGE_ONLY_NONCOPLANAR'
        elif dot(na,nb)>0:status='COPLANAR_INTERIOR_OVERLAP'
        else:status='SHARED_EDGE_ONLY_COPLANAR'
        pts=v[np.unique(np.r_[f[i],f[j]])];center=pts.mean(0);edgelen=float(np.linalg.norm(v[shared[1]]-v[shared[0]]));sides=[float(np.linalg.norm(np.cross(v[shared[1]]-v[shared[0]],v[k]-v[shared[0]]))/max(edgelen,1e-30)) for k in [ai,bi]]
        dist,_,sf=closest(source,[center]);label=int(raw['face_label'][sf[0]])
        pairs.append(dict(id=num,face_ids=[i,j],shared_vertex_ids=shared.tolist(),center=center.tolist(),status=status,exact_determinant_sign=1 if det>0 else -1 if det<0 else 0,shared_edge_length_mm=edgelen,triangle_altitudes_mm=sides,extent_mm=float(np.ptp(pts,axis=0).max()),nearest_original_fdi=label,distance_to_original_mm=float(dist[0])))
    write(OUT/'pair_details.json',pairs);print('exact shared-edge check',dict((s,sum(x['status']==s for x in pairs)) for s in set(x['status'] for x in pairs)),flush=True)
    # Complete component inventory, no component removed from saved proxy.
    a=f.ravel();bb=np.roll(f,-1,axis=1).ravel();n,labels=connected_components(coo_matrix((np.ones(len(a),np.int8),(a,bb)),shape=(len(v),len(v))).tocsr(),directed=False);fl=labels[f[:,0]];counts=np.bincount(fl);main=int(np.argmax(counts));tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2;components=[]
    for cid in np.flatnonzero(np.arange(n)!=main):
        ids=np.flatnonzero(fl==cid);vs=v[np.unique(f[ids])];t=tri[ids]-vs.mean(0);vol=float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6);center=vs.mean(0);distance,_,sf=closest(source,[center]);components.append(dict(id=int(cid),face_ids=ids.tolist(),faces=len(ids),center=center.tolist(),bounds=[vs.min(0).tolist(),vs.max(0).tolist()],extent_mm=float(np.ptp(vs,axis=0).max()),area_mm2=float(area[ids].sum()),signed_volume_mm3=vol,nearest_original_fdi=int(raw['face_label'][sf[0]]),distance_to_original_mm=float(distance[0])))
    components.sort(key=lambda x:-x['area_mm2']);write(OUT/'component_details.json',components);np.savez_compressed(OUT/'component_ids.npz',face_component=fl,main_component=main)
    # Expanded localization: every original vertex, plus original 20000 area samples.
    dist,pos,near=closest(loc,raw['vertices']);np.savez_compressed(OUT/'all_original_vertex_distances.npz',distance_mm=dist,closest_xyz=pos,closest_proxy_face=near)
    regions=[]
    for idx in np.argsort(dist)[::-1]:
        center=raw['vertices'][idx]
        if all(np.linalg.norm(center-np.array(x['center']))>2 for x in regions):
            regions.append(dict(id=len(regions)+1,source_vertex=int(idx),center=center.tolist(),closest_point=pos[idx].tolist(),distance_mm=float(dist[idx]),source_vertex_fdi=int(raw['vertex_labels'][idx]),proxy_face=int(near[idx])))
            if len(regions)==8:break
    write(OUT/'deviation_regions.json',regions)
    stats={}
    for label in sorted(np.unique(raw['vertex_labels'])):
        dd=dist[raw['vertex_labels']==label];stats[str(int(label))]=dict(count=len(dd),median=float(np.median(dd)),p95=float(np.quantile(dd,.95)),max=float(dd.max()))
    write(OUT/'vertex_distance_by_fdi.json',stats)
    print('vertex max',regions[0],flush=True)

def sampling():
    d=np.load(P/'poisson.npz');v=d['vertices'];f=d['faces'];comp=np.load(OUT/'component_ids.npz');main=f[comp['face_component']==comp['main_component']];rng=np.random.default_rng(2026101703);groups=[];pointsets=[]
    for group,records in [('pair',read(OUT/'pair_details.json')),('component',read(OUT/'component_details.json')),('deviation',read(OUT/'deviation_regions.json'))]:
        for r in records:
            center=np.array(r['center']);radius=max(.03,min(.5,r['extent_mm']*.6)) if group=='pair' else (max(.05,r['extent_mm']*.65) if group=='component' else 2.)
            points=rng.uniform(-radius,radius,(1000,3))+center
            groups.append(dict(kind=group,id=r['id'],start=sum(len(x) for x in pointsets),count=len(points),radius_mm=radius));pointsets.append(points)
    xyz=np.concatenate(pointsets);inside=contains(v,f,xyz);inside_main=contains(v,main,xyz);base=np.load(B/'proxy_geometry.npz');inside_old=contains(base['vertices'],base['faces'],xyz);dist,_,_=closest(locator(v,f),xyz)
    rotated=[]
    for k in range(3):
        q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,0]*=np.linalg.det(q);rotated.append(contains((v-v.mean(0))@q,f,(xyz-v.mean(0))@q));print('rotation check',k+1,flush=True)
    mismatch=np.any(np.asarray(rotated)!=inside[None,:],axis=0)
    for g in groups:
        s=slice(g['start'],g['start']+g['count']);g.update(inside_count=int(inside[s].sum()),rotation_disagreements=int(mismatch[s].sum()),rotation_disagreements_beyond_0_0001mm=int((mismatch[s]&(dist[s]>1e-4)).sum()),full_vs_main_label_changes=int((inside[s]!=inside_main[s]).sum()),old_vs_poisson_label_changes=int((inside[s]!=inside_old[s]).sum()))
    np.savez_compressed(OUT/'local_diagnostic_points.npz',xyz=xyz,inside_poisson=inside,inside_main_only=inside_main,inside_old_proxy=inside_old,distance_to_poisson=dist,rotation_disagreement=mismatch)
    globaldata=np.load(P/'space_labels_DIAGNOSTIC_ONLY.npz');gxyz=globaldata['xyz'];gi=globaldata['inside'];gm=contains(v,main,gxyz);old=np.load(B/'space_candidates_DIAGNOSTIC_ONLY.npz');assert np.array_equal(gxyz,old['xyz']);go=old['inside_after'];changed=np.flatnonzero(gi!=go);write(OUT/'global_change_points.json',dict(indices=changed.tolist(),xyz=gxyz[changed].tolist()))
    write(OUT/'sampling.json',dict(seed=2026101703,local_count=len(xyz),groups=groups,local_rotation_disagreements=int(mismatch.sum()),local_rotation_disagreements_beyond_0_0001mm=int((mismatch&(dist>1e-4)).sum()),global_count=len(gxyz),global_full_vs_main_label_changes=int((gi!=gm).sum()),global_old_vs_poisson_label_changes=len(changed),note='Main-component-only mesh exists in memory solely as a counterfactual; saved proxy unchanged. Old-vs-new changes are sensitivity, not ground-truth errors.'))
    print('sampling done',len(changed),'global old-vs-new',int((gi!=gm).sum()),'component changes',flush=True)

def shot(meshes,center,field,eye,title,points=None):
    pl=pv.Plotter(off_screen=True,window_size=(640,640));pl.set_background('#edf1f5')
    for v,f,color,opacity,edges in meshes:
        if len(f):pl.add_mesh(pd(v,f),color=color,opacity=opacity,show_edges=edges,edge_color='#18384a',ambient=.4)
    if points:
        for p,color in points:pl.add_mesh(pv.Sphere(radius=field*.012,center=p),color=color)
    eye=np.asarray(eye);eye=eye/max(np.linalg.norm(eye),1e-20);up=np.array([0.,0.,1.])
    if abs(eye@up)>.9:up=np.array([0.,1.,0.])
    pl.camera_position=[center+eye*field*5,center,up];pl.enable_parallel_projection();pl.camera.parallel_scale=field/2;im=Image.fromarray(pl.screenshot(return_img=True));pl.close();dr=ImageDraw.Draw(im);dr.rectangle((0,0,640,43),fill='#edf1f5');dr.text((12,7),title,font=font(22),fill='#18384a');scale=field/5;dr.line((18,605,146,605),fill='#18384a',width=3);dr.text((18,570),f'{scale:.6g} mm',font=font(18),fill='#18384a');return im

def render():
    d=np.load(P/'poisson.npz');v=d['vertices'];f=d['faces'];raw=np.load(RAW/'raw_index_preserved.npz');rv=raw['vertices'];rf=raw['faces'];fl=np.load(OUT/'component_ids.npz')['face_component'];pairs=read(OUT/'pair_details.json');comps=read(OUT/'component_details.json');regions=read(OUT/'deviation_regions.json');basis=raw['basis'];eye=basis@np.array([.5,.7,-.8]);centers=v[f].mean(1);rc=rv[rf].mean(1)
    def local(fscent,center,radius):return np.linalg.norm(fscent-center,axis=1)<radius
    overview=Image.new('RGB',(1920,740),'#edf1f5');dr=ImageDraw.Draw(overview);center=(rv.min(0)+rv.max(0))/2
    for j,(name,records,color) in enumerate([('47 对相邻面候选',pairs,'#cf3655'),('26 个小闭合片',comps,'#e37b24'),('偏差最高的 8 个区域',regions,'#2866d7')]):
        pts=[(np.array(r['center']),color) for r in records];overview.paste(shot([(v,f,'#d5c7af',1,False)],center,90,eye,name,pts),(640*j,0))
    dr.text((20,675),'标记球为定位示意，尺寸经过放大；真实尺度见逐项局部图。所有坐标与距离均为原始毫米。',font=font(25),fill='#18384a');overview.save(OUT/'overview.png')
    for group,records in [('pairs',pairs),('components',comps),('deviation',regions)]:
        for r in records:
            c=np.array(r['center']);field=8 if group!='deviation' else 12;canvas=Image.new('RGB',(1280,740),'#edf1f5');dd=ImageDraw.Draw(canvas)
            if group=='pairs':
                ids=r['face_ids'];highlight=f[ids];canvas.paste(shot([(v,f[local(centers,c,field)],'#d5c7af',1,False),(v,highlight,'#cf3655',1,True)],c,field,eye,f'P{r["id"]:02d} · 周围 8 mm'),(0,0))
                a=v[f[ids[0]]];norm=np.cross(a[1]-a[0],a[2]-a[0]);norm/=max(np.linalg.norm(norm),1e-30);zoom=max(r['extent_mm']*1.8,1e-5)
                canvas.paste(shot([(v,f[ids[:1]],'#cf3655',.8,True),(v,f[ids[1:]],'#287ac0',.8,True)],c,zoom,norm,f'P{r["id"]:02d} · 两个三角面真实比例'),(640,0));note=f'{r["status"]}；最近原表面 FDI={r["nearest_original_fdi"]}；高度 '+str([round(x,9) for x in r['triangle_altitudes_mm']])+' mm'
            elif group=='components':
                cf=f[r['face_ids']];canvas.paste(shot([(v,f[local(centers,c,field)],'#d5c7af',.3,False),(v,cf,'#e37b24',1,True)],c,field,eye,f'C{r["id"]:02d} · 周围 8 mm'),(0,0))
                canvas.paste(shot([(v,cf,'#e37b24',1,True),(rv,rf[local(rc,c,max(.5,r['extent_mm']*2))],'#6aa8c3',.2,False)],c,max(.2,r['extent_mm']*2.5),eye,f'C{r["id"]:02d} · 闭合片放大'),(640,0));note=f'面积 {r["area_mm2"]:.6f} mm²；有向体积 {r["signed_volume_mm3"]:.9f} mm³；最近原表面 FDI={r["nearest_original_fdi"]}'
            else:
                pp=np.array(r['closest_point']);canvas.paste(shot([(rv,rf[local(rc,c,field)],'#d5c7af',1,False)],c,field,eye,f'D{r["id"]} · 原扫描',[(c,'#cf3655')]),(0,0));canvas.paste(shot([(v,f[local(centers,c,field)],'#6aa8c3',1,False)],c,field,eye,f'D{r["id"]} · 泊松代理',[(c,'#cf3655'),(pp,'#23a070')]),(640,0));note=f'距离 {r["distance_mm"]:.6f} mm；原顶点 FDI={r["source_vertex_fdi"]}；红球为原扫描点，绿球为代理最近点'
            dd.text((15,665),note,font=font(17),fill='#18384a');canvas.save(OUT/group/f'{r["id"]:02d}.png')
        print('rendered',group,len(records),flush=True)

def finish():
    pairs=read(OUT/'pair_details.json');cs=read(OUT/'component_details.json');rs=read(OUT/'deviation_regions.json');s=read(OUT/'sampling.json');stats=read(OUT/'vertex_distance_by_fdi.json')
    assert all(sha(x['path'])==x['sha256'] for x in read(OUT/'input_hashes.json'))
    local_comp_changes=sum(g['full_vs_main_label_changes'] for g in s['groups'] if g['kind']=='component')
    summary=dict(pair_status_counts={k:sum(r['status']==k for r in pairs) for k in sorted(set(r['status'] for r in pairs))},minor_component_count=len(cs),minor_abs_volume_sum_mm3=sum(abs(c['signed_volume_mm3']) for c in cs),local_component_label_changes=local_comp_changes,local_component_test_count=26000,largest_vertex_deviation=rs[0],vertex_distance_by_fdi=stats,sampling=s,original_and_proxy_inputs_unchanged=True,human_review='PENDING')
    write(OUT/'summary.json',summary)
    parts=['<!doctype html><meta charset="utf-8"><title>泊松代理问题定位与空间采样影响</title><style>body{font:18px/1.7 system-ui;max-width:1500px;margin:30px auto;background:#edf1f5;color:#18384a}img{max-width:100%}pre{white-space:pre-wrap}summary{cursor:pointer;padding:8px}table{border-collapse:collapse}td,th{border:1px solid #b0bdc8;padding:7px}</style><h1>00OMSZGW_lower：泊松代理局部验收</h1><p>只新增诊断。原扫描、旧代理、泊松代理哈希不变。下述小片移除仅是内存中的分类对照，没有保存或替换模型。</p><img src="overview.png">',
        '<h2>47 对相邻三角面的复核</h2><p>47 对均共享两个顶点。对保存的 IEEE 浮点坐标转成精确有理数，检查四点是否共面；非共面且共享边的两个非退化三角形，只能在共享边上相交。此判定不使用前一轮的微小内缩阈值。</p><pre>'+json.dumps(summary['pair_status_counts'],ensure_ascii=False,indent=2)+'</pre>',
        '<h2>空间采样影响</h2><p>局部每项 1000 点、3 次旋转复查；离表面 0.0001 mm 以内的点单独标识。旧代理与泊松标签不同代表敏感性，不代表其中一个是真值。有限样本零变化不等于影响严格为零。</p><pre>'+json.dumps({k:v for k,v in s.items() if k!='groups'},ensure_ascii=False,indent=2)+'</pre>',
        '<h2>小闭合片</h2><p>全部 '+str(len(cs))+' 片的绝对有向体积之和 '+str(summary['minor_abs_volume_sum_mm3'])+' mm³。各片位置、大小、局部标签影响见下。</p>',
        '<h2>最大偏差定位</h2><p>从原来的 2 万个表面抽样点扩展到全部原始顶点。这里是已测顶点最大值，不是连续三角面上的数学全局最大值；FDI=0 表示牙龈/未标牙区域。</p><pre>'+json.dumps(stats,ensure_ascii=False,indent=2)+'</pre>']
    mapping={(g['kind'],g['id']):g for g in s['groups']}
    for folder,kind,title,records in [('deviation','deviation','偏差区域',rs),('components','component','小闭合片',cs),('pairs','pair','相邻面候选',pairs)]:
        parts.append('<h2>'+title+'：逐项放大</h2>')
        for r in records:
            g=mapping[(kind,r['id'])];parts.append(f'<details><summary>{title} {r["id"]} · 旋转不一致 {g["rotation_disagreements"]}/1000 · 仅主分量标签变化 {g["full_vs_main_label_changes"]}/1000 · 旧/新代理变化 {g["old_vs_poisson_label_changes"]}/1000</summary><a href="{folder}/{r["id"]:02d}.png"><img loading="lazy" src="{folder}/{r["id"]:02d}.png"></a><pre>'+json.dumps(r,ensure_ascii=False,indent=2)+'</pre></details>')
    parts.append('<p><a href="summary.json">完整摘要</a> · <a href="sampling.json">逐项采样指标</a> · <a href="pair_details.json">精确共享边判定</a></p>')
    intro='<div style="padding:18px;background:#fff1d8"><b>先看结论：</b>47 对均为正常共享边，精确判定未发现内部重叠，撤销前一轮这 47 对的重叠嫌疑。81,000 个局部点的 3 次旋转检查全部一致。<br>小闭合片对全域 500,000 点：0 个标签变化；对小片附近 26,000 点：'+str(local_comp_changes)+' 个变化（约 '+f'{local_comp_changes/26000:.2%}'+'）。因此全域均匀采样影响小，但局部/近表面采样不可忽略。<br>原扫描顶点最大偏差 1.499 mm 位于 FDI=0；37 牙最大 0.606 mm，47 牙最大 0.393 mm。旧代理与泊松代理在全域 500,000 点中有 137 个标签变化（0.0274%），这是形状敏感性，不是真值错误率。<br>建议先审核小闭合片及 D1、D6 等高偏差区域，近表面采样暂保留不确定标记；本轮未删片、未修网格、未修改任何训练标签。</div><h2>三个代表放大图</h2><p>D1：最大偏差，FDI=0</p><img src="deviation/01.png"><p>D6：37 牙的局部偏差</p><img src="deviation/06.png"><p>C9：最大的小闭合片，负有向体积提示内腔边界的可能性，不自动当作外部碎片删除</p><img src="components/09.png">'
    page='\n'.join(parts).replace('<img src="overview.png">',intro+'<h2>全局定位</h2><img src="overview.png">')
    (OUT/'index.html').write_text(page,encoding='utf-8');shutil.copyfile(__file__,OUT/'implementation.py');write(OUT/'delivery_manifest.json',[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file()]);print(json.dumps({k:v for k,v in summary.items() if k not in ['sampling','vertex_distance_by_fdi']},ensure_ascii=False,indent=2))

if __name__=='__main__':globals()[sys.argv[1]]()
