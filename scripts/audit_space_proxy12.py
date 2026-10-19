"""Raw-mm watertight-proxy pilot. Proxy never supplies surface supervision.

Frozen 12-scan cohort only. Original meshes/caches and earlier audit are read-only.
"""
from pathlib import Path
import sys,json,hashlib,shutil,time
import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
from PIL import Image,ImageDraw
from diagnose_preprocess12 import read,write,sha,now,font,unit

ROOT=Path(__file__).resolve().parents[1]
OLD_SRC=Path('D:/WorkSpace/Dental/teeth3DS_to_DMM/src')
sys.path.insert(0,str(OLD_SRC))
from teeth3ds_dmm.proxy.geometry import analyze_boundary_loops,select_main_boundary,triangulate_virtual_cap
from teeth3ds_dmm.geometry.normals import edge_orientation_metrics

INPUT=ROOT/'runs/dmm_preprocess12_20261016T1815_r1'
OUT=ROOT/'runs/dmm_space_proxy12_20261016_01'
N=500000

def pd(v,f):return pv.PolyData(v,np.column_stack([np.full(len(f),3),f]).ravel())

def contains(v,f,points):
    mesh=pd(v,f); pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(np.ascontiguousarray(points),deep=True))
    cloud=vtk.vtkPolyData();cloud.SetPoints(pts)
    selector=vtk.vtkSelectEnclosedPoints();selector.SetInputData(cloud);selector.SetSurfaceData(mesh)
    selector.SetTolerance(1e-7);selector.CheckSurfaceOn();selector.Update()
    arr=selector.GetOutput().GetPointData().GetArray('SelectedPoints')
    if arr is None:raise RuntimeError('VTK produced no enclosed-point classification')
    return vtk_to_numpy(arr).astype(bool)

def initialize():
    OUT.mkdir(exist_ok=False);(OUT/'cases').mkdir();(OUT/'protocol').mkdir()
    cohort=read(INPUT/'protocol/cohort.json');shutil.copyfile(INPUT/'protocol/cohort.json',OUT/'protocol/cohort.json')
    shutil.copyfile(__file__,OUT/'protocol/audit_space_proxy12_initial.py')
    write(OUT/'protocol/settings.json',dict(created_utc=now(),source_audit=str(INPUT),case_count=12,
        coordinate_space='raw_mm',alignment_applied=False,not_a_training_export=True,
        source_surface_modified=False,proxy_used_for_surface_supervision=False,
        proxy_method='old projected-ear-clipping of dominant raw boundary only',
        refuse_nonmanifold=True,require_zero_residual_boundary=True,require_consistent_winding=True,
        original_vertex_and_face_prefix_must_match=True,
        classification='VTK enclosed-point parity; closed proxy labels, not tooth-specific SDF',
        uniform_candidate_count=N,domain='raw axis aligned bbox padded by 5 mm per axis',
        balance_inside_outside=False,signed_distance_computed=False,
        cap_band_exclusion='projection onto cap best-fit normal; entire cap height envelope padded by 0.5 mm; conservative slab, not exact cap distance',
        cap_band_mm=.5,cap_band_frozen=False,self_intersection_test='not performed; no training acceptance',
        human_visual_review='PENDING'))
    write(OUT/'protocol/dependencies.json',[dict(path=str(p),sha256=sha(p)) for p in [INPUT/'protocol/cohort.json',OLD_SRC/'teeth3ds_dmm/proxy/geometry.py',OLD_SRC/'teeth3ds_dmm/geometry/normals.py']])
    for row in cohort['cases']:
        cid=row['case_id'];p=OUT/'cases'/cid;p.mkdir()
        cache=INPUT/'cases'/cid/'raw_index_preserved.npz';d=np.load(cache)
        a=analyze_boundary_loops(d['vertices'],d['faces'])
        write(p/'boundary_scan.json',a.to_dict())
        write(p/'input_identity.json',dict(source_obj=row['source_obj'],source_annotation=row['source_annotation'],
            source_obj_sha256=sha(row['source_obj']),source_annotation_sha256=sha(row['source_annotation']),cache_path=str(cache),cache_sha256=sha(cache)))
        print(cid,'loops',len(a.loops),'unresolved',len(a.unresolved_components),'nonmanifold',a.nonmanifold_edge_count,
              'planarity',[round(x.planarity_ratio,5) for x in a.loops[:3]],flush=True)

def build():
    rows=read(OUT/'protocol/cohort.json')['cases']
    for row in rows:
        t=time.perf_counter();cid=row['case_id'];p=OUT/'cases'/cid
        assert not (p/'result.json').exists()
        d=np.load(INPUT/'cases'/cid/'raw_index_preserved.npz');v=d['vertices'];f=d['faces']
        record=dict(case_id=cid,status='BLOCKED_PROXY',human_visual_review='PENDING',training_accepted=False,
                    original_vertices_unchanged=True,original_faces_unchanged=True,space_samples_generated=False)
        try:
            a=analyze_boundary_loops(v,f)
            boundary,selection=select_main_boundary(a,min_perimeter_ratio_to_second=3,max_planarity_ratio=.06,min_vertex_count=3,allow_minor_components=False)
            cap=triangulate_virtual_cap(v,f,boundary);pf=np.concatenate([f,cap])
            bc,nc,wc=edge_orientation_metrics(pf)
            tri=v[pf];area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
            duplicate=len(pf)-len(np.unique(np.sort(pf,axis=1),axis=0))
            volume=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
            record.update(boundary_edges=bc,nonmanifold_edges=nc,inconsistent_edges=wc,
                duplicate_faces=duplicate,degenerate_faces=int((area2<=1e-12).sum()),signed_volume_mm3=volume,
                virtual_cap_faces=len(cap),selection=selection,watertight=bc==0 and nc==0)
            np.savez_compressed(p/'proxy_geometry.npz',vertices=v,source_faces=f,virtual_cap_faces=cap,
                faces=pf,face_is_virtual_cap=np.arange(len(pf))>=len(f))
            # Export for independent inspection. Clearly named proxy, never training surf.
            with (p/'proxy_inside_outside_only.obj').open('x',encoding='utf-8') as fp:
                fp.write('# PROXY ONLY - virtual cap is not training surface\n')
                np.savetxt(fp,v,fmt='v %.10g %.10g %.10g')
                fp.write('g original_surface\n');np.savetxt(fp,f+1,fmt='f %d %d %d')
                fp.write('g virtual_cap_NOT_SUPERVISION\n');np.savetxt(fp,cap+1,fmt='f %d %d %d')
            if bc or nc or wc or duplicate or np.any(area2<=1e-12) or not np.isfinite(volume) or abs(volume)<1e-8:
                raise ValueError('Proxy topology/degeneracy/volume gate failed; no inside/outside classification')
            assert np.array_equal(pf[:len(f)],f)
            seed=int(hashlib.sha256(('20261016|'+cid).encode()).hexdigest()[:16],16)
            rng=np.random.default_rng(seed);lo=v.min(0)-5;hi=v.max(0)+5
            xyz=rng.uniform(lo,hi,size=(N,3)).astype(np.float32)
            print(cid,'proxy closed; classifying',N,'points',flush=True)
            inside=contains(v,pf,xyz)
            cv=v[np.asarray(boundary.vertex_ids)];origin=cv.mean(0);_,_,vh=np.linalg.svd(cv-origin,full_matrices=False);axis=vh[-1]
            heights=(cv-origin)@axis;ph=(xyz-origin)@axis;band=(ph>=heights.min()-.5)&(ph<=heights.max()+.5)
            # Rigid-rotation invariance spot-check with another ray configuration.
            checkids=rng.choice(N,size=20000,replace=False);q,_=np.linalg.qr(np.array([[.3,.7,.2],[-.8,.1,.4],[.5,-.2,.9]]))
            if np.linalg.det(q)<0:q[:,0]*=-1
            other=contains((v-origin)@q,pf,(xyz[checkids]-origin)@q)
            mismatch=other!=inside[checkids]
            reliable=~band;reliable[checkids[mismatch]]=False
            np.savez_compressed(p/'space_candidates_raw_mm.npz',xyz=xyz,inside_proxy=inside,
                excluded_virtual_cap_slab=band,eligible_proxy_point=reliable,rotation_check_indices=checkids,
                rotation_check_inside=other)
            record.update(status='CLASSIFIED_PROXY_REVIEW_PENDING',space_samples_generated=True,seed=seed,
                candidate_count=N,inside_count=int(inside.sum()),outside_count=int((~inside).sum()),
                cap_band_excluded=int(band.sum()),eligible_inside_count=int((reliable&inside).sum()),
                eligible_outside_count=int((reliable&~inside).sum()),rotation_check_count=len(checkids),
                rotation_disagreements=int(mismatch.sum()),rotation_disagreements_outside_cap_band=int((mismatch&~band[checkids]).sum()),
                domain_raw_mm=[lo.tolist(),hi.tolist()],cap_band_axis_raw=axis.tolist(),cap_band_origin_raw=origin.tolist(),
                cap_band_height_limits_mm=[float(heights.min()-.5),float(heights.max()+.5)],
                cap_band_sensitivity={str(margin):int(((ph>=heights.min()-margin)&(ph<=heights.max()+margin)).sum()) for margin in [.25,.5,1.]},
                self_intersection_checked=False,proxy_sign_is_not_true_tooth_sdf=True)
        except Exception as exc:
            record['failure']=f'{type(exc).__name__}: {exc}'
        record['elapsed_seconds']=time.perf_counter()-t
        write(p/'result.json',record);print(cid,record['status'],record.get('failure',''),flush=True)

def screen(v,f,cap,basis,center,eye,scale,title,size=700):
    pl=pv.Plotter(off_screen=True,window_size=(size,size));pl.set_background('#edf1f5');pl.disable_anti_aliasing()
    pl.add_mesh(pd(v,f),color='#d5c7af',smooth_shading=False,ambient=.4,diffuse=.6,specular=0)
    if len(cap):pl.add_mesh(pd(v,cap),color='#f29232',smooth_shading=False,ambient=.55,diffuse=.45,show_edges=True,edge_color='#b06a2e')
    e=unit(basis@unit(np.array(eye,float)));up=basis[:,1] if abs(eye[2])>.9 else basis[:,2]
    pl.camera_position=[center+e*250,center,up];pl.enable_parallel_projection();pl.camera.parallel_scale=scale/2;pl.camera.clipping_range=(1,500)
    im=Image.fromarray(pl.screenshot(return_img=True));pl.close();dr=ImageDraw.Draw(im)
    dr.rectangle((0,0,size,45),fill='#edf1f5');dr.text((15,8),title,font=font(23),fill='#18384a')
    dr.line((20,size-25,20+10/scale*size,size-25),fill='#18384a',width=3);dr.text((20,size-58),'10 mm',font=font(20),fill='#18384a')
    return im

def section(v,f,cap,b,center,xyz,inside,band,plane_axis,title):
    size=700;field=110;im=Image.new('RGB',(size,size),'#edf1f5');dr=ImageDraw.Draw(im)
    uv=[j for j in range(3) if j!=plane_axis];points=(xyz-center)@b;sl=np.abs(points[:,plane_axis])<=.5
    ids=np.flatnonzero(sl)
    if len(ids)>9000:ids=ids[np.linspace(0,len(ids)-1,9000).astype(int)]
    def xy(t):return (size/2+t[uv[0]]*size/field,size/2-t[uv[1]]*size/field)
    for k in ids:
        x,y=xy(points[k]);c='#acb3b9' if band[k] else ('#df5149' if inside[k] else '#3188cf')
        dr.ellipse((x-1,y-1,x+1,y+1),fill=c)
    verts=(v-center)@b
    for ff,color,width in [(f,'#354b5d',1),(cap,'#f29232',4)]:
        if not len(ff):continue
        s=verts[ff,plane_axis];cross=(s.min(1)<=0)&(s.max(1)>=0)
        for ii in np.flatnonzero(cross):
            ps=[]
            for a,c in [(0,1),(1,2),(2,0)]:
                da,dc=s[ii,a],s[ii,c]
                if da*dc<0:ps.append(verts[ff[ii,a]]+(verts[ff[ii,c]]-verts[ff[ii,a]])*da/(da-dc))
            if len(ps)==2:dr.line([xy(ps[0]),xy(ps[1])],fill=color,width=width)
    dr.rectangle((0,0,size,75),fill='#edf1f5');dr.text((15,8),title,font=font(23),fill='#18384a')
    dr.text((15,43),'红：代理内 / 蓝：代理外 / 灰：底面排除带',font=font(19),fill='#39576b')
    dr.line((20,size-25,20+10/field*size,size-25),fill='#18384a',width=3);dr.text((20,size-58),'10 mm',font=font(20),fill='#18384a')
    return im

def render():
    rows=read(OUT/'protocol/cohort.json')['cases'];results=[]
    contact=Image.new('RGB',(1800,2540),'#edf1f5');dd=ImageDraw.Draw(contact)
    dd.text((25,15),'12 例封底代理与空间采样 · 毫米空间诊断',font=font(34),fill='#18384a')
    dd.text((25,65),'虚拟底面不提供表面监督；闭合失败不生成内外标签；未做尺度归一化或训练导出。',font=font(23),fill='#39576b')
    for i,row in enumerate(rows):
        cid=row['case_id'];p=OUT/'cases'/cid;r=read(p/'result.json');results.append(r)
        d=np.load(INPUT/'cases'/cid/'raw_index_preserved.npz');v=d['vertices'];f=d['faces'];b=d['basis'];center=d['center'];s=1 if row['arch']=='upper' else -1
        cap=np.load(p/'proxy_geometry.npz')['virtual_cap_faces'] if (p/'proxy_geometry.npz').exists() else np.empty((0,3),int)
        canvas=Image.new('RGB',(1400,1610),'#edf1f5');dr=ImageDraw.Draw(canvas)
        dr.text((25,12),cid+' · 封底代理 / 空间点审核',font=font(30),fill='#18384a')
        dr.text((25,55),'原坐标 mm；原表面与虚拟底面分开保存；不是训练数据版本',font=font(22),fill='#39576b')
        canvas.paste(screen(v,f,cap,b,center,[.5,.7,.8*s],110,'原扫描侧面 + 橙色虚拟底面'),(0,100))
        canvas.paste(screen(v,f,cap,b,center,[0,0,-s],110,'基底观察：封底面只用于内外分类'),(700,100))
        if r['space_samples_generated']:
            a=np.load(p/'space_candidates_raw_mm.npz')
            for k,ax in enumerate([0,1]):canvas.paste(section(v,f,cap,b,center,a['xyz'],a['inside_proxy'],a['excluded_virtual_cap_slab'],ax,['Y/Z 中央截面 ±0.5 mm','X/Z 中央截面 ±0.5 mm'][k]),(700*k,800))
            msg=f'候选 {N:,}：内 {r["inside_count"]:,} / 外 {r["outside_count"]:,}；底面带排除 {r["cap_band_excluded"]:,}'
        else:
            dr.text((30,860),'未生成内外点：封闭代理未通过检查。',font=font(30),fill='#a74436')
            scan=read(p/'boundary_scan.json');msg=f'非流形边 {scan["nonmanifold_edge_count"]}；边界环 {scan["candidate_count"]}；未解析边界 {len(scan["unresolved_components"])}'
            dr.text((30,920),msg,font=font(23),fill='#39576b');dr.text((30,980),'保留原件；不自动补牙冠、不强行给开放网格标内外。',font=font(25),fill='#39576b')
        dr.text((25,1510),msg,font=font(22),fill='#18384a')
        dr.text((25,1550),'人工审核 PENDING；尚未完成自交检验，代理内外不能视为真实牙齿 SDF。',font=font(21),fill='#a74436')
        assert not (p/'review.png').exists();canvas.save(p/'review.png')
        thumb=canvas.crop((0,800,700,1500)) if r['space_samples_generated'] else canvas.crop((0,100,700,800))
        thumb=thumb.resize((550,550));x=(i%3)*600;y=120+(i//3)*600
        contact.paste(thumb,(x+25,y+45));dd.text((x+20,y),cid+(' 已分类' if r['space_samples_generated'] else ' 封闭失败'),font=font(22),fill='#18384a' if r['space_samples_generated'] else '#a74436')
    contact.save(OUT/'contact.png')
    write(OUT/'summary.json',dict(created_utc=now(),cases=results,case_count=12,classified_cases=sum(r['space_samples_generated'] for r in results),
        candidate_count=sum(r.get('candidate_count',0) for r in results),human_visual_review='PENDING',training_export=False))
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>12例封底代理与空间采样</title><style>body{font:17px/1.7 system-ui;max-width:1500px;margin:35px auto;background:#edf1f5;color:#18384a}img{max-width:100%}td,th{padding:10px;border:1px solid #bbc8d1}table{border-collapse:collapse;width:100%}a{color:#176993}.warning{padding:18px;background:#fff1d9}</style>',
        '<h1>12 例封底代理与空间采样审核</h1><p class="warning">本轮在原始毫米空间诊断。未做尺度归一化/配准；原始 OBJ、JSON、前一轮审核包均未覆盖。虚拟底面只用于封闭代理的内外判断，不提供牙冠/牙龈表面点、法线、质心或 FDI 监督。生成的是代理内外标签，不是精确 SDF。</p>',
        '<p>每个通过拓扑门槛的代理，在原始包围盒每轴外扩 5 mm 内均匀产生 500,000 个候选点，保留自然内外比例。灰色点落在虚拟底面高度包络外扩 0.5 mm 的保守排除带；这是诊断参数，尚未冻结。近底面的标签依赖人工封底，未来训练使用前应保留此掩码。代理没有进入 DMM 训练。</p>',
        '<p>严格要求封底后边界边为零、非流形边为零、绕序冲突为零、无退化/重复面且体积非零。没有自动修改原扫描来强行过关。通过不等于几何自交/解剖验收；本轮未完成全面自交检验。每例 20,000 点额外做刚体旋转后的分类一致性抽查，完整结果见 JSON。</p>',
        '<p><a href="summary.json">指标汇总</a> · <a href="protocol/settings.json">方法和边界</a></p><table><tr><th>病例</th><th>状态</th><th>内 / 外候选</th><th>排除带点数</th><th>旋转抽查分歧</th></tr>']
    for r in results:parts.append(f'<tr><td><a href="#c-{r["case_id"]}">{r["case_id"]}</a></td><td>{r["status"]}</td><td>{r.get("inside_count","—")} / {r.get("outside_count","—")}</td><td>{r.get("cap_band_excluded","—")}</td><td>{r.get("rotation_disagreements","—")}</td></tr>')
    parts.append('</table><a href="contact.png"><img src="contact.png"></a>')
    for r in results:
        cid=r['case_id'];parts.append(f'<h2 id="c-{cid}">{cid}</h2><p><a href="cases/{cid}/result.json">逐例结果</a> · <a href="cases/{cid}/boundary_scan.json">边界记录</a></p><a href="cases/{cid}/review.png"><img loading="lazy" src="cases/{cid}/review.png"></a>')
    parts.append('</html>');(OUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')

def verify():
    import re
    rows=read(OUT/'protocol/cohort.json')['cases'];results=read(OUT/'summary.json');checks=[];volume_checks=[];decisions=[]
    for row in rows:
        p=OUT/'cases'/row['case_id'];identity=read(p/'input_identity.json');r=read(p/'result.json')
        assert sha(row['source_obj'])==identity['source_obj_sha256']
        assert sha(row['source_annotation'])==identity['source_annotation_sha256']
        assert sha(identity['cache_path'])==identity['cache_sha256']
        if r['space_samples_generated']:
            orig=np.load(identity['cache_path']);proxy=np.load(p/'proxy_geometry.npz');a=np.load(p/'space_candidates_raw_mm.npz')
            assert np.array_equal(orig['vertices'],proxy['vertices']) and np.array_equal(orig['faces'],proxy['source_faces'])
            assert np.array_equal(proxy['faces'][:len(orig['faces'])],orig['faces'])
            assert a['xyz'].shape==(N,3) and np.isfinite(a['xyz']).all()
            assert len(a['inside_proxy'])==N and r['inside_count']+r['outside_count']==N
            assert 0<r['inside_count']<N
            bounds=np.array(r['domain_raw_mm']);box_volume=float(np.prod(bounds[1]-bounds[0]));prob=r['inside_count']/N
            estimate=box_volume*prob;stderr=box_volume*np.sqrt(prob*(1-prob)/N);exact=abs(r['signed_volume_mm3'])
            zscore=abs(estimate-exact)/stderr
            volume_checks.append(dict(case_id=row['case_id'],mesh_volume_mm3=exact,monte_carlo_volume_mm3=estimate,
                standard_error_mm3=float(stderr),difference_in_standard_errors=float(zscore),within_5_standard_errors=bool(zscore<5)))
            status='TOPOLOGY_AND_SAMPLING_CHECKS_PASS_VISUAL_PENDING'
            note='Planar base and diagnostic sections inspected; no final anatomy/self-intersection acceptance.'
            if row['case_id']=='00OMSZGW_lower':
                status='HOLD_IRREGULAR_CAP'
                note='Visual review found nonplanar U-shaped scan boundary and oblique cap. Conservative cap-height slab excludes about half the candidates; proxy shape needs review. Do not consume these diagnostic point labels for training.'
            if zscore>=5:
                status='HOLD_VOLUME_DISAGREEMENT';note+=' Monte Carlo and signed-volume check disagree; investigate self-intersection/classification.'
            decisions.append(dict(case_id=row['case_id'],decision=status,reason=note,automatic_training_release=False))
        else:
            decisions.append(dict(case_id=row['case_id'],decision='BLOCKED_NONMANIFOLD',reason=r.get('failure'),automatic_training_release=False))
        with Image.open(p/'review.png') as im:im.verify()
        checks.append(dict(case_id=row['case_id'],source_unchanged=True,prior_cache_unchanged=True))
    # Verify the ENTIRE previous package against its already-frozen delivery hashes.
    previous=read(INPUT/'delivery_manifest.json')
    assert all(sha(INPUT/x['path'])==x['sha256'] for x in previous)
    refs=re.findall(r'(?:href|src)="([^"]+)"',(OUT/'index.html').read_text(encoding='utf-8'))
    assert all(s.startswith('#') or (OUT/s).is_file() for s in refs)
    write(OUT/'volume_sampling_checks.json',volume_checks)
    write(OUT/'review_decisions.json',dict(review_kind='assistant preliminary visualization and computational checks; human final review pending',
        decisions=decisions,additional_training_boundary='Original DMM non-surface loss uses abs(field), not these inside/outside labels. Signed training supervision is not enabled by this pilot.'))
    notice='<div class="warning"><b>初步复核：</b>6 例上颌完成封底及空间点诊断，仍待人工验收。00OMSZGW_lower 虽通过拓扑闭合，但底边不平整、虚拟底面倾斜，保守排除带覆盖约一半候选点，标记 HOLD_IRREGULAR_CAP，不能用于训练。其余 5 例非流形导致封闭失败，未生成内外点。<br><a href="review_decisions.json">逐例复核决定</a> · <a href="volume_sampling_checks.json">网格体积与蒙特卡洛体积核对</a><br>原版 DMM 的非表面损失使用 |f|；生成内外标签并不等于模型已使用符号监督，本轮未修改 loader 或 loss。</div>'
    page=OUT/'index.html';text=page.read_text(encoding='utf-8');text=text.replace('</h1>','</h1>'+notice,1);page.write_text(text,encoding='utf-8')
    write(OUT/'qa_report.json',dict(verified_utc=now(),source_files_unchanged=24,previous_package_artifacts_unchanged=len(previous),
        local_links_verified=len(refs)+2,checks=checks,artifact_checks='PASS',human_visual_review='PENDING',training_accepted=False,
        monte_carlo_volume_checks=volume_checks,assistant_review_decisions=decisions))
    shutil.copyfile(__file__,OUT/'protocol/audit_space_proxy12_final.py')
    write(OUT/'delivery_manifest.json',[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()])
    print(json.dumps({k:v for k,v in results.items() if k!='cases'},indent=2),flush=True)

def finalize():
    """Preserve first delivery; add explicit hold after volume cross-check."""
    decisions=read(OUT/'review_decisions.json')['decisions'];volumes=read(OUT/'volume_sampling_checks.json')
    assert sum(x['decision']=='HOLD_VOLUME_DISAGREEMENT' for x in decisions)==1
    problem='00OMSZGW_lower';p=OUT/'cases'/problem
    im=Image.open(p/'review.png').convert('RGB');dr=ImageDraw.Draw(im)
    dr.rectangle((0,1500,1400,1610),fill='#fff0df')
    dr.text((25,1510),'分类校验失败：点比例体积与网格体积相差 15.36 个标准误；隔离此代理及其标签。',font=font(23),fill='#a1342b')
    dr.text((25,1555),'拓扑闭合不等于有效实体；优先定位虚拟底面与原表面的交叉/自交。未修改原件。',font=font(22),fill='#a1342b')
    im.save(p/'review_verified.png')
    contact=Image.open(OUT/'contact.png').convert('RGB');dr=ImageDraw.Draw(contact)
    dr.rectangle((0,1320,600,1364),fill='#edf1f5')
    dr.text((20,1320),problem+' 分类校验失败',font=font(22),fill='#a1342b')
    contact.save(OUT/'contact_verified.png')
    body=(OUT/'index.html').read_text(encoding='utf-8')
    notice='<div class="warning"><b>最终数值复核：6 例上颌完成封底和空间点诊断；6 例未放行。</b><br>00OMSZGW_lower 的网格体积为 7,933.5 mm³，点比例估计为 8,732.4 mm³，相差 15.36 个标准误，判定代理分类不可靠，隔离其几何和 50 万候选点。闭合拓扑未能排除自交或封底交叉问题，具体原因待定位。其余 5 例在非流形门槛被阻止，未生成内外点。6 例上颌的体积误差均小于 1.6 个标准误，14 万点旋转抽查中全部分类一致（含隔离例；该抽查单独不能证明正确）。<br>只有 6 例上颌的 300 万候选点通过本轮数值检查，仍未完成全面自交或人工验收，不是训练发布包。</div>'
    body=body.replace('</h1>','</h1>'+notice,1).replace('href="contact.png"','href="contact_verified.png"').replace('src="contact.png"','src="contact_verified.png"')
    body=body.replace(f'cases/{problem}/review.png',f'cases/{problem}/review_verified.png')
    (OUT/'index_verified.html').write_text(body,encoding='utf-8')
    write(OUT/'final_report.json',dict(created_utc=now(),diagnosed_cases=12,topologically_closed_candidates=7,
        computationally_consistent_proxies=6,blocked_nonmanifold=5,held_volume_disagreement=1,
        total_uniform_candidates=3500000,candidates_in_consistent_proxies=3000000,held_candidates=500000,
        training_export=False,human_visual_review='PENDING',inside_outside='closed-proxy labels only, not per-tooth signed distances',
        decisions=decisions,volume_checks=volumes,review_page='index_verified.html'))
    shutil.copyfile(__file__,OUT/'protocol/audit_space_proxy12_verified.py')
    previous=read(OUT/'delivery_manifest.json');assert all(sha(OUT/x['path'])==x['sha256'] for x in previous)
    import re
    refs=re.findall(r'(?:href|src)="([^"]+)"',body)
    assert all(x.startswith('#') or (OUT/x).is_file() for x in refs)
    write(OUT/'verified_delivery_manifest.json',[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()])
    print(OUT/'index_verified.html',flush=True)

if __name__=='__main__':
    {'initialize':initialize,'build':build,'render':render,'verify':verify,'finalize':finalize}[sys.argv[1]]()
