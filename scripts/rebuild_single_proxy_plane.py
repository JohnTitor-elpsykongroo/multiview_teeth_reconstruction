"""Single-case proxy-only planar base; all inputs immutable."""
from pathlib import Path
import sys,time,json,shutil
import numpy as np
import vtk
from PIL import Image,ImageDraw
from audit_space_proxy12 import pd,contains,screen,section,analyze_boundary_loops,select_main_boundary,edge_orientation_metrics
from diagnose_preprocess12 import read,write,sha,now,font
from teeth3ds_dmm.proxy.geometry import _ear_clip,_orient_cap_against_source,_polygon_self_intersects

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/dmm_proxy_plane_00OMSZGW_20261016_01'
OLD=ROOT/'runs/dmm_space_proxy12_20261016_01'
RAW=ROOT/'runs/dmm_preprocess12_20261016T1815_r1'
CID='00OMSZGW_lower'

def build():
    OUT.mkdir(exist_ok=False)
    inputs=[RAW/'cases'/CID/'raw_index_preserved.npz',OLD/'cases'/CID/'proxy_geometry.npz',OLD/'cases'/CID/'space_candidates_raw_mm.npz']
    ident=read(OLD/'cases'/CID/'input_identity.json')
    inputs += [Path(ident['source_obj']),Path(ident['source_annotation'])]
    write(OUT/'input_hashes.json',[dict(path=str(p),sha256=sha(p)) for p in inputs])
    d=np.load(inputs[0]);v=d['vertices'];f=d['faces'];b,_=select_main_boundary(analyze_boundary_loops(v,f),min_perimeter_ratio_to_second=3,max_planarity_ratio=.06,min_vertex_count=3,allow_minor_components=False)
    ids=np.asarray(b.vertex_ids);ring=v[ids];origin=ring.mean(0);_,_,vh=np.linalg.svd(ring-origin,full_matrices=False);axis=vh[-1]
    crown=v[np.isin(d['vertex_labels'],list(range(31,38))+list(range(41,48)))].mean(0)
    if np.dot(crown-origin,axis)<0:axis=-axis
    h=(v-origin)@axis;plane=float(h.min()-2.0)
    foot=ring+(plane-(ring-origin)@axis)[:,None]*axis
    uv=(ring-origin)@vh[:2].T
    assert not _polygon_self_intersects(uv)
    verts=np.concatenate([v,foot]);new=np.arange(len(v),len(v)+len(ids));side=[]
    for i in range(len(ids)):
        j=(i+1)%len(ids);side.extend([[ids[i],ids[j],new[j]],[ids[i],new[j],new[i]]])
    side=_orient_cap_against_source(np.asarray(side),f,set(b.edge_ids))
    cap=new[np.asarray(_ear_clip(uv))]
    cap=_orient_cap_against_source(cap,side,{tuple(sorted((int(new[i]),int(new[(i+1)%len(new)])))) for i in range(len(new))})
    faces=np.concatenate([f,side,cap]);assert edge_orientation_metrics(faces)==(0,0,0)
    assert np.array_equal(verts[:len(v)],v) and np.array_equal(faces[:len(f)],f)
    np.savez_compressed(OUT/'proxy_geometry.npz',vertices=verts,faces=faces,source_faces=f,side_faces=side,bottom_faces=cap,axis=axis,origin=origin,plane=plane)
    with (OUT/'proxy_ONLY_inside_outside.obj').open('x') as fp:
        fp.write('# PROXY ONLY: not surface/normal training geometry\n');np.savetxt(fp,verts,fmt='v %.10g %.10g %.10g')
        for name,fs in [('original',f),('virtual_sidewalls',side),('virtual_bottom',cap)]:fp.write('g '+name+'\n');np.savetxt(fp,fs+1,fmt='f %d %d %d')
    write(OUT/'construction.json',dict(case_id=CID,created_utc=now(),method='boundary extrusion along best-fit boundary normal to common plane',axis_towards_crown=axis.tolist(),origin=origin.tolist(),plane_height_mm=plane,plane_clearance_beyond_all_original_vertices_mm=2.,boundary_height_range_mm=[float(((ring-origin)@axis).min()),float(((ring-origin)@axis).max())],source_vertices=len(v),source_faces=len(f),side_faces=len(side),bottom_faces=len(cap),original_geometry_preserved=True,training_export=False,human_review='PENDING'))
    print('built',len(verts),len(faces),flush=True)

def intersections(v,f,start=0):
    """AABB broad phase, VTK triangle tests; adjacent faces inset to exclude common boundary."""
    mesh=pd(v,f);loc=vtk.vtkStaticCellLocator();loc.SetDataSet(mesh);loc.BuildLocator()
    tri=v[f];lo=tri.min(1);hi=tri.max(1);query=vtk.vtkIdList();pairs=[];adj=[];tested=0;t=time.time()
    for i in range(start,len(f)):
        bounds=np.column_stack([lo[i]-1e-9,hi[i]+1e-9]).ravel();loc.FindCellsWithinBounds(bounds,query)
        js=np.array([query.GetId(k) for k in range(query.GetNumberOfIds())],dtype=np.int64);js=js[js<i]
        js=js[np.all(hi[js]>=lo[i]-1e-9,axis=1)&np.all(lo[js]<=hi[i]+1e-9,axis=1)]
        for j in js:
            shared=np.any(f[i,:,None]==f[j,None,:]);a=tri[i];b=tri[j]
            if shared:
                # Relative inset 1e-7: ignores intended common edge/vertex contacts.
                a=a+(a.mean(0)-a)*1e-7;b=b+(b.mean(0)-b)*1e-7
            tested+=1
            if vtk.vtkTriangle.TrianglesIntersect(*a,*b):
                (adj if shared else pairs).append([int(i),int(j)])
        if (i-start)%50000==0:print('intersection scan',i,'/',len(f),'hits',len(pairs),len(adj),flush=True)
    return dict(nonadjacent_pairs=pairs,adjacent_overlap_candidates=adj,triangle_tests=tested,elapsed_seconds=time.time()-t,numerical_method='VTK TrianglesIntersect; AABB 1e-9 mm; shared-index faces inset 1e-7 relative, not exact arithmetic',scope_start_face=start)

def check(which):
    d=np.load(OUT/'proxy_geometry.npz');n=len(d['source_faces'])
    if which=='source':v=d['vertices'];f=d['source_faces'];start=0
    elif which=='old':o=np.load(OLD/'cases'/CID/'proxy_geometry.npz');v=o['vertices'];f=o['faces'];start=n
    else:v=d['vertices'];f=d['faces'];start=n
    result=intersections(v,f,start);write(OUT/f'intersections_{which}.json',result)
    print(which,len(result['nonadjacent_pairs']),len(result['adjacent_overlap_candidates']),flush=True)

def verify():
    """Secondary separating-axis check, especially for near-coplanar cap faces."""
    d=np.load(OUT/'proxy_geometry.npz');o=np.load(OLD/'cases'/CID/'proxy_geometry.npz')
    for key,v,f in [('source',d['vertices'],d['source_faces']),('old',o['vertices'],o['faces']),('new',d['vertices'],d['faces'])]:
        result=read(OUT/f'intersections_{key}.json');out={}
        for name in ['nonadjacent_pairs','adjacent_overlap_candidates']:
            kept=[]
            for i,j in result[name]:
                a=v[f[i]].copy();b=v[f[j]].copy()
                if name=='adjacent_overlap_candidates':a+=(a.mean(0)-a)*1e-7;b+=(b.mean(0)-b)*1e-7
                ea=np.roll(a,-1,axis=0)-a;eb=np.roll(b,-1,axis=0)-b;na=np.cross(ea[0],ea[1]);nb=np.cross(eb[0],eb[1])
                axes=np.concatenate([[na,nb],np.cross(ea[:,None],eb[None,:]).reshape(-1,3),np.cross(ea,na),np.cross(eb,nb)])
                norms=np.linalg.norm(axes,axis=1);axes=axes[norms>1e-15]/norms[norms>1e-15,None]
                pa=a@axes.T;pb=b@axes.T
                separated=np.any((pa.max(0)<pb.min(0)-1e-10)|(pb.max(0)<pa.min(0)-1e-10))
                if not separated:kept.append([i,j])
            out[name]=kept
        out['method']='VTK candidates filtered by independent triangle separating axes including coplanar axes; 1e-10 mm tolerance; shared faces inset 1e-7 relative'
        out['candidate_counts']={k:len(result[k]) for k in ['nonadjacent_pairs','adjacent_overlap_candidates']}
        write(OUT/f'intersections_verified_{key}.json',out)
        print(key,out['candidate_counts'],'verified',len(out['nonadjacent_pairs']),len(out['adjacent_overlap_candidates']),flush=True)

def classify():
    d=np.load(OUT/'proxy_geometry.npz');o=np.load(OLD/'cases'/CID/'proxy_geometry.npz');v=d['vertices'];f=d['faces'];rng=np.random.default_rng(2026101601)
    lo=np.minimum(v.min(0),o['vertices'].min(0))-5;hi=np.maximum(v.max(0),o['vertices'].max(0))+5
    xyz=rng.uniform(lo,hi,(500000,3));volbox=float(np.prod(hi-lo));res={};outs={}
    for name,a,fs in [('before',o['vertices'],o['faces']),('after',v,f)]:
        mask=contains(a,fs,xyz);outs[name]=mask;p=float(mask.mean());mc=volbox*p;se=volbox*np.sqrt(p*(1-p)/len(xyz));t=a[fs]-a.mean(0);vol=float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)
        ids=rng.choice(len(xyz),20000,replace=False);q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,0]*=np.linalg.det(q)
        other=contains((a-a.mean(0))@q,fs,(xyz[ids]-a.mean(0))@q)
        res[name]=dict(signed_volume_mm3=vol,monte_carlo_volume_mm3=mc,standard_error_mm3=se,difference_standard_errors=abs(mc-abs(vol))/se,inside_count=int(mask.sum()),rotation_check_count=len(ids),rotation_disagreements=int((other!=mask[ids]).sum()),boundary_nonmanifold_inconsistent=list(edge_orientation_metrics(fs)))
    # Independent signed-solid-angle check, including interior points rather than
    # only the overwhelmingly exterior uniform sample.
    checkids=np.unique(np.concatenate([rng.choice(len(xyz),128,replace=False),rng.choice(np.flatnonzero(outs['after']),128,replace=False)]))
    windings={}
    for name,a,fs in [('before',o['vertices'],o['faces']),('after',v,f)]:
        tri=a[fs];values=[]
        for p in xyz[checkids]:
            u=tri[:,0]-p;w=tri[:,1]-p;z=tri[:,2]-p
            nu=np.linalg.norm(u,axis=1);nw=np.linalg.norm(w,axis=1);nz=np.linalg.norm(z,axis=1)
            num=np.einsum('ij,ij->i',u,np.cross(w,z))
            den=nu*nw*nz+np.einsum('ij,ij->i',u,w)*nz+np.einsum('ij,ij->i',w,z)*nu+np.einsum('ij,ij->i',z,u)*nw
            values.append(float(np.arctan2(num,den).sum()/(2*np.pi)))
        ws=np.asarray(values);windings[name]=ws
        res[name]['solid_angle_check_count']=len(checkids)
        res[name]['solid_angle_parity_disagreements']=int(((np.abs(ws)>.5)!=outs[name][checkids]).sum())
        res[name]['solid_angle_max_integer_deviation']=float(np.abs(ws-np.round(ws)).max())
        print('solid-angle',name,'done',flush=True)
    np.savez_compressed(OUT/'space_candidates_DIAGNOSTIC_ONLY.npz',xyz=xyz,inside_before=outs['before'],inside_after=outs['after'],independent_check_indices=checkids,winding_before=windings['before'],winding_after=windings['after'])
    write(OUT/'classification.json',dict(candidate_count=len(xyz),domain=[lo.tolist(),hi.tolist()],seed=2026101601,results=res,changed_label_count=int((outs['before']!=outs['after']).sum()),note='Same points and domain for before/after. Not exported for training. Proxy volume intentionally changes.'))
    print(json.dumps(res,indent=2),flush=True)

def render(preview_only=False):
    d=np.load(OUT/'proxy_geometry.npz');o=np.load(OLD/'cases'/CID/'proxy_geometry.npz');raw=np.load(RAW/'cases'/CID/'raw_index_preserved.npz');basis=raw['basis'];v=d['vertices'];center=(v.min(0)+v.max(0))/2;field=90.
    out=Image.new('RGB',(1400,1540),'#edf1f5');dr=ImageDraw.Draw(out);dr.text((22,12),CID+' · 旧补面 / 统一底平面与侧壁',font=font(30),fill='#18384a')
    for k,(eye,label) in enumerate([([.5,.7,-.8],'斜侧视'),([0,0,1],'底部视图')]):
        for j,(a,fs,cap,title) in enumerate([(o['vertices'],o['source_faces'],o['virtual_cap_faces'],'修改前'),(v,d['source_faces'],np.concatenate([d['side_faces'],d['bottom_faces']]),'修改后')]):
            out.paste(screen(a,fs,cap,basis,center,eye,field,title+' · '+label),(700*j,65+700*k))
    dr.text((20,1480),'米色：原扫描；橙色：虚拟侧壁/底面。相同视野，仅代理副本发生变化。',font=font(24),fill='#18384a');out.save(OUT/'before_after.png')
    if preview_only:return
    # Show detected source/cap crossing faces on both proxies, same camera.
    im=Image.new('RGB',(1400,790),'#edf1f5');dr=ImageDraw.Draw(im);dr.text((20,10),'相交检查定位 · 橙色为检测命中的三角面',font=font(29),fill='#18384a')
    for j,(key,a,fs) in enumerate([('old',o['vertices'],o['faces']),('new',v,d['faces'])]):
        r=read(OUT/f'intersections_verified_{key}.json');src=read(OUT/'intersections_verified_source.json');pairs=r['nonadjacent_pairs']+r['adjacent_overlap_candidates']+src['nonadjacent_pairs']+src['adjacent_overlap_candidates'];ids=np.unique(np.asarray(pairs,dtype=int).ravel());hit=fs[ids] if len(ids) else fs[:0]
        im.paste(screen(a,fs,hit,basis,center,[.5,.7,-.8],field,('修改前' if j==0 else '修改后')+f' · 命中 {len(ids)} 个面'),(j*700,65))
    im.save(OUT/'intersection_locations.png')
    samples=np.load(OUT/'space_candidates_DIAGNOSTIC_ONLY.npz');im=Image.new('RGB',(1400,1500),'#edf1f5');dr=ImageDraw.Draw(im)
    dr.text((20,10),'同一批空间点 · 前后截面对照（中央截面 ±0.5 mm）',font=font(28),fill='#18384a')
    for k,ax in enumerate([0,1]):
        for j,(name,a,fs,cap) in enumerate([('before',o['vertices'],o['source_faces'],o['virtual_cap_faces']),('after',v,d['source_faces'],np.concatenate([d['side_faces'],d['bottom_faces']]))]):
            panel=section(a,fs,cap,basis,center,samples['xyz'],samples['inside_'+name],np.zeros(len(samples['xyz']),bool),ax,('修改前' if j==0 else '修改后')+f' · {"YZ" if ax==0 else "XZ"}')
            dd=ImageDraw.Draw(panel);dd.rectangle((0,40,700,74),fill='#edf1f5');dd.text((15,43),'红：代理内 / 蓝：代理外；橙线：虚拟面',font=font(19),fill='#39576b')
            im.paste(panel,(j*700,65+k*700))
    im.save(OUT/'inside_outside_sections.png')

def finish():
    assert all(sha(x['path'])==x['sha256'] for x in read(OUT/'input_hashes.json'))
    checks={k:read(OUT/f'intersections_verified_{k}.json') for k in ['source','old','new']};cl=read(OUT/'classification.json');new=cl['results']['after']
    hits=lambda x:len(x['nonadjacent_pairs'])+len(x['adjacent_overlap_candidates'])
    passed=hits(checks['source'])==0 and hits(checks['new'])==0 and new['difference_standard_errors']<=5 and new['rotation_disagreements']==0 and new['solid_angle_parity_disagreements']==0
    summary=dict(status='NUMERICAL_CHECKS_PASS_HUMAN_PENDING' if passed else 'HOLD_INTERSECTION_OR_CLASSIFICATION',human_review='PENDING',original_inputs_unchanged=True,source_intersection_pairs=hits(checks['source']),old_virtual_intersection_pairs=hits(checks['old']),new_virtual_intersection_pairs=hits(checks['new']),classification=cl,training_export=False)
    write(OUT/'summary.json',summary)
    txt=f'''<!doctype html><meta charset="utf-8"><title>00OMSZGW 代理封底复核</title><style>body{{font:18px/1.7 system-ui;max-width:1450px;margin:30px auto;background:#edf1f5;color:#18384a}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style><h1>00OMSZGW_lower：统一底平面和侧壁</h1><p>仅新增代理副本；原始顶点及面保持原样。底平面沿边界最佳拟合平面的法线方向，放在所有原始顶点之外 2 mm。原版代理保留。本页不代表训练验收。</p><p>状态：{summary['status']}。原始面相交候选 {hits(checks['source'])} 对；旧补面相关 {hits(checks['old'])} 对；新侧壁/底面相关 {hits(checks['new'])} 对。相邻面以微小内缩排除共享边接触，属于浮点数值检查，不是精确算术证明。</p><img src="before_after.png"><h2>相交定位</h2><img src="intersection_locations.png"><h2>同一批 50 万空间点检查</h2><pre>{json.dumps(cl,ensure_ascii=False,indent=2)}</pre><p>内外标签变化包含底部代理体积变化，不能当作误分类率。即使封底通过，也只用于空间点内外判断，虚拟面不进入表面或法线监督。</p><a href="proxy_ONLY_inside_outside.obj">新代理 OBJ</a> · <a href="summary.json">审核摘要</a> · <a href="construction.json">构造参数</a>'''
    txt=txt.replace('<h2>同一批 50 万空间点检查</h2>','<h2>同一批 50 万空间点检查</h2><img src="inside_outside_sections.png">')
    txt=txt.replace('相邻面以微小内缩排除共享边接触，属于浮点数值检查，不是精确算术证明。','VTK 初筛后使用独立的分离轴检查复核，剔除了近共面三角形的假阳性。相邻面以微小内缩排除共享边接触；这是浮点数值检查，不是精确算术证明。新侧壁/底面通过本轮检查，但原扫描表面仍存在相交问题，因此整个代理保留 HOLD。')
    (OUT/'index.html').write_text(txt,encoding='utf-8');shutil.copyfile(__file__,OUT/'implementation.py')
    write(OUT/'delivery_manifest.json',[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file()]);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    action=sys.argv[1]
    if action in ['source','old','new']:check(action)
    elif action=='preview':render(True)
    else:globals()[action]()
