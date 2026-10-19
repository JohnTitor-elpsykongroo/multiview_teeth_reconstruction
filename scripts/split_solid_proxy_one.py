"""FDI-based dental collar / outer proxy split, independent Make Solid, CSG union."""
from pathlib import Path
import sys,json,shutil
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image,ImageDraw
from diagnose_preprocess12 import read,write,sha,font
from audit_space_proxy12 import pd,contains,screen,edge_orientation_metrics
from localize_poisson_proxy_issues import locator,closest
import audit_make_solid_one as audit
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/dmm_ms_split_00OMSZGW_20261017_01'
BASE=audit.BASE;RAW=audit.RAW;OLD=ROOT/'runs/dmm_make_solid_00OMSZGW_20261017_01'

def partition():
    raw=np.load(RAW);b=np.load(BASE/'proxy_geometry.npz');v=b['vertices'];f=b['faces'];tooth=np.zeros(len(f),bool);tooth[:len(raw['faces'])]=(raw['vertex_labels'][raw['faces']]>0).any(1);dist=cKDTree(raw['vertices'][raw['vertex_labels']>0]).query(v[f].mean(1))[0]
    dental=tooth|(dist<=2.);outer=(~tooth)&(dist>=1.);assert np.all(dental|outer);assert not np.any(tooth&outer)
    write(OUT/'input_hashes.json',[dict(path=str(x),sha256=sha(x)) for x in [RAW,BASE/'proxy_geometry.npz',OLD/'solid_main.npz']])
    np.savez_compressed(OUT/'partition.npz',face_is_dental=dental,face_is_outer=outer,face_is_overlap=dental&outer,distance_to_tooth_vertex_mm=dist)
    for name,mask in [('dental',dental),('outer',outer)]:
        used,idx=np.unique(f[mask],return_inverse=True);vv=v[used];ff=idx.reshape(-1,3)
        with (OUT/(name+'_input.obj')).open('x') as fp:
            fp.write('# INPUT PART FOR PROXY ONLY; raw coordinates in mm\n');np.savetxt(fp,vv,fmt='v %.10g %.10g %.10g');np.savetxt(fp,ff+1,fmt='f %d %d %d')
    write(OUT/'partition.json',dict(dental_faces=int(dental.sum()),outer_faces=int(outer.sum()),overlap_faces=int((dental&outer).sum()),method='All faces touching an FDI>0 source vertex plus faces whose centroid is within 2 mm of a tooth vertex; outer excludes tooth faces and starts at centroid distance 1 mm; distance is Euclidean to labeled vertices, not geodesic',all_faces_covered=True,tooth_faces_in_dental_only=True,expected_overlap_band_mm=[1,2],scope='Only proxy; separate solids retain all nonzero-volume components until boolean union'))
    center=(v.min(0)+v.max(0))/2;im=Image.new('RGB',(1400,780),'#edf1f5');dr=ImageDraw.Draw(im)
    for j,(mask,title) in enumerate([(dental,'牙体 + 2 mm 邻域'),(outer,'外围牙龈 + 侧壁 + 底面')]):im.paste(screen(v,f[mask],f[dental&outer],raw['basis'],center,[.5,.7,-.8],90,title),(700*j,0))
    dr.text((20,720),'橙色：两部分共有的重叠带；两份输入均保留原始坐标。',font=font(25),fill='#18384a');im.save(OUT/'partition.png');print('partition saved',flush=True)

def union():
    sys.path.insert(0,str(OUT/'deps'));import manifold3d as mm
    solids=[];records=[]
    for name in ['dental','outer']:
        par=read(OUT/('split_'+name)/'parameters.json');assert par['status']=='EXPORTED';mesh=pv.read(par['export']).triangulate();v,inv=np.unique(np.asarray(mesh.points,dtype=np.float64),axis=0,return_inverse=True);f=inv[mesh.faces.reshape(-1,4)[:,1:]]
        n,lab=connected_components(coo_matrix((np.ones(f.size,np.int8),(f.ravel(),np.roll(f,-1,axis=1).ravel())),shape=(len(v),len(v))),directed=False);fl=lab[f[:,0]];kept=[];components=[]
        for i in range(n):
            fs=f[fl==i];t=v[fs];c=t.mean((0,1));t=t-c;vol=float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6);keep=abs(vol)>1e-8
            components.append(dict(id=i,faces=len(fs),signed_volume_mm3=vol,kept=keep))
            if keep:kept.append(fs)
        ff=np.concatenate(kept);used,idx=np.unique(ff,return_inverse=True);v=v[used];f=idx.reshape(-1,3);top=edge_orientation_metrics(f);assert top==(0,0,0),top
        solid=mm.Manifold(mm.Mesh(vert_properties=np.asarray(v,dtype=np.float32),tri_verts=np.asarray(f,dtype=np.uint32)));status=str(solid.status());assert status=='Error.NoError',status
        np.savez_compressed(OUT/(name+'_solid.npz'),vertices=v,faces=f);solids.append(solid);records.append(dict(part=name,components=components,manifold_status=status,volume_mm3=solid.volume()))
    intersection=solids[0]^solids[1];iv=intersection.volume();assert iv>0,('No volumetric overlap',iv)
    merged=solids[0]+solids[1];assert str(merged.status())=='Error.NoError';m=merged.to_mesh();v=np.asarray(m.vert_properties)[:,:3].astype(np.float64);f=np.asarray(m.tri_verts).astype(np.int64)
    # Retain all union shells first; do not silently discard teeth.
    np.savez_compressed(OUT/'solid_main.npz',vertices=v,faces=f);pd(v,f).save(OUT/'split_union_PROXY_ONLY.ply')
    n,lab=connected_components(coo_matrix((np.ones(f.size,np.int8),(f.ravel(),np.roll(f,-1,axis=1).ravel())),shape=(len(v),len(v))),directed=False);fl=lab[f[:,0]];counts=np.bincount(fl);write(OUT/'components.json',[dict(id=i,faces=int(c),kept=True) for i,c in enumerate(counts)])
    write(OUT/'union.json',dict(parts=records,volumetric_overlap_mm3=iv,union_volume_mm3=merged.volume(),union_components=int(n),union_faces=len(f),manifold_status=str(merged.status()),method='Manifold3D boolean union; all nonzero-volume part components retained; no final smoothing'))
    print('union',len(v),len(f),'components',n,'overlap mm3',iv,flush=True)

def evaluate():
    audit.OUT=OUT;oldwrite=audit.write
    def writer(p,x):
        if Path(p).name=='evaluation.json':
            x['method']='FDI dental collar / outer proxy split; Make Solid Accurate 512/512 independently; Manifold3D union';x['components_after']=read(OUT/'union.json')['union_components']
        oldwrite(p,x)
    audit.write=writer;audit.evaluate();audit.intersection_check()
    d=np.load(OUT/'solid_main.npz');rng=np.random.default_rng(2026101705);b=np.load(BASE/'proxy_geometry.npz');old=np.load(OLD/'solid_main.npz');records=[]
    for reg in read(OLD/'local_sampling.json'):
        xyz=rng.uniform(-1.5,1.5,(10000,3))+reg['center'];inside=contains(d['vertices'],d['faces'],xyz);oi=contains(old['vertices'],old['faces'],xyz);bi=contains(b['vertices'],b['faces'],xyz);records.append(dict(fdi=reg['fdi'],center=reg['center'],count=len(xyz),vs_unsplit_make_solid=int((inside!=oi).sum()),vs_planar=int((inside!=bi).sum()),inside=int(inside.sum())))
    write(OUT/'fixed_local_sampling.json',records)

def report():
    raw=np.load(RAW);original=np.load(BASE/'proxy_geometry.npz');new=np.load(OUT/'solid_main.npz');old=np.load(OLD/'solid_main.npz');r=read(OUT/'evaluation.json');baseline=read(OLD/'evaluation.json');ix=read(OUT/'intersection_check.json');u=read(OUT/'union.json');basis=raw['basis'];center=(original['vertices'].min(0)+original['vertices'].max(0))/2
    im=Image.new('RGB',(1400,770),'#edf1f5')
    for col,name in enumerate(['dental','outer']):
        d=np.load(OUT/(name+'_solid.npz'));im.paste(screen(d['vertices'],d['faces'],np.empty((0,3),int),basis,center,[.5,.7,-.8],90,('牙体' if col==0 else '外围')+'单独 Make Solid'),(700*col,0))
    ImageDraw.Draw(im).text((20,720),'两部分分别实体化；下一步进行布尔并集，不能仅拼接三角面。',font=font(24),fill='#18384a');im.save(OUT/'parts_solid.png')
    im=Image.new('RGB',(1400,1490),'#edf1f5');ImageDraw.Draw(im).text((20,7),'整体实体化 vs 分区实体化后并集',font=font(29),fill='#18384a')
    for row,eye in enumerate([[.5,.7,-.8],[0,0,1]]):
        for col,(d,title) in enumerate([(old,'之前：整体 Make Solid'),(new,'本次：分区 Make Solid + 并集')]):im.paste(screen(d['vertices'],d['faces'],np.empty((0,3),int),basis,center,eye,90,title),(col*700,50+row*700))
    im.save(OUT/'before_after.png');olddev=np.load(OLD/'deviation.npz');dev=np.load(OUT/'deviation.npz')
    for reg in read(OLD/'local_sampling.json'):
        k=reg['fdi'];c=np.asarray(reg['center']);idx=np.argmin(np.linalg.norm(raw['vertices']-c,axis=1));eye=olddev['closest_xyz'][idx]-c;eye/=np.linalg.norm(eye);up=basis[:,2]
        if abs(eye@up)>.9:up=basis[:,1]
        canvas=Image.new('RGB',(1800,680),'#edf1f5')
        for col,(vv,ff,title) in enumerate([(raw['vertices'],raw['faces'],'原扫描'),(old['vertices'],old['faces'],'整体 Make Solid'),(new['vertices'],new['faces'],'分区后实体并集')]):
            sel=np.linalg.norm(vv[ff].mean(1)-c,axis=1)<5;mesh=pd(vv,ff[sel]).clip(normal=eye,origin=c+eye*.08,invert=True);pl=pv.Plotter(off_screen=True,window_size=(600,600));pl.set_background('#edf1f5');pl.add_mesh(mesh,color='#d5c7af',ambient=.4);pl.add_mesh(pv.Sphere(radius=.04,center=c),color='#c52d50');pl.camera_position=[c+eye*50,c,up];pl.enable_parallel_projection();pl.camera.parallel_scale=2.5;pic=Image.fromarray(pl.screenshot(return_img=True));pl.close();dd=ImageDraw.Draw(pic);dd.rectangle((0,0,600,43),fill='#edf1f5');dd.text((12,7),title,font=font(22),fill='#18384a');canvas.paste(pic,(600*col,0))
        ImageDraw.Draw(canvas).text((15,620),f'FDI {k}：同位置、同剖切平面，5 mm 视野；显示剖切开口不是网格漏洞。',font=font(25),fill='#18384a');canvas.save(OUT/f'local_{k}.png')
    partition=np.load(OUT/'partition.npz');faces=original['faces'][partition['face_is_overlap']];seam=np.unique(faces);seam=seam[seam<len(raw['vertices'])];ds=dev['distance_mm'][seam];seamstats=dict(vertex_count=len(seam),p95_mm=float(np.quantile(ds,.95)),max_mm=float(ds.max()))
    write(OUT/'seam_deviation.json',seamstats)
    pl=pv.Plotter(off_screen=True,window_size=(1400,850));pl.set_background('#edf1f5');pl.add_mesh(pd(new['vertices'],new['faces']),color='#d5c7af',ambient=.4);pts=pv.PolyData(raw['vertices'][seam]);pts['distance_mm']=ds;pl.add_mesh(pts,scalars='distance_mm',cmap='turbo',clim=[0,.5],point_size=5,render_points_as_spheres=True,scalar_bar_args={'title':'Original overlap-band distance (mm), clipped 0.5'});eye=basis@np.array([.5,.7,-.8]);pl.camera_position=[center+eye*250,center,basis[:,2]];pl.enable_parallel_projection();pl.camera.parallel_scale=45;pl.screenshot(OUT/'seam.png');pl.close()
    html='<!doctype html><meta charset="utf-8"><title>分区 Make Solid 验收</title><style>body{font:18px/1.7 system-ui;max-width:1500px;margin:25px auto;background:#edf1f5;color:#18384a}img{max-width:100%}td,th{border:1px solid #abc;padding:10px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>00OMSZGW_lower：牙体与外围分别 Make Solid，再做实体并集</h1><p>根据 FDI 牙号定义牙体，附加 2 mm 欧氏邻域；外围从距牙体 1 mm 处开始，留下 1–2 mm 重叠带。两边均使用 Accurate 512/512、零偏移、零增厚。使用 Manifold3D 做实体并集，不做合并后的全局平滑。</p><p>全部原始三角面至少进入一份输入。各部分仅删除体积绝对值 ≤ 1e-8 mm³ 的退化闭合片；其他实体均参与并集，避免误删牙体。只处理本例代理，不覆盖原扫描、不跑全量、不导出训练数据。</p><table><tr><th>方法</th><th>37 原顶点 P95 / 最大 mm</th><th>47 原顶点 P95 / 最大 mm</th><th>20k 原表面 P95 mm</th></tr>'
    for title,e in [('整体 Make Solid',baseline),('分区 Make Solid + 并集',r)]:
        t=e['original_vertex_distances_by_fdi'];html+=f'<tr><td>{title}</td><td>{t["37"]["p95"]:.4f} / {t["37"]["max"]:.4f}</td><td>{t["47"]["p95"]:.4f} / {t["47"]["max"]:.4f}</td><td>{e["original_surface_samples_mm"]["p95"]:.4f}</td></tr>'
    html+='</table><h2>1. 分区输入</h2><img src="partition.png"><h2>2. 分别实体化</h2><img src="parts_solid.png"><h2>3. 并集合并后</h2><img src="before_after.png"><h2>4. 固定后牙局部剖切</h2><img src="local_37.png"><img src="local_47.png"><h2>5. 原重叠带到最终代理的距离</h2><img src="seam.png"><pre>'+json.dumps(seamstats,ensure_ascii=False,indent=2)+'</pre><h2>数值检查</h2><pre>'+json.dumps(r,ensure_ascii=False,indent=2)+'</pre><pre>'+json.dumps(ix,ensure_ascii=False,indent=2)+'</pre><h2>固定局部空间点</h2><pre>'+json.dumps(read(OUT/'fixed_local_sampling.json'),ensure_ascii=False,indent=2)+'</pre><details><summary>并集记录</summary><pre>'+json.dumps(u,ensure_ascii=False,indent=2)+'</pre></details><p>距离不是连续表面的 Hausdorff 上界；不同代理的分类分歧不是错误率。数值自交筛查不是精确数学证明。人工验收待定。</p><p><a href="split_union_PROXY_ONLY.ply">合并代理 PLY</a> · <a href="partition.json">分区定义</a></p>'
    (OUT/'index.html').write_text(html,encoding='utf-8');assert all(sha(x['path'])==x['sha256'] for x in read(OUT/'input_hashes.json'));assert all(sha(x['path'])==x['sha256'] for x in read(BASE/'input_hashes.json'));write(OUT/'source_hash_verification.json',dict(all_unchanged=True));shutil.copyfile(__file__,OUT/'implementation.py');shutil.copyfile(ROOT/'scripts/mm_tune_solid.ps1',OUT/'api_implementation.ps1');write(OUT/'delivery_manifest.json',[dict(path=str(x.relative_to(OUT)),sha256=sha(x)) for x in OUT.rglob('*') if x.is_file() and 'deps' not in x.parts and 'wheels' not in x.parts]);print('report ready',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
