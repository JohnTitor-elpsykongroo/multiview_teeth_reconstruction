"""Twelve-case fixed-reference similarity diagnostic; no training publication.

Reuses the legacy canonical-frame builder and SVD solver unchanged. Lower FDI
is adapted only at the upper-only legacy builder API, never in source/output.
"""
from pathlib import Path
import sys,json,csv,hashlib,shutil,math
import numpy as np
import pyvista as pv
from PIL import Image,ImageDraw
from vtkmodules.vtkRenderingCore import vtkCoordinate
from diagnose_preprocess12 import read,write,sha,now,font,unit

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/dmm_preprocess12_20261016T1815_r1'
SPACE=ROOT/'runs/dmm_space_proxy12_20261016_01'
OUT=ROOT/'runs/dmm_fixed_reference12_20261016_01'
OLD=Path('D:/WorkSpace/Dental/teeth3DS_to_DMM/src')
sys.path.insert(0,str(OLD))
from teeth3ds_dmm.alignment.canonical import build_canonical_frame
from teeth3ds_dmm.alignment.similarity import solve_similarity

REFS={'upper':'01328DDN_upper','lower':'01JNZKAX_lower'}
COLORS=['#168b98','#e08b36','#7268b4','#c35d82','#54944c','#596eaa']
def labels(arch):return [q*10+i for q in ([1,2] if arch=='upper' else [3,4]) for i in range(1,8)]
def mesh(v,f):return pv.PolyData(v,np.column_stack([np.full(len(f),3),f]).ravel())
def transform(x,t):return t['scale']*(np.asarray(x)@np.asarray(t['rotation']).T)+np.asarray(t['translation'])
def homogeneous(t):
    a=np.eye(4);a[:3,:3]=t['scale']*np.array(t['rotation']);a[:3,3]=t['translation'];return a
def centroids(d,arch):
    return {i:np.average(d['face_centers'][mask],axis=0,weights=d['face_area'][mask])
            for i in labels(arch) if (mask:=(d['face_label']==i)&d['strict']).any()}

def initialize():
    OUT.mkdir(exist_ok=False);(OUT/'protocol').mkdir();(OUT/'cases').mkdir();(OUT/'references').mkdir()
    cohort=read(SOURCE/'protocol/cohort.json');assert len(cohort['cases'])==12
    splitpath=Path('D:/WorkSpace/Dental/data/Teeth3DS_DualArch_v1/reports/patient_splits.json');splits=read(splitpath)
    oldtest={x.split('_')[0] for x in read('D:/WorkSpace/Dental/data/Teeth3DS_DMM/05_dataset_profiles/upper_v1.3_strict_fixed_offsurface/splits/test_split.json')}
    for row in cohort['cases']:
        assert splits[row['patient_id']]=='train' and row['patient_id'] not in oldtest
    shutil.copyfile(SOURCE/'protocol/cohort.json',OUT/'protocol/cohort.json')
    shutil.copyfile(__file__,OUT/'protocol/implementation_initial.py')
    dependencies=[splitpath,SOURCE/'protocol/cohort.json',SOURCE/'delivery_manifest.json',SPACE/'verified_delivery_manifest.json',
        OLD/'teeth3ds_dmm/alignment/canonical.py',OLD/'teeth3ds_dmm/alignment/similarity.py',OLD/'teeth3ds_dmm/schemas/alignment.py',ROOT/'docs/dmm_preprocessing_replan_20261016.md']
    write(OUT/'protocol/dependencies.json',[dict(path=str(p),sha256=sha(p)) for p in dependencies])
    write(OUT/'protocol/frozen_diagnostic_choices.json',dict(created_utc=now(),case_count=12,refs=REFS,
        reference_selection_frozen_before_alignment=True,old_reference_case='6XSU5W4B_upper',old_reference_current_split=splits['6XSU5W4B'],
        reference_change_reason='Use the old fixed-reference algorithm, but avoid the old reference now assigned to held-out test. Independent upper/lower development references.',
        upper_reference_reason='14 target teeth, complete control in both old and current training, no original topology conflict, not a size-extreme selection',
        lower_reference_reason='14 target teeth and conventional scan base; other full lower candidates have irregular-base proxy failure or posterior-crown normal/shape alert. This reference has 39 gum nonmanifold edges: allowed only for centroid-based diagnostic alignment, not training acceptance.',
        reference_status='FIXED_FOR_DIAGNOSTIC_RUN_ONLY',human_reference_acceptance='PENDING',
        centroid_method='strict face area weighted, per-tooth equal weights in alignment',
        min_common_teeth=6,bilateral_and_anterior_posterior_required=True,
        method='legacy SVD fixed reference, x_model=s_i R_i x_mm+t_i, proper rotation and positive uniform scale',
        canonical_reference_bbox_max_side=1.9,canonical_axes='+X patient-left; +Y anterior; +Z X cross Y (upper crown +Z, lower crown -Z)',
        before_overlay='Apply SAME reference canonical transform to every RAW case, no case-specific centering, rotation, or scaling',
        relative_scale='alpha=s_case/s_reference; dimensionless size adjustment into reference-equivalent mm',
        residual_mm='post residual_model/s_case, case-mm equivalent; not correspondence accuracy or preprocessing damage',
        source_geometry_unchanged=True,no_per_tooth_fit=True,no_nonrigid_warp=True,no_clamp=True,no_mirror=True,
        sampling='No new surface or spatial sampling; existing proxy/space candidates follow the same transform with previous hold statuses retained',
        training_export=False))
    for arch,cid in REFS.items():
        d=np.load(SOURCE/'cases'/cid/'raw_index_preserved.npz');c=centroids(d,arch);assert len(c)==14
        # The legacy dataclass only accepts upper FDI. This bijection changes API
        # identifiers only; right/left anatomy and all coordinates are untouched.
        api={i:(i if arch=='upper' else (20+i%10 if i//10==3 else 10+i%10)) for i in c}
        oldc={api[i]:tuple(p) for i,p in c.items()}
        frame=build_canonical_frame(reference_case_id=cid,centroids=oldc,mesh_vertices=d['vertices'],
            right_group=tuple(range(11,18)),left_group=tuple(range(21,28)),anterior_group=(11,12,21,22),posterior_group=(16,17,26,27),
            reference_bbox_margin=.95,manual_orientation_confirmed=False)
        ref=dict(case_id=cid,arch=arch,transform=frame.transform.to_dict(),
            basis_columns=np.column_stack([frame.basis_x,frame.basis_y,frame.basis_z]).tolist(),
            centroids_raw_mm={str(k):v.tolist() for k,v in c.items()},
            target_centroids_model={str(k):list(frame.reference_centroids[api[k]]) for k in c},
            legacy_api_identifier_adapter=api,actual_output_fdi_unchanged=True,
            original_bbox_half_extent_mm=frame.bbox_half_extent,reference_bbox_margin=.95,
            human_orientation_confirmation='PENDING',reference_status='FIXED_DIAGNOSTIC_ONLY')
        write(OUT/'references'/f'{arch}.json',ref)
        print(arch,cid,'reference model/mm',frame.transform.scale,flush=True)

def align():
    rows=read(OUT/'protocol/cohort.json')['cases'];space_dec={x['case_id']:x for x in read(SPACE/'review_decisions.json')['decisions']}
    refs={a:read(OUT/'references'/f'{a}.json') for a in REFS};results=[];residual_rows=[];global_extent=1.
    for row in rows:
        cid=row['case_id'];arch=row['arch'];p=OUT/'cases'/cid;p.mkdir(exist_ok=False)
        sourcepath=SOURCE/'cases'/cid/'raw_index_preserved.npz';d=np.load(sourcepath);v=d['vertices'];f=d['faces'];c=centroids(d,arch)
        prior=read(SOURCE/'cases'/cid/'metrics.json');ref=refs[arch];target={int(k):np.array(x) for k,x in ref['target_centroids_model'].items()}
        common=sorted(c.keys()&target.keys());qs=[1,2] if arch=='upper' else [3,4]
        assert len(common)>=6 and all(any(i//10==q for i in common) for q in qs)
        assert any(i%10 in [1,2] for i in common) and any(i%10 in [6,7] for i in common)
        x=np.array([c[i] for i in common]);y=np.array([target[i] for i in common]);sv=np.linalg.svd(x-x.mean(0),compute_uv=False)
        assert sv[1]/sv[0]>1e-4,'Near-collinear centroid anchors'
        fit=solve_similarity(x,y);t=fit.to_dict();s=t['scale'];R=np.array(t['rotation']);tr=np.array(t['translation'])
        H=homogeneous(t);Hi=np.linalg.inv(H);va=fit.apply(v);vb=transform(v,ref['transform'])
        before=transform(x,ref['transform']);after=fit.apply(x)
        rb=np.linalg.norm(before-y,axis=1);ra=np.linalg.norm(after-y,axis=1)
        assert (ra@ra)<=(rb@rb)+1e-12,'Fit failed to improve its objective'
        roundtrip=(va-tr)@R/s;error=np.linalg.norm(roundtrip-v,axis=1)
        norm=d['face_normals']@R.T;norm/=np.linalg.norm(norm,axis=1,keepdims=True)
        tri=va[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);cross/=np.linalg.norm(cross,axis=1,keepdims=True)
        ndot=np.einsum('ij,ij->i',norm,cross);assert ndot.min()>1-1e-5
        # Whole-mesh checks, not sampling: no discarded base, wisdom or outlier.
        outside=np.any(np.abs(va)>1+1e-10,axis=1);lab=d['vertex_labels'];tooth=np.isin(lab,labels(arch));gum=lab==0
        edges=f[:,[0,1]];srcdist=np.linalg.norm(v[edges[:,1]]-v[edges[:,0]],axis=1);dstdist=np.linalg.norm(va[edges[:,1]]-va[edges[:,0]],axis=1)
        length_relative=np.max(np.abs(dstdist/(s*srcdist)-1))
        assert length_relative<1e-7 and error.max()<1e-5
        alpha=s/ref['transform']['scale'];pertooth=[]
        for i,label in enumerate(common):
            item=dict(case_id=cid,arch=arch,fdi=label,before_model=float(rb[i]),after_model=float(ra[i]),after_case_mm=float(ra[i]/s),
                reference_center_model=y[i].tolist(),before_center_model=before[i].tolist(),after_center_model=after[i].tolist(),
                delta_model=(after[i]-y[i]).tolist())
            pertooth.append(item);residual_rows.append(item)
            assert np.linalg.norm(c[label]-prior['tooth_statistics'][str(label)]['center_raw_mm'])<1e-10
        m=dict(case_id=cid,arch=arch,reference_case=ref['case_id'],is_reference=cid==ref['case_id'],common_fdi=common,
            transform=t,T_model_from_raw_mm=H.tolist(),T_raw_mm_from_model=Hi.tolist(),
            scale_model_per_mm=s,mm_per_model_unit=1/s,relative_scale_to_reference=alpha,size_adjustment_percent=100*(alpha-1),
            before_rmse_model=float(np.sqrt(np.mean(rb**2))),after_rmse_model=float(np.sqrt(np.mean(ra**2))),
            after_rmse_case_mm=float(np.sqrt(np.mean(ra**2))/s),max_residual_model=float(ra.max()),max_residual_case_mm=float(ra.max()/s),
            per_tooth=pertooth,source_centroid_singular_values_mm=sv.tolist(),source_centroid_second_to_first=float(sv[1]/sv[0]),
            vertices=len(v),faces=len(f),outside_domain_vertices=int(outside.sum()),outside_domain_fraction=float(outside.mean()),
            outside_target_tooth_vertices=int((outside&tooth).sum()),outside_gum_vertices=int((outside&gum).sum()),
            outside_other_vertices=int((outside&~(tooth|gum)).sum()),max_abs_model_coordinate=float(np.abs(va).max()),
            inverse_roundtrip_max_mm=float(error.max()),normal_vs_recomputed_min_dot=float(ndot.min()),uniform_edge_scale_max_relative_error=float(length_relative),
            prior_topology=prior['topology'],prior_reason=prior['previous_reason'],wisdom_fdi=prior['wisdom_fdi'],
            missing_target_fdi=prior['missing_target_fdi'],space_proxy_decision=space_dec[cid]['decision'],
            raw_cache_sha256=sha(sourcepath),source_obj_sha256=sha(row['source_obj']),source_annotation_sha256=sha(row['source_annotation']),
            human_visual_review='PENDING',training_accepted=False,geometry_status='ALIGNED_FOR_DIAGNOSIS')
        np.savez_compressed(p/'aligned_geometry.npz',vertices_model=va,faces=f,vertex_labels=lab,vertex_instances=d['vertex_instances'],
            face_normals_model=norm,face_label=d['face_label'],strict=d['strict'],outside_domain_vertex_mask=outside)
        sp=SPACE/'cases'/cid
        if (sp/'space_candidates_raw_mm.npz').exists():
            a=np.load(sp/'space_candidates_raw_mm.npz');xyz=fit.apply(a['xyz']);ab={k:a[k] for k in a.files if k!='xyz'}
            ab.update(xyz_model=xyz,outside_canonical_cube=np.any(np.abs(xyz)>1,axis=1))
            np.savez_compressed(p/'previous_space_candidates_transformed.npz',**ab)
            g=np.load(sp/'proxy_geometry.npz');np.savez_compressed(p/'previous_proxy_transformed.npz',vertices_model=fit.apply(g['vertices']),
                source_faces=g['source_faces'],virtual_cap_faces=g['virtual_cap_faces'],faces=g['faces'],face_is_virtual_cap=g['face_is_virtual_cap'])
            m['transformed_previous_space_candidates']=len(xyz)
            m['previous_space_domain_note']='Rigid-similarity transform of old raw-mm box candidates; NOT a fresh uniform sample of canonical [-1,1]^3. Previous hold remains effective.'
        else:m['transformed_previous_space_candidates']=0
        write(p/'metrics.json',m);write(p/'transform.json',dict(case_id=cid,**t,T_model_from_raw_mm=H.tolist(),T_raw_mm_from_model=Hi.tolist(),units='model/mm',human_acceptance='PENDING'))
        results.append(m);global_extent=max(global_extent,float(np.abs(va).max()),float(np.abs(vb).max()))
        print(cid,f's={s:.7f} alpha={alpha:.4f} RMSE {m["before_rmse_model"]:.4f}->{m["after_rmse_model"]:.4f} ({m["after_rmse_case_mm"]:.3f}mm) outside={int(outside.sum())}',flush=True)
    # Shared frame for ALL cases and BOTH before/after; no per-case auto framing.
    half=math.ceil((global_extent+.06)*4)/4
    write(OUT/'protocol/display.json',dict(half_field_model=half,full_field_model=2*half,center_model=[0,0,0],
        shared_across_all_cases=True,before_transform='reference canonical transform only'))
    write(OUT/'results.json',dict(created_utc=now(),cases=results,refs=REFS,source_cases=12,training_export=False))
    fields=['case_id','arch','reference_case','scale_model_per_mm','mm_per_model_unit','relative_scale_to_reference','size_adjustment_percent',
        'before_rmse_model','after_rmse_model','after_rmse_case_mm','max_residual_case_mm','outside_domain_vertices','outside_target_tooth_vertices','outside_gum_vertices']
    with (OUT/'case_metrics.csv').open('x',encoding='utf-8-sig',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
    with (OUT/'per_tooth_residuals.csv').open('x',encoding='utf-8-sig',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=['case_id','arch','fdi','before_model','after_model','after_case_mm'],extrasaction='ignore');w.writeheader();w.writerows(residual_rows)

def picture(items,eye,up,half,title,points=None,size=760,domain=False):
    pl=pv.Plotter(off_screen=True,window_size=(size,size));pl.set_background('#edf1f5');pl.enable_depth_peeling(number_of_peels=10,occlusion_ratio=0)
    for v,f,color,opacity in items:
        pl.add_mesh(mesh(v,f),color=color,opacity=opacity,smooth_shading=False,ambient=.45,diffuse=.55,specular=0)
    if domain:pl.add_mesh(pv.Box(bounds=(-1,1,-1,1,-1,1)),style='wireframe',color='#cb5554',line_width=2,lighting=False)
    if points is not None:
        a,z,labs=points
        if len(a):
            line=pv.PolyData(np.concatenate([a,z]),lines=np.column_stack([np.full(len(a),2),np.arange(len(a)),np.arange(len(a))+len(a)]).ravel())
            pl.add_mesh(line,color='#e38b2d',line_width=3,lighting=False)
            pl.add_points(a,color='#078b98',point_size=7,render_points_as_spheres=True)
            pl.add_points(z,color='#8b4cac',point_size=7,render_points_as_spheres=True)
    pl.camera_position=[unit(np.array(eye))*8,[0,0,0],up];pl.enable_parallel_projection();pl.camera.parallel_scale=half;pl.camera.clipping_range=(.1,30)
    im=Image.fromarray(pl.screenshot(return_img=True));dr=ImageDraw.Draw(im)
    if points is not None and labs:
        coord=vtkCoordinate();coord.SetCoordinateSystemToWorld()
        for pt,label in zip(points[1],points[2]):
            coord.SetValue(*pt);x,y=coord.GetComputedDoubleDisplayValue(pl.renderer);x=int(x);y=size-int(y)
            dr.rounded_rectangle((x-16,y-11,x+16,y+11),radius=3,fill='white');dr.text((x,y),str(label),font=font(16),anchor='mm',fill='#4f2b69')
    pl.close();dr.rectangle((0,0,size,47),fill='#edf1f5');dr.text((15,10),title,font=font(24),fill='#18384a')
    dr.line((22,size-25,22+.5/(2*half)*size,size-25),fill='#18384a',width=3);dr.text((22,size-60),'0.5 model unit',font=font(20),fill='#18384a')
    return im

def render_cases():
    rows=read(OUT/'results.json')['cases'];half=read(OUT/'protocol/display.json')['half_field_model'];refs={a:read(OUT/'references'/f'{a}.json') for a in REFS}
    refmeshes={a:np.load(OUT/'cases'/cid/'aligned_geometry.npz') for a,cid in REFS.items()}
    for m in rows:
        cid=m['case_id'];arch=m['arch'];p=OUT/'cases'/cid;ref=refs[arch];rd=refmeshes[arch]
        d=np.load(p/'aligned_geometry.npz');src=np.load(SOURCE/'cases'/cid/'raw_index_preserved.npz')
        # Tooth surface only in overlay, full geometry remains in aligned cache.
        f=d['faces'][np.isin(d['face_label'],labels(arch))];rf=rd['faces'][np.isin(rd['face_label'],labels(arch))]
        vb=transform(src['vertices'],ref['transform']);va=d['vertices_model'];vr=rd['vertices_model'];s=1 if arch=='upper' else -1
        canvas=Image.new('RGB',(1520,1850),'#edf1f5');dr=ImageDraw.Draw(canvas)
        dr.text((25,15),cid+' · 固定参考整颌相似对齐',font=font(31),fill='#18384a')
        dr.text((25,62),f'紫：参考 {ref["case_id"]}；青：病例；橙线：同 FDI 质心残差',font=font(23),fill='#39576b')
        dr.text((25,102),'前图只施加所有病例共用的参考坐标变换；前后视角、中心、显示比例完全相同。',font=font(22),fill='#39576b')
        a=np.array([x['after_center_model'] for x in m['per_tooth']]);before=np.array([x['before_center_model'] for x in m['per_tooth']]);target=np.array([x['reference_center_model'] for x in m['per_tooth']])
        for k,(eye,up,name) in enumerate([([0,0,s],[0,1,0],'咬合'),([0,1,.15*s],[0,0,s],'前方')]):
            for j,(v,pts,stage) in enumerate([(vb,before,'对齐前'),(va,a,'对齐后')]):
                im=picture([(vr,rf,'#9d71b2',.52),(v,f,'#149ba8',.6)],eye,up,half,stage+' · '+name,
                    points=(pts,target,m['common_fdi'] if k==0 else []))
                canvas.paste(im,(j*760,150+k*760))
                if j==1 and k==0:im.save(p/'after_occlusal.png')
        dr.text((25,1695),f's={m["scale_model_per_mm"]:.7f} model/mm；相对参考调整 {m["size_adjustment_percent"]:+.2f}%；1 model={m["mm_per_model_unit"]:.3f} mm（本例）',font=font(23),fill='#18384a')
        dr.text((25,1740),f'质心 RMSE：{m["before_rmse_model"]:.4f} → {m["after_rmse_model"]:.4f} model；对齐后 {m["after_rmse_case_mm"]:.3f} 病例等效 mm',font=font(23),fill='#18384a')
        dr.text((25,1785),'残差包含真实牙形/排列差异；不是牙冠重建误差。仅显示目标牙；全部牙龈/智齿保存在原拓扑结果中。',font=font(21),fill='#9b4d39')
        assert not (p/'overlay_before_after.png').exists();canvas.save(p/'overlay_before_after.png')
        allf=d['faces'];out=d['outside_domain_vertex_mask'];ffout=out[allf].any(1)
        dom=picture([(va,allf[~ffout],'#beb9af',1),(va,allf[ffout],'#df554d',1)], [.6,.8,.7*s],[0,0,s],half,
            f'整网格域检查 · {cid}',domain=True)
        dom.save(p/'canonical_domain.png')
        print('render',cid,flush=True)
    contact=Image.new('RGB',(1800,2530),'#edf1f5');dr=ImageDraw.Draw(contact)
    dr.text((25,15),'12 例对齐后叠加 · 紫色为各颌固定参考',font=font(34),fill='#18384a')
    dr.text((25,65),'每例详图含对齐前后及两个视角；各牙位等权，整颌单一相似变换；未逐牙移动。',font=font(23),fill='#39576b')
    for i,m in enumerate(rows):
        x=i%3*600;y=120+i//3*600;im=Image.open(OUT/'cases'/m['case_id']/'after_occlusal.png').resize((550,550))
        contact.paste(im,(x+25,y+35));dr.text((x+20,y),m['case_id'],font=font(23),fill='#18384a')
    contact.save(OUT/'after_contact.png')

def charts():
    rows=read(OUT/'results.json')['cases'];im=Image.new('RGB',(1800,1100),'#edf1f5');dr=ImageDraw.Draw(im)
    dr.text((30,18),'尺度调整与对齐残差 · 两个参考病例的零残差不代表泛化性能',font=font(32),fill='#18384a')
    dr.text((30,68),'左：相对参考的尺寸调整 α=s_i/s_ref；右：同一参考坐标中质心 RMSE。灰为前，青为后。',font=font(23),fill='#39576b')
    maxpct=max(15,math.ceil(max(abs(r['size_adjustment_percent']) for r in rows)/5)*5);maxrm=max(r['before_rmse_model'] for r in rows)*1.1
    cx=600;span=250
    for tick in [-maxpct,0,maxpct]:
        x=cx+tick/maxpct*span;dr.line((x,150,x,1000),fill='#ccd5dc',width=1);dr.text((x,120),f'{tick:+.0f}%',font=font(20),anchor='mm',fill='#39576b')
    for i,m in enumerate(rows):
        y=170+i*66;dr.text((25,y+2),m['case_id']+(' [参考]' if m['is_reference'] else ''),font=font(21),fill='#18384a')
        end=cx+m['size_adjustment_percent']/maxpct*span;dr.rectangle((min(cx,end),y,max(cx,end)+1,y+22),fill='#168f9c')
        dr.text((875,y),f'{m["relative_scale_to_reference"]:.4f} ×',font=font(20),fill='#18384a')
        for j,(key,color) in enumerate([('before_rmse_model','#a6afb8'),('after_rmse_model','#168f9c')]):
            w=m[key]/maxrm*450;dr.rectangle((1060,y+j*19,1060+w,y+j*19+14),fill=color)
        dr.text((1530,y),f'{m["before_rmse_model"]:.4f} → {m["after_rmse_model"]:.4f}',font=font(19),fill='#18384a')
    dr.text((30,1035),'s_i 单位是 model/mm；α 是相对参考的尺寸调整倍率，不是各轴分别归一化。完整数值见 CSV。',font=font(23),fill='#39576b')
    im.save(OUT/'scale_and_rmse.png')
    heat=Image.new('RGB',(1800,1170),'#edf1f5');dr=ImageDraw.Draw(heat)
    dr.text((25,15),'逐牙对齐后质心残差 · 病例毫米等价值 r_model / s_i',font=font(33),fill='#18384a')
    vmax=max(1,math.ceil(max(x['after_case_mm'] for m in rows for x in m['per_tooth'])))
    dr.text((25,65),f'统一色阶 0–{vmax} mm，仅用于显示；— 表示缺少对应牙号；38/48 保留几何但不参与 14 牙位配准。',font=font(23),fill='#39576b')
    for group,arch in enumerate(['upper','lower']):
        y0=145+group*475;dr.text((25,y0),('上颌' if arch=='upper' else '下颌')+' / '+REFS[arch],font=font(25),fill='#18384a')
        ls=labels(arch)
        for k,label in enumerate(ls):dr.text((385+k*89,y0+46),str(label),font=font(22),anchor='mm',fill='#18384a')
        for j,m in enumerate(r for r in rows if r['arch']==arch):
            y=y0+78+j*54;dr.text((25,y+10),m['case_id'],font=font(21),fill='#18384a');values={x['fdi']:x['after_case_mm'] for x in m['per_tooth']}
            for k,label in enumerate(ls):
                x=345+k*89;val=values.get(label)
                if val is None:c='#d9dfe4';text='—'
                else:
                    u=min(val/vmax,1);c=tuple(np.round(np.array([234,244,236])*(1-u)+np.array([227,113,80])*u).astype(int));text=f'{val:.2f}'
                dr.rectangle((x,y,x+82,y+44),fill=c,outline='#b8c7cf');dr.text((x+41,y+22),text,font=font(20),anchor='mm',fill='#18384a')
    dr.text((25,1100),'残差来自病例与参考之间的形态/排列差异和质心对应；不自动据此剔除拥挤、缺牙或大尺寸病例。',font=font(23),fill='#39576b')
    heat.save(OUT/'per_tooth_residual_heatmap.png')

def aggregate():
    rows=read(OUT/'results.json')['cases'];half=read(OUT/'protocol/display.json')['half_field_model']
    for arch in REFS:
        ref=read(OUT/'references'/f'{arch}.json');s=1 if arch=='upper' else -1;items_before=[];items_after=[]
        selected=[r for r in rows if r['arch']==arch]
        for i,m in enumerate(selected):
            d=np.load(OUT/'cases'/m['case_id']/'aligned_geometry.npz');src=np.load(SOURCE/'cases'/m['case_id']/'raw_index_preserved.npz');f=d['faces'][np.isin(d['face_label'],labels(arch))]
            items_before.append((transform(src['vertices'],ref['transform']),f,COLORS[i],.32));items_after.append((d['vertices_model'],f,COLORS[i],.32))
        im=Image.new('RGB',(1520,1820),'#edf1f5');dr=ImageDraw.Draw(im)
        dr.text((25,15),('上颌' if arch=='upper' else '下颌')+' 6 例整体叠加 · 同一视野',font=font(32),fill='#18384a')
        for k,(eye,up,title) in enumerate([([0,0,s],[0,1,0],'咬合'),([0,1,.1*s],[0,0,s],'前方')]):
            for j,(items,stage) in enumerate([(items_before,'对齐前'),(items_after,'对齐后')]):im.paste(picture(items,eye,up,half,stage+' · '+title),(760*j,90+760*k))
        for i,m in enumerate(selected):
            x=25+(i%3)*500;y=1650+(i//3)*60;dr.rectangle((x,y,x+22,y+22),fill=COLORS[i]);dr.text((x+34,y-3),m['case_id'],font=font(23),fill='#18384a')
        dr.text((25,1780),'颜色对应病例；重叠用于看整体离散度，局部细节请查看每例与固定参考的双病例图。',font=font(21),fill='#39576b')
        im.save(OUT/f'{arch}_all_before_after.png')

def finish():
    rows=read(OUT/'results.json')['cases'];cohort=read(OUT/'protocol/cohort.json')['cases']
    # Real operator checks plus full source/upstream artifact integrity checks.
    for source,manifest in [(SOURCE,'delivery_manifest.json'),(SPACE,'verified_delivery_manifest.json')]:
        assert all(sha(source/x['path'])==x['sha256'] for x in read(source/manifest))
    for m,row in zip(rows,cohort):
        assert m['case_id']==row['case_id']
        assert sha(row['source_obj'])==m['source_obj_sha256'] and sha(row['source_annotation'])==m['source_annotation_sha256']
        a=np.load(OUT/'cases'/m['case_id']/'aligned_geometry.npz');o=np.load(SOURCE/'cases'/m['case_id']/'raw_index_preserved.npz')
        assert np.array_equal(a['faces'],o['faces']) and np.array_equal(a['vertex_labels'],o['vertex_labels']) and np.array_equal(a['vertex_instances'],o['vertex_instances'])
        assert np.allclose(np.array(m['T_raw_mm_from_model'])@np.array(m['T_model_from_raw_mm']),np.eye(4),atol=1e-10)
    # Independent centroid solver equivalence in inverse direction, no extra fit variant.
    inversechecks=[]
    for m in rows:
        c=centroids(np.load(SOURCE/'cases'/m['case_id']/'raw_index_preserved.npz'),m['arch']);x=np.array([c[i] for i in m['common_fdi']]);y=np.array([z['reference_center_model'] for z in m['per_tooth']])
        # Do not refit reverse similarity (which has different LS scale). Apply the
        # stored inverse to the actual transformed centroids, not to reference shape.
        actual=transform(x,m['transform']);H=np.array(m['T_raw_mm_from_model']);back=actual@H[:3,:3].T+H[:3,3]
        inversechecks.append(float(np.linalg.norm(back-x,axis=1).max()))
    summary=dict(created_utc=now(),case_count=12,refs=REFS,all_proper_positive_similarity=True,
        max_inverse_error_mm=max(m['inverse_roundtrip_max_mm'] for m in rows),min_normal_dot=min(m['normal_vs_recomputed_min_dot'] for m in rows),
        source_files_unchanged=24,earlier_two_packages_unchanged=True,aligned_cases=12,
        nonreference_rmse_case_mm_range=[min(m['after_rmse_case_mm'] for m in rows if not m['is_reference']),max(m['after_rmse_case_mm'] for m in rows if not m['is_reference'])],
        relative_size_adjustment_percent_range=[min(m['size_adjustment_percent'] for m in rows),max(m['size_adjustment_percent'] for m in rows)],
        outside_domain_cases=[dict(case_id=m['case_id'],all_vertices=m['outside_domain_vertices'],target_teeth=m['outside_target_tooth_vertices'],gum=m['outside_gum_vertices']) for m in rows if m['outside_domain_vertices']],
        previous_space_candidates_transformed=sum(m['transformed_previous_space_candidates'] for m in rows),
        new_surface_samples_generated=False,new_canonical_cube_space_samples_generated=False,training_export=False,human_visual_review='PENDING')
    write(OUT/'qa_summary.json',summary)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>12例固定参考尺度与对齐审核</title><style>body{font:17px/1.7 system-ui;max-width:1600px;margin:32px auto;background:#edf1f5;color:#18384a}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:8px;border:1px solid #bbc9d2;text-align:left}a{color:#176a99}.notice{padding:18px;background:#fff1d8}summary{cursor:pointer;font-weight:bold}</style>',
        '<h1>12 例固定参考：统一尺度与整颌对齐</h1><p class="notice">已实际计算并保存 12 例相似变换、变换后全网格和法线。上颌参考 01328DDN_upper，下颌参考 01JNZKAX_lower。复用旧版固定参考算法；旧参考 6XSU5W4B_upper 当前属于 test，因此不用于这轮开发。下颌参考存在牙龈基底非流形，仅作为本轮质心配准的固定诊断参考；未获得人工或训练验收。上下颌独立坐标，不代表恢复咬合关系。</p>',
        '<p>每牙用原始 strict 三角面的面积质心；至少 6 个共同牙位且覆盖双侧/前后；按牙位等权求 x_model=sᵢ Rᵢ x_mm+tᵢ，sᵢ&gt;0，det(R)=+1。参考最长包围盒边为 1.90 model；+X 患者左，+Y 前，+Z=X×Y，下颌牙冠朝 -Z。没有逐牙移动、非刚性变形、裁切、二次包围盒归一化或镜像。</p>',
        '<p><b>如何看前后图：</b>前图给所有原始病例施加同一个参考坐标变换，没有逐例预对齐；后图使用各自拟合的整颌相似变换。全部图共用相机中心和视野。紫为参考、青为病例、橙线为对应牙位中心残差。叠加图隐藏牙龈以方便检查牙冠；域检查和保存结果保留全部原始面。</p>',
        '<p><b>单位：</b>sᵢ 为 model/mm；α=sᵢ/s_ref 表示相对参考尺寸调整。1 model=1/sᵢ mm 随病例变化，不能宣称全部都是固定 50 mm。逐牙毫米残差用 r_model/sᵢ，是病例毫米等价值，不是配准真值误差或牙冠损伤。</p>',
        '<nav><a href="scale_and_rmse.png">尺度与残差总图</a> · <a href="per_tooth_residual_heatmap.png">逐牙残差</a> · <a href="case_metrics.csv">病例 CSV</a> · <a href="per_tooth_residuals.csv">逐牙 CSV</a> · <a href="qa_summary.json">检查摘要</a> · <a href="protocol/frozen_diagnostic_choices.json">参考选择和方法</a></nav>',
        '<h2>尺度变化和残差</h2><img src="scale_and_rmse.png"><h2>逐牙残差</h2><img src="per_tooth_residual_heatmap.png">',
        '<h2>每例数值与域占用</h2><table><tr><th>病例</th><th>s model/mm</th><th>相对尺寸调整</th><th>RMSE 前→后 model</th><th>后 RMSE / 最大 mm</th><th>越界顶点 全部/目标牙/牙龈</th></tr>']
    for m in rows:
        cid=m['case_id'];parts.append(f'<tr><td><a href="#c-{cid}">{cid}</a></td><td>{m["scale_model_per_mm"]:.7f}</td><td>{m["size_adjustment_percent"]:+.2f}%</td><td>{m["before_rmse_model"]:.4f} → {m["after_rmse_model"]:.4f}</td><td>{m["after_rmse_case_mm"]:.3f} / {m["max_residual_case_mm"]:.3f}</td><td>{m["outside_domain_vertices"]} / {m["outside_target_tooth_vertices"]} / {m["outside_gum_vertices"]}</td></tr>')
    parts.append('</table><p class="notice">越界没有被裁切。高残差可能来自真实排列、缺牙、扫描覆盖或标注问题，不能自动判为坏数据。封底失败、智齿、历史法线警报等上游状态全部保留；对齐成功不解除这些限制。</p>')
    for arch in REFS:parts.append(f'<h2>{"上颌" if arch=="upper" else "下颌"} 6 例前后叠加</h2><a href="{arch}_all_before_after.png"><img src="{arch}_all_before_after.png"></a>')
    for m in rows:
        cid=m['case_id'];parts.append(f'<h2 id="c-{cid}">{cid}</h2><p><a href="cases/{cid}/transform.json">正逆变换</a> · <a href="cases/{cid}/metrics.json">逐牙指标及上游警报</a></p><a href="cases/{cid}/overlay_before_after.png"><img loading="lazy" src="cases/{cid}/overlay_before_after.png"></a><details><summary>全部几何域检查（红色面含越界顶点，红框为 [-1,1]³）</summary><img loading="lazy" src="cases/{cid}/canonical_domain.png"></details>')
    parts.append('<h2>已有空间点如何处理</h2><p>上一轮 7 例、350 万空间候选点及对应封底代理已施加完全相同的变换；内外与排除带标签保持不变，00OMSZGW_lower 的分类失败状态仍然有效。这些点原先在各例毫米包围盒内产生，变换后不是统一 [-1,1]³ 的新均匀样本。本轮没有重新空间采样，也没有执行正式表面采样或训练导出。</p><p>所有原始 OBJ/JSON 与前两轮审核包哈希核对一致。保存了全部原始面索引、牙号、实例、逆矩阵；单位法线与变换后面叉积重新核对。人工视觉验收状态：PENDING。</p></html>')
    (OUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    import re
    refs=re.findall(r'(?:href|src)="([^"]+)"','\n'.join(parts));assert all(x.startswith('#') or (OUT/x).is_file() for x in refs)
    for p in OUT.rglob('*.png'):
        with Image.open(p) as im:im.verify()
    write(OUT/'review_records.json',[dict(case_id=m['case_id'],stage='scale_and_fixed_reference_alignment',decision='PENDING',reviewer=None,
        image=f'cases/{m["case_id"]}/overlay_before_after.png',notes='') for m in rows])
    shutil.copyfile(__file__,OUT/'protocol/implementation_final.py')
    write(OUT/'delivery_manifest.json',[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()])
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    {'initialize':initialize,'align':align,'render':render_cases,'charts':charts,'aggregate':aggregate,'finish':finish}[sys.argv[1]]()
