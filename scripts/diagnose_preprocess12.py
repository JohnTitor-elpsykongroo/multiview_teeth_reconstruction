"""Bounded, read-only Teeth3DS audit. No repair, fitting, sampling or dataset export.

Run select once, analyze once, render each stage once, then finish. Every output
directory is new; source OBJ vertex and triangle order are retained exactly.
"""
from pathlib import Path
import argparse, collections, datetime, hashlib, html, json, shutil, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('D:/WorkSpace/Dental/data')
NEW = DATA/'Teeth3DS_DualArch_v1'
OLD = DATA/'Teeth3DS_DMM/05_dataset_profiles/upper_v1.3_strict_fixed_offsurface'
FONT = 'C:/Windows/Fonts/msyh.ttc'
STAGES = ['geometry', 'semantics', 'normals']
TITLES = ['01 原始几何', '02 原始牙号与边界', '03 原始面绕序法线']
SIZE = 800
FIELD = 110.0

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, x):
    p = Path(p)
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def font(n): return ImageFont.truetype(FONT,n)
def unit(v): return v/max(float(np.linalg.norm(v)),1e-12)

def select(out):
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol').mkdir(); (out/'cases').mkdir()
    splits=read(NEW/'reports/patient_splits.json')
    oldtest={x.split('_')[0] for x in read(OLD/'splits/test_split.json')}
    oldtrain={x.split('.')[0] for x in read(OLD/'splits/train_split.json') if '__mirror' not in x}
    rows=[read(p) for p in sorted((NEW/'work/cases').glob('*/prepared.json'))]
    # Failed historical preparations can lack patient_id; they cannot establish
    # patient-level train eligibility and are excluded from this selection pool.
    byid={r['case_id']:r for r in rows}
    eligible=[r for r in rows if splits.get(r.get('patient_id'))=='train' and r.get('patient_id') not in oldtest]
    allhash=collections.defaultdict(list)
    for r in rows: allhash[r.get('source_sha256','')].append(r['case_id'])
    protected={r['case_id'] for r in rows if splits.get(r.get('patient_id'))!='train' or r.get('patient_id') in oldtest}
    eligible=[r for r in eligible if not (set(allhash[r.get('source_sha256','')]) & protected)]
    elig={r['case_id'] for r in eligible}; chosen=[]; used=set()
    def add(cid, why):
        assert cid in elig and byid[cid]['patient_id'] not in used, cid
        r=byid[cid]; used.add(r['patient_id'])
        bundle=read(ROOT/f'.runtime/dmm_r4_best_val_20261015/package/model_bundle/{r["arch"]}/model.json')
        mids=bundle.get('training_case_ids',[])
        chosen.append(dict(case_id=cid,patient_id=r['patient_id'],arch=r['arch'],selection_reason=why,
            source_obj=r['obj'],source_annotation=r['annotation'],assigned_split=splits[r['patient_id']],
            old_profile_train=cid in oldtrain,old_profile_test_patient=False,
            r4_training_id_present=cid in mids,previous_status=r['status'],previous_reason=r.get('reason'),
            previous_metadata=str(NEW/'work/cases'/cid/'prepared.json'),
            previous_source_sha256=r.get('source_sha256'),previous_annotation_sha256=r.get('annotation_sha256'),
            metadata_exact_geometry_matches=allhash.get(r.get('source_sha256',''),[])))
    add('01328DDN_upper','新旧训练交集；历史完整 14 牙对照')
    add('01346914_upper','新旧训练交集；历史缺牙/覆盖不足对照')
    add('016N696F_upper','历史法线拒绝；保留异常原件')
    add('017UXKRN_upper','新流程历史 INWARD_NORMALS_27；复核判据')
    def width(r):
        c=r.get('centers_raw',{}); a,b=('17','27') if r['arch']=='upper' else ('37','47')
        return float(np.linalg.norm(np.array(c[a])-c[b])) if a in c and b in c else None
    for q,name in [(0.1,'较窄'),(.9,'较宽')]:
        pool=[r for r in eligible if r['arch']=='upper' and r['status']=='PREPARED' and r['case_id'] in oldtrain and r['patient_id'] not in used and sum(r.get('presence',{}).values())==14 and width(r) is not None]
        pool.sort(key=lambda r:(width(r),r['case_id']))
        r=pool[round(q*(len(pool)-1))]
        add(r['case_id'],f'新旧训练交集；历史 17–27 质心距离 {q:.0%} 分位附近（{name}）')
        chosen[-1]['selection_width_mm']=width(r); chosen[-1]['selection_pool_n']=len(pool)
    add('00OMSZGW_lower','历史完整 14 牙下颌对照')
    add('0132CR0A_lower','历史缺牙/覆盖不足下颌对照')
    add('013TGCFK_lower','新流程历史 INWARD_NORMALS_37；复核判据')
    add('013NUWYR_lower','含智齿的范围/牙号审核；不自动删除')
    pool=[r for r in eligible if r['arch']=='lower' and r['status']=='PREPARED' and r['patient_id'] not in used and r.get('original_topology',{}).get('nonmanifold_edges',0)>0]
    pool.sort(key=lambda r:r['case_id'])
    assert pool, 'No topology challenge candidate'
    add(pool[0]['case_id'],'历史原网格非流形、后续曾清理；本次保留清理前原件')
    pool=[r for r in eligible if r['arch']=='lower' and r['status']=='PREPARED' and r['patient_id'] not in used and sum(r.get('presence',{}).values())==14 and width(r) is not None]
    pool.sort(key=lambda r:(width(r),r['case_id']))
    r=pool[round(.9*(len(pool)-1))]
    add(r['case_id'],'历史 37–47 质心距离 90% 分位附近；较宽下颌')
    chosen[-1]['selection_width_mm']=width(r); chosen[-1]['selection_pool_n']=len(pool)
    assert len(chosen)==12 and len(used)==12
    write(out/'protocol/cohort.json',dict(created_utc=now(),scope='12 scans from 12 patients; 6 upper and 6 lower; no paired-arch claim',
        sampling='purposeful diagnostic challenge cohort, not prevalence estimate',
        selection_frozen_before_raw_visual_review=True,patient_grouping=True,
        protected='exclude current val/test patients, old upper test patients, known exact raw geometry duplicates crossing these groups',
        duplicate_limit='exact hashes from prior metadata only; selected files rehashed; near-duplicate anatomy not established',
        metadata_files_scanned=len(rows),raw_scans_to_process=12,cases=chosen))
    paths=[NEW/'reports/patient_splits.json',OLD/'splits/test_split.json',OLD/'splits/train_split.json',NEW/'reports/build_config.json',ROOT/'docs/dmm_preprocessing_replan_20261016.md']
    write(out/'protocol/evidence.json',[dict(path=str(p),sha256=sha(p)) for p in paths])
    shutil.copyfile(__file__,out/'protocol/diagnose_preprocess12.py')
    write(out/'protocol/settings.json',dict(unit='mm',unit_evidence=read(NEW/'reports/build_config.json')['unit_evidence'],
        shared_orthographic_width_mm=FIELD,pixels=SIZE,geometry_modified=False,repairs=False,
        point_sampling=False,alignment=False,mirroring=False,training=False,
        camera_note='FDI-derived display camera only. No preprocessing registration. Raw coordinates unchanged.',
        normal_note='cross(v1-v0,v2-v0), no flip; view-facing colors are not outwardness truth; radial score is heuristic',
        human_visual_review='PENDING'))
    print(json.dumps(chosen,ensure_ascii=False,indent=2),flush=True)

def obj_read(path):
    v=[]; f=[]; vn=0
    with Path(path).open(encoding='utf-8') as inp:
        for line in inp:
            t=line.partition('#')[0].split()
            if not t: continue
            if t[0]=='v': v.append([float(x) for x in t[1:4]])
            elif t[0]=='vn': vn+=1
            elif t[0]=='f':
                if len(t)!=4: raise ValueError('Nontriangle face: do not silently triangulate')
                a=[int(x.split('/')[0]) for x in t[1:]]
                if 0 in a: raise ValueError('OBJ index 0')
                f.append([x-1 if x>0 else len(v)+x for x in a])
    v=np.asarray(v,np.float64); f=np.asarray(f,np.int64)
    assert len(v) and len(f) and np.isfinite(v).all() and f.min()>=0 and f.max()<len(v)
    return v,f,vn

def analyze(out):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    cohort=read(out/'protocol/cohort.json')
    for row in cohort['cases']:
        cid=row['case_id']; p=out/'cases'/cid; p.mkdir(exist_ok=False)
        before={k:sha(row[k]) for k in ['source_obj','source_annotation']}
        v,f,vnc=obj_read(row['source_obj']); j=read(row['source_annotation'])
        lab=np.asarray(j['labels'],np.int32); inst=np.asarray(j['instances'],np.int32)
        assert len(lab)==len(inst)==len(v),cid
        assert j['id_patient']==row['patient_id'] and j['jaw']==row['arch'],cid
        tri=v[f]; cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); twice=np.linalg.norm(cross,axis=1)
        n=cross/np.maximum(twice[:,None],1e-30); area=twice/2; fc=tri.mean(1)
        fl=lab[f]; strict=(fl==fl[:,0:1]).all(1)
        majority=(fl[:,0]==fl[:,1])|(fl[:,0]==fl[:,2])|(fl[:,1]==fl[:,2])
        face_label=np.where(fl[:,0]==fl[:,1],fl[:,0],np.where(fl[:,0]==fl[:,2],fl[:,0],np.where(fl[:,1]==fl[:,2],fl[:,1],-1)))
        edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
        se=np.sort(edges,axis=1); ue,inv,cnt=np.unique(se,axis=0,return_inverse=True,return_counts=True)
        orientation=np.where(edges[:,0]<edges[:,1],1,-1)
        osum=np.bincount(inv,weights=orientation,minlength=len(ue))
        be=ue[cnt==1]; ne=ue[cnt>2]; ie=ue[(cnt==2)&(osum!=0)]
        _,dupcount=np.unique(np.sort(f,axis=1),axis=0,return_counts=True)
        valid=set([0]+([10*q+i for q in [1,2] for i in range(1,9)] if row['arch']=='upper' else [10*q+i for q in [3,4] for i in range(1,9)]))
        invalid=sorted(set(lab.tolist())-valid)
        qleft=2 if row['arch']=='upper' else 3; qright=1 if row['arch']=='upper' else 4
        stats={}; centers={}
        for label in sorted(set(lab.tolist())):
            mask=(face_label==label)&strict&(area>1e-12)
            if not mask.any(): continue
            a=area[mask]; cen=np.average(fc[mask],axis=0,weights=a); centers[str(label)]=cen.tolist()
            radial=fc[mask]-cen; radial/=np.maximum(np.linalg.norm(radial,axis=1,keepdims=True),1e-30)
            dot=(radial*n[mask]).sum(1)
            verts=np.flatnonzero(lab==label); es=ue[(lab[ue[:,0]]==label)&(lab[ue[:,1]]==label)]
            remap=np.full(len(v),-1,np.int32); remap[verts]=np.arange(len(verts))
            graph=coo_matrix((np.ones(len(es)),(remap[es[:,0]],remap[es[:,1]])),shape=(len(verts),len(verts))).tocsr()
            nc,cl=connected_components(graph,directed=False); sizes=np.sort(np.bincount(cl))[::-1]
            iid,ic=np.unique(inst[verts],return_counts=True)
            stats[str(label)]=dict(vertices=len(verts),strict_faces=int(mask.sum()),strict_area_mm2=float(a.sum()),
                center_raw_mm=cen.tolist(),radial_dot_median_unweighted=float(np.median(dot)),
                radial_negative_area_fraction=float(a[dot<0].sum()/a.sum()),
                labeled_vertex_components=int(nc),largest_component_vertex_fraction=float(sizes[0]/len(verts)),
                component_vertex_sizes=sizes[:12].tolist(),instances=dict(zip(map(str,iid),map(int,ic))))
        def avg(ids): return np.mean([centers[str(i)] for i in ids if str(i) in centers],axis=0)
        left=[qleft*10+i for i in range(1,8)]; right=[qright*10+i for i in range(1,8)]
        x=unit(avg(left)-avg(right)); y0=avg([qleft*10+i for i in [1,2]]+[qright*10+i for i in [1,2]])-avg([qleft*10+i for i in [6,7]]+[qright*10+i for i in [6,7]])
        y=unit(y0-x*np.dot(x,y0)); z=unit(np.cross(x,y)); basis=np.stack([x,y,z],axis=1)
        assert np.isfinite(basis).all() and np.linalg.det(basis)>.999
        projected=v@basis; center=((projected.max(0)+projected.min(0))/2)@basis.T
        prior=read(row['previous_metadata']); instance_labels={}
        for ii in np.unique(inst): instance_labels[str(ii)]=np.unique(lab[inst==ii]).tolist()
        metric=dict(case_id=cid,arch=row['arch'],selection_reason=row['selection_reason'],
            source_hashes_before=before,source_hashes_match_prior=dict(source_obj=before['source_obj']==row['previous_source_sha256'],source_annotation=before['source_annotation']==row['previous_annotation_sha256']),
            vertices=len(v),faces=len(f),obj_vn_records=vnc,computed_normal_source='original triangle winding, not OBJ vn',
            bounds_raw_mm=[v.min(0).tolist(),v.max(0).tolist()],raw_axis_spans_mm=np.ptp(v,axis=0).tolist(),
            display_basis_spans_mm=np.ptp(projected,axis=0).tolist(),surface_area_mm2=float(area.sum()),
            zero_area_faces=int((area<=1e-12).sum()),duplicate_face_excess=int(np.maximum(dupcount-1,0).sum()),
            topology=dict(boundary_edges=len(be),nonmanifold_edges=len(ne),inconsistent_edges=len(ie)),
            vertex_labels=sorted(set(lab.tolist())),invalid_fdi=invalid,
            observed_target_teeth=sum(i in valid and i!=0 and i%10!=8 for i in set(lab.tolist())),
            missing_target_fdi=sorted(valid-set(lab.tolist())-{0,qleft*10+8,qright*10+8}),
            wisdom_fdi=sorted(i for i in set(lab.tolist()) if i%10==8 and i in valid),
            strict_faces=int(strict.sum()),majority_boundary_faces=int((majority&~strict).sum()),ambiguous_faces=int((~majority).sum()),
            boundary_area_fraction=float(area[~strict].sum()/area.sum()),ambiguous_area_fraction=float(area[~majority].sum()/area.sum()),
            gum_instance_zero_disagreement_vertices=int(((lab==0)!=(inst==0)).sum()),instance_to_labels=instance_labels,
            tooth_statistics=stats,previous_status=row['previous_status'],previous_reason=row['previous_reason'],
            previous_original_topology=prior.get('original_topology'),previous_after_topology=prior.get('topology'),
            previous_local_surface_filter=prior.get('local_surface_filter'),
            display_center_raw_mm=center.tolist(),display_basis_columns=basis.tolist(),
            geometry_changed=False,automatic_status='DIAGNOSED_REVIEW_PENDING',human_visual_review='PENDING')
        np.savez_compressed(p/'raw_index_preserved.npz',vertices=v,faces=f,vertex_labels=lab,vertex_instances=inst,
            face_normals=n,face_area=area,face_centers=fc,face_label=face_label,strict=strict,majority=majority,
            boundary_edges=be,nonmanifold_edges=ne,inconsistent_edges=ie,basis=basis,center=center)
        write(p/'metrics.json',metric)
        print(f'{cid}: {len(v)} vertices, {len(f)} faces; topology={metric["topology"]}; labels={metric["vertex_labels"]}',flush=True)

PALETTE=np.array([[225,103,82],[239,165,66],[222,196,73],[111,178,91],[55,174,153],[58,149,203],[111,120,205],[181,111,186],
                  [181,64,74],[190,111,36],[161,155,54],[61,132,79],[40,126,128],[51,99,162],[100,75,162],[147,76,147]],np.uint8)
def label_color(label,arch):
    if label==0:return [190,180,177]
    qs=[1,2] if arch=='upper' else [3,4]
    if label//10 not in qs or label%10 not in range(1,9):return [240,25,210]
    return PALETTE[qs.index(label//10)*8+label%10-1]

def render_one(row,p,stage):
    import pyvista as pv
    from vtkmodules.vtkRenderingCore import vtkCoordinate
    d=np.load(p/'raw_index_preserved.npz'); m=read(p/'metrics.json')
    v=d['vertices']; f=d['faces']; n=d['face_normals']; cen=d['center']; basis=d['basis']
    facearr=np.column_stack([np.full(len(f),3),f]).ravel()
    mesh=pv.PolyData(v,facearr); s=1 if row['arch']=='upper' else -1
    views=[('咬合面',np.array([0,0,s]),np.array([0,1,0])),('基底面',np.array([0,0,-s]),np.array([0,1,0])),
           ('前方',np.array([0,1,.15*s]),np.array([0,0,s])),('斜侧',np.array([.7,.8,.8*s]),np.array([0,0,s]))]
    canvas=Image.new('RGB',(SIZE*2,1900),'#edf1f5'); draw=ImageDraw.Draw(canvas)
    draw.text((28,16),f'{TITLES[STAGES.index(stage)]} | {row["case_id"]}',font=font(32),fill='#142c40')
    draw.text((28,65),'原始毫米坐标；统一 110 mm 视野；无清理/补洞/缩放；仅调整显示相机',font=font(23),fill='#39576b')
    draw.text((28,102),row['selection_reason'],font=font(21),fill='#39576b')
    captions=[]
    for vi,(name,eye,up) in enumerate(views):
        e=unit(basis@unit(eye)); u=unit(basis@up)
        pl=pv.Plotter(off_screen=True,window_size=(SIZE,SIZE));pl.set_background('#edf1f5');pl.disable_anti_aliasing()
        kw=dict(smooth_shading=False,show_scalar_bar=False,show_edges=False,lighting=True,ambient=.4,diffuse=.6,specular=0)
        if stage=='geometry':
            pl.add_mesh(mesh,color='#d9c9ad',**kw)
            for key,color in [('boundary_edges','#1b2430'),('nonmanifold_edges','#f49e13'),('inconsistent_edges','#e32192')]:
                ed=d[key]
                if len(ed):pl.add_mesh(pv.PolyData(v,lines=np.column_stack([np.full(len(ed),2),ed]).ravel()),color=color,line_width=3,lighting=False)
        elif stage=='semantics':
            rgb=np.array([label_color(int(t),row['arch']) for t in d['face_label']],np.uint8)
            rgb[d['majority']&~d['strict']]=[245,245,245];rgb[~d['majority']]=[235,20,205]
            pl.add_mesh(mesh,scalars=rgb,rgb=True,preference='cell',**kw)
        else:
            dots=n@e;rgb=np.where((dots>=0)[:,None],[41,174,172],[224,61,125]).astype(np.uint8)
            rgb[d['face_area']<=1e-12]=[245,180,25]
            pl.add_mesh(mesh,scalars=rgb,rgb=True,preference='cell',lighting=True,ambient=.35,diffuse=.65,specular=0,show_scalar_bar=False)
            rng=np.random.default_rng(20261016); ar=d['face_area'];ids=rng.choice(len(f),size=min(1100,len(f)),replace=False,p=ar/ar.sum())
            arrows=pv.PolyData(d['face_centers'][ids]);arrows['direction']=n[ids]
            pl.add_mesh(arrows.glyph(orient='direction',scale=False,factor=1.5),color='#203443',lighting=False)
        pl.camera_position=[cen+e*250,cen,u];pl.enable_parallel_projection();pl.camera.parallel_scale=FIELD/2
        pl.camera.clipping_range=(1,500)
        arr=pl.screenshot(return_img=True); im=Image.fromarray(arr).convert('RGB'); dr=ImageDraw.Draw(im)
        dr.text((20,12),name,font=font(26),fill='#142c40')
        # Project tooth centroids with the same VTK camera. Labels are identifiers,
        # not visible-surface landmarks; occluded labels omitted outside occlusal views.
        if stage=='semantics' and vi==0:
            coord=vtkCoordinate();coord.SetCoordinateSystemToWorld()
            for label,st in m['tooth_statistics'].items():
                if label=='0':continue
                coord.SetValue(*st['center_raw_mm']); px,py=coord.GetComputedDoubleDisplayValue(pl.renderer)
                px=int(px);py=SIZE-int(py)
                dr.rounded_rectangle((px-18,py-14,px+18,py+14),radius=4,fill='white',outline='#182c3d')
                dr.text((px,py),label,font=font(20),anchor='mm',fill='#132c3f')
        bar=round(10/FIELD*SIZE);dr.line((24,SIZE-26,24+bar,SIZE-26),fill='#152b3d',width=4)
        dr.text((24,SIZE-59),'10 mm',font=font(20),fill='#152b3d')
        pl.close();canvas.paste(im,((vi%2)*SIZE,145+(vi//2)*SIZE));captions.append(im)
    foot={'geometry':'黑：开边界（扫描裁切也会产生）  橙：非流形边  品红：相邻面绕序不一致',
          'semantics':'颜色：FDI；灰褐：牙龈；白：2/3 多数票边界面；品红：三顶点全异/非法号',
          'normals':'青：朝当前观察者；红：背向观察者；箭头：原面法线 1.5 mm（仅显示）'}[stage]
    draw.text((28,1760),foot,font=font(22),fill='#203f55')
    draw.text((28,1800),f'V={m["vertices"]:,} F={m["faces"]:,} | 边界/非流形/绕序冲突={list(m["topology"].values())}',font=font(22),fill='#203f55')
    draw.text((28,1840),'图像已生成 ≠ 人工验收通过。法线红色 ≠ 自动证明朝内；开放牙冠的径向判据仅为线索。',font=font(21),fill='#9c3d3b')
    assert not (p/f'{stage}.png').exists();canvas.save(p/f'{stage}.png')
    captions[0].save(p/f'{stage}_occlusal.png')

def render(out,stage):
    rows=read(out/'protocol/cohort.json')['cases']
    for row in rows:
        render_one(row,out/'cases'/row['case_id'],stage)
        print(f'{stage}: {row["case_id"]}',flush=True)
    contact(out,stage)

def contact(out,stage):
    rows=read(out/'protocol/cohort.json')['cases']
    contact=Image.new('RGB',(1800,2520),'#edf1f5');dr=ImageDraw.Draw(contact)
    dr.text((25,12),TITLES[STAGES.index(stage)]+' · 12 例总览（6 上颌 / 6 下颌）',font=font(34),fill='#142c40')
    dr.text((25,58),'同一毫米视野；相机按牙号定位，方向待审核；详图含四视角。人工验收 PENDING。',font=font(22),fill='#39576b')
    for i,r in enumerate(rows):
        x=(i%3)*600;y=110+(i//3)*600
        im=Image.open(out/'cases'/r['case_id']/f'{stage}_occlusal.png').resize((550,550))
        contact.paste(im,(x+25,y+35));dr.text((x+20,y),r['case_id'],font=font(24),fill='#142c40')
    assert not (out/f'{stage}_contact.png').exists();contact.save(out/f'{stage}_contact.png')

def finish(out):
    shutil.copyfile(__file__,out/'protocol/diagnose_preprocess12_final.py')
    rows=read(out/'protocol/cohort.json')['cases']; metrics=[];unchanged=[]
    (out/'protocol/historical_case_metadata').mkdir(exist_ok=False)
    evidence=[]
    for r in rows:
        m=read(out/'cases'/r['case_id']/'metrics.json');metrics.append(m)
        src=Path(r['previous_metadata'])
        shutil.copyfile(src,out/'protocol/historical_case_metadata'/f'{r["case_id"]}.json')
        evidence.append(dict(path=str(src),sha256=sha(src)))
        after={k:sha(r[k]) for k in ['source_obj','source_annotation']}
        unchanged.append(dict(case_id=r['case_id'],hashes_after=after,unchanged=after==m['source_hashes_before']))
        for s in STAGES:
            with Image.open(out/'cases'/r['case_id']/f'{s}.png') as im: im.verify()
    assert all(x['unchanged'] for x in unchanged)
    for arch in ['upper','lower']:
        bp=ROOT/f'.runtime/dmm_r4_best_val_20261015/package/model_bundle/{arch}/model.json'
        evidence.append(dict(path=str(bp),sha256=sha(bp)))
    write(out/'protocol/supplementary_evidence.json',evidence)
    write(out/'protocol/source_integrity_after.json',unchanged)
    write(out/'summary.json',dict(finished_utc=now(),case_count=12,patient_count=12,source_unchanged=True,
        human_visual_review='PENDING',training_accepted=False,scope='raw geometry, FDI, winding normals only',cases=metrics))
    reviews=[dict(case_id=r['case_id'],stage=s,reviewer=None,reviewed_utc=None,decision='PENDING',notes='',image=f'cases/{r["case_id"]}/{s}.png') for r in rows for s in STAGES]
    write(out/'review_records.json',reviews)
    lines=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>DMM 12 例原始数据诊断审核</title><style>body{font:16px/1.7 system-ui;margin:36px auto;max-width:1500px;background:#f4f6f8;color:#203748}h1,h2{color:#18384a}a{color:#116a9a}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border:1px solid #ccd6dd;text-align:left}img{max-width:100%;height:auto}summary{cursor:pointer;font-weight:bold}.warn{padding:16px;background:#fff1d8}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}</style>',
        '<h1>DMM：12 例原始数据诊断审核包</h1><p>范围：12 位患者的 12 份原始扫描，6 上颌、6 下颌。开发诊断样本；不估计全量异常率。避开当前 val/test 和旧上颌 test 患者，以及历史哈希可识别的跨组完全重复原件。</p>',
        '<p class="warn">人工验收：PENDING。未清理、补洞、翻法线、重新编号、配准、采样、镜像或训练。只在本目录生成缓存和审核图。24 个原始 OBJ/JSON 的运行前后 SHA256 一致。相机方向由 FDI 推导，仅服务显示，不能用来独立证明 FDI 左右正确；需原始解剖/源标注交叉审核。</p>',
        '<p>单位沿用已有 build_config.json 的用户确认 mm；没有独立标尺重新测量。所有全颌图共享 110 mm 正交视野。法线由原三角面绕序计算，无翻转。青/红表示朝向/背向相机，不是朝外/朝内真值。径向统计仅用于查找可疑区域。</p>',
        '<nav><a href="#geometry">① 原始几何</a> · <a href="#semantics">② 牙号</a> · <a href="#normals">③ 法线</a> · <a href="protocol/cohort.json">冻结名单</a> · <a href="summary.json">全部指标</a> · <a href="review_records.json">36 项待审核记录</a></nav>',
        '<h2>逐例摘要</h2><table><tr><th>病例 / 选入原因</th><th>目标牙覆盖</th><th>原始拓扑：边界 / 非流形 / 绕序冲突</th><th>原始面法线径向线索</th></tr>']
    for r,m in zip(rows,metrics):
        neg=[f'{k}: {v["radial_dot_median_unweighted"]:.3f}' for k,v in m['tooth_statistics'].items() if k!='0' and v['radial_dot_median_unweighted']<0]
        lines.append(f'<tr><td><a href="#case-{r["case_id"]}">{r["case_id"]}</a><br>{html.escape(r["selection_reason"])}</td><td>{m["observed_target_teeth"]}/14<br>未标注: {m["missing_target_fdi"]}<br>智齿: {m["wisdom_fdi"]}</td><td>{list(m["topology"].values())}</td><td>负中位数：{html.escape("; ".join(neg) or "无")}<br>不能单凭该值判定或翻转</td></tr>')
    lines.append('</table><h2>本轮初步判断与证据边界</h2><ul><li>12 例均未发现非法 FDI、实例混号、零面积面或重复面；常规双面邻接绕序冲突为零。这不等于解剖牙号已经人工确认。</li><li>5 例原始下颌存在非流形边，端点标注均为牙龈 0；3 例进入 R4 的病例历史上仅在该区域清理过少量面，不能据此声称牙冠被削坏。</li><li>017UXKRN 的 27 与 013TGCFK 的 37 有明显凹陷/有限牙冠覆盖，局部截面法线仍朝向表面外侧；负径向统计不能单独证明整牙反向。两例历史已排除且未进入当前 R4 训练，不能直接解释当前模板变差。</li><li>016N696F 历史按名单拒绝，本次所有目标牙径向中位数为正，也无普通邻接绕序冲突；旧拒绝原因尚未复现，仍保留待核。</li><li>当前优先处理牙冠完整性/覆盖范围和审核判据的区分。未发现足以把新旧模板差异归因于普遍牙号错乱或法线翻转的证据；下一步仍须在同一批数据上追踪尺度、对齐与采样，不能提前归因模型结构。</li></ul><p>上述为自动指标加助手初步目视判断，人工放行状态仍为 PENDING。重点审核 36 张四视图；局部近景只覆盖下面 4 个牙位。</p>')
    lines.append('<h2>原始牙号覆盖</h2><a href="fdi_coverage.png"><img src="fdi_coverage.png"></a><h2>法线警报的局部证据</h2>')
    for item in read(out/'local_evidence.json')['closeups']:
        lines.append(f'<details><summary>{item["case_id"]} / FDI {item["fdi"]}：{html.escape(item["reason"])}</summary><a href="{item["image"]}"><img loading="lazy" src="{item["image"]}"></a></details>')
    for s,t in zip(STAGES,TITLES):
        lines.append(f'<h2 id="{s}">{t}</h2><a href="{s}_contact.png"><img src="{s}_contact.png"></a>')
    for r,m in zip(rows,metrics):
        cid=r['case_id'];lines.append(f'<h2 id="case-{cid}">{cid}</h2><p>{html.escape(r["selection_reason"])} | 历史状态 {m["previous_status"]}: {m["previous_reason"]} | <a href="cases/{cid}/metrics.json">详细指标与原坐标</a></p><div class="grid">')
        for s,t in zip(STAGES,TITLES):lines.append(f'<div><a href="cases/{cid}/{s}.png">{t}<img loading="lazy" src="cases/{cid}/{s}.png"></a></div>')
        lines.append('</div>')
    lines.append('<h2>验收填写与后续边界</h2><p>按 review_records.json 的每例每阶段记录审核人、时间、结论及异常位置。几何：区分扫描开放边与非流形；牙号：核对解剖身份、牙龈边界、缺牙/智齿；法线：结合前后视图、相邻绕序、局部近景核验。未决病例保留，不自动剔除。此次结果不能归因模型架构或参数；需要验收数据后的单因素训练对照。</p><p>本轮未执行尺度归一化、对齐、镜像、表面采样及 loader 导出。这些阶段等待本批原始审核后在独立版本继续。</p></html>')
    (out/'index.html').write_text('\n'.join(lines),encoding='utf-8')
    files=[p for p in out.rglob('*') if p.is_file()]
    write(out/'artifact_manifest.json',[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)])
    print(out/'index.html',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['select','analyze','render','finish']);ap.add_argument('--out',required=True);ap.add_argument('--stage',choices=STAGES)
    args=ap.parse_args();out=Path(args.out).resolve()
    assert out.is_relative_to((ROOT/'runs').resolve()),'Output must be an isolated repo runs directory'
    if args.action=='select':select(out)
    elif args.action=='analyze':analyze(out)
    elif args.action=='render':render(out,args.stage)
    elif args.action=='finish':finish(out)
