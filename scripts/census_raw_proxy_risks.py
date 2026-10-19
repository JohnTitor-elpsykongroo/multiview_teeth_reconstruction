"""Read-only whole-source screening. Numerical candidates, not exclusion labels."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='1'
import sys,json,time,re,hashlib,traceback,collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/dmm_raw_risk_census_20261017_01'
DATA=Path('D:/WorkSpace/Dental/data/Teeth3DS')
NEW=Path('D:/WorkSpace/Dental/data/Teeth3DS_DualArch_v1')
sys.path.insert(0,str(ROOT/'runs/dmm_meshlab_proxy_00OMSZGW_20261017_01/deps'))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def raw(p):
    data=Path(p).read_bytes();s=data.decode('utf-8-sig');vs=re.findall(r'^v\s+([^\r\n]+)',s,re.M);fs=re.findall(r'^f\s+([^\r\n]+)',s,re.M)
    if any('/' in x or '#' in x for x in fs):
        from diagnose_preprocess12 import obj_read
        v,f,_=obj_read(p)
    else:
        v=np.fromstring(' '.join(vs),sep=' ',dtype=np.float64);f=np.fromstring(' '.join(fs),sep=' ',dtype=np.int64)
        width=len(vs[0].split());assert width in [3,6] and len(v)==width*len(vs) and len(f)==3*len(fs),'Nontriangle or inconsistent OBJ fields'
        v=v.reshape(-1,width)[:,:3].copy();f=f.reshape(-1,3);f=np.where(f<0,len(v)+f,f-1)
    assert np.isfinite(v).all() and f.min()>=0 and f.max()<len(v)
    return v,f,hashlib.sha256(data).hexdigest()

def scan(row):
    t=time.time();r=dict(row);cid=row['case_id']
    try:
        import pymeshlab as ml
        v,f,h=raw(row['obj']);annotation=Path(row['annotation']).read_bytes();j=json.loads(annotation.decode('utf-8-sig'));labels=np.asarray(j['labels'],dtype=np.int32);assert len(labels)==len(v),'Annotation length mismatch';assert j['jaw']==row['arch'];assert j['id_patient']==row['patient_id'];valid={0}|{10*q+i for q in ([1,2] if row['arch']=='upper' else [3,4]) for i in range(1,9)};assert set(labels).issubset(valid),'FDI invalid'
        ms=ml.MeshSet();ms.add_mesh(ml.Mesh(vertex_matrix=v,face_matrix=f.astype(np.int32)));ms.compute_selection_by_self_intersections_per_face();bad=ms.current_mesh().face_selection_array();assert len(bad)==len(f);top=ms.get_topological_measures()
        tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5;face_labels=labels[f];tooth=(face_labels>0).any(1);tree=cKDTree(v[labels>0]);distance=tree.query(tri.mean(1),workers=1,distance_upper_bound=2.+1e-12)[0];near=(~tooth)&(distance<=2.);critical=tooth|near
        regions={}
        for name,mask in [('all',np.ones(len(f),bool)),('tooth',tooth),('near_gum_2mm',near),('critical',critical),('peripheral',~critical)]:
            regions[name]=dict(faces=int(mask.sum()),area_mm2=float(area[mask].sum()),selected_faces=int((bad&mask).sum()),selected_area_mm2=float(area[bad&mask].sum()),selected_area_fraction=float(area[bad&mask].sum()/max(area[mask].sum(),1e-30)))
        fdi={}
        for k in np.unique(labels):
            if k:
                mask=(face_labels==k).any(1)&bad;fdi[str(int(k))]=dict(selected_faces=int(mask.sum()),selected_area_mm2=float(area[mask].sum()))
        r.update(status='SCREENED',source_sha256=h,annotation_sha256=hashlib.sha256(annotation).hexdigest(),vertices=len(v),faces=len(f),regions=regions,per_fdi=fdi,topology={k:(x.item() if hasattr(x,'item') else x) for k,x in top.items()},seconds=time.time()-t)
        # Locate candidates without retaining full source geometry or modifying it.
        np.savez_compressed(OUT/'cases'/(cid+'_selected.npz'),face_ids=np.flatnonzero(bad),face_centers=tri[bad].mean(1),face_area_mm2=area[bad],critical=critical[bad])
    except Exception as e:r.update(status='UNSCREENED',error=str(e),seconds=time.time()-t)
    save(OUT/'cases'/(cid+'.json'),r);return cid,r['status'],r['seconds']

def inventory():
    OUT.mkdir(exist_ok=False);(OUT/'cases').mkdir();metadata={p.parent.name:read(p) for p in (NEW/'work/cases').glob('*/prepared.json')};splits=read(NEW/'reports/patient_splits.json');trained=set()
    for arch in ['upper','lower']:
        path=ROOT/f'.runtime/dmm_r4_best_val_20261015/package/model_bundle/{arch}/model.json';trained.update(read(path).get('training_case_ids',[]))
    rows=[]
    for arch in ['upper','lower']:
        for obj in sorted((DATA/arch).glob('*/*.obj')):
            cid=obj.stem;patient=obj.parent.name;m=metadata.get(cid,{})
            rows.append(dict(case_id=cid,patient_id=patient,arch=arch,obj=str(obj),annotation=str(obj.with_suffix('.json')),previous_status=m.get('status','NO_METADATA'),previous_reason=m.get('reason'),assigned_split=splits.get(patient,'UNKNOWN'),r4_training=cid in trained))
    assert len({x['case_id'] for x in rows})==len(rows)
    save(OUT/'inventory.json',rows);save(OUT/'protocol.json',dict(scope='All raw upper/lower OBJ files under Teeth3DS; no repair, no sampling export, no source mutation',method='MeshLab numerical self-intersection face selection on original indexed mesh; FDI>0 face vertices define tooth; gingival face centroid within 2 mm of tooth vertices defines near gum',primary_comparison='selected critical area fraction versus frozen 00OMSZGW_lower baseline; also report absolute area and threshold sensitivity',limitations='Numerical intersection screen is not classification ground truth. Normal open trim boundaries are not exclusion criteria. Missing annotations remain unknown. Patient and arch-scan denominators differ.',workers=4,created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    print('Inventory',len(rows),collections.Counter(x['arch'] for x in rows),'patients',len({x['patient_id'] for x in rows}),flush=True)

def run():
    rows=read(OUT/'inventory.json');pending=[x for x in rows if not (OUT/'cases'/(x['case_id']+'.json')).exists()]
    pending.sort(key=lambda x:(x['case_id']!='00OMSZGW_lower',x['case_id']))
    if pending and pending[0]['case_id']=='00OMSZGW_lower':print(scan(pending.pop(0)),flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(scan,x):x for x in pending}
        for i,future in enumerate(as_completed(futures),1):
            try:cid,status,secs=future.result()
            except Exception as e:print('WORKER_ERROR',futures[future]['case_id'],repr(e),flush=True);continue
            if i%25==0:print('completed',i,'/',len(pending),'last',cid,status,round(secs,2),flush=True)

def archive_parser_errors():
    dest=OUT/'parser_attempt_01';dest.mkdir(exist_ok=False)
    for p in (OUT/'cases').glob('*.json'):
        if read(p).get('error')=='Nontriangle or non-XYZ OBJ':p.rename(dest/p.name)
    row=next(x for x in read(OUT/'inventory.json') if x['case_id']=='00OMSZGW_lower');print(scan(row),flush=True)
    from diagnose_preprocess12 import obj_read
    v,f,_=raw(row['obj']);vv,ff,_=obj_read(row['obj']);assert np.array_equal(v,vv) and np.array_equal(f,ff);print('Parser verified against original indexed reader',flush=True)

def summarize():
    rows=read(OUT/'inventory.json');results=[];missing=[]
    for row in rows:
        p=OUT/'cases'/(row['case_id']+'.json')
        if p.exists():results.append(read(p))
        else:missing.append(row['case_id'])
    ok=[x for x in results if x['status']=='SCREENED'];unknown=[x for x in results if x['status']!='SCREENED'];ref=next(x for x in ok if x['case_id']=='00OMSZGW_lower');threshold=ref['regions']['critical']['selected_area_fraction'];absolute=ref['regions']['critical']['selected_area_mm2']
    flagged=sorted([x for x in ok if x['regions']['critical']['selected_area_fraction']>=threshold],key=lambda x:-x['regions']['critical']['selected_area_fraction']);patients={x['patient_id'] for x in flagged}
    def group(items):return dict(scans=len(items),patients=len({x['patient_id'] for x in items}),by_arch=dict(collections.Counter(x['arch'] for x in items)),r4_training_scans=sum(x['r4_training'] for x in items),previously_prepared_scans=sum(x['previous_status']=='PREPARED' for x in items),assigned_train_scans=sum(x['assigned_split']=='train' for x in items))
    summary=dict(complete=len(results)==len(rows),inventory=group(rows),screened=group(ok),unknown=group(unknown),missing=missing,reference=dict(case_id=ref['case_id'],critical_selected_fraction=threshold,critical_selected_area_mm2=absolute,tooth_selected_fraction=ref['regions']['tooth']['selected_area_fraction']),at_least_reference=group(flagged),any_critical_candidates=group([x for x in ok if x['regions']['critical']['selected_faces']>0]),peripheral_only_candidates=group([x for x in ok if x['regions']['critical']['selected_faces']==0 and x['regions']['peripheral']['selected_faces']>0]),no_selected_faces=group([x for x in ok if x['regions']['all']['selected_faces']==0]),threshold_sensitivity={},absolute_area_at_least_reference=group([x for x in ok if x['regions']['critical']['selected_area_mm2']>=absolute]),both_relative_and_absolute=group([x for x in flagged if x['regions']['critical']['selected_area_mm2']>=absolute]),tooth_fraction_at_least_reference=group([x for x in ok if x['regions']['tooth']['selected_area_fraction']>=ref['regions']['tooth']['selected_area_fraction']]),patient_level_removal_impact=group([x for x in rows if x['patient_id'] in patients]),unknown_reasons=dict(collections.Counter('MISSING_ANNOTATION' if 'No such file' in x.get('error','') else x.get('error','UNKNOWN') for x in unknown)),interpretation='Risk screening only, not confirmed inside/outside errors or approved deletion; no source file changed')
    for label,factor in [('half_reference',.5),('reference',1.),('double_reference',2.)]:summary['threshold_sensitivity'][label]=group([x for x in ok if x['regions']['critical']['selected_area_fraction']>=threshold*factor])
    peripheral_threshold=ref['regions']['peripheral']['selected_area_fraction'];summary['reference']['peripheral_selected_fraction']=peripheral_threshold
    summary['peripheral_at_least_reference']=group([x for x in ok if x['regions']['peripheral']['selected_area_fraction']>=peripheral_threshold])
    summary['peripheral_high_but_critical_below_reference']=group([x for x in ok if x['regions']['peripheral']['selected_area_fraction']>=peripheral_threshold and x['regions']['critical']['selected_area_fraction']<threshold])
    summary['whole_mesh_at_least_reference']=group([x for x in ok if x['regions']['all']['selected_area_fraction']>=ref['regions']['all']['selected_area_fraction']])
    summary['approximately_reference_0_4_percent']=group([x for x in ok if x['regions']['critical']['selected_area_fraction']>=.004])
    save(OUT/'peripheral_at_least_reference_cases.json',[x for x in ok if x['regions']['peripheral']['selected_area_fraction']>=peripheral_threshold])
    approx_patients={x['patient_id'] for x in ok if x['regions']['critical']['selected_area_fraction']>=.004}
    summary['approximately_reference_patient_removal_impact']=group([x for x in rows if x['patient_id'] in approx_patients])
    save(OUT/'approximately_reference_0_4_percent_cases.json',sorted([x for x in ok if x['regions']['critical']['selected_area_fraction']>=.004],key=lambda x:-x['regions']['critical']['selected_area_fraction']))
    (OUT/'approximately_reference_ids_REVIEW_ONLY.txt').write_text('\n'.join(x['case_id'] for x in sorted(ok,key=lambda x:-x['regions']['critical']['selected_area_fraction']) if x['regions']['critical']['selected_area_fraction']>=.004)+'\n',encoding='utf-8')
    # Existing source hashes are checked against the historical preparation ledger where present.
    integrity=collections.Counter()
    for x in ok:
        p=NEW/'work/cases'/x['case_id']/'prepared.json';m=read(p) if p.exists() else {}
        for field in ['source_sha256','annotation_sha256']:
            integrity[field+('_missing_history' if not m.get(field) else '_matches' if m[field]==x[field] else '_differs')]+=1
    summary['historical_hash_comparison']=dict(integrity)
    save(OUT/'summary.json',summary);save(OUT/'candidate_cases.json',flagged);save(OUT/'near_reference_candidates.json',sorted([x for x in ok if x['regions']['critical']['selected_area_fraction']>=threshold*.5],key=lambda x:-x['regions']['critical']['selected_area_fraction']));save(OUT/'unknown_cases.json',unknown);(OUT/'candidate_scan_ids_REVIEW_ONLY.txt').write_text('\n'.join(x['case_id'] for x in flagged)+'\n',encoding='utf-8');save(OUT/'all_case_metrics.json',results)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def gallery():
    summary=read(OUT/'summary.json');assert summary['complete'];render_gallery(read(OUT/'near_reference_candidates.json'))

def preview():
    rows=[]
    for p in (OUT/'cases').glob('*.json'):
        try:r=read(p)
        except json.JSONDecodeError:continue
        if r['status']=='SCREENED' and r['regions']['critical']['selected_area_fraction']>=.004414118581276514*.5:rows.append(r)
    render_gallery(sorted(rows,key=lambda r:-r['regions']['critical']['selected_area_fraction']))

def render_gallery(flagged):
    from PIL import Image,ImageDraw
    import pyvista as pv
    from audit_space_proxy12 import pd
    from diagnose_preprocess12 import font
    chosen=flagged if len(flagged)<=30 else flagged[:12]+flagged[-4:];(OUT/'gallery').mkdir(exist_ok=True)
    for r in chosen:
        cid=r['case_id']
        if (OUT/'gallery'/(cid+'.png')).exists():continue
        v,f,h=raw(r['obj']);assert h==r['source_sha256'];j=read(r['annotation']);labels=np.asarray(j['labels']);centers={int(k):v[labels==k].mean(0) for k in np.unique(labels) if k};left=[k for k in centers if k//10 in [2,3]];right=[k for k in centers if k//10 in [1,4]];anterior=[k for k in centers if k%10<=2];posterior=[k for k in centers if k%10>=6]
        if left and right and anterior and posterior:
            x=np.mean([centers[k] for k in right],0)-np.mean([centers[k] for k in left],0);x/=np.linalg.norm(x);y=np.mean([centers[k] for k in anterior],0)-np.mean([centers[k] for k in posterior],0);y-=x*(x@y);y/=np.linalg.norm(y);z=np.cross(x,y);basis=np.stack([x,y,z],1)
        else:_,_,vh=np.linalg.svd(v-v.mean(0),full_matrices=False);basis=vh.T
        c=((v@basis).min(0)+(v@basis).max(0))/2@basis.T;sel=np.load(OUT/'cases'/(cid+'_selected.npz'));bad=sel['face_ids'];critical=sel['critical'];im=Image.new('RGB',(1400,780),'#edf1f5')
        for col,ey in enumerate([[.5,.7,-.8],[-.5,.2,.8]]):
            pl=pv.Plotter(off_screen=True,window_size=(700,700));pl.set_background('#edf1f5');pl.add_mesh(pd(v,f),color='#d5c7af',ambient=.4)
            if (~critical).any():pl.add_mesh(pd(v,f[bad[~critical]]),color='#e59326',ambient=.6)
            if critical.any():pl.add_mesh(pd(v,f[bad[critical]]),color='#d62651',ambient=.6)
            eye=basis@np.asarray(ey);pl.camera_position=[c+eye*250,c,basis[:,2]];pl.enable_parallel_projection();pl.camera.parallel_scale=45;im.paste(Image.fromarray(pl.screenshot(return_img=True)),(700*col,0));pl.close()
        dr=ImageDraw.Draw(im);dr.text((18,705),f'{cid} · 关键区候选面积占比 {r["regions"]["critical"]["selected_area_fraction"]*100:.3f}% · 红：牙体/近牙龈；橙：外围',font=font(24),fill='#18384a');im.save(OUT/'gallery'/(cid+'.png'));print('rendered',cid,flush=True)
    save(OUT/'gallery_cases.json',[x['case_id'] for x in chosen])

def report():
    import html,shutil
    s=read(OUT/'summary.json');flagged=read(OUT/'candidate_cases.json');gallery_ids=read(OUT/'gallery_cases.json');n=s['inventory']['scans'];evaluated=s['screened']['scans'];k=len(flagged)
    content='<!doctype html><meta charset="utf-8"><title>原始网格异常分布统计</title><style>body{font:18px/1.7 system-ui;max-width:1450px;margin:25px auto;background:#edf1f5;color:#18384a}img{max-width:100%}td,th{border:1px solid #abc;padding:10px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>类似 00OMSZGW 的原始网格：全量只读筛查</h1>'
    content+=f'<p>原始扫描 {n} 份（950 上颌、950 下颌；{s["inventory"]["patients"]} 位患者）。完成区域筛查 {evaluated} 份，无法完成 {s["unknown"]["scans"]} 份，未扫描 {len(s["missing"])} 份。扫描份数不等于患者数。</p><p><b>按牙齿及近牙龈自交候选面积比例达到 00OMSZGW 的标准，共 {k} 份，占可筛查扫描的 {100*k/max(evaluated,1):.2f}%。其中当前 R4 训练病例 {s["at_least_reference"]["r4_training_scans"]} 份。</b></p>'
    content+='<p>“类似”在这里有明确限定：原始网格自交数值筛查，按 FDI 牙体和距牙体顶点 2 mm 的牙龈面质心分区；关键区候选面积比例 ≥ '+f'{s["reference"]["critical_selected_fraction"]*100:.4f}%'+ '。这是风险候选数量，不是已确认内外分类错误的病例数量。没有封底、修复、训练、删除或移动原始数据。</p><p>普通开放裁切边界不计为失败；智齿、缺牙、旧流程法线拒绝也不自动算入本类。标注缺失等无法筛查的扫描列为未知，不能算通过。数值检查可能有退化接触等误报；最终剔除前应核验候选的关键区域。</p><h2>阈值敏感性</h2><table><tr><th>标准</th><th>扫描数</th><th>患者数</th><th>上/下颌</th><th>当前 R4 训练</th></tr>'
    for label,g in s['threshold_sensitivity'].items():content+=f'<tr><td>{label}</td><td>{g["scans"]}</td><td>{g["patients"]}</td><td>{g["by_arch"]}</td><td>{g["r4_training_scans"]}</td></tr>'
    content+='</table><p>为避免把 0.440% 与 0.4414% 人为分成两类，另报告接近基准的 0.4% 阈值：'+str(s['approximately_reference_0_4_percent']['scans'])+' 份，<a href="approximately_reference_0_4_percent_cases.json">完整名单</a>。原先严格基准统计仍保留，阈值不是自动淘汰规则。</p><p>外围区域达到 00OMSZGW 自身外围比例的共 '+str(s['peripheral_at_least_reference']['scans'])+' 份，其中 '+str(s['peripheral_high_but_critical_below_reference']['scans'])+' 份的关键区尚低于基准。外围问题单列，不能自动等同于牙体附近分类风险。</p><h2>达到基准的候选清单</h2><table><tr><th>扫描 ID</th><th>关键区候选面积比例</th><th>面积 mm²</th><th>历史状态</th><th>划分 / 当前训练</th></tr>'
    for r in read(OUT/'near_reference_candidates.json'):
        a=r['regions']['critical'];cid=r['case_id'];content+=f'<tr><td><a href="#g_{cid}">{cid}</a></td><td>{a["selected_area_fraction"]*100:.4f}%</td><td>{a["selected_area_mm2"]:.3f}</td><td>{r["previous_status"]}</td><td>{r["assigned_split"]} / {r["r4_training"]}</td></tr>'
    content+='</table><p><a href="candidate_scan_ids_REVIEW_ONLY.txt">待复核扫描 ID 清单（没有执行剔除）</a> · <a href="candidate_cases.json">候选完整指标</a> · <a href="near_reference_candidates.json">放宽到一半基准的清单</a> · <a href="unknown_cases.json">无法筛查清单</a> · <a href="all_case_metrics.json">全部指标</a></p><h2>候选定位图（含一半基准以上的较轻对照）</h2><p>每图相同 90 mm 视野、两面观察；红色为关键区候选面，橙色为外围候选面。色块不是体积错误范围。该图集是定位资料，尚未逐病例人工确认。</p>'
    for cid in gallery_ids:content+=f'<h3 id="g_{cid}">{cid}</h3><img loading="lazy" src="gallery/{cid}.png">'
    content+='<h2>完整统计与边界</h2><pre>'+html.escape(json.dumps(s,ensure_ascii=False,indent=2))+'</pre>'
    content=content.replace('达到基准的候选清单</h2>','候选与较轻对照清单（含一半基准以上）</h2>').replace('待复核扫描 ID 清单（没有执行剔除）','严格基准扫描 ID 清单（没有执行剔除）')
    content+='<p><a href="approximately_reference_ids_REVIEW_ONLY.txt">接近或超过基准（≥0.4%）的扫描 ID 清单</a></p>'
    protocol=read(OUT/'protocol.json');protocol.update(workers_final=8,reader_support='OBJ XYZ or XYZRGB; vertex and face order preserved; baseline reader checked against original parser',distance_query='Exact nearest labeled tooth vertex within 2 mm; capped search equivalent to unbounded 2 mm predicate',threshold_note='Original exact 00OMSZGW threshold retained; 0.4 percent reported additionally to include almost identical 0.440 percent cases without a sharp arbitrary boundary')
    save(OUT/'protocol.json',protocol)
    (OUT/'index.html').write_text(content,encoding='utf-8');shutil.copyfile(__file__,OUT/'implementation.py');save(OUT/'delivery_manifest.json',[dict(path=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.rglob('*') if p.is_file() and 'parser_attempt_01' not in p.parts and p.name!='delivery_manifest.json']);print('report ready',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
