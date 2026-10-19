"""Additional local evidence, derived only from the frozen twelve raw caches."""
import sys, shutil
from pathlib import Path
import numpy as np
import pyvista as pv
from PIL import Image,ImageDraw
from diagnose_preprocess12 import read,write,font,unit,label_color,STAGES

OUT=Path(sys.argv[1]).resolve()
assert OUT.is_relative_to((Path(__file__).resolve().parents[1]/'runs').resolve())
PICKS=[('01328DDN_upper',27,'完整对照'),('017UXKRN_upper',27,'历史 INWARD_NORMALS_27'),
       ('013TGCFK_lower',37,'历史 INWARD_NORMALS_37'),('016N696F_upper',17,'历史名单拒绝；本次最小径向中位数牙位')]
records=[]
for cid,tooth,why in PICKS:
    p=OUT/'cases'/cid;d=np.load(p/'raw_index_preserved.npz');m=read(p/'metrics.json')
    v=d['vertices'];f=d['faces'];n=d['face_normals'];fc=d['face_centers'];b=d['basis'];s=1 if m['arch']=='upper' else -1
    target=(d['face_label']==tooth)&d['strict'];center=np.array(m['tooth_statistics'][str(tooth)]['center_raw_mm'])
    radial=fc-center;radial/=np.maximum(np.linalg.norm(radial,axis=1,keepdims=True),1e-20);dots=(radial*n).sum(1)
    mesh=pv.PolyData(v,np.column_stack([np.full(len(f),3),f]).ravel())
    canvas=Image.new('RGB',(1600,1850),'#edf1f5');dr=ImageDraw.Draw(canvas)
    dr.text((25,16),f'{cid} · FDI {tooth} · 原始局部证据',font=font(34),fill='#18384a')
    st=m['tooth_statistics'][str(tooth)]
    dr.text((25,66),f'{why} | 面径向中位数 {st["radial_dot_median_unweighted"]:.3f} | 负值面积 {st["radial_negative_area_fraction"]:.1%}',font=font(22),fill='#39576b')
    for idx,title in enumerate(['原始几何 + 目标牙号（24 mm 视野）','原面法线：青朝相机 / 红背相机','径向点积：蓝为正 / 橙为负（仅启发式）']):
        pl=pv.Plotter(off_screen=True,window_size=(800,800));pl.set_background('#edf1f5');pl.disable_anti_aliasing()
        e=unit(b@unit(np.array([.3,.25,s])));rgb=np.full((len(f),3),[185,180,172],np.uint8)
        if idx==0:rgb[d['face_label']==tooth]=label_color(tooth,m['arch'])
        elif idx==1:rgb[target]=np.where(((n@e)[target]>=0)[:,None],[41,174,172],[224,61,125])
        else:rgb[target]=np.where((dots[target]>=0)[:,None],[45,137,206],[239,137,46])
        pl.add_mesh(mesh,scalars=rgb,rgb=True,preference='cell',lighting=True,ambient=.35,diffuse=.65,show_scalar_bar=False)
        if idx==1:
            ids=np.flatnonzero(target);ids=ids[np.linspace(0,len(ids)-1,min(150,len(ids))).astype(int)]
            pts=pv.PolyData(fc[ids]);pts['n']=n[ids];pl.add_mesh(pts.glyph(orient='n',scale=False,factor=.8),color='#18384a',lighting=False)
        pl.camera_position=[center+e*150,center,b[:,1]];pl.enable_parallel_projection();pl.camera.parallel_scale=12;pl.camera.clipping_range=(1,350)
        im=Image.fromarray(pl.screenshot(return_img=True));pl.close();dd=ImageDraw.Draw(im)
        dd.text((15,14),title,font=font(22),fill='#18384a');dd.line((25,765,25+5/24*800,765),fill='#18384a',width=4);dd.text((25,730),'5 mm',font=font(20),fill='#18384a')
        canvas.paste(im,((idx%2)*800,120+(idx//2)*800))
    # Exact triangle/plane intersections through target centroid, normal display X.
    # Segment normal is the original triangle normal projected into the Y/Z plane.
    panel=Image.new('RGB',(800,800),'#edf1f5');dd=ImageDraw.Draw(panel)
    dd.text((15,14),'穿过质心的 Y/Z 截面 + 原面法线',font=font(24),fill='#18384a')
    dd.text((15,50),'橙：目标牙；灰：其余原表面；箭头长 0.8 mm',font=font(20),fill='#39576b')
    signed=(v-center)@b[:,0];ft=signed[f];mask=(ft.min(1)<0)&(ft.max(1)>0)
    ids=np.flatnonzero(mask);segments=[]
    def xy(pt):
        z=(pt-center)@b
        return np.array([400+z[1]*800/24,440-s*z[2]*800/24])
    for fi in ids:
        pts=[]
        for a,c in [(0,1),(1,2),(2,0)]:
            da,dc=ft[fi,a],ft[fi,c]
            if da*dc<0:pts.append(v[f[fi,a]]+(v[f[fi,c]]-v[f[fi,a]])*da/(da-dc))
        if len(pts)!=2:continue
        mid=np.mean(pts,axis=0);pp=xy(mid)
        if np.any(pp<75) or np.any(pp>760):continue
        color='#df842a' if target[fi] else '#a4a7ab'
        dd.line([tuple(xy(pts[0])),tuple(xy(pts[1]))],fill=color,width=3 if target[fi] else 2)
        if target[fi]:segments.append((fi,mid))
    for fi,mid in segments[::max(1,len(segments)//30)]:
        a=xy(mid);c=xy(mid+n[fi]*.8);dd.line([tuple(a),tuple(c)],fill='#243c52',width=2)
        ve=unit(c-a);side=np.array([-ve[1],ve[0]])
        dd.polygon([tuple(c),tuple(c-ve*7+side*3),tuple(c-ve*7-side*3)],fill='#243c52')
    dd.ellipse((396,436,404,444),fill='#d92654');dd.text((408,435),'质心',font=font(18),fill='#c02c4c')
    dd.line((25,765,25+5/24*800,765),fill='#18384a',width=4);dd.text((25,730),'5 mm',font=font(20),fill='#18384a')
    canvas.paste(panel,(800,920));dr.text((25,1740),'只截取显示窗口和数学截面；没有裁剪/改写源网格。截面是开放表面，不补成封闭体。',font=font(23),fill='#39576b')
    dr.text((25,1785),'径向为负可由凹面、残冠/部分扫描或质心位置导致；不能据此自动翻法线。人工验收待定。',font=font(23),fill='#9c3d3b')
    path=p/f'local_fdi_{tooth}.png';assert not path.exists();canvas.save(path)
    records.append(dict(case_id=cid,fdi=tooth,reason=why,image=str(path.relative_to(OUT)),section_target_segments=len(segments),status='REVIEW_PENDING'))
    print(cid,tooth,flush=True)

rows=read(OUT/'protocol/cohort.json')['cases'];all_labels=[10*q+i for q in [1,2,3,4] for i in range(1,9)]
im=Image.new('RGB',(1700,820),'#edf1f5');dr=ImageDraw.Draw(im)
dr.text((25,15),'原始牙号覆盖矩阵 · 缺号不等于确诊缺牙',font=font(34),fill='#18384a')
dr.text((25,65),'彩色：该号存在；空白：本颌目标号未标注；灰色：其他颌；38/48 保留显示。',font=font(22),fill='#39576b')
for k,t in enumerate(all_labels):dr.text((310+k*42,120),str(t),font=font(19),fill='#18384a')
checks=[]
for i,r in enumerate(rows):
    p=OUT/'cases'/r['case_id'];m=read(p/'metrics.json');d=np.load(p/'raw_index_preserved.npz');y=160+i*49
    dr.text((25,y+5),r['case_id'],font=font(22),fill='#18384a')
    for k,t in enumerate(all_labels):
        applicable=(t//10 in ([1,2] if r['arch']=='upper' else [3,4]))
        c=tuple(label_color(t,r['arch'])) if t in m['vertex_labels'] else ('white' if applicable else '#d5dce2')
        dr.rectangle((305+k*42,y,343+k*42,y+38),fill=c,outline='#b3c1cb')
    checks.append(dict(case_id=r['case_id'],nonmanifold_endpoint_fdi=np.unique(d['vertex_labels'][d['nonmanifold_edges']]).tolist(),
        largest_labeled_component_min_fraction=min(v['largest_component_vertex_fraction'] for k,v in m['tooth_statistics'].items() if k!='0')))
im.save(OUT/'fdi_coverage.png');write(OUT/'local_evidence.json',dict(closeups=records,topology_semantic_crosscheck=checks))
shutil.copyfile(__file__,OUT/'protocol/diagnose_preprocess12_details.py')
