"""Assemble measured sampling-audit findings and scientific figures."""
from pathlib import Path
import json,html,hashlib,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_space_sampling_20261018 import OUT,DATA,PROJECT,read,write,quant

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
g=read(OUT/'global_summary.json');d=read(OUT/'domain_summary.json');dr=read(OUT/'domain_entries.json')
entries=read(OUT/'global_entries.json');deep=read(OUT/'deep_entries.json');selected=read(OUT/'deep_selection.json')
exact=read(OUT/'exact_distance.json');contract=read(OUT/'dmm_space_contract.json')
assert not [r for r in deep if 'error' in r]
assert len({r['case_id'].split('__mirror')[0] for r in deep})==len(selected)
assert not g['errors'] and not d['errors']
original=[r for r in deep if not r['parent_id']]
current_exact=[r for r in exact['results'] if r['variant']=='current_uniform']
loader4096=[np.mean([b['near_2mm'] for b in r['official_loader_batches']['4096']]) for r in original]
loader8192=[np.mean([b['near_2mm'] for b in r['official_loader_batches']['8192']]) for r in original]
oldqc=Path(r'D:\WorkSpace\Dental\data\Teeth3DS_DMM\05_dataset_profiles\upper_v1.3_strict_fixed_offsurface\qc\validation_report.json')
oldmethod=read(oldqc)['free_space_method']
dataset_unchanged=hashlib.sha256((DATA/'case_results.json').read_bytes()).hexdigest()==g['dataset_sha256_before']
summary=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),space_entries=g['entry_count'],space_points=g['point_count'],global_random_generation_flags=len(g['flagged']),
    original_cases_deep=len(original),entries_deep=len(deep),deep_scope='training only, selected extremes/quantiles by arch, scale, RMS, domain fraction; not a random population sample',
    exact_original_cases=len(current_exact),exact_near_2mm_fraction=quant([r['near_2mm_fraction'] for r in current_exact]),
    official_loader_4096_near_2mm_20_draw_mean=quant(loader4096),official_loader_8192_near_2mm_20_draw_mean=quant(loader8192),
    affected_domain_entries=d['all']['affected_entries'],affected_domain_originals=d['original']['affected_entries'],affected_domain_mirrors=d['mirror']['affected_entries'],
    domain_outside_tooth_points=d['all']['outside_tooth_points'],domain_outside_gum_points=d['all']['outside_gum_points'],pass_domain_affected_entries=d['PASS']['affected_entries'],
    legacy_active_profile_free_space_method=oldmethod,finite_sdf_column_loss_difference_max=max(x['max_total_loss_difference'] for x in contract['arches']),
    no_dataset_manifest_changes=dataset_unchanged,no_dataset_or_model_write_commands=True,training_started=False,
    conclusion='No random-generation, slot-label, mirror-warp or whole-cloud reuse defect found. Uniform cube distribution matches this inspected legacy active profile. Confirmed domain mismatch affects only exported gingiva samples, not tooth samples. Near-surface coverage varies materially; this is a design tradeoff pending controlled evaluation, not demonstrated training failure.')
write(OUT/'AUDIT_SUMMARY.json',summary)

fig,axes=plt.subplots(2,2,figsize=(13,8.8),layout='constrained')
a=axes[0,0];tr=[r for r in entries if r['split']=='train' and not r['parent_id']]
a.hist([r['mm_per_model_unit'] for r in tr],bins=32,color='#3a9287',alpha=.85)
a.set(title='训练原例：模型单位对应的毫米尺度',xlabel='mm / model unit',ylabel='病例数')
a=axes[0,1];a.hist([r['joint_8cube_reduced_chi2'] for r in entries],bins=35,color='#4e7eaa');a.axvline(1,color='#b47c3c',ls='--');a.set(title='全 3,001 池：8×8×8 联合分布检查',xlabel='reduced χ²（均匀假设期望约 1）',ylabel='条目数')
a=axes[1,0];x=np.arange(len(original));a.plot(x,loader4096,'o',ms=4,color='#387f91',label='总4096 → 空间2048');a.plot(x,loader8192,'o',ms=4,color='#b78940',label='总8192 → 空间4096');a.set(title=f'{len(original)} 个分层选例：每例20次真实 loader 抽样',xlabel='选例索引（非随机总体样本）',ylabel='距稠密观测表面点池 ≤2 mm 的数量');a.legend(fontsize=9)
a=axes[1,1];width=.35;x=np.arange(2);a.bar(x-width/2,[d['original']['affected_entries'],d['mirror']['affected_entries']],width,label='牙龈越界',color='#b88459');a.bar(x+width/2,[d['original']['affected_tooth_entries'],d['mirror']['affected_tooth_entries']],width,label='牙齿越界',color='#4e9a87');a.set(xticks=x,xticklabels=['原例 / 1671','镜像 / 1330'],title='实际导出表面点与空间采样域的覆盖差异',ylabel='受影响条目数');a.legend()
fig.savefig(OUT/'audit_overview.png',dpi=150);plt.close(fig)

# Visualize the largest measured gingiva-domain mismatch; no mesh is repaired or cropped.
worst=max((r for r in dr if r['split']=='train' and not r['parent_id']),key=lambda r:r['max_distance_mm'])
row=next(r for r in read(DATA/'case_results.json') if r['case_id']==worst['case_id']);entry=row['entries'][0]
with np.load(entry['samples_path']) as z:surf=z['surf'];space=np.concatenate((z['pos'][:2500,:3],z['neg'][:2500,:3]))
idx=np.linspace(0,len(surf)-1,18000).astype(int);points=surf[idx,:3];outside=np.any(np.abs(points)>1,axis=1)
fig,axs=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
for ax,(i,j,name) in zip(axs,[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
    ax.scatter(space[:,i],space[:,j],s=.6,c='#9eb5c0',alpha=.18,rasterized=True)
    ax.scatter(points[~outside,i],points[~outside,j],s=.6,c='#398c7c',alpha=.65,rasterized=True)
    ax.scatter(points[outside,i],points[outside,j],s=2,c='#ce674d',alpha=.9,rasterized=True)
    ax.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],c='#486e87',lw=1.3);ax.set_aspect('equal');ax.set(title=name,xlabel='XYZ'[i],ylabel='XYZ'[j]);ax.grid(alpha=.15)
fig.suptitle(f"{worst['case_id']} · 空间点仅在蓝框内；橙色为域外牙龈（显示子集）",fontsize=12)
fig.savefig(OUT/'domain_example.png',dpi=150);plt.close(fig)

def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
facts=[('训练入口','D:/WorkSpace/Dental/DMM/train_dmm.py','load_ram=True；实际调用 DMM.forward'),('RAM loader','D:/WorkSpace/Dental/DMM/data/data_with_labels.py','表面比例0.5；余下pos/neg各一半；两槽mask均=-1'),('网络入口','D:/WorkSpace/Dental/DMM/networks/dmm_net.py','读取 [:3] 坐标及 [4] 牙号；不读取有限的 [3] SDF值'),('空间损失','D:/WorkSpace/Dental/DMM/networks/loss.py','exp(-100*abs(pred_sdf)) 以及梯度模长约束；不是点到真值SDF回归'),('旧版生效报告',str(oldqc),f'free_space_method={oldmethod}'),('旧版配置','D:/WorkSpace/Dental/teeth3DS_to_DMM/configs/upper_baseline.yaml','uniform cube；loader_compatibility_only；true_inside_outside=false')]
body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>DMM 空间点采样只读审查</title><style>body{{font:15px/1.85 "Microsoft YaHei",sans-serif;color:#203943;max-width:1120px;margin:36px auto;padding:0 24px;background:#f8faf9}}h1,h2{{line-height:1.4}}h1{{font-size:27px}}h2{{margin-top:32px;font-size:20px}}.note{{padding:18px 22px;background:#e8f2ee;border-left:4px solid #378d7b}}.warn{{background:#fbf1df;padding:18px;border-left:4px solid #b38740}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{text-align:left;border:1px solid #d6e1dd;padding:8px;overflow-wrap:anywhere}}th{{background:#eef4f1}}img{{max-width:100%;border:1px solid #dce6e1;border-radius:5px}}code{{background:#eaf0ed;padding:2px 5px}}a{{color:#227c70}}li{{margin:7px 0}}.small{{font-size:12px;color:#71898a}}</style></head><body>
<h1>DMM 空间点采样：只读审查</h1><p>2026-10-18 · output/latest · 3,001导出条目 / 1,671原例 · 没有训练、改数据或扩大排除范围。</p>
<div class="note"><b>结论：</b>没有发现空间随机数生成、整池重复、分槽标记或镜像空间变换错误。当前全域均匀方式与检查到的旧版生效 profile 一致。确认的覆盖不一致是部分牙龈表面在立方体外；牙齿表面未发现该问题。近表面空间点数量存在显著病例差异，这属于需要控制实验的采样设计问题，尚不能判定为训练失败原因。</div>
<h2>1. pos/neg 到底是否提供正负 SDF 监督</h2>
<p>结论仅限本机当前实际训练路径。两槽名称仍决定从哪个点池抽样，但槽位身份不作为正负监督传入损失。有限的第4列SDF值不参与网络计算；该列在loader中仍用于过滤NaN，所以不能说它在任何意义上都“无用”。网络读取坐标、FDI/非表面牙号和单独的is_on_surf掩码。</p>
{table(['证据','路径','核对结果'],facts)}
<p>上下颌各取一个真实RAM-loader批次，固定坐标、标签、掩码、网络和latent，仅将第4列从0改为±17，再反转正负。两个网络的总损失及各日志项均完全相同（最大总损失差0）。另以纯非表面样本验证整体损失对pred_sdf正负的对称性。随机初始化网络、没有优化器、没有保存模型，参数逐个张量核对未变化。</p>
<p class="warn"><b>不能跨越的推论：</b>不读取正负标签 ≠ 内外分布无关。若某版本先按内外分类再各取一半，分类仍会改变空间监督分布，不能擅自删掉。此次检查的旧版实际profile本身是offsurface_compat，不能把它描述为已采用内外平衡的版本。表面法线和组件间表面约束仍含方向/符号信息，不能泛称整个DMM没有内外信息。</p>
<h2>2. 全量空间点生成检查</h2>
{table(['检查','范围','结果'],[['点数 / dtype / 有限值 / 立方体边界 / 两列占位','全部3,001池 / 1,500,500,000点','未发现异常'],['独立种子逐点重放','全部3,001池，500,000点/池','逐元素一致'],['种子、整池及两槽内容SHA比较','3,001种子 / 3,001整池 / 6,002槽','无重复种子或整池/槽'],['池内坐标重复','每池4096点子集；深查条目全500k','没有发现重复；不冒充全库逐行去重'],['8×8×8联合均匀统计','全部3,001池',f"reduced χ² 中位数 {g['reduced_chi2']['p50']:.3f}；范围 {g['reduced_chi2']['min']:.3f}–{g['reduced_chi2']['max']:.3f}"],['镜像空间点','独立种子全量重放','在最终规范立方体重新采样；未被组合配准拉伸/平移']])}
<img src="audit_overview.png" alt="全量空间分布、尺度、真实loader近表面覆盖与域外表面统计">
<h2>3. 确认的域覆盖不一致：牙龈，不是牙齿</h2>
{table(['范围','条目','含域外表面的条目','域外牙齿点','域外牙龈点'],[[k,d[key]['entries'],d[key]['affected_entries'],d[key]['outside_tooth_points'],d[key]['outside_gum_points']] for k,key in [('原例','original'),('镜像','mirror'),('计算PASS','PASS'),('待复核REVIEW','REVIEW')]])}
<p>全部域外表面点共937,785个，其中533,481个距离立方体超过1病例毫米；最大到立方体距离9.51mm。它们仍有表面与法线监督，缺少的是该立方体外区域的非表面采样。222原例加180镜像共402条目均已是REVIEW。所有已导出的牙齿表面点、全部计算PASS条目表面点均在域内。该结果不说明源牙龈解剖正确，也不需要据此修补少数网格或增加永久排除。</p>
<img src="domain_example.png" alt="最大牙龈域覆盖差异示例">
<h2>4. 近表面覆盖：病例差异显著，但尚未做质量归因</h2>
<p>仅从训练侧按尺度、残差、越界比例及历史共同例选取{len(original)}原例，另外检查{len(deep)-len(original)}镜像；这是刻意覆盖极端情况的非随机样本，不能用其均值估计全库总体。每例从实际点池抽20,000空间点，距稠密观测表面采样池的距离是到真实三角表面的距离上界，不是有符号距离。</p>
<p>另对{len(current_exact)}原例各4096个空间点计算到原始保留三角面的精确最近距离（无符号、无需封口）；通过质心/半径分组和最近顶点上界保证候选覆盖，并与小网格全三角暴力法对照，最大差{exact['brute_force_validation_max_error']:.2g}mm。选例中距真实表面≤2mm比例为{min(r['near_2mm_fraction'] for r in current_exact)*100:.2f}%–{max(r['near_2mm_fraction'] for r in current_exact)*100:.2f}%。此范围不是全库分位数。</p>
<p>实际官方RAM-loader在总采样4096和8192两种<b>诊断情景</b>下分别给出2048和4096空间点；新项目尚无正式训练，不能把情景参数当成已生效训练设置。4096情景每例20次抽样的≤2mm点数均值范围{min(loader4096):.1f}–{max(loader4096):.1f}，选例中位{np.median(loader4096):.1f}。8192情景大致增加一倍近表面点。增大存盘池本身并不会在固定每步抽样数下同比提高局部密度。</p>
<p>训练原例1模型单位对应{g['training_original_scale_mm']['min']:.2f}–{g['training_original_scale_mm']['max']:.2f}mm；固定模型立方体对应的物理体积与每立方毫米采样密度因病例而异。DMM梯度约束也在表面点计算，其他组件表面点亦参与组件非表面项，因此不能把“近表面空间点少”理解为“这些位置完全没有梯度/几何监督”。</p>
{table(['病例','当前均匀点 ≤2mm','旧版生效profile ≤2mm','说明'],[[cid,f"{next(r['near_2mm_fraction'] for r in exact['results'] if r['case_id']==cid and r['variant']=='current_uniform')*100:.2f}%",f"{next(r['near_2mm_fraction'] for r in exact['results'] if r['case_id']==cid and r['variant']=='legacy_fixed_offsurface')*100:.2f}%",'同原始毫米表面；每种独立4096点，参考/尺度不同，非单变量因果实验'] for cid in dict.fromkeys(r['case_id'] for r in exact['results'] if r['variant']=='legacy_fixed_offsurface')])}
<h2>5. 建议与执行边界</h2><ol><li>保留旧版兼容槽位与当前原始产物；无需因pos/neg名称重做内外分类或网格封口。</li><li>明确训练/提取的规范域是否要求覆盖牙龈裁切区。若要覆盖，在训练侧研究统一参考域或原表面周围的额外空间覆盖；不要用逐例裁剪/钳位掩盖问题。</li><li>若牙冠细节是主要目标，先做固定模型、病例、总点数、尺度的采样对照，比较纯均匀与全局＋近表面混合。混合比例与距离带需要实测，不能直接写入正式数据。近点仍为非表面约束，不能太靠近零面而与非零惩罚冲突，也不能冒充真SDF/内外标签。</li><li>确认正式训练SamplesPerScene预算后再评估每牙和近表面覆盖；本次没有改训练配置、重采样正式数据、修网格或扩大排除。</li></ol>
<h2>证据文件</h2><ul>{''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['AUDIT_SUMMARY.json','global_summary.json','global_entries.json','domain_summary.json','domain_entries.json','deep_selection.json','deep_entries.json','exact_distance.json','dmm_space_contract.json'])}</ul><p class="small">case_results.json 前后SHA一致：{dataset_unchanged}。本次只写独立 runs/space_sampling_audit_20261018_01 与审查脚本。全部数据和模型保持不变；计算检查与输入分布差异不代表训练或解剖质量验收。</p></body></html>'''
(OUT/'REPORT_ZH.html').write_text(body,encoding='utf-8')
write(OUT/'artifact_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='artifact_hashes.json'})
print(json.dumps(summary,ensure_ascii=False,indent=2))
