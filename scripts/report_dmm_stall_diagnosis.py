"""Consolidate bounded causal controls into a Chinese report and audit."""
import hashlib
import html
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT=Path(__file__).resolve().parents[1]
OUT=Path((PROJECT/'tmp/dmm_diagnostic_current.txt').read_text('utf-8-sig'))
def read(name):return json.loads((OUT/name).read_text('utf-8-sig'))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
OLD=Path(read('plan.json')['source_run'])
primary=read('shape_backtrack_probes.json');follow=read('shape_backtrack_followup.json')
fixed=read('matching_refresh_separation.json');fresh=read('matching_refresh_fresh_mesh.json')
capacity=read('component_geometry_evaluation.json');precision=read('root_precision_control.json')
models=read('model_and_population_audit.json');history=read('history_analysis.json');pose=read('pose_vs_shape_3d.json')
assert read('step_probe_state.json')['status']=='COMPLETE'
assert read('step_followup_state.json')['status']=='COMPLETE'
assert len(read('component_capacity.json')['results'])==4
base=primary['baseline']['loss']
by_scale={r['scale']:r for r in primary['rows']}
for row in follow['rows']:by_scale[row['scale']]=row
fd=(by_scale[1/128]['fixed_chart']['frozen_total']-by_scale[-1/128]['fixed_chart']['frozen_total'])/(2/128)
relative=abs(fd-primary['slope'])/abs(primary['slope'])
summary=dict(conclusion='主要已定位的停滞原因是采样/验收流程的不连续；数值求根也是实际阻塞，但不足以解释全部停滞。未发现权重文件绑定错误或梯度断链，完整模型容量/泛化尚未排除。',
    native_shape_steps=history['groups']['shape'],latent_rms=history['final_latents'],
    shape_direction_autograd=primary['slope'],shape_direction_frozen_fd=fd,fd_relative_difference=relative,
    minimum_native_scale=1/128,smaller_accepted_diagnostic_scale=1/1024,
    baseline=base,fixed_sampling_control=fixed['means'],fresh_mesh_sampling_control=fresh['means'],
    quality_accepted=False,original_run_unchanged_pending_audit=True)
(OUT/'diagnosis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

fig,axes=plt.subplots(1,2,figsize=(12,4.5))
keys=('frozen_samples_frozen_plan','frozen_samples_new_plan','new_samples_new_plan')
x=np.arange(3)
for record,offset,color,label in ((fixed,-.18,'#287da8','Fixed surface chart'),(fresh,.18,'#dd8452','Fresh surface mesh')):
    values=[(record['means'][key]-record['means']['baseline'])*1e6 for key in keys]
    bars=axes[0].bar(x+offset,values,.36,color=color,label=label)
    axes[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
axes[0].axhline(0,color='#555555',linewidth=.8)
axes[0].set_xticks(x,['Fixed pixels\nfixed OT','Fixed pixels\nnew OT','New pixels\nnew OT'])
axes[0].set_ylabel('Image objective change (1e-6); below zero is better')
axes[0].set_title('Same shape update (scale 1/128), different scoring')
axes[0].legend(fontsize=8)
scales=[1/(2**i) for i in range(8)]+[1/1024]
values=[(by_scale[s]['fixed_chart']['loss']-base)*1e6 for s in scales]
axes[1].bar(np.arange(len(scales)),values,color=['#a8a8a8']*8+['#31826e'])
axes[1].axhline(0,color='#555555',linewidth=.8)
axes[1].scatter([0],[values[0]],color='#b6232e',marker='x',s=95,zorder=4,label='Fresh mesh: 2 roots fail')
axes[1].scatter([8],[(by_scale[1/1024]['fresh_topology']['loss']-base)*1e6],color='#31826e',marker='D',s=45,zorder=4,label='Fresh check at 1/1024 improves')
axes[1].set_xticks(np.arange(len(scales)),['1','1/2','1/4','1/8','1/16','1/32','1/64','1/128','1/1024'],rotation=45)
axes[1].set_title('Native scoring rejects all 8 allowed scales')
axes[1].set_ylabel('Total objective change (1e-6)')
axes[1].legend(fontsize=8)
fig.suptitle('Frozen R4 checkpoint | one replayed shape direction | no new model training')
fig.tight_layout();fig.savefig(OUT/'sampling_and_backtracking.png',dpi=160);plt.close(fig)

def cap_score(label,phase):return next(r['rms_mm'] for r in capacity['rows'] if r['label']==label and r['phase']==phase)
step=by_scale[1/128];tiny=by_scale[1/1024]
table='\n'.join(f"FDI {label}: {cap_score(label,'before'):.3f} -> {cap_score(label,'after'):.3f} mm" for label in (11,27,31,47))
text=f'''DMM 双颌拟合停滞原因诊断（L428SD7J）

结论
{summary['conclusion']}
优先修拟合流程和数值实现，再判断是否需要重新训练。当前结果不能证明新权重整体失效，也不能证明其已达到几何质量要求。

1. 已定位的主因：优化方向与候选评分使用了不同的采样状态
原运行20个形状步骤只接受1步；最终标准化q的RMS为上颌0.000494、下颌0.000476，形状几乎停留在均值。
形状梯度最大范数持续约0.0020–0.0027，明显高于1e-6阈值，属于停滞，不能叫收敛。
复现的总损失 {base:.12f} 与原检查点完全一致。
冻结采样/匹配后，自动微分方向导数 {primary['slope']:.9f}，中心有限差分 {fd:.9f}，相对差约{relative*100:.2f}%。
这支持这一个检查点/方向上的解码器—隐式表面—渲染梯度链可用，不是全面的梯度正确性证明。

同一小步（系数1/128）：
总损失：基准 {base:.9f}；固定样本与匹配 {step['fixed_chart']['frozen_total']:.9f}；
重新采样/匹配 {step['fixed_chart']['loss']:.9f}；强制新网格并重新采样 {step['fresh_topology']['loss']:.9f}。
固定表面chart时，有6/15视图的预测像素样本索引发生变化。
进一步拆分后的图像损失：
基准 {fixed['means']['baseline']:.9f}
固定样本、固定OT {fixed['means']['frozen_samples_frozen_plan']:.9f}
固定样本、重算OT {fixed['means']['frozen_samples_new_plan']:.9f}
重新采样、重算OT {fixed['means']['new_samples_new_plan']:.9f}
仅重算OT没有使这一步变坏，重新选择样本才将下降翻转成上升。
新网格下固定样本并重算OT仍下降至 {fresh['means']['frozen_samples_new_plan']:.9f}；
全图密集语义项也从 {fresh['means']['baseline_dense']:.9f} 降至 {fresh['means']['candidate_dense']:.9f}。
因此不能把这一次拒绝简单归因于Marching Cubes不可微；新网格会进一步放大采样后的评分跳变。

2. 回溯范围不足，且整组latent共享一步接受/拒绝
原配置latent_step=0.1、backtracks=8，只试系数1到1/128。
完整回放8档：最大步长的旧chart损失已下降，但新表面上颌3485点中2点无法在信任域内包围根，导致整组更新被拒。
其余7档主要被重新采样后的损失上升拒绝。
同一方向继续缩小到1/1024（每牙方向L2步长通常约0.0000977），旧chart总损失 {tiny['fixed_chart']['loss']:.9f}，新网格 {tiny['fresh_topology']['loss']:.9f}，均可下降。
这个对照说明现有回溯范围遗漏可接受小步；但采样得分存在跳变，不能把这一个小步当作已经获得稳定、几何正确的重建。

3. 数值求根是独立的真实问题
原部署源码在q=0时，上颌报信任域错误，下颌报根未收敛；同一权重用隔离的阻尼/区间求根+float64场可通过原1e-6门槛。
将这些双精度根转为float32坐标并用单精度场求值，上/下颌最大残差分别为
{precision['upper']['float32_at_same_roots_max_abs']:.3g} / {precision['lower']['float32_at_same_roots_max_abs']:.3g}；32/214个点超过1e-6。
这里同时包含场求值与坐标量化误差，不能把数字解释成纯场求值误差或精确几何距离。
这解释了默认入口失败和部分大步拒绝；它不能解释那些已成功求根、却因采样评分上升而被拒的小步。

4. 初始化有偏差，但不是停滞的充分解释
原试验因成本使用观测质心射线+均值Kabsch的第一个候选，没有完成全部旋转候选的渲染排名。
可见遮挡轮廓质心不是同一解剖点，多视图三角化与均值配准会有系统偏差。
仅用于诊断的3D真值14牙质心刚性校正：初始上下颌需约2.39°/2.89°旋转校正；
校正后的逐牙质心RMS仍约2.84/3.39 mm。它是质心Kabsch对照，不是最优表面配准，也不等同于已知真值位姿。
按同一刚性校正规则，初始/最终形状表面RMS约1.8110/1.8114 mm，进一步支持形状基本没有改变。
初始化和牙列形状差异需要改进，但在固定当前位姿的受控小步中已复现采样导致拒绝，不能只归咎初值差。

5. 新权重：未发现文件或梯度断链，但不能完全免责
上/下颌权重及统计文件哈希均与各自model.json绑定一致；导出模型的训练病例名单与本机train病例ID集合相同。
本机manifest字节哈希与远端记录不同，因此不把本机manifest当作远端原件。
导出记录有88/86个验证病例，无失败项；验证分数约0.605/0.590 mm。
这是直接3D latent拟合后的单向观测到生成表面采样均值，不能与本次双向精确三角面采样RMS横向比较；其标记仍是“未质量验收”。
L428SD7J在本机预处理中虽split=train，但两颌均因WISDOM_TEETH_EXCLUDED被排除；导出训练病例名单也不包含它。
屏蔽智齿像素并不自动消除剩余牙列形态的分布差异。

固定新权重、绕过图像采样与法线求根的4牙直接3D对照：80步Adam，只优化对应牙的10维q；
使用诊断用3D质心Kabsch位姿，512拟合表面点+512独立面积样本。几何另用MC64网格、每方向1000点的精确点到三角面距离检查。
实际组件网格对称表面RMS：
{table}
11、31、47号牙有明显可优化形状响应；27号牙改善极小。
因此权重没有整体“失去形状能力”，但27号牙的表达上限/局部优化/残余位姿问题仍未区分。
这些是单组件、带3D真值的诊断，不是完整混合双颌的图像重建，也不构成独立泛化验收。

建议的修复次序
第一：让每次内层回溯使用同一组样本索引/分层成员与对应关系；外层更新后再刷新，记录刷新前后目标。保留全图密集语义和真实可见性检查。
第二：把局部代理下降与重新提取后的真实评分分开记录；处理采样刷新噪声后再决定接受，避免把采样变化当成几何劣化。
第三：增加细步长范围/自适应trust region，并记录每档损失、失败arch/组件/点数；不要单纯放宽根残差、域或可见性门槛。
第四：对新chart失败点局部加密/重建法线chart，保留精度对照。仅增加MC分辨率或迭代数不能修复采样目标不连续。
第五：完成这些修复后，用同一病例、同一固定预算做一次对照，再单独评价新权重容量与更换初始化的收益。

边界
原运行、部署包及模型保持只读。本诊断没有修改生产拟合器或重新训练网络。
所有3D真值只用于明确标记的归因对照；不能用于宣称原图像拟合成功。
完整数据见本目录JSON、日志和source_snapshot；首次探针的import错误日志单独保留，不作为模型失败证据。
'''
(OUT/'REPORT_ZH.txt').write_text(text,encoding='utf-8')
css='body{font:16px/1.7 system-ui;max-width:1250px;margin:30px auto;padding:0 24px;color:#182936;background:#f5f7f9}pre{white-space:pre-wrap;background:white;padding:24px;border-radius:10px}img{width:100%;background:white;border-radius:8px}h1{font-size:28px}a{color:#176da3}'
page=f'<!doctype html><meta charset="utf-8"><title>DMM停滞原因诊断</title><style>{css}</style><h1>DMM 双颌拟合：形状停滞原因诊断</h1><pre>{html.escape(text)}</pre>'
for name,title in (('sampling_and_backtracking.png','采样刷新与回溯拒绝'),('capacity_controls.png','同一权重的直接3D形状响应'),('capacity_geometry.png','组件几何对照（灰色目标、蓝色模型）')):
    page+=f'<h2>{title}</h2><a href="{name}"><img src="{name}"></a>'
page+='<h2>证据文件</h2><ul>'
for name in ('diagnosis_summary.json','shape_backtrack_probes.json','shape_backtrack_followup.json','matching_refresh_separation.json','matching_refresh_fresh_mesh.json','root_precision_control.json','history_analysis.json','pose_vs_shape_3d.json','model_and_population_audit.json','component_capacity.json','component_geometry_evaluation.json'):
    page+=f'<li><a href="{name}">{name}</a></li>'
page+='</ul>'
(OUT/'REPORT_ZH.html').write_text(page,encoding='utf-8')

# Verify all prior user-facing results and native result/checkpoint bytes.
delivery=json.loads((OLD/'DELIVERY.json').read_text('utf-8'))
old_changes=[name for name,record in delivery['files'].items() if digest(OLD/name)!=record['sha256']]
package=PROJECT/'.runtime/dmm_r4_best_val_20261015/package'
package_checks=json.loads((package/'SHA256SUMS.json').read_text('utf-8'))
package_changes=[name for name,expected in package_checks.items() if digest(package/name)!=expected]
assert not old_changes and not package_changes,(old_changes,package_changes)
audit=dict(prior_delivery_checked=len(delivery['files']),prior_delivery_changed=old_changes,package_checked=len(package_checks),package_changed=package_changes,
    original_fitting_code_unchanged=digest(OLD/'runtime_source/DMM/dmm/fitting.py')==digest(package/'third_party/DMM/dmm/fitting.py'))
(OUT/'integrity_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
summary['original_run_unchanged_pending_audit']=False
summary['integrity_audit']=audit
(OUT/'diagnosis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
names=['dmm_stall_common.py','probe_dmm_stall_step.py','probe_dmm_pose_and_history.py','audit_dmm_stall_models.py','prepare_dmm_capacity_targets.py','probe_dmm_component_capacity.py','evaluate_dmm_stall_capacity.py','probe_dmm_matching_refresh.py','report_dmm_stall_diagnosis.py']
registry={}
for name in names:
    path=PROJECT/'scripts'/name;shutil.copy2(path,snapshot/name);registry[name]=digest(path)
(snapshot/'SHA256SUMS.json').write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='COMPLETE',report=str(OUT/'REPORT_ZH.html'),audit=audit),indent=2),flush=True)
