"""Save observed-label comparisons, fitted meshes and an honest pilot report."""
import colorsys
import hashlib
import html
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

PROJECT = Path(__file__).resolve().parents[1]
RUN = Path((PROJECT / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
FIT = RUN / sys.argv[1]
BASELINE = RUN / 'fit_pilot_004' if 'resume' in FIT.name else FIT
OUT = RUN / ('review_' + FIT.name)
OUT.mkdir(exist_ok=False)
result = json.loads((FIT / 'result.json').read_text('utf-8'))
plan = json.loads((RUN / 'plan.json').read_text('utf-8'))
spec = json.loads((RUN / 'pilot_scene_spec.json').read_text('utf-8'))
labels_order = np.array([0, *range(11, 18), *range(21, 28), *range(31, 38), *range(41, 48)], np.uint8)
palette = np.array([[.07, .07, .09], *[colorsys.hsv_to_rgb((i * .618) % 1, .65, .95) for i in range(28)]])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'savefig.dpi': 150})


def read_json(path):
    return json.loads(path.read_text('utf-8'))


def prediction(path):
    with np.load(path) as data:
        pred = labels_order[data['semantics'].argmax(-1)]
        truth, valid = data['labels'].copy(), data['valid'].copy()
    return pred, truth, valid


def edges(mask):
    return (mask != np.roll(mask, 1, 0)) | (mask != np.roll(mask, -1, 0)) | (mask != np.roll(mask, 1, 1)) | (mask != np.roll(mask, -1, 1))


def overlay(rgb, target, pred, valid):
    image = rgb.astype(float) / 255
    observed = edges(target) & (target != 0) & valid
    predicted = edges(pred) & (pred != 0) & valid
    image[~valid] = image[~valid] * .65 + np.array([.5, .1, .55]) * .35
    image[observed] = [1, .85, .05]
    image[predicted] = [.05, 1, 1]
    return image


initial_report = read_json(BASELINE / 'initial_render/report.json')
final_report = read_json(FIT / 'final_render/report.json')
views = list(final_report['views'])
rows, silhouette_rows, data_cache = [], [], {}
rgb_by_id = {r['view_id']: r['rgb'] for r in spec['views']}
for view in views:
    filename = final_report['views'][view]['file']
    final, truth, valid = prediction(FIT / 'final_render' / filename)
    initial, _, _ = prediction(BASELINE / 'initial_render' / initial_report['views'][view]['file'])
    with Image.open(rgb_by_id[view]) as image:
        rgb = np.asarray(image.convert('RGB'))
    for label in sorted((set(np.unique(final[valid])) | set(np.unique(truth[valid])) | set(np.unique(initial[valid]))) - {0, 255}):
        target = (truth == label) & valid
        before, after = (initial == label) & valid, (final == label) & valid
        def iou(pred):
            union = (target | pred).sum()
            return float((target & pred).sum()/union) if union else None
        rows.append(dict(view=view, label=int(label), target_pixels=int(target.sum()), initial_iou=iou(before), final_iou=iou(after)))
    for arch in ('both', 'upper', 'lower'):
        def foreground(value):
            if arch == 'upper':
                return (value >= 11) & (value <= 27) & valid
            if arch == 'lower':
                return (value >= 31) & (value <= 47) & valid
            return (value != 0) & (value != 255) & valid
        a = foreground(truth)
        def silhouette(value):
            b = foreground(value)
            return float((a & b).sum() / (a | b).sum()) if (a | b).any() else None
        silhouette_rows.append(dict(view=view, arch=arch, target_pixels=int(a.sum()),
            initial_predicted_pixels=int(foreground(initial).sum()), final_predicted_pixels=int(foreground(final).sum()),
            initial_iou=silhouette(initial), final_iou=silhouette(final)))
    data_cache[view] = rgb, initial, final, truth, valid


def comparison(selected, name):
    figure, axes = plt.subplots(len(selected), 5, figsize=(15, 3.05 * len(selected)), squeeze=False)
    for i, view in enumerate(selected):
        rgb, initial, final, truth, valid = data_cache[view]
        channels = np.zeros_like(truth)
        for k, label in enumerate(labels_order):
            channels[truth == label] = k
        colored = palette[channels]
        colored[~valid] = [.55, .15, .6]
        error = np.zeros((*truth.shape, 3))
        error[valid & (truth == final) & (truth != 0)] = [.2, .7, .35]
        error[valid & (truth != final)] = [.95, .22, .2]
        error[~valid] = [.35, .2, .4]
        images = (rgb, colored, overlay(rgb, truth, initial, valid), overlay(rgb, truth, final, valid), error)
        titles = ('Synthetic RGB', 'Observed FDI / purple=ignore', 'Initial: yellow=target cyan=model',
                  'Final: yellow=target cyan=model', 'Valid pixels: red=label error')
        for j, (image, title) in enumerate(zip(images, titles)):
            axes[i, j].imshow(image)
            axes[i, j].set_title(title if i == 0 else '')
            axes[i, j].axis('off')
        axes[i, 0].text(-.04, .5, view, transform=axes[i, 0].transAxes, rotation=90, va='center')
    figure.suptitle('L428SD7J | frozen R4 DMM | 15-view synthetic pilot | ' + result['status'], fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, .97))
    figure.savefig(OUT / name)
    plt.close(figure)


comparison([v for v in ('oral007', 'oral008', 'oral009') if v in data_cache], 'comparison_key_views.png')
comparison(views, 'comparison_all_15_views.png')


def load_mesh(folder):
    base = BASELINE if folder == 'initial_render' else FIT
    with np.load(base / folder / 'mesh.npz') as data:
        return {key: data[key].copy() for key in data.files}


meshes = {name: load_mesh(name + '_render') for name in ('initial', 'final')}
for phase, mesh in meshes.items():
    vertices, faces = mesh['vertices_world_mm'], mesh['faces']
    color = (palette[mesh['semantics'].argmax(-1)] * 255).astype(np.uint8)
    trimesh.Trimesh(vertices, faces, vertex_colors=color, process=False).export(OUT / (phase + '_dual_arch_world_mm.ply'))
    if phase == 'final':
        for index, arch in enumerate(('upper', 'lower')):
            subset = trimesh.Trimesh(vertices, faces[mesh['face_arch'] == index], vertex_colors=color, process=False)
            subset.remove_unreferenced_vertices()
            subset.export(OUT / (arch + '_world_mm.ply'))

figure = plt.figure(figsize=(14, 8))
angles = [(0, -90, 'Anterior'), (25, -125, 'Oblique'), (80, -90, 'Occlusal direction')]
all_vertices = np.concatenate([m['vertices_world_mm'] for m in meshes.values()])
center = (all_vertices.min(0) + all_vertices.max(0)) / 2
radius = max(float(np.ptp(all_vertices, axis=0).max()) / 2, 1)
for i, (phase, mesh) in enumerate(meshes.items()):
    v, f = mesh['vertices_world_mm'], mesh['faces']
    color = palette[mesh['semantics'].argmax(-1)][f].mean(1)
    for j, (elev, azim, title) in enumerate(angles):
        ax = figure.add_subplot(2, 3, i*3+j+1, projection='3d')
        collection = Poly3DCollection(v[f], facecolors=color, edgecolor='none', linewidth=0, alpha=1, rasterized=True)
        ax.add_collection3d(collection)
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(phase.capitalize() + ' | ' + title)
        ax.set_axis_off()
figure.suptitle('World coordinates in mm | blended tooth + gum surfaces | color denotes semantic component')
figure.tight_layout()
figure.savefig(OUT / 'geometry_initial_final.png')
plt.close(figure)

history = read_json(FIT / 'checkpoint.json')['history']
figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot([r['before'] for r in history], label='Before step', alpha=.6)
axes[0].plot([r['after'] for r in history], label='After step')
axes[0].set(xlabel='Iteration', ylabel='Total objective', title='Fixed-budget optimization')
axes[0].legend()
for arch, color in (('upper', '#3275ab'), ('lower', '#d3734c')):
    chosen = [r for r in rows if (r['label'] < 30) == (arch == 'upper') and r['initial_iou'] is not None and r['final_iou'] is not None]
    axes[1].scatter([r['initial_iou'] for r in chosen], [r['final_iou'] for r in chosen], s=13, alpha=.65, label=arch, color=color)
axes[1].plot([0, 1], [0, 1], '--', color='gray')
axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Initial tooth/view IoU', ylabel='Final tooth/view IoU', title='Above diagonal = improved')
axes[1].legend()
figure.tight_layout()
figure.savefig(OUT / 'optimization_and_iou.png')
plt.close(figure)

initial_scores = [r['initial_iou'] for r in rows if r['target_pixels'] > 0]
final_scores = [r['final_iou'] for r in rows if r['target_pixels'] > 0]
silhouette_summary = {arch: {phase: float(np.mean([r[phase + '_iou'] for r in silhouette_rows
    if r['arch'] == arch and r['target_pixels'] > 0])) for phase in ('initial', 'final')}
    for arch in ('both', 'upper', 'lower')}
summary = dict(status=result['status'], case=plan['case'], fit=str(FIT), iterations=result['iterations'],
    fit_resolution=[256, 256], views=len(views), budget=dict(pose=10, shape=10, joint=20),
    mean_initial_tooth_view_iou=float(np.mean(initial_scores)), mean_final_tooth_view_iou=float(np.mean(final_scores)),
    iou_mean_scope='Same observed GT tooth-view pairs, valid pixels only; core minimum gate unchanged',
    observed_tooth_view_pairs=len(initial_scores), silhouette_iou=silhouette_summary,
    silhouette_scope='Same observed target-positive views per arch; no-observation false positives retained in per_view_silhouette pixel counts',
    min_initial_tooth_view_iou=min(r['initial_iou'] for r in rows if r['initial_iou'] is not None), min_final_tooth_view_iou=result['final_metrics']['min_tooth_iou'],
    baseline_render=str(BASELINE / 'initial_render'),
    resume_scope='Native strict identity-checked resume; original pre-optimization baseline used in comparisons',
    accepted_steps=sum(r['accepted'] for r in history), unresolved_count=len(result['unresolved']),
    final_image_loss=result['final_image_loss'], frozen_unobserved=result['frozen_unobserved'],
    model_epochs=dict(upper=240, lower=360),
    input_scope='Oracle visible FDI and calibrated synthetic cameras; development case, source provenance split=train',
    initialization='Observed FDI centroid rays and mean-model Kabsch, no target 3D/pose',
    execution_changes='Isolated safeguarded float64 implicit roots/derivatives, float32 raster; unchanged weights and original gates',
    original_default_attempt='Failed during initialization with normal-root trust/convergence errors; retained',
    geometry_accuracy='No independently accepted 3D geometry error computed; IoU is image fit only', quality_accepted=False,
    per_tooth_view=rows, per_view_silhouette=silhouette_rows, artifacts=sorted(p.name for p in OUT.iterdir()))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
text = f'''新 DMM 双颌拟合试验：L428SD7J
实际状态：{summary['status']}
模型：上颌 epoch 240；下颌 epoch 360；网络权重冻结。
输入：全部15视图，512原图一致缩放到256；FDI/valid 最近邻；K/A同步缩放。
预算：pose 10，shape 10，joint 20；实际完成 {summary['iterations']} 步，接受 {summary['accepted_steps']} 步。
逐牙/视图 IoU 均值：{summary['mean_initial_tooth_view_iou']:.6f} -> {summary['mean_final_tooth_view_iou']:.6f}
逐牙/视图最小 IoU：{summary['min_initial_tooth_view_iou']:.6f} -> {summary['min_final_tooth_view_iou']:.6f}
牙齿整体轮廓 IoU（15视图均值）：{silhouette_summary['both']['initial']:.6f} -> {silhouette_summary['both']['final']:.6f}
逐牙均值使用同一组实际有观测的牙/视图，忽略区不计入；最小值沿用求解器原门槛。
未解决可见性项：{summary['unresolved_count']}

先看 comparison_key_views.png 的观测轮廓（黄色）与拟合轮廓（青色），
再看 geometry_initial_final.png 的三维形状。彩色为组件语义，不是误差热图。
PLY文件含牙龈，是混合表面；分颌文件不是逐牙解剖分割结果，坐标单位为mm。

原部署包默认入口在自动初始化阶段失败；完整失败日志和数值检查均保留。
本次结果来自独立执行副本：阻尼/区间法线求根，双精度局部隐式场，float32光栅化。
所有原始权重、统计文件和部署源码未改；没有放宽1e-6残差、0.1信任域或0.85图像阈值。
初值来自观测质心与均值模型配准，没有使用目标三维网格或真值位姿。
RGB只用于追溯和叠加可视化；拟合目标使用FDI语义、投影XY和已知相机，不检验RGB自动识别或相机恢复。
续跑使用原生身份检查；执行缓存/CPU卸载的单独源码和数值对照记录保存在运行根目录。
这是开发病例的oracle标签+已知相机试验，不能作为独立泛化或三维解剖质量验收。
'''
(OUT / 'RESULT_ZH.txt').write_text(text, encoding='utf-8')
images = [('comparison_key_views.png', 'Observed contours versus initial/final reconstruction'),
          ('geometry_initial_final.png', 'Initial and final 3D blended surfaces'),
          ('optimization_and_iou.png', 'Optimization and image IoU'),
          ('comparison_all_15_views.png', 'All 15 views')]
page = '<!doctype html><meta charset="utf-8"><title>DMM trial result</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#f6f7f9;color:#18202a}pre{white-space:pre-wrap;background:white;padding:20px}img{width:100%;background:white}a{color:#1769a8}</style><h1>DMM dual-arch synthetic pilot</h1><pre>' + html.escape(text) + '</pre>'
for name, title in images:
    page += f'<h2>{html.escape(title)}</h2><a href="{name}"><img src="{name}"></a>'
page += '<p><a href="final_dual_arch_world_mm.ply">Final mesh PLY</a> · <a href="summary.json">Metrics JSON</a></p>'
(OUT / 'index.html').write_text(page, encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k not in ('per_tooth_view', 'per_view_silhouette', 'artifacts')}, indent=2))
print(OUT)
