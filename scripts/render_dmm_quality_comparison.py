"""Render exact saved surfaces with the same display framing and material."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/dmm_mean_quality_audit_20261016'
FONT='C:/Windows/Fonts/msyh.ttc'

def render(v,f,eye,size=700):
    v=(v-(v.max(0)+v.min(0))/2)/(v[:,0].max()-v[:,0].min())*1.7
    z=np.asarray(eye,dtype=float);z/=np.linalg.norm(z)
    up=np.array([0.,0.,1.]) if abs(z[2])<.95 else np.array([0.,1.,0.])
    x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x);r=np.array([x,y,z]).T
    p=v@r;t=p[f]
    normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-10)
    normal*=np.where(normal[:,2:3]<0,-1,1)
    light=np.array([-.35,.45,.82]);light/=np.linalg.norm(light)
    shade=.30+.65*np.maximum(normal@light,0)+.05*np.maximum(normal[:,2],0)
    colors=np.clip(shade[:,None]*[243,226,197],0,255).astype(np.uint8)
    xy=t[:,:,:2]*size/2.15;xy[:,:,0]+=size/2;xy[:,:,1]=size/2-xy[:,:,1]
    im=Image.new('RGB',(size,size),'#e8eef2');draw=ImageDraw.Draw(im)
    for i in np.argsort(t[:,:,2].mean(1)):
        draw.polygon([tuple(p) for p in xy[i]],fill=tuple(colors[i]))
    return im

def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 128
    specs=[('old_saved_307','旧 DMM · 已保存 epoch 307','原始保存网格；生成分辨率未留运行记录'),
           ('r4_saved48','R4 · 刚才展示的初始网格','48³ + 法线方向求根'),
           (f'r4_mean_n{n}',f'R4 · 同一均值，重新提取 {n}³','直接查询隐式场；无平滑、无补洞')]
    initial=np.load(ROOT/'runs/dmm_static_trial_20261015T121206Z_4ccf8f/fit_pilot_004/initial_render/mesh.npz')
    params=json.loads((ROOT/'runs/dmm_static_trial_20261015T121206Z_4ccf8f/fit_pilot_004/starting_parameters.json').read_text())
    ff=initial['faces'][initial['face_arch']==0];ids,rev=np.unique(ff,return_inverse=True);T=np.asarray(params['upper']['T_world_from_arch'])
    vv=(initial['vertices_world_mm'][ids]-T[:3,3])@np.linalg.inv(T[:3,:3]).T
    np.savez_compressed(OUT/'r4_saved48.npz',vertices=vv,faces=rev.reshape(-1,3))
    canvas=Image.new('RGB',(2100,1670),'#e8eef2');draw=ImageDraw.Draw(canvas)
    draw.text((30,20),'新旧 DMM 上颌：实际网格对照',font=ImageFont.truetype(FONT,42),fill='#172c3e')
    draw.text((30,82),'统一材质、视角、显示宽度；各模型原始坐标尺度不同，此图不比较毫米尺寸。',font=ImageFont.truetype(FONT,24),fill='#516878')
    for col,(stem,title,subtitle) in enumerate(specs):
        d=np.load(OUT/f'{stem}.npz');v=d['vertices'];f=d['faces']
        # Legacy upper convention has crown-facing +Z and anterior +Y.
        # Rotate its display 180 degrees about X to match R4's crown-facing -Z.
        if stem.startswith('old'):v=v*np.array([1,-1,-1])
        for row,eye in enumerate([(0,0,-1),(.55,-.8,-.9)]):
            canvas.paste(render(v,f,eye),(col*700,200+row*700))
        draw.text((col*700+24,140),title,font=ImageFont.truetype(FONT,27),fill='#172c3e')
        draw.text((col*700+24,180),subtitle,font=ImageFont.truetype(FONT,19),fill='#516878')
        draw.text((col*700+24,1600),f'{len(v):,} 顶点 / {len(f):,} 三角面',font=ImageFont.truetype(FONT,23),fill='#172c3e')
    canvas.save(OUT/f'comparison_n{n}.png');print(OUT/f'comparison_n{n}.png')

if __name__=='__main__':main()
