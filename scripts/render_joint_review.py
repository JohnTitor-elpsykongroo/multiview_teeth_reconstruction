"""Scientific review of joint images and canonical versus world surface errors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from run_forward_check import PROJECT_ROOT, load_ply
from evaluate_shape_only import sampled_points, nearest_triangle_distances
from shape_only_common import world_meshes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_root',type=Path)
    parser.add_argument('--suffix',default='')
    args=parser.parse_args()
    root=args.run_root.resolve(strict=True)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('review must be in project runs')
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.suffix):
        raise ValueError('invalid figure suffix')
    semantic_path=root/f'semantic_review{args.suffix}.png'
    geometry_path=root/f'geometry_review{args.suffix}.png'
    if any(path.exists() for path in [semantic_path,geometry_path]):
        raise FileExistsError('review images already exist')
    evaluation=json.loads((root/'evaluation.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    fit=Path(provenance['fit_input'])
    manifest=json.loads((fit/'manifest.json').read_text())
    truth_root=fit.parent/'truth'
    truth=json.loads((truth_root/'manifest.json').read_text())
    labels=manifest['active_labels']
    targets={v['camera']:np.asarray(Image.open(fit/v['path'])) for v in manifest['masks']}
    heldout=json.loads((truth_root/truth['heldout_camera_file']).read_text())['name']
    targets[heldout]=np.asarray(Image.open(truth_root/truth['heldout_mask_file']))
    fig,axes=plt.subplots(2,len(targets),figsize=(14,8),facecolor='#f5f6f8')
    for column,(name,target) in enumerate(targets.items()):
        predictions=[np.asarray(Image.open(root/'renders'/(f'heldout_{stage}' if name==heldout else stage)/f'{name}_predicted_labels.png')) for stage in ['initial','final']]
        union=(target>0)|(predictions[0]>0)|(predictions[1]>0)
        yy,xx=np.nonzero(union)
        x0,x1=max(0,xx.min()-12),min(union.shape[1],xx.max()+13)
        y0,y1=max(0,yy.min()-12),min(union.shape[0],yy.max()+13)
        for row,(stage,pred) in enumerate(zip(['initial','final'],predictions)):
            rgb=np.full((*pred.shape,3),20,dtype=np.uint8)
            rgb[(target>0)&(pred>0)]=(225,225,225)
            rgb[(target>0)&(pred>0)&(target!=pred)]=(251,190,48)
            rgb[(target>0)&(pred==0)]=(242,70,83)
            rgb[(target==0)&(pred>0)]=(60,146,245)
            scores=[np.count_nonzero((pred==k)&(target==k))/np.count_nonzero((pred==k)|(target==k)) for k in labels if np.count_nonzero((pred==k)|(target==k))]
            axes[row,column].imshow(rgb[y0:y1,x0:x1],interpolation='nearest')
            axes[row,column].set_title(f'{stage} | {name}\nFDI mean IoU = {np.mean(scores):.4f}',fontsize=11)
            axes[row,column].axis('off')
    err=evaluation['pose_errors']['final']
    fig.suptitle(f'Joint: 40 latent + 6 pose variables | {evaluation["status"]}\nFinal pose errors: {err["rotation_degrees"]:.3f} deg, {err["translation_dmm"]:.5f} DMM units',fontsize=13)
    fig.text(.5,.025,'White: same FDI | Red: target only | Blue: prediction only | Yellow: wrong FDI | 65 deg held out',ha='center',fontsize=10)
    fig.subplots_adjust(left=.01,right=.99,bottom=.09,top=.86,hspace=.5,wspace=.06)
    fig.savefig(semantic_path,dpi=160); plt.close(fig)

    gt={k:load_ply(truth_root/'meshes'/f'tooth{k}.ply') for k in labels}
    gt_pose=json.loads((truth_root/truth['pose_file']).read_text())['arch_to_world']
    points,distances={},{}
    for frame in ['canonical','world']:
        target_meshes=gt if frame=='canonical' else world_meshes(gt,gt_pose)
        for stage in ['initial','final']:
            meshes={k:load_ply(root/'meshes'/stage/f'tooth{k}.ply') for k in labels}
            if frame=='world':
                matrix=json.loads((root/f'{stage}_pose.json').read_text())['arch_to_world']
                meshes=world_meshes(meshes,matrix)
            p,d=[],[]
            for label in labels:
                sampled=sampled_points(meshes[label],1500,510+label)
                p.append(sampled); d.append(nearest_triangle_distances(sampled,target_meshes[label]))
            points[(frame,stage)]=np.concatenate(p); distances[(frame,stage)]=np.concatenate(d)
    fig=plt.figure(figsize=(14,11))
    fig.subplots_adjust(left=.01,right=.93,bottom=.07,top=.88,hspace=.28,wspace=.18)
    for column,frame in enumerate(['canonical','world']):
        all_points=np.r_[points[(frame,'initial')],points[(frame,'final')]]
        center=(all_points.min(axis=0)+all_points.max(axis=0))/2
        radius=np.ptp(all_points,axis=0).max()/2*1.1
        norm=Normalize(0,np.percentile(distances[(frame,'initial')],95))
        panels=[]
        for row,stage in enumerate(['initial','final']):
            ax=fig.add_subplot(2,2,row*2+column+1,projection='3d'); panels.append(ax)
            p=points[(frame,stage)]
            artist=ax.scatter(*p.T,c=distances[(frame,stage)],cmap='magma',norm=norm,s=3,depthshade=False,rasterized=True)
            for axis,value in zip('xyz',center):
                getattr(ax,'set_'+axis+'lim')(value-radius,value+radius)
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=30,azim=-60)
            ax.set_title(f'{stage} | {frame} geometry',fontsize=12)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z'); ax.tick_params(labelsize=8)
        fig.colorbar(artist,ax=panels,shrink=.6,label=f'{frame} distance (DMM units)',extend='max')
    fig.suptitle('Joint: canonical shape and world geometry errors\n1500 area-sampled points/tooth; initial P95 scale shared within each column',fontsize=13)
    fig.savefig(geometry_path,dpi=170); plt.close(fig)
    print(json.dumps({'status':evaluation['status'],'figures':[str(semantic_path),str(geometry_path)]}))


if __name__=='__main__':
    main()
