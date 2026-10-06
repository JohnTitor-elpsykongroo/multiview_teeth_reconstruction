"""Visual QA of prepared mask perturbations; no fitting or geometry-truth access."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_forward_check import PROJECT_ROOT,sha256


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('batch_root',type=Path); args=parser.parse_args()
    root=args.batch_root.resolve(strict=True)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()) or (root/'input_review.png').exists():
        raise ValueError('invalid root or figure exists')
    plan=json.loads((root/'plan.json').read_text())
    control=Path(plan['control_run']); p=json.loads((control/'provenance.json').read_text()); fit=Path(p['fit_input'])
    manifest=json.loads((fit/'manifest.json').read_text())
    variants=[('control',fit,manifest)]
    for item in plan['jobs']:
        if item['kind']=='initialization':
            continue
        f=root/'jobs'/item['id']/'case/fit_input'; m=json.loads((f/'manifest.json').read_text())
        variants.append((item['id'],f,m))
    cameras=[m['camera'] for m in manifest['masks']]
    fig,axes=plt.subplots(len(cameras),len(variants),figsize=(12,10),squeeze=False)
    for row,name in enumerate(cameras):
        original=np.asarray(Image.open(fit/next(m['path'] for m in manifest['masks'] if m['camera']==name)))
        yy,xx=np.nonzero(original); x0,x1=xx.min()-8,xx.max()+9; y0,y1=yy.min()-8,yy.max()+9
        for column,(identifier,f,m) in enumerate(variants):
            ax=axes[row,column]; ax.axis('off')
            item=next((item for item in m['masks'] if item['camera']==name),None)
            if item is None:
                ax.text(.5,.5,'NOT PROVIDED TO FITTER',ha='center',va='center',fontsize=10); ax.set_title(f'{identifier}\n{name}: dropped'); continue
            if sha256(f/item['path'])!=item['sha256']:
                raise ValueError('prepared mask changed')
            target=np.asarray(Image.open(f/item['path']))
            rgb=np.full((*original.shape,3),20,dtype=np.uint8)
            rgb[(original==target)&(target>0)]=(220,220,220)
            rgb[(original>0)&(target==0)]=(242,70,83)
            rgb[(original==0)&(target>0)]=(60,146,245)
            rgb[(original>0)&(target>0)&(original!=target)]=(250,190,48)
            ax.imshow(rgb[max(0,y0):y1,max(0,x0):x1],interpolation='nearest')
            ax.set_title(f'{identifier}\n{name}: {np.count_nonzero(original!=target)} changed pixels',fontsize=10)
    fig.suptitle('Prepared observations vs clean original masks\nWhite: unchanged tooth | Red: removed pixels | Blue: added pixels | Yellow: relabeled',fontsize=13)
    fig.subplots_adjust(top=.86,bottom=.03,left=.01,right=.99,hspace=.35,wspace=.13)
    fig.savefig(root/'input_review.png',dpi=160); plt.close(fig)
    print(json.dumps({'figure':str(root/'input_review.png')}))


if __name__=='__main__':
    main()
