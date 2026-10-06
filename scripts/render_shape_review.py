"""Export a scientific before/after review figure after independent evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_root',type=Path)
    parser.add_argument('--output',default='shape_review.png',help='filename inside this run directory')
    parser.add_argument('--evaluation-file',default='evaluation.json',help='relative evaluation path inside this run')
    args=parser.parse_args()
    root=args.run_root.resolve(strict=True)
    evaluation_path=(root/args.evaluation_file).resolve(strict=True)
    if not evaluation_path.is_relative_to(root):
        raise ValueError('evaluation must belong to this run')
    evaluation=json.loads(evaluation_path.read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    fit=Path(provenance['fit_input'])
    manifest=json.loads((fit/'manifest.json').read_text())
    truth=json.loads((fit.parent/'truth/manifest.json').read_text())
    active=manifest['active_labels']
    targets={v['camera']:np.asarray(Image.open(fit/v['path'])) for v in manifest['masks']}
    targets['occlusal_65deg']=np.asarray(Image.open(fit.parent/'truth'/truth['heldout_mask_file']))
    names=list(targets)
    if Path(args.output).name != args.output:
        raise ValueError('output must be a filename inside the run')
    output=root/args.output
    if output.exists():
        raise FileExistsError(output)
    fig,axes=plt.subplots(2,4,figsize=(14,8.4),facecolor='#f5f6f8')
    for column,name in enumerate(names):
        arrays=[]
        for stage in ['initial','final']:
            folder=f'heldout_{stage}' if name=='occlusal_65deg' else stage
            arrays.append(np.asarray(Image.open(root/'renders'/folder/f'{name}_predicted_labels.png')))
        union=np.isin(targets[name],active)|np.isin(arrays[0],active)|np.isin(arrays[1],active)
        yy,xx=np.nonzero(union)
        x0,x1=max(0,xx.min()-12),min(union.shape[1],xx.max()+13)
        y0,y1=max(0,yy.min()-12),min(union.shape[0],yy.max()+13)
        for row,(stage,pred) in enumerate(zip(['initial','final'],arrays)):
            actual=np.isin(targets[name],active)
            predicted=np.isin(pred,active)
            rgb=np.zeros((*pred.shape,3),dtype=np.uint8)+20
            rgb[actual&predicted]=(225,225,225)
            rgb[actual&predicted&(targets[name]!=pred)]=(251,190,48)
            rgb[actual&~predicted]=(242,70,83)
            rgb[predicted&~actual]=(60,146,245)
            values=[]
            for label in active:
                a,b=targets[name]==label,pred==label
                union_count=np.count_nonzero(a|b)
                if union_count:
                    values.append(np.count_nonzero(a&b)/union_count)
            axes[row,column].imshow(rgb[y0:y1,x0:x1],interpolation='nearest')
            axes[row,column].set_title(f'{stage} | {name}\nactive tooth IoU = {np.mean(values):.4f}',fontsize=11)
            axes[row,column].axis('off')
    fig.suptitle(f'Shape-only: {len(active)} teeth / {evaluation["active_latent_dimensions"]} latent dimensions; pose and cameras fixed\nFDI '+', '.join(map(str,active)),fontsize=14)
    fig.text(.5,.025,'White: same FDI   |   Red: target only   |   Blue: prediction only   |   Yellow: wrong FDI   |   65 deg held out',ha='center',fontsize=10)
    fig.subplots_adjust(left=.01,right=.99,bottom=.09,top=.88,hspace=.65,wspace=.065)
    fig.savefig(output,dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps({'status':evaluation['status'],'figure':str(output)}))


if __name__=='__main__':
    main()
