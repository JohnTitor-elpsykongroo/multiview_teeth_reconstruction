"""Fixed LPS front/side/occlusal atlas of evaluation meshes (Pillow, no GPU).

Dominant face labels and painter ordering are diagnostic visualization, not the
soft-semantic differentiable renderer used for later multiview fitting.
"""
import argparse
import colorsys
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def preview(evaluation_root, output):
    root = Path(evaluation_root)
    records = json.loads((root/'evaluation.json').read_text())['cases']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for index, record in enumerate(records):
        image = Image.new('RGB', (1200, 480), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((16, 12), record['case_id']+' | '+record['status'], fill='black')
        mesh = root/record.get('artifact_directory', f'case_{index:04d}')/'mesh.npz'
        if not mesh.is_file():
            draw.text((16, 60), record.get('error', 'mesh unavailable'), fill='red')
        else:
            with np.load(mesh, allow_pickle=False) as data:
                vertices = data['vertices_model'] * data['model_unit_mm']
                faces, labels = data['faces'], data['face_labels']
                observed = data['observed_points_model'] * data['model_unit_mm']
            center = (vertices.max(0)+vertices.min(0))/2
            extent = max(float(np.ptp(vertices, axis=0).max()), 1.)
            scale = 330/extent
            for panel, (name, axes, depth, reverse) in enumerate((
                    ('front: X/Z, from anterior', (0,2), 1, True),
                    ('side: Y/Z, from patient right', (1,2), 0, True),
                    ('occlusal: X/Y, from superior', (0,1), 2, False))):
                offset = panel*400
                draw.text((offset+12, 40), name, fill='black')
                uv = (vertices[:,axes]-center[list(axes)])*scale
                uv[:,1] *= -1
                uv += [offset+200, 260]
                depths = vertices[faces,depth].mean(1)
                order = np.argsort(depths)
                if reverse:
                    order = order[::-1]
                for face in order:
                    label = int(labels[face])
                    color = (170,170,170) if label == 0 else tuple(int(c*255) for c in colorsys.hsv_to_rgb((label*0.61803398875)%1, .55, .85))
                    draw.polygon([tuple(p) for p in uv[faces[face]]], fill=color)
                observed_uv = (observed[:,axes]-center[list(axes)])*scale
                observed_uv[:,1] *= -1
                observed_uv += [offset+200, 260]
                for x,y in observed_uv[::max(1,len(observed_uv)//600)]:
                    draw.point((int(x),int(y)), fill=(20,20,20))
            draw.text((16, 455), 'Fixed axes; shared scale. Black: evaluation scan samples. Face colors: dominant FDI. Gum: gray.', fill='black')
        image.save(output/f'case_{index:04d}.png')
    return len(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(dict(previews=preview(args.evaluation, args.output), output=args.output)))
