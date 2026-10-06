"""Fresh-directory diagnostic artifacts for the native DMM renderer."""
from pathlib import Path
import colorsys

import numpy as np
from PIL import Image, ImageDraw

from dmm import CHANNEL_FDI
from dmm.provenance import source_fingerprint
from dmm.validation import require, sha256, write_json


def write_render_artifacts(output, scene, rendered, mesh, matching=None, gradients=None):
    output = Path(output)
    require(not output.exists(), "render output must be a new directory")
    require(set(rendered) == {v.view_id for v in scene.observations}, "render artifact views mismatch")
    output.mkdir(parents=True, exist_ok=False)
    def array(value):
        return value.detach().cpu().numpy()
    np.savez_compressed(output / "mesh.npz", vertices_world_mm=array(mesh.vertices_world_mm),
                        faces=array(mesh.faces), semantics=array(mesh.semantics), face_arch=array(mesh.face_arch))
    palette = np.array([[.07, .07, .09], *[colorsys.hsv_to_rgb((i * .618) % 1, .65, .95) for i in range(28)]])
    rows, diagnostics, files = [], {}, ["mesh.npz"]
    for index, obs in enumerate(scene.observations):
        item = rendered[obs.view_id]
        sem, xy, depth, coverage = map(array, (item.semantics, item.xy_pixels, item.depth_mm, item.coverage))
        name = f"view_{index:03d}.npz"
        np.savez_compressed(output / name, semantics=sem, xy_pixels=xy, depth_mm=depth,
                            coverage=coverage, face_index=array(item.face_index), K=obs.K,
                            T_camera_from_world=obs.T_camera_from_world, labels=obs.labels, valid=obs.valid)
        files.append(name)
        diagnostics[obs.view_id] = dict(file=name, visible_pixels=int((coverage > 0).sum()), **item.metadata)
        h, w = depth.shape
        mask_image = sem @ palette
        valid = np.isfinite(depth)
        shade = np.zeros_like(depth)
        if valid.any():
            shade[valid] = .25 + .7 * (depth[valid].max() - depth[valid]) / max(float(np.ptp(depth[valid])), 1e-6)
        depth_image = np.repeat(shade[..., None], 3, -1)
        xy_image = np.zeros((h, w, 3))
        xy_image[..., :2] = np.clip(xy / [max(w - 1, 1), max(h - 1, 1)], 0, 1)
        xy_image *= coverage[..., None]
        panels = []
        for title, data in (("Soft semantics", mask_image), ("Depth (gum included)", depth_image), ("Material XY: R=u G=v", xy_image)):
            panel = Image.new("RGB", (240, 266), (245, 245, 245))
            panel.paste(Image.fromarray((np.clip(data, 0, 1) * 255).astype(np.uint8)).resize((240, 240), Image.Resampling.NEAREST), (0, 26))
            ImageDraw.Draw(panel).text((6, 7), title, fill="black")
            panels.append(panel)
        row = Image.new("RGB", (720, 290), "white")
        ImageDraw.Draw(row).text((6, 6), f"View {index}: {obs.view_id}", fill="black")
        for j, panel in enumerate(panels): row.paste(panel, (j * 240, 24))
        rows.append(row)
    image = Image.new("RGB", (720, 290 * len(rows)), "white")
    for i, row in enumerate(rows): image.paste(row, (0, 290 * i))
    image.save(output / "preview.png")
    files.append("preview.png")
    report = dict(status="RENDER_DIAGNOSTIC_NOT_RECONSTRUCTION", scene_id=scene.manifest["scene_id"],
                  source_sha256=source_fingerprint(), channel_fdi=[0, *CHANNEL_FDI],
                  mesh=mesh.diagnostics, views=diagnostics, matching=matching, gradients=gradients,
                  artifacts={name: sha256(output / name) for name in files})
    write_json(output / "report.json", report)
    return report
