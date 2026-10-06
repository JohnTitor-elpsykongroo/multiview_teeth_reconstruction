"""Add cropped tooth review sheets and an occlusal view to a forward run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from run_forward_check import (colorize, load_ply, make_cameras, panel_image,
                               rasterize)


def zoom(image: Image.Image, labels: np.ndarray) -> Image.Image:
    yy, xx = np.nonzero(labels)
    if not len(xx):
        raise ValueError("empty teeth-only mask")
    padding = 18
    x0 = max(0, int(xx.min()) - padding)
    x1 = min(image.width, int(xx.max()) + padding + 1)
    y0 = max(0, int(yy.min()) - padding)
    y1 = min(image.height, int(yy.max()) + padding + 1)
    crop = image.crop((x0, y0, x1, y1))
    fitted = ImageOps.contain(crop, (640, 360), method=Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (640, 360), (0, 0, 0))
    canvas.paste(fitted, ((640 - fitted.width) // 2, (360 - fitted.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "FORWARD_RENDERED_VISUAL_REVIEW_REQUIRED":
        raise ValueError(f"unexpected forward status: {report['status']}")
    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    variants = ("zero", "training_case")
    for variant in variants:
        semantic_panels = []
        shaded_panels = []
        for camera in report["cameras"]:
            name = camera["name"]
            folder = root / "renders" / variant
            label_map = np.asarray(Image.open(folder / f"{name}_teeth_labels.png"))
            semantic = Image.open(folder / f"{name}_teeth_semantic.png").convert("RGB")
            shaded = Image.open(folder / f"{name}_teeth_shaded.png").convert("RGB")
            semantic_panels.append((f"{variant} {name}", zoom(semantic, label_map)))
            shaded_panels.append((f"{variant} {name}", zoom(shaded, label_map)))
        panel_image(semantic_panels, root / f"{variant}_teeth_semantic_zoom.png")
        panel_image(shaded_panels, root / f"{variant}_teeth_shaded_zoom.png")

    review_config = dict(config)
    review_config["camera_elevation_degrees"] = 65
    review_config["camera_azimuth_degrees"] = [0]
    mesh_sets = {}
    for variant in variants:
        folder = root / "meshes" / variant
        mesh_sets[variant] = {
            int(label): load_ply(folder / ("gum.ply" if int(label) == 0 else f"tooth{label}.ply"))
            for label in report["variants"][variant]
        }
    camera = make_cameras(mesh_sets["zero"], review_config)[0]
    camera["name"] = "occlusal_65deg"
    (root / "review_camera.json").write_text(json.dumps(camera, indent=2), encoding="utf-8")
    semantic_panels = []
    shaded_panels = []
    fdi_panels = []
    pixel_counts = {}
    for variant in variants:
        label_map, depth, shade, info = rasterize(mesh_sets[variant], camera, include_gum=False)
        folder = root / "renders" / variant
        Image.fromarray(label_map, "L").save(folder / "occlusal_65deg_teeth_labels.png")
        np.savez_compressed(folder / "occlusal_65deg_teeth_depth.npz", depth=depth)
        semantic = colorize(label_map)
        shaded = Image.fromarray(shade, "L").convert("RGB")
        semantic.save(folder / "occlusal_65deg_teeth_semantic.png")
        shaded.save(folder / "occlusal_65deg_teeth_shaded.png")
        semantic_panels.append((variant, zoom(semantic, label_map)))
        shaded_panels.append((variant, zoom(shaded, label_map)))
        annotated = semantic.copy()
        draw = ImageDraw.Draw(annotated)
        R = np.asarray(camera["R_world_to_camera"])
        t = np.asarray(camera["t_world_to_camera"])
        K = np.asarray(camera["K"])
        for label, (xyz, _) in mesh_sets[variant].items():
            if label == 0:
                continue
            center_cam = R @ xyz.mean(axis=0) + t
            if center_cam[2] <= 0:
                continue
            pixel = K @ (center_cam / center_cam[2])
            u, v = int(round(pixel[0])), int(round(pixel[1]))
            draw.ellipse((u - 11, v - 10, u + 11, v + 10), fill=(0, 0, 0))
            draw.text((u - 7, v - 6), str(label), fill=(255, 255, 255))
        fdi_panels.append((variant, zoom(annotated, label_map)))
        pixel_counts[variant] = info["visible_pixel_counts"]
    panel_image(semantic_panels, root / "occlusal_semantic_review.png")
    panel_image(shaded_panels, root / "occlusal_shaded_review.png")
    panel_image(fdi_panels, root / "occlusal_fdi_review.png")
    (root / "review_view_metrics.json").write_text(
        json.dumps({"camera": camera, "visible_pixel_counts": pixel_counts}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "REVIEW_VIEWS_RENDERED", "run_root": str(root)}))


if __name__ == "__main__":
    main()
