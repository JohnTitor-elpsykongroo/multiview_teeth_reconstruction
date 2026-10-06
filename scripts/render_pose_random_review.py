"""Keep the complete image viewport in random-pose visual review artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", type=Path)
    args = parser.parse_args()
    root = args.batch_root.resolve(strict=True)
    summary = json.loads((root / "review" / "summary.json").read_text())
    review = root / "review"
    selected = []
    for level in summary["levels"]:
        measured = [row for row in summary["trials"] if row["level"] == level["id"] and "final_iou" in row]
        if measured:
            selected.append(min(measured, key=lambda row: row["final_iou"]))
    width, image_height, block_height = 1440, 383, 810
    contact = Image.new("RGB", (width, len(selected) * block_height), "#f4f5f7")
    for index, row in enumerate(selected):
        block = Image.new("RGB", (width, block_height), "#f4f5f7")
        draw = ImageDraw.Draw(block)
        draw.text((10, 5), f"{row['trial']} | initial {row['initial_rotation_degrees']:.2f} deg, translation {row['initial_translation_dmm']:.4f} DMM | IoU {row['initial_iou']:.4f} -> {row['final_iou']:.4f}", fill="black")
        for stage_index, stage in enumerate(["initial", "final"]):
            image = Image.open(Path(row["fit_root"]) / "renders" / stage / "overlay_sheet.png").convert("RGB")
            block.paste(image.resize((width, image_height), Image.Resampling.LANCZOS), (0, 30 + stage_index * image_height))
            draw.text((5, 57 + stage_index * image_height), stage.upper(), fill="white")
        block.save(review / f"{row['level']}_full_comparison.png")
        contact.paste(block, (0, index * block_height))
    contact.save(review / "full_comparison.png")
    # Check every final semantic image, with two trials per row and all cameras
    # included. This complements occupancy overlays with tooth ID inspection.
    measured = [row for row in summary["trials"] if "final_iou" in row]
    semantics = Image.new("RGB", (1920, ((len(measured) + 1) // 2) * 280), "#f4f5f7")
    draw = ImageDraw.Draw(semantics)
    for index, row in enumerate(measured):
        x, y = (index % 2) * 960, (index // 2) * 280
        draw.text((x + 5, y + 5), row["trial"], fill="black")
        image = Image.open(Path(row["fit_root"]) / "renders" / "final" / "semantic_sheet.png").convert("RGB")
        semantics.paste(image.resize((960, 255), Image.Resampling.LANCZOS), (x, y + 25))
    semantics.save(review / "all_final_semantics.png")
    record = {"selection": "lowest final IoU per predeclared level; complete image viewport retained",
              "selected_trials": [row["trial"] for row in selected],
              "semantic_trial_count": len(measured),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (review / "full_visual_selection.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
