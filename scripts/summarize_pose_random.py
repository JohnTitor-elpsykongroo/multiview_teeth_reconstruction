"""Summarize every random pose trial, retaining numerical and execution failures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    return {"min": min(values), "median": float(np.median(values)), "max": max(values)} if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", type=Path)
    args = parser.parse_args()
    root = args.batch_root.resolve(strict=True)
    plan = read_json(root / "plan.json")
    state = read_json(root / "status.json")
    review = root / "review"
    review.mkdir(exist_ok=True)
    rows = []
    for job in plan["jobs"]:
        entry = state["jobs"].get(job["id"], {})
        row = {"trial": job["id"], "level": job["level"],
               "initial_rotation_degrees": job["rotation_angle_degrees"],
               "initial_translation_dmm": job["translation_norm_dmm"],
               "status": entry.get("status", "PENDING"), "fit_root": entry.get("fit_root"),
               "error": entry.get("error")}
        if entry.get("fit_root") and (Path(entry["fit_root"]) / "evaluation.json").exists():
            fit = Path(entry["fit_root"])
            evaluation = read_json(fit / "evaluation.json")
            provenance = read_json(fit / "provenance.json")
            expected_inputs = plan["input_sha256"]
            config = read_json(job["config_path"])
            fit_input = Path(config["fit_input"])
            manifest = read_json(fit_input / "manifest.json")
            assert provenance["config_sha256"] == job["config_sha256"]
            assert provenance["fit_manifest_sha256"] == expected_inputs[str((fit_input / "manifest.json").resolve())]
            assert provenance["camera_sha256"] == expected_inputs[str((fit_input / manifest["camera_file"]).resolve())]
            for item in manifest["masks"]:
                assert provenance["mask_sha256"][item["camera"]] == expected_inputs[str((fit_input / item["path"]).resolve())]
            for label, sha in provenance["fixed_mesh_sha256"].items():
                assert sha == expected_inputs[str((Path(config["fixed_meshes"]) / f"tooth{label}.ply").resolve())]
            row.update({"final_rotation_error_degrees": evaluation["final_rotation_error_degrees"],
                        "final_translation_error_dmm": evaluation["final_translation_error_dmm"],
                        "initial_iou": evaluation["initial_mean_tooth_iou"],
                        "final_iou": evaluation["final_mean_tooth_iou"],
                        "iou_gain": evaluation["mean_tooth_iou_gain"],
                        "legacy_gain_check_pass": evaluation["mean_tooth_iou_gain"] >= plan["legacy_gain_threshold"],
                        "checks": evaluation["checks"], "provenance_verified": True})
        rows.append(row)
    levels = []
    for level in plan["definition"]["levels"]:
        subset = [row for row in rows if row["level"] == level["id"]]
        measured = [row for row in subset if "final_iou" in row]
        levels.append({"id": level["id"], "rotation_range": level["rotation_degrees"],
                       "translation_range": level["translation_dmm"], "trial_count": len(subset),
                       "passes": sum(row["status"] == "POSE_ONLY_PASS" for row in subset),
                       "failures": [row["trial"] for row in subset if row["status"] != "POSE_ONLY_PASS"],
                       "rotation_error_degrees": stats([row["final_rotation_error_degrees"] for row in measured]),
                       "translation_error_dmm": stats([row["final_translation_error_dmm"] for row in measured]),
                       "final_iou": stats([row["final_iou"] for row in measured])})
    measured = [row for row in rows if "final_iou" in row]
    passes = sum(row["status"] == "POSE_ONLY_PASS" for row in rows)
    summary = {"status": "POSE_RANDOM_ALL_PASS" if passes == len(rows) else "POSE_RANDOM_HAS_FAILURES",
               "trial_count": len(rows), "pass_count": passes, "pass_rate": passes / len(rows),
               "acceptance": plan["acceptance"], "acceptance_reason": plan["definition"]["acceptance_reason"],
               "legacy_gain_threshold": plan["legacy_gain_threshold"],
               "legacy_gain_check_pass_count": sum(row.get("legacy_gain_check_pass", False) for row in rows),
               "frozen_inputs_unchanged": all(digest(path) == sha for path, sha in plan["input_sha256"].items()),
               "archived_code_unchanged": all(digest(root / "code_snapshot" / name) == sha
                                              for name, sha in plan["code_sha256"].items()),
               "initial_iou": stats([row["initial_iou"] for row in measured]),
               "final_iou": stats([row["final_iou"] for row in measured]),
               "rotation_error_degrees": stats([row["final_rotation_error_degrees"] for row in measured]),
               "translation_error_dmm": stats([row["final_translation_error_dmm"] for row in measured]),
               "levels": levels, "trials": rows,
               "scope": "one fixed DMM upper arch, three known cameras, same-model noiseless masks; sampled initialization robustness only"}
    (review / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# 固定 shape 与相机的随机位姿恢复", "", f"状态：`{summary['status']}`；通过 **{passes}/{len(rows)}**。", "",
             f"随机种子：{plan['definition']['seed']}。每档 4 组，旋转轴和平移方向独立均匀分布在单位球面，角度及平移范数在预设区间均匀采样。所有初值在拟合前写入 plan.json。", "",
             "固定原上颌 14 牙、正面及左右 30° 相机、原逐牙 FDI 掩码；原质心粗配准和 5 轮轮廓优化代码未改动。每次拟合结束后才执行真值评价。", "",
             "验收：旋转误差 ≤1°、平移误差 ≤0.02 DMM 单位、最终逐牙平均 IoU ≥0.85、IoU 不下降。原单次实验的 gain≥0.5 作为诊断记录，不用于本批次验收，因为较小初始扰动可能已经有较高 IoU。", "",
             "| 档位 | 初始旋转范围 | 初始平移范数范围 | 通过 | 最终 IoU 范围 | 最大旋转误差 | 最大平移误差 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for level in levels:
        if level["final_iou"]:
            iou = f"{level['final_iou']['min']:.4f}–{level['final_iou']['max']:.4f}"
            rot = f"{level['rotation_error_degrees']['max']:.4f}°"
            trans = f"{level['translation_error_dmm']['max']:.6f}"
        else:
            iou = rot = trans = "未测得"
        lines.append(f"| {level['id']} | {level['rotation_range']}° | {level['translation_range']} | {level['passes']}/{level['trial_count']} | {iou} | {rot} | {trans} |")
    lines += ["", "| 试验 | 初始旋转 | 初始平移 | 初始 IoU | 最终 IoU | 最终旋转误差 | 最终平移误差 | 状态 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        fit_relative = Path(row["fit_root"]).relative_to(root).as_posix() if row.get("fit_root") else None
        label = f"[{row['trial']}](../{fit_relative}/evaluation.json)" if fit_relative else row["trial"]
        if "final_iou" in row:
            lines.append(f"| {label} | {row['initial_rotation_degrees']:.2f}° | {row['initial_translation_dmm']:.4f} | {row['initial_iou']:.4f} | {row['final_iou']:.4f} | {row['final_rotation_error_degrees']:.4f}° | {row['final_translation_error_dmm']:.6f} | {row['status']} |")
        else:
            lines.append(f"| {label} | {row['initial_rotation_degrees']:.2f}° | {row['initial_translation_dmm']:.4f} | — | — | — | — | {row['status']} |")
    lines += ["", "该通过率是这 16 个预先保存的随机初值的实测结果，不是连续扰动空间的收敛保证。shape 与目标掩码来自同一生成模型，相机已知；尚不能推断未知相机、真实分割误差或 Pose+Shape 联合优化的稳定性。DMM 平移单位未换算为毫米。", ""]
    (review / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if measured:
        colors = {"small": "#419d78", "medium": "#3672b8", "large": "#dd9b35", "stress": "#bc547a"}
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for row in measured:
            color = colors[row["level"]]
            axes[0].scatter(row["initial_iou"], row["final_iou"], c=color)
            axes[1].scatter(row["initial_rotation_degrees"], row["final_rotation_error_degrees"], c=color)
            axes[2].scatter(row["initial_translation_dmm"], row["final_translation_error_dmm"], c=color)
        axes[0].axhline(0.85, color="gray", ls="--", label="acceptance")
        axes[0].set(xlabel="Initial mean tooth IoU", ylabel="Final mean tooth IoU", ylim=(0, 1.02))
        axes[1].axhline(1, color="gray", ls="--")
        axes[1].set(xlabel="Initial rotation angle (degrees)", ylabel="Final rotation error (degrees)")
        axes[2].axhline(0.02, color="gray", ls="--")
        axes[2].set(xlabel="Initial translation norm (DMM units)", ylabel="Final translation error (DMM units)")
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.suptitle(f"Fixed shape/cameras: {passes}/{len(rows)} random starts passed")
        figure.savefig(review / "metrics.png", dpi=160)
        plt.close(figure)
        selected = []
        for level in levels:
            candidates = [row for row in measured if row["level"] == level["id"]]
            if candidates:
                selected.append(min(candidates, key=lambda row: row["final_iou"]))
        # Crop white header / empty image margins while retaining all cameras.
        canvas = Image.new("RGB", (1920, len(selected) * 370), "#f4f5f7")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(selected):
            y = index * 370
            title = f"{row['trial']} | angle {row['initial_rotation_degrees']:.1f} deg, translation {row['initial_translation_dmm']:.3f} | IoU {row['initial_iou']:.3f} -> {row['final_iou']:.3f}"
            draw.text((12, y + 5), title, fill="black")
            for stage_index, stage in enumerate(["initial", "final"]):
                image = Image.open(Path(row["fit_root"]) / "renders" / stage / "overlay_sheet.png").convert("RGB")
                image = image.crop((0, 190, image.width, min(image.height, 390))).resize((1920, 165))
                canvas.paste(image, (0, y + 30 + stage_index * 165))
                draw.text((8, y + 35 + stage_index * 165), stage, fill="white")
        canvas.save(review / "worst_per_level.png")
        (review / "visual_selection.json").write_text(json.dumps({"selection": "lowest final IoU in each predeclared level", "trials": [row["trial"] for row in selected]}, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "passed": passes, "total": len(rows),
                      "summary": str(review / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
