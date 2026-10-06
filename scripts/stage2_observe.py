"""Stage-two commands; inference never accepts supervision or truth paths."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-input")
    validate.add_argument("input_manifest", type=Path)
    infer = sub.add_parser("infer")
    infer.add_argument("input_manifest", type=Path)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--long-edge", type=int, default=512)
    infer.add_argument("--confidence-threshold", type=float, default=.6)
    infer.add_argument("--margin-threshold", type=float, default=.1)
    infer.add_argument("--allow-fixture", action="store_true")
    infer.add_argument("--occlusion-threshold", type=float, default=.3)
    infer.add_argument("--allow-no-occlusion-head", action="store_true", help="legacy checkpoint; explicitly exposed-only scenes")
    train = sub.add_parser("train")
    train.add_argument("dataset", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--device", default="cpu")
    train.add_argument("--long-edge", type=int, default=512)
    train.add_argument("--seed", type=int, default=0)
    preflight = sub.add_parser("training-preflight")
    preflight.add_argument("dataset", type=Path)
    args = p.parse_args()
    if args.command == "validate-input":
        from tooth_observation.data import PhotoScene
        scene = PhotoScene(args.input_manifest)
        for name in scene.rows:
            scene.load(name)
        result = {"status": "INPUT_VALID", "scene_id": scene.manifest["scene_id"], "views": len(scene.rows)}
    elif args.command == "infer":
        from tooth_observation.model import ModelPredictor
        from tooth_observation.pipeline import observe
        predictor = ModelPredictor(args.checkpoint, args.device, args.long_edge, allow_no_occlusion_head=args.allow_no_occlusion_head)
        result = observe(args.input_manifest, args.output, predictor, args.confidence_threshold, args.margin_threshold, args.allow_fixture,
                         occlusion_threshold=args.occlusion_threshold)
        result = {"status": result["status"], "manifest": str(args.output / "manifest.json"), "views": len(result["views"])}
    else:
        from tooth_observation.training import preflight_dataset, train_baseline
        if args.command == "training-preflight":
            splits = preflight_dataset(args.dataset)
            result = {"status": "TRAINING_INPUT_VALID", "scenes": {k: len(v) for k, v in splits.items()}}
        else:
            result = train_baseline(args.dataset, args.output, args.epochs, args.device, args.long_edge, args.seed)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
