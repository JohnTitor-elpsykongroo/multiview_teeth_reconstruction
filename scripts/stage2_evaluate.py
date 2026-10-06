"""Explicit prediction/annotation evaluation, separate from inference."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tooth_observation.evaluation import evaluate_predictions

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prediction_manifest", type=Path)
    p.add_argument("--input-manifest", type=Path, required=True)
    p.add_argument("--annotations-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-fixture", action="store_true")
    a = p.parse_args()
    r = evaluate_predictions(a.prediction_manifest, a.input_manifest, a.annotations_manifest, a.output, allow_fixture=a.allow_fixture)
    print(json.dumps({"status": r["status"], "report": str(a.output / "report.json")}, ensure_ascii=False))
