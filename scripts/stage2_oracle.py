"""Use existing Blender FDI/tissue annotations, with explicit oracle provenance."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tooth_observation.oracle import package_dataset
from tooth_observation.data import read_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--presence-policy", type=Path, help="Optional explicit stage-three existence policy; never taken from truth")
    a = p.parse_args()
    r = package_dataset(a.dataset, a.output, read_json(a.presence_policy) if a.presence_policy else None)
    print(json.dumps({"status": r["status"], "scenes": len(r["scenes"]), "views": r["view_count"], "report": str(a.output / "report.json")}))
