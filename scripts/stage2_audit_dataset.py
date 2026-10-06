"""Read-only Blender acceptance; annotations/truth only in this audit command."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tooth_observation.audit import audit_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--geometry", action="store_true", help="explicitly read truth for sampled depth-to-mesh checks")
    args = p.parse_args()
    result = audit_dataset(args.dataset, args.output, geometry=args.geometry)
    print(json.dumps({"status": result["status"], "report": str(args.output / "report.json")}, ensure_ascii=False))
