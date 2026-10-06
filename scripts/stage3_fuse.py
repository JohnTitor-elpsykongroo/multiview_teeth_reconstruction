"""Build observation diagnostics and, with explicit inputs, a DMM candidate."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observation_fusion.contract import inference_path, read_json
from observation_fusion.pipeline import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--presence-policy", required=True, help="JSON: explicit_external or assume_all_present with origin")
    parser.add_argument("--assembly-config", help="Optional model_bundle paths and independently sourced poses; paths relative to config")
    parser.add_argument("--allow-fixture", action="store_true", help="debug only; output remains marked fixture")
    parser.add_argument("--allow-oracle", action="store_true", help="explicit Blender annotation observations; provenance remains oracle")
    args = parser.parse_args()
    config = None
    if args.assembly_config:
        path = inference_path(args.assembly_config)
        config = read_json(path)
        if config.get("model_bundle"):
            config["model_bundle"] = {k: str(inference_path(path.parent / v)) for k, v in config["model_bundle"].items()}
    report = prepare(args.observations, args.output, read_json(inference_path(args.presence_policy)),
                     config=config, allow_fixture=args.allow_fixture, allow_oracle=args.allow_oracle)
    print(f"{report['status']}: fit_ready={report['fit_ready']}; pending={','.join(report['pending'])}")
    print(str(Path(args.output).resolve() / "report.json"))


if __name__ == "__main__":
    main()
