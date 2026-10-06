"""Explicitly validate and copy a stage-3 candidate into a fresh fitting package."""
from pathlib import Path
import shutil
from dmm.scene import load_scene
from dmm.pixels import EDGE
from dmm.validation import read_json, write_json, require, resolve_ref, reference
from dmm.provenance import source_fingerprint


def prepare_candidate(candidate, output):
    source = Path(candidate).resolve()
    require(source.name == "candidate_manifest.json", "candidate manifest required")
    scene = load_scene(source, "cpu", allow_candidate=True)
    require(all(v.metadata.get("pixel_convention") == EDGE for v in scene.observations), "candidate must declare edge pixel convention")
    output = Path(output).resolve()
    require(not output.exists(), "scene package output already exists")
    # Traverse explicit JSON references only; never scan/copy truth or annotations.
    roots = [source.parent, source.parent.parent/"model_bundle", source.parent.parent/"initialization"]
    copied = set()
    def copy_file(path):
        relative = path.relative_to(source.parent.parent)
        target = output/relative
        if path in copied: return
        copied.add(path)
        if path.suffix == ".json":
            data = read_json(path)
            def walk(value):
                if isinstance(value, dict):
                    if set(value) == {"path", "sha256"}: copy_file(resolve_ref(path, value, roots))
                    else:
                        for child in value.values(): walk(child)
                elif isinstance(value, list):
                    for child in value: walk(child)
            walk(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    copy_file(source)
    data = read_json(source)
    data.update(status="INPUT_VALIDATED_NOT_RECONSTRUCTED", fit_ready=True,
                preparation=dict(source_sha256=source_fingerprint(), source_candidate=reference(source, output/"fit_input/manifest.json"),
                                 scope="explicit_pixel_compatibility; no_quality_claim"))
    write_json(output/"fit_input/manifest.json", data)
    # The preparation provenance reference is not read by the inference loader.
    load_scene(output/"fit_input/manifest.json", "cpu")
    return dict(status="INPUT_VALIDATED_NOT_RECONSTRUCTED", manifest=str(output/"fit_input/manifest.json"))
