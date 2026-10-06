import hashlib
import subprocess

from dmm import DMM_ROOT


def source_fingerprint():
    digest = hashlib.sha256()
    files = [DMM_ROOT / "train_dmm.py", DMM_ROOT / "dmm_cli.py"]
    for name in ("dmm", "networks", "training", "data", "utils", "third_party/torchmeta"):
        files.extend((DMM_ROOT / name).rglob("*.py"))
    for path in sorted(files, key=lambda p: p.relative_to(DMM_ROOT).as_posix()):
        digest.update(path.relative_to(DMM_ROOT).as_posix().encode())
        digest.update(path.read_text(encoding="utf-8-sig").encode("utf-8"))
    return digest.hexdigest()


def upstream_commit():
    import json
    import re
    record = DMM_ROOT / "UPSTREAM.json"
    if record.is_file():
        data = json.loads(record.read_text(encoding="utf-8-sig"))
        if data.get("repository") != "https://github.com/cong-yi/DMM" or not re.fullmatch(r"[0-9a-f]{40}", data.get("commit", "")):
            raise ValueError("invalid DMM upstream provenance")
        return data["commit"]
    command = ["git", "-c", f"safe.directory={DMM_ROOT.as_posix()}", "-C", str(DMM_ROOT)]
    # Never mistake the enclosing project's HEAD for the DMM donor revision.
    from pathlib import Path
    top = Path(subprocess.check_output(command + ["rev-parse", "--show-toplevel"], text=True).strip()).resolve()
    if top != DMM_ROOT.resolve():
        raise ValueError("vendored DMM needs UPSTREAM.json")
    return subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()


def project_revision():
    """Project provenance is separate from the upstream donor and source digest."""
    from pathlib import Path
    root = DMM_ROOT.parent.parent.resolve()
    command = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    try:
        top = Path(subprocess.check_output(command + ["rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL).strip()).resolve()
        if top != root: return None
        commit = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(command + ["status", "--porcelain", "--untracked-files=no"], text=True).strip())
        return dict(commit=commit, tracked_files_dirty=dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def decoder_contract():
    """Conservative executable identity of the decoder, independent of fit code.

    Full source provenance remains mandatory. Any decoder change requires an
    explicit migration; changing render/OT/optimizer code does not change weights.
    """
    files = [DMM_ROOT / "utils/math.py", DMM_ROOT / "dmm/__init__.py",
             DMM_ROOT / "dmm/validation.py", DMM_ROOT / "dmm/provenance.py"]
    for folder in ("networks", "third_party/torchmeta"):
        files.extend((DMM_ROOT / folder).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(DMM_ROOT).as_posix()):
        digest.update(path.relative_to(DMM_ROOT).as_posix().encode())
        digest.update(path.read_text(encoding="utf-8-sig").encode())
    return dict(version="coupled_decoder_api_v1", decoder_sha256=digest.hexdigest(),
                query="normalized_sigmoid_blend_present_components", model_unit_mm=50,
                statistics="per_component_cholesky_original_train_v1")


def validate_decoder_binding(metadata):
    from dmm.validation import require
    if "decoder_contract" in metadata:
        require(metadata["decoder_contract"] == decoder_contract(), "decoder compatibility mismatch")
    else:
        # Legacy bundles are not silently upgraded or waived.
        require(metadata["local_source_sha256"] == source_fingerprint(),
                "legacy bundle/runtime source mismatch; use the recorded source revision")


def training_runtime(device):
    """Versions and numeric modes needed to interpret/reproduce an optimizer run."""
    import platform
    import torch
    import numpy
    import scipy
    import skimage
    gpu = str(device).startswith('cuda')
    return dict(python=platform.python_version(), system=platform.system(),
                torch=str(torch.__version__), numpy=numpy.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
                cuda=torch.version.cuda if gpu else None, device_type='cuda' if gpu else 'cpu',
                gpu=torch.cuda.get_device_name(device) if gpu else None,
                capability=list(torch.cuda.get_device_capability(device)) if gpu else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                matmul_precision=torch.get_float32_matmul_precision())
