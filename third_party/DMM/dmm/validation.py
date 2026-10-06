"""Shared validation; file references are relative to their containing JSON."""
import hashlib
import json
from pathlib import Path

import numpy as np

from dmm import CONTRACT_ID, VERSION, PROFILE, TEETH


class ContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda x: (_ for _ in ()).throw(ContractError(f"nonfinite JSON: {x}")))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def header(data, kind):
    require(data.get("contract_id") == CONTRACT_ID, "contract_id mismatch")
    require(data.get("version") == VERSION, "unsupported contract version")
    require(data.get("artifact_kind") == kind, f"expected artifact_kind={kind}")
    require(data.get("representation_profile") == PROFILE, "representation profile mismatch")


def stamped(kind, **kwargs):
    return dict(contract_id=CONTRACT_ID, version=VERSION, artifact_kind=kind,
                representation_profile=PROFILE, **kwargs)


def reference(path, owner):
    import os
    return {"path": Path(os.path.relpath(Path(path).resolve(), Path(owner).resolve().parent)).as_posix(),
            "sha256": sha256(path)}


def resolve_ref(owner, ref, allowed_roots=None):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "reference requires path and sha256")
    require(isinstance(ref["path"], str) and bool(ref["path"]), "empty reference path")
    relative = Path(ref["path"])
    require(not relative.is_absolute() and not relative.drive, "references must be relative")
    path = (Path(owner).resolve().parent / relative).resolve()
    if allowed_roots is not None:
        require(any(path.is_relative_to(Path(root).resolve()) for root in allowed_roots),
                f"reference outside loader allowlist: {path}")
    require(path.is_file(), f"missing input file: {path}")
    require(isinstance(ref["sha256"], str) and sha256(path) == ref["sha256"].lower(), f"SHA256 mismatch: {path}")
    return path


def array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), f"invalid {name}: expected finite {shape}")
    return result


def rigid(value, name):
    matrix = array(value, (4, 4), name)
    rotation = matrix[:3, :3]
    require(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5, rtol=0)
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            and abs(np.linalg.det(rotation) - 1) <= 1e-5, f"{name} must be rigid without scale/reflection")
    return matrix


def presence_map(value, arch):
    require(arch in TEETH, f"unknown arch: {arch}")
    require(isinstance(value, dict) and set(value) == {str(x) for x in TEETH[arch]}, f"{arch}: presence needs all 14 FDI keys")
    require(all(type(x) is bool for x in value.values()), "presence accepts true/false only")
    return {int(k): v for k, v in value.items()}


def domain(value):
    box = array(value, (2, 3), "sampling_domain_model")
    require(np.all(box[1] > box[0]), "sampling domain needs positive extent")
    return box


def nonempty(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be a nonempty string")
    return value
