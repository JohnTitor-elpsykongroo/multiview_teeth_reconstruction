"""Run a reproducible and resumable random-start pose campaign on frozen inputs.

The original fitter and evaluator run from an archived code snapshot; neither
algorithm is changed. Every realized random start is saved before fitting.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE_NAMES = ["run_pose_only.py", "run_forward_check.py", "evaluate_pose_only.py",
              "run_pose_random.py", "summarize_pose_random.py"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def input_hashes(config):
    fit = Path(config["fit_input"])
    manifest_path = fit / "manifest.json"
    manifest = read_json(manifest_path)
    paths = [manifest_path, fit / manifest["camera_file"]]
    paths += [fit / item["path"] for item in manifest["masks"]]
    paths += [Path(config["fixed_meshes"]) / f"tooth{label}.ply"
              for label in manifest["visible_fdi_ids"]]
    return {str(path.resolve()): sha256(path) for path in paths}


def prepare(plan_path):
    definition = read_json(plan_path)
    base_path = (ROOT / definition["base_config"]).resolve(strict=True)
    base = read_json(base_path)
    batch = ROOT / "runs" / (dt.datetime.now(dt.timezone.utc).strftime("pose_random_%Y%m%dT%H%M%SZ_")
                              + sha256(plan_path)[:8])
    batch.mkdir(parents=True, exist_ok=False)
    snapshot = batch / "code_snapshot"
    snapshot.mkdir()
    for name in CODE_NAMES:
        shutil.copyfile(ROOT / "scripts" / name, snapshot / name)
    count = len(definition["levels"]) * int(definition["trials_per_level"])
    streams = np.random.SeedSequence(definition["seed"]).spawn(count)
    jobs = []
    for level in definition["levels"]:
        for index in range(int(definition["trials_per_level"])):
            stream = streams[len(jobs)]
            rng = np.random.default_rng(stream)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            angle = float(rng.uniform(*level["rotation_degrees"]))
            distance = float(rng.uniform(*level["translation_dmm"]))
            job_id = f"{level['id']}_{index + 1:02d}"
            job_root = batch / "jobs" / job_id
            job_root.mkdir(parents=True)
            config = copy.deepcopy(base)
            config["initial_rotation_degrees_xyz"] = (axis * angle).tolist()
            config["initial_translation_dmm"] = (direction * distance).tolist()
            config["acceptance"].update(definition.get("acceptance_overrides", {}))
            config_path = job_root / "config.json"
            write_json(config_path, config)
            jobs.append({"id": job_id, "level": level["id"], "spawn_key": list(stream.spawn_key),
                         "rotation_axis": axis.tolist(), "rotation_angle_degrees": angle,
                         "translation_direction": direction.tolist(), "translation_norm_dmm": distance,
                         "config_path": str(config_path), "config_sha256": sha256(config_path)})
    plan = {"created_at_utc": utc_now(), "definition": definition, "base_config": base,
            "base_config_sha256": sha256(base_path), "jobs": jobs,
            "random_distribution": "independent isotropic unit axes/directions; uniform angle/norm per interval",
            "rng": "NumPy PCG64 with SeedSequence independent spawned streams",
            "input_sha256": input_hashes(base),
            "code_sha256": {name: sha256(snapshot / name) for name in CODE_NAMES},
            "python_executable": sys.executable,
            "legacy_gain_threshold": base["acceptance"]["min_mean_tooth_iou_gain"],
            "acceptance": config["acceptance"]}
    write_json(batch / "plan.json", plan)
    write_json(batch / "status.json", {"status": "PREPARED", "jobs": {}, "created_at_utc": utc_now()})
    return batch, plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "configs" / "pose_random.json")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.resume:
        batch = args.resume.resolve(strict=True)
        plan = read_json(batch / "plan.json")
    else:
        batch, plan = prepare(args.plan.resolve(strict=True))
    if not batch.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("campaign must be inside this project runs directory")
    snapshot = batch / "code_snapshot"
    for name, digest in plan["code_sha256"].items():
        if sha256(snapshot / name) != digest:
            raise ValueError(f"archived execution code changed: {name}")
    for path, digest in plan["input_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"frozen campaign input changed: {path}")
    for job in plan["jobs"]:
        if sha256(job["config_path"]) != job["config_sha256"]:
            raise ValueError(f"realized trial config changed: {job['id']}")
    print(json.dumps({"status": "POSE_RANDOM_PREPARED", "batch_root": str(batch),
                      "trial_count": len(plan["jobs"])}), flush=True)
    if args.prepare_only:
        return
    # O_EXCL prevents concurrent controllers. A stale lock is reclaimable only
    # once its process has exited; completed jobs are then skipped on resume.
    lock_path = batch / "controller.lock"
    if lock_path.exists():
        lock = read_json(lock_path)
        if os.name == "nt":
            import ctypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            handle = kernel.OpenProcess(0x1000, False, int(lock["pid"]))
            if handle:
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                kernel.CloseHandle(handle)
                raise RuntimeError(f"controller PID {lock['pid']} is still running")
        else:
            try:
                os.kill(int(lock["pid"]), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("controller is still running")
        lock_path.unlink()
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created_at_utc": utc_now()}, handle)
    state = read_json(batch / "status.json")
    mutex = threading.Lock()

    def update(job_id, **details):
        with mutex:
            state["jobs"].setdefault(job_id, {}).update(details)
            state["updated_at_utc"] = utc_now()
            write_json(batch / "status.json", state)

    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

    def execute(job):
        jid = job["id"]
        job_root = batch / "jobs" / jid
        old = state["jobs"].get(jid, {})
        if old.get("terminal"):
            return
        attempts = sorted((batch / "runs").glob(f"pose_only_*_{job['config_sha256'][:8]}"))
        # Never restart a child that survived an interrupted controller.
        if old.get("child_pid") and os.name == "nt":
            import ctypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            handle = kernel.OpenProcess(0x1000, False, int(old["child_pid"]))
            if handle:
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                kernel.CloseHandle(handle)
                raise RuntimeError(f"{jid} child PID still running; resume after it exits")
        completed = [path for path in attempts if (path / "fit_report.json").exists()]
        fit_root = completed[-1] if completed else None
        try:
            if fit_root is None:
                update(jid, phase="FIT", child_pid=None, started_at_utc=utc_now(), attempts=[str(p) for p in attempts])
                with (job_root / "fit.log").open("a", encoding="utf-8") as handle:
                    process = subprocess.Popen([sys.executable, "-u", str(snapshot / "run_pose_only.py"),
                                                "--config", job["config_path"]], cwd=ROOT,
                                               stdout=handle, stderr=subprocess.STDOUT, env=env,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    update(jid, child_pid=process.pid)
                    return_code = process.wait()
                update(jid, child_pid=None)
                attempts = sorted((batch / "runs").glob(f"pose_only_*_{job['config_sha256'][:8]}"))
                if return_code != 0:
                    raise RuntimeError(f"fitter exit code {return_code}; see fit.log")
                completed = [path for path in attempts if (path / "fit_report.json").exists()]
                if not completed:
                    raise RuntimeError("fitter did not produce a completed report")
                fit_root = completed[-1]
            update(jid, phase="EVALUATE", fit_root=str(fit_root), attempts=[str(p) for p in attempts])
            evaluation_path = fit_root / "evaluation.json"
            if not evaluation_path.exists():
                with (job_root / "evaluate.log").open("a", encoding="utf-8") as handle:
                    process = subprocess.Popen([sys.executable, str(snapshot / "evaluate_pose_only.py"),
                                                str(fit_root)], cwd=ROOT, stdout=handle,
                                               stderr=subprocess.STDOUT, env=env,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    update(jid, child_pid=process.pid)
                    return_code = process.wait()
                update(jid, child_pid=None)
                if return_code != 0:
                    raise RuntimeError(f"evaluator exit code {return_code}; see evaluate.log")
            evaluation = read_json(evaluation_path)
            status = evaluation["status"]
            update(jid, status=status, phase="COMPLETED", terminal=True, finished_at_utc=utc_now())
            print(json.dumps({"trial": jid, "status": status,
                              "rotation_error_degrees": evaluation["final_rotation_error_degrees"],
                              "translation_error_dmm": evaluation["final_translation_error_dmm"],
                              "final_iou": evaluation["final_mean_tooth_iou"]}), flush=True)
        except Exception as exc:
            update(jid, status="EXECUTION_FAILED", phase="FAILED", terminal=True,
                   error=f"{type(exc).__name__}: {exc}", child_pid=None,
                   attempts=[str(p) for p in attempts], finished_at_utc=utc_now())
            print(json.dumps({"trial": jid, "status": "EXECUTION_FAILED", "error": str(exc)}), flush=True)

    try:
        state.update(status="RUNNING", controller_pid=os.getpid(), updated_at_utc=utc_now())
        write_json(batch / "status.json", state)
        with ThreadPoolExecutor(max_workers=int(plan["definition"]["workers"])) as pool:
            futures = [pool.submit(execute, job) for job in plan["jobs"]]
            for future in as_completed(futures):
                future.result()
        for path, digest in plan["input_sha256"].items():
            if sha256(path) != digest:
                raise RuntimeError(f"input changed during campaign: {path}")
        state.update(status="COMPLETED", controller_pid=None, frozen_inputs_unchanged=True,
                     finished_at_utc=utc_now())
        write_json(batch / "status.json", state)
        result = subprocess.run([sys.executable, str(snapshot / "summarize_pose_random.py"), str(batch)],
                                cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
