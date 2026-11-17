"""Unified background training and benchmark task launcher.

The five ``train_*`` MCP tools and the Qlib benchmark runner share one shape:
validate a repository or YAML input, build an external command, run it in a
background thread, and let the caller poll progress later.  Previously each tool
duplicated this logic and carried an unused FastAPI service whose endpoints the
MCP path could never reach, so a launched job had no way to report its status.
This module is the single, functional implementation: it builds the exact same
external commands as before, but stores every task in a process-level registry
that ``training_get_task`` can read back, and it removes the dead HTTP layer.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

_MAX_LOG_LINES = 2000

_tasks: dict[str, dict[str, Any]] = {}
_tasks_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

def _task_snapshot(task: dict[str, Any], *, tail: int) -> dict[str, Any]:
    log = task["log"]
    lines = list(log)
    tail = max(0, int(tail))
    return {
        "task_id": task["task_id"],
        "tool": task["tool"],
        "status": task["status"],
        "command": task["command"],
        "created_at": task["created_at"],
        "pid": task.get("pid"),
        "exit_code": task.get("exit_code"),
        "error": task.get("error"),
        "log_tail": lines[-tail:] if tail else [],
        "output": task.get("output"),
    }


def get_training_status(task_id: str, *, tail: int = 50) -> dict[str, Any]:
    """Return a task snapshot, or a ``not_found`` notice for an unknown id."""
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return {
            "status": "not_found",
            "task_id": task_id,
            "tool": None,
            "command": None,
            "created_at": None,
            "pid": None,
            "exit_code": None,
            "error": "no such task id",
            "log_tail": [],
            "output": None,
        }
    return _task_snapshot(task, tail=tail)


def list_training_tasks() -> list[dict[str, Any]]:
    """Return a launch-only summary of all known tasks, newest first."""
    with _tasks_lock:
        tasks = list(_tasks.values())
    summaries = [
        {
            "task_id": t["task_id"],
            "tool": t["tool"],
            "status": t["status"],
            "created_at": t["created_at"],
        }
        for t in tasks
    ]
    summaries.sort(key=lambda row: row["created_at"] or "", reverse=True)
    return summaries


# ---------------------------------------------------------------------------
# External command construction (preserves the original per-tool CLI shape)
# ---------------------------------------------------------------------------

def _repo(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default)).resolve()


def _resolve_config_file(repo: Path, config_file: str | None) -> Path | None:
    if not config_file:
        return None
    path = Path(config_file)
    if not path.is_absolute():
        path = repo / path
    return path


def _build_qcm_command(repo: Path, options: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(repo / "train_qcm.py"),
        "--model", str(options.get("model", "qrdqn")),
        "--seed", str(options.get("seed", 0)),
        "--pool", str(options.get("pool", 20)),
        "--std-lam", str(options.get("std_lam", 1.0)),
    ]


def _build_aff_command(repo: Path, options: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, str(repo / "train_AFF.py")]
    config_file = _resolve_config_file(repo, options.get("config_file"))
    if config_file is not None:
        if not config_file.exists():
            raise ValueError(f"Configuration file does not exist: {config_file}")
        cmd.extend(["--config", str(config_file)])
    config_params = options.get("config_params") or {}
    for key, value in config_params.items():
        cmd.append(f"--{key}={value}")
    return cmd


def _build_gfn_command(repo: Path, options: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, str(repo / "train_gfn.py")]
    config_file = _resolve_config_file(repo, options.get("config_file"))
    if config_file is not None:
        if not config_file.exists():
            raise ValueError(f"Configuration file does not exist: {config_file}")
        cmd.extend(["--config", str(config_file)])
    for key, value in (options.get("config_params") or {}).items():
        if value is None:
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _build_gp_command(repo: Path, options: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, str(repo / "train_GP.py")]
    config_file = _resolve_config_file(repo, options.get("config_file"))
    if config_file is not None and config_file.exists():
        cmd.extend(["--config", str(config_file)])
    for key, value in (options.get("config_params") or {}).items():
        arg_name = str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{arg_name}")
        else:
            cmd.extend([f"--{arg_name}", str(value)])
    return cmd


def _build_ppo_command(repo: Path, options: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, str(repo / "train_ppo.py")]
    config_file = _resolve_config_file(repo, options.get("config_file"))
    if config_file is not None and config_file.exists():
        cmd.extend(["--config", str(config_file)])
    for key, value in (options.get("config_params") or {}).items():
        if not str(key).replace("_", "").isalnum():
            raise ValueError(f"Invalid parameter name: {key}")
        cmd.extend([f"--{key}", str(value)])
    return cmd


_SPECS: dict[str, dict[str, Any]] = {
    "train_qcm": {
        "label": "AlphaQCM",
        "repo": ("FTU_ALPHAQCM_REPO", "external/AlphaQCM"),
        "pythonpath_dirs": (),
        "build": _build_qcm_command,
    },
    "train_AFF": {
        "label": "AlphaForge",
        "repo": ("FTU_ALPHAFORGE_REPO", "external/AlphaForge"),
        "pythonpath_dirs": (),
        "build": _build_aff_command,
    },
    "train_gfn_AlphaSAGE": {
        "label": "AlphaSAGE",
        "repo": ("FTU_ALPHASAGE_REPO", "external/AlphaSAGE"),
        "pythonpath_dirs": ("src",),
        "build": _build_gfn_command,
    },
    "train_GP_AlphaSAGE": {
        "label": "AlphaSAGE",
        "repo": ("FTU_ALPHASAGE_REPO", "external/AlphaSAGE"),
        "pythonpath_dirs": ("src",),
        "build": _build_gp_command,
    },
    "train_PPO_AlphaSAGE": {
        "label": "AlphaSAGE",
        "repo": ("FTU_ALPHASAGE_REPO", "external/AlphaSAGE"),
        "pythonpath_dirs": ("src",),
        "build": _build_ppo_command,
    },
}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(task_id: str, cwd: Path, cmd: list[str], spec: dict[str, Any]) -> None:
    env = os.environ.copy()
    pythonpath = str(cwd)
    for extra in spec.get("pythonpath_dirs") or ():
        pythonpath = str(cwd / extra) + os.pathsep + pythonpath
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        with _tasks_lock:
            _tasks[task_id]["pid"] = proc.pid
        for line in proc.stdout:
            line = line.rstrip()
            with _tasks_lock:
                _tasks[task_id]["log"].append(line)
        code = proc.wait()
        with _tasks_lock:
            _tasks[task_id]["exit_code"] = code
            _tasks[task_id]["status"] = "completed" if code == 0 else "failed"
            if code != 0:
                _tasks[task_id]["error"] = f"training process exit code: {code}"
    except Exception as exc:  # noqa: BLE001 - task registry must always finalize
        with _tasks_lock:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = f"training task exception: {exc}"


def _spawn(tool: str, label: str, cmd: list[str], cwd: Path, pythonpath_dirs: tuple[str, ...]) -> dict[str, Any]:
    task_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    with _tasks_lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "tool": tool,
            "status": "running",
            "command": " ".join(cmd),
            "created_at": created_at,
            "pid": None,
            "exit_code": None,
            "error": None,
            "log": deque(maxlen=_MAX_LOG_LINES),
            "output": None,
        }
    spec = {"pythonpath_dirs": pythonpath_dirs}
    threading.Thread(target=_worker, args=(task_id, cwd, cmd, spec), daemon=True).start()
    return {
        "status": "launched",
        "tool": tool,
        "task_id": task_id,
        "command": " ".join(cmd),
        "message": f"{label} training launched in the background",
        "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# Public launchers
# ---------------------------------------------------------------------------

def launch_training(
    name: str, *, config_file: str | None = None, config_params: dict[str, Any] | None = None,
    task_name: str = "training", **fixed: Any,
) -> dict[str, Any]:
    """Validate a training repository and start its external training script."""
    spec = _SPECS[name]
    repo = _repo(*spec["repo"])
    if not repo.exists():
        raise FileNotFoundError(
            f"{spec['label']} training repository does not exist: {repo}. "
            f"The {name} tool wraps an external research project that is not bundled; "
            f"clone it into this path or set the matching FTU_*_REPO variable."
        )
    options: dict[str, Any] = {"config_file": config_file, "config_params": config_params}
    options.update(fixed)
    cmd = spec["build"](repo, options)
    return _spawn(name, spec["label"], cmd, repo, spec["pythonpath_dirs"])


def launch_qlib_benchmark(
    *, yaml_path: str, provider_uri: str | None = None, experiment_name: str | None = None,
    task_name: str = "qlib_training",
) -> dict[str, Any]:
    """Run a Qlib benchmark YAML through ``qlib.cli.run`` in the background."""
    if not yaml_path:
        raise ValueError("yaml_path is required")
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")

    if provider_uri:
        _update_yaml_provider_uri(yaml_file, provider_uri)

    cmd = [sys.executable, "-m", "qlib.cli.run", str(yaml_file)]
    if experiment_name:
        cmd.extend(["--experiment_name", str(experiment_name)])
    cwd = yaml_file.parent
    return _spawn("qlib_benchmark_runner", "Qlib benchmark", cmd, cwd, ())


def _update_yaml_provider_uri(yaml_file: Path, provider_uri: str) -> tuple[str, str]:
    import yaml

    config = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
    old_uri = (config.get("qlib_init") or {}).get("provider_uri", "Not set")
    config.setdefault("qlib_init", {})["provider_uri"] = provider_uri
    with open(yaml_file, "w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False, allow_unicode=True)
    return old_uri, provider_uri


def list_qlib_models() -> dict[str, list[str]]:
    """Scan the configured Qlib benchmarks directory for model YAML files."""
    benchmarks_dir = Path(os.environ.get("FTU_QLIB_BENCHMARKS", "external/qlib_benchmark/benchmarks"))
    models: dict[str, list[str]] = {}
    if not benchmarks_dir.exists():
        return models
    for model_dir in sorted(benchmarks_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        yaml_files = sorted(str(f) for f in model_dir.glob("workflow_config_*.yaml"))
        if yaml_files:
            models[model_dir.name] = yaml_files
    return models
