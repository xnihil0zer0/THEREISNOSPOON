"""Paths, strict-JSON helpers, stream-target rules and the run_job.sh GPU dispatcher for the broadcast console."""
from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlsplit

from . import DEFAULT_RTMP_INGEST
from .chat_vj_director import CHAT_VJ_SYSTEM_PROMPT_NO_CHAT, CHAT_VJ_SYSTEM_PROMPT_WITH_CHAT

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT, STATIC_DIR = PACKAGE_DIR.parent, PACKAGE_DIR / "static"
ARTIFACTS_DIR = Path(os.environ.get("THEREISNOSPOON_ARTIFACTS", "/home/xnihil0zer0/AI-Data/THEREISNOSPOON/artifacts"))
RESEARCH = Path("/home/xnihil0zer0/AI-Data/ARBOR-Research")
RUN_JOB = RESEARCH / "adversarial_audit/tools/run_job.sh"
STUDY_V2_DIR = RESEARCH / "claude_morpho_coevolution/blackboard/fl2va_chain_videos/taguchi_v2"
BASELINE_JSON = ARTIFACTS_DIR / "study_v2_baseline_metrics.json"
ARTIFACT_TYPES = {".mp4": "video/mp4", ".png": "image/png", ".json": "application/json"}
HUD_KEYS = ("grid_16x16_ratio", "temporal_motion_mean_l1", "active_motion_frame_fraction", "temporal_accel_ratio",
            "vae_17f_pulse_ratio", "mean_laplacian_sharpness", "audio_rms_dbfs")
REPORT_KEYS = ("format_valid", "grid_16x16_pass", "quality_pass", "format_checks", "nb_frames", "fps", "duration")
MODES = {True: ("chat_vj_10_lines", "Chat VJ", CHAT_VJ_SYSTEM_PROMPT_WITH_CHAT),
         False: ("creative_and_strange_no_chat", "Creative & Strange", CHAT_VJ_SYSTEM_PROMPT_NO_CHAT)}
JOBS = {"vj": ("any", "14G", "10"), "render": ("both", "24G", "30")}
DEFAULT_AUTHOR = "@console_viewer"
Job = Callable[[str, dict[str, Any]], dict[str, Any]]


def finite(value: Any) -> Any:
    """Make a value strict-JSON safe: numpy to Python, NaN and +-inf to None, paths to strings."""
    if callable(getattr(value, "tolist", None)):
        value = value.tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return str(value) if isinstance(value, Path) else value


def read_json(path: Path) -> Any:
    with contextlib.suppress(OSError, ValueError):
        return json.loads(path.read_text())
    return None


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(finite(data), indent=2, allow_nan=False))
    os.replace(tmp, path)


def signature(path: Path) -> list[Any]:
    st = path.stat()
    return [path.name, st.st_size, st.st_mtime_ns]


def receipt_path(clip: Path) -> Path:
    return ARTIFACTS_DIR / f"{clip.stem}_quality.json"


def sheet_path(clip: Path) -> Path:
    return ARTIFACTS_DIR / f"{clip.stem}_vs_study_v2.png"


def artifact_url(path: Path) -> str:
    """Cache-busted /artifacts URL, so the player reloads when a clip is replaced under the same name."""
    return f"/artifacts/{quote(path.name)}?v={path.stat().st_mtime_ns}"


def latest_clip() -> Path | None:
    stamped = []
    for path in ARTIFACTS_DIR.glob("*.mp4"):
        with contextlib.suppress(OSError):
            stamped.append((path.stat().st_mtime_ns, path))
    return max(stamped)[1] if stamped else None


def resolve_artifact(filename: str) -> Path | None:
    """Map a request name to a regular .mp4, .png or .json directly inside ARTIFACTS_DIR, else None."""
    root = ARTIFACTS_DIR.resolve()
    path = (root / filename).resolve()
    ok = path.parent == root and path.suffix.lower() in ARTIFACT_TYPES and path.is_file()
    return path if ok and not path.name.startswith(".") else None


def normalize_lines(lines: list[Any]) -> list[dict[str, str]]:
    """Accept {author, message} rows or bare strings, trimmed to the YouTube chat limits."""
    rows = []
    for line in lines:
        row = line if isinstance(line, dict) else {"message": line}
        message = str(row.get("message") or "").strip()[:200]
        if message:
            rows.append({"author": str(row.get("author") or DEFAULT_AUTHOR).strip()[:40], "message": message})
    return rows


def stream_target(url: str | None, key: str | None) -> str | None:
    """Compose base URL + stream key. A bare default ingest without a key leaves the target unchanged."""
    url, key = (url or "").strip(), (key or "").strip()
    if any(ch.isspace() or ch in "/?#" for ch in key):
        raise ValueError("The stream key must be a single token without spaces or slashes")
    target = f"{(url or DEFAULT_RTMP_INGEST).rstrip('/')}/{key}" if key else url
    if not target or target.rstrip("/") == DEFAULT_RTMP_INGEST:
        return None
    parts = urlsplit(target)
    if parts.scheme not in ("rtmp", "rtmps") and (parts.scheme or Path(target).suffix.lower() not in (".flv", ".m3u8")):
        raise ValueError("The target must be rtmp://, rtmps:// or a local .flv or .m3u8 path")
    return target


def run_gpu_job(kind: str, request: dict[str, Any], local_job: Job) -> tuple[dict[str, Any], str]:
    """Run a job behind the mandatory run_job.sh GPU queue, or in-process where the launcher is unavailable."""
    mode, mem, minutes = JOBS[kind]
    if not RUN_JOB.exists() or os.environ.get("THEREISNOSPOON_INPROCESS_JOBS") == "1":
        return local_job(kind, request), "in-process"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".job-{kind}-", dir=ARTIFACTS_DIR) as tmp:
        req, res = Path(tmp) / "request.json", Path(tmp) / "result.json"
        write_json(req, request)
        comfy_py = Path("/mnt/ai-data/ComfyUI2/llm2/bin/python")
        py_bin = str(comfy_py if comfy_py.exists() else Path(sys.executable))
        cmd = [str(RUN_JOB), mode, "--name", f"THEREISNOSPOON-{kind}", "--mem", mem, "--minutes", minutes, "--",
               py_bin, "-m", "thereisnospoon.server", "--job", kind, str(req), str(res)]
        path = os.pathsep.join(filter(None, (str(PROJECT_ROOT), os.environ.get("PYTHONPATH"))))
        env = dict(os.environ, PYTHONPATH=path, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True,
                              text=True, timeout=3600, check=False)
        result = read_json(res)
    if not isinstance(result, dict):
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise RuntimeError(f"{kind} job exited with status {proc.returncode}: {tail}")
    return result, f"run_job.sh {mode}"


def study_v2_baseline(cached: dict[str, Any] | None, evaluate: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
    """Mean HUD metrics over every Study v2 clip, recomputed only when the reference set changes."""
    refs = sorted(STUDY_V2_DIR.glob("*.mp4"))
    if not refs:
        raise FileNotFoundError(f"No Study v2 reference clips in {STUDY_V2_DIR}")
    sig = [signature(p) for p in refs]
    if cached and cached.get("signature") == sig:
        return cached
    reports = {p.name: finite(evaluate(p)) for p in refs}
    mean = {}
    for key in HUD_KEYS:
        values = [r[key] for r in reports.values() if isinstance(r.get(key), (int, float))]
        mean[key] = sum(values) / len(values) if values else None
    per_clip = {name: {k: r.get(k) for k in HUD_KEYS + REPORT_KEYS[:3]} for name, r in reports.items()}
    baseline = {"signature": sig, "clips": [p.name for p in refs], "mean": mean, "per_clip": per_clip}
    write_json(BASELINE_JSON, baseline)
    return baseline


def quality_view(clip: Path | None, baseline: dict[str, Any] | None) -> dict[str, Any]:
    """HUD payload: the latest clip's receipt (when it still matches the file) next to the Study v2 means."""
    base = baseline or {}
    state: dict[str, Any] = {"clip": None, "clean": {}, "study_v2": base.get("mean") or {},
                             "study_v2_clips": len(base.get("clips") or []), "study_v2_dir": str(STUDY_V2_DIR)}
    receipt = read_json(receipt_path(clip)) if clip else None
    with contextlib.suppress(OSError):
        if clip and isinstance(receipt, dict) and receipt.get("signature") == signature(clip):
            clean = receipt["clean"]
            state.update({k: clean.get(k) for k in REPORT_KEYS}, clip=clip.name, measured_at=receipt["measured_at"],
                         clean={k: clean.get(k) for k in HUD_KEYS}, ffprobe_ok=clean.get("format_valid"))
    return state


def clip_view(clip: Path | None) -> dict[str, Any]:
    state: dict[str, Any] = {"filename": None, "url": None, "size_bytes": None, "sheet_url": None, "receipt_url": None}
    with contextlib.suppress(OSError):
        if clip is not None:
            state.update(filename=clip.name, url=artifact_url(clip), size_bytes=clip.stat().st_size)
            for name, path in (("sheet_url", sheet_path(clip)), ("receipt_url", receipt_path(clip))):
                state[name] = artifact_url(path) if path.is_file() else None
    return state
