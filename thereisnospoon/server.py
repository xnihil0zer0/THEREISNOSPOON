"""THEREISNOSPOON FastAPI orchestrator: live chat, Chat VJ, clean 10 s clips, quality HUD and YouTube RTMP.

Serve: uvicorn thereisnospoon.server:app --host 127.0.0.1 --port 8090
GPU work (Chat VJ synthesis, H3 renders) runs as run_job.sh subprocesses of the job entry point
python -m thereisnospoon.server --job <vj|render> <request.json> <result.json>
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from . import CHANNEL_HANDLE, __version__
from .broadcast_console import BroadcastConsole, ConsoleError
from .broadcast_support import (ARTIFACT_TYPES, ARTIFACTS_DIR, STATIC_DIR, finite, latest_clip, normalize_lines,
                                resolve_artifact)
from .chat_vj_director import ChatVJDirector
from .clean_10s_generator import Clean10sClipGenerator
from .quality_metrics import compare_against_study_v2, evaluate_clip_quality
from .rtmp_broadcaster import YouTubeRTMPBroadcaster
from .youtube_chat_poller import YouTubeChatPoller


_DIRECTOR: ChatVJDirector | None = None
_GENERATOR: Clean10sClipGenerator | None = None


def execute_job(kind: str, request: dict[str, Any]) -> dict[str, Any]:
    """Body of one GPU job: Chat VJ direction, then for renders a single clean 10 s H3 clip."""
    global _DIRECTOR, _GENERATOR
    result: dict[str, Any] = {}
    if _DIRECTOR is None:
        _DIRECTOR = ChatVJDirector()
    if _GENERATOR is None:
        _GENERATOR = Clean10sClipGenerator()
    try:
        if kind == "vj" or not request.get("prompt"):
            scene = _DIRECTOR.direct(request.get("lines") or [])
            result["vj"] = {"mode": scene.mode, "chat_lines": list(scene.chat_lines), "scene": scene.scene,
                            "spoken_line": scene.spoken_line, "dialogue_tag": scene.dialogue_tag,
                            "soundscape": scene.soundscape, "music": scene.music, "prompt": scene.prompt,
                            "samples": [list(s) for s in scene.samples], "elapsed_ms": scene.elapsed_ms}
        if kind == "render":
            prompt = request.get("prompt") or result["vj"]["prompt"]
            out = _GENERATOR.generate(prompt, Path(request["out_mp4"]), seed=int(request["seed"]))
            result["clip"] = str(out)
            result["render_sec"] = _GENERATOR.last_render_sec
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def worker_main() -> int:
    """Persistent resident GPU worker loop launched through run_job.sh both."""
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        res = execute_job(str(payload.get("kind") or "render"), dict(payload.get("request") or {}))
        sys.stdout.write(json.dumps(finite(res), allow_nan=False) + "\n")
        sys.stdout.flush()
    return 0


def job_main(argv: list[str]) -> int:
    """GPU job entry point, launched through run_job.sh by broadcast_support.run_gpu_job."""
    kind, request_path, result_path = argv
    result = execute_job(kind, json.loads(Path(request_path).read_text()))
    Path(result_path).write_text(json.dumps(finite(result), allow_nan=False))
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        sys.exit(worker_main())
    if len(sys.argv) != 5 or sys.argv[1] != "--job" or sys.argv[2] not in ("vj", "render"):
        sys.exit("usage: python -m thereisnospoon.server --job <vj|render> <request.json> <result.json>")
    sys.exit(job_main(sys.argv[2:]))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


class SafeJSONResponse(JSONResponse):
    """Starlette refuses NaN and infinity, which silent audio produces for dBFS. Send them as null."""

    def render(self, content: Any) -> bytes:
        return super().render(finite(content))


console = BroadcastConsole(YouTubeChatPoller(), YouTubeRTMPBroadcaster, evaluate_clip_quality,
                           compare_against_study_v2, execute_job)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("THEREISNOSPOON_AUTOPOLL", "1") != "0":
        console.poller.start()
        console.polling = True
    console.schedule_quality(latest_clip())
    yield
    console.shutdown()


app = FastAPI(title="THEREISNOSPOON Autonomous Neural Chat VJ Broadcast Console", version=__version__,
              lifespan=lifespan, default_response_class=SafeJSONResponse)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(ConsoleError)
async def console_error(_: Request, exc: ConsoleError) -> SafeJSONResponse:
    return SafeJSONResponse({"detail": exc.detail}, status_code=exc.status)


class ChatInject(BaseModel):
    lines: list[dict[str, Any] | str] = Field(default_factory=list, max_length=50)
    author: str | None = None
    message: str | None = None
    clear: bool = False


class PreviewRequest(BaseModel):
    use_chat: bool = True


class RenderRequest(BaseModel):
    seed: int | None = Field(default=None, ge=0, lt=2 ** 31)
    fresh_prompt: bool = False


class StreamConfig(BaseModel):
    action: Literal["start", "stop", "configure"] = "configure"
    rtmp_url: str | None = Field(default=None, max_length=500)
    stream_key: str | None = Field(default=None, max_length=200)
    autoloop: bool | None = None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    """Channel, last-10 chat, latest Chat VJ prompt, current clip URL, quality vs Study v2 and RTMP telemetry."""
    return console.status()


@app.post("/api/chat/inject")
def chat_inject(req: ChatInject) -> dict[str, Any]:
    """Add lines to the last-10 window, or clear it with clear=true (then add any lines given)."""
    single = [{"author": req.author, "message": req.message}] if req.message else []
    rows = normalize_lines([*req.lines, *single])
    if not rows and not req.clear:
        raise HTTPException(400, "Send lines (or author and message), or clear=true")
    if req.clear:
        console.poller.clear_chat()
    added = [row for row in rows if console.poller.inject_chat_message(row["author"], row["message"])]
    return {"cleared": req.clear, "added": len(added), "chat": console.chat()}


@app.post("/api/chat/poll")
def chat_poll() -> dict[str, Any]:
    """Poll @THEREISNOSPOON-t4j right now, overriding the poller's own back-off timer."""
    poller = console.poller
    poller._next_poll = 0.0
    try:
        rows = poller.poll_once()
    except Exception as exc:
        raise HTTPException(502, f"Live chat poll failed: {exc}") from exc
    note = poller.last_error or ("" if poller.is_live else f"{CHANNEL_HANDLE} is not live right now")
    return {"new_messages": len(rows), "live": bool(poller.is_live), "video_id": poller.video_id or None,
            "broadcast_status": poller.broadcast_status, "note": note, "chat": console.chat()}


@app.post("/api/vj/preview-prompt")
def vj_preview(req: PreviewRequest | None = None) -> dict[str, Any]:
    """Run the ChatVJDirector on the current window (empty chat gives Creative & Strange mode)."""
    use_chat = req is None or req.use_chat
    return console.direct(console.poller.get_last_10_chat_lines() if use_chat else [])


@app.post("/api/clip/generate")
def clip_generate(req: RenderRequest | None = None) -> dict[str, Any]:
    """Queue one clean 10 s render from the latest unrendered VJ prompt, or from a fresh one."""
    req = req or RenderRequest()
    return console.start_render(req.seed, req.fresh_prompt)


@app.post("/api/quality/refresh")
def quality_refresh() -> dict[str, Any]:
    clip = latest_clip()
    started = console.schedule_quality(clip, force=True)
    what = clip.name if clip else "the Study v2 baseline"
    return {"started": started, "message": f"Measuring {what} vs Study v2" if started else "Already measuring"}


@app.post("/api/stream/configure")
def stream_configure(cfg: StreamConfig) -> dict[str, Any]:
    """Set the RTMP key or URL, then start, stop or only save the YouTube broadcaster target."""
    return console.configure_stream(cfg.action, cfg.rtmp_url, cfg.stream_key, cfg.autoloop)


@app.get("/artifacts/{filename}")
def artifact(filename: str) -> FileResponse:
    path = resolve_artifact(filename)
    if path is None:
        raise HTTPException(404, "No such .mp4, .png or .json artifact")
    return FileResponse(path, media_type=ARTIFACT_TYPES[path.suffix.lower()])
