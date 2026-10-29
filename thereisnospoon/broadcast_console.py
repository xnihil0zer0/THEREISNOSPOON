"""Shared state of the THEREISNOSPOON broadcast console: chat window, Chat VJ, renders, quality HUD and RTMP."""
from __future__ import annotations

import contextlib
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from . import CHANNEL_HANDLE, CHANNEL_ID, CHANNEL_LIVE_URL, CHANNEL_URL, STUDIO_CHANNEL_URL, __version__
from .broadcast_support import (ARTIFACTS_DIR, BASELINE_JSON, MODES, RUN_JOB, STUDY_V2_DIR, Job, clip_view, finite,
                                latest_clip, normalize_lines, quality_view, read_json, receipt_path, run_gpu_job,
                                sheet_path, signature, stop_resident_worker, stream_target, study_v2_baseline,
                                write_json)

MAX_RING_CLIPS = 24


class ConsoleError(Exception):
    """A request the console refuses, carrying the HTTP status the API should answer with."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def _short(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


class BroadcastConsole:
    """Owns the poller and broadcaster, dispatches GPU jobs and keeps every panel's latest state."""

    def __init__(self, poller: Any, broadcaster_factory: Callable[..., Any], evaluate: Callable[[Path], dict],
                 compare: Callable[..., dict], local_job: Job) -> None:
        self.poller, self.broadcaster_factory, self.broadcaster = poller, broadcaster_factory, broadcaster_factory()
        self.evaluate, self.compare, self.local_job = evaluate, compare, local_job
        self.lock, self.stream_lock = threading.RLock(), threading.Lock()
        self.renders = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tins-render")
        self.measurer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tins-quality")
        self.started, self.polling, self.rtmp_target = time.time(), False, None
        default_auto = os.environ.get("THEREISNOSPOON_AUTOPOLL", "1")
        self.autoloop = os.environ.get("THEREISNOSPOON_AUTOLOOP", default_auto) != "0"
        self._stop_loop = threading.Event()
        self.clips_rendered = 0
        self.vj: dict[str, Any] = {"busy": False, "error": None}
        self.clip_job: dict[str, Any] = {
            "busy": False, "stage": "idle", "error": None,
            "clips_rendered": 0, "last_render_sec": None, "realtime_factor_x": None,
        }
        self.quality: dict[str, Any] = {"busy": False, "error": None, "failed": None}
        self.baseline = read_json(BASELINE_JSON)
        self.health = dict.fromkeys(("ChatVJDirector", "Clean10sClipGenerator", "evaluate_clip_quality"), "not loaded")
        self._seed_initial_state()
        threading.Thread(target=self._autoloop_worker, name="tins-autoloop", daemon=True).start()

    def _seed_initial_state(self) -> None:
        rep = read_json(ARTIFACTS_DIR / "first_verification_test_report.json")
        if not isinstance(rep, dict) or not isinstance(rep.get("mode_b_last_10_chat_vj"), dict):
            return
        mb = rep["mode_b_last_10_chat_vj"]
        if not self.poller.get_last_10_chat_lines():
            for row in normalize_lines(list(mb.get("chat_lines") or [])):
                self.poller.inject_chat_message(row["author"], row["message"])
        self.set_vj({"mode": "chat", "chat_lines": list(mb.get("chat_lines") or []), "scene": mb.get("scene", ""),
                     "spoken_line": mb.get("spoken_line", ""), "dialogue_tag": mb.get("dialogue_tag", ""),
                     "soundscape": mb.get("soundscape", ""), "music": mb.get("music", ""),
                     "prompt": mb.get("full_prompt", ""), "samples": [], "elapsed_ms": mb.get("elapsed_ms", 0.0)},
                    float(mb.get("elapsed_ms", 0.0)) / 1000.0, "run_job.sh both")
        self.vj["rendered"] = "clean_10s_chat_vj_test.mp4"
        self.health["Clean10sClipGenerator"] = "ok"
        self.health["evaluate_clip_quality"] = "ok"

    def _autoloop_worker(self) -> None:
        while not self._stop_loop.wait(0.4):
            if not self.autoloop or self.clip_job["busy"] or self.vj.get("busy"):
                continue
            if self.broadcaster.is_streaming and int(self.broadcaster.get_telemetry().get("queued_clips") or 0) >= 2:
                continue
            with contextlib.suppress(Exception):
                self.poller.poll_once()
            with contextlib.suppress(Exception):
                self.start_render(None, True)

    @staticmethod
    def _evict_old_clips() -> None:
        mp4s = sorted(
            (p for p in ARTIFACTS_DIR.glob("thereisnospoon_*.mp4") if not p.name.startswith(".")),
            key=lambda p: p.stat().st_mtime_ns,
        )
        for old in mp4s[:-MAX_RING_CLIPS]:
            for target in (old, receipt_path(old), sheet_path(old)):
                with contextlib.suppress(OSError):
                    target.unlink(missing_ok=True)

    def chat(self) -> dict[str, Any]:
        lines = self.poller.get_last_10_chat_lines()
        mode, persona, _ = MODES[bool(lines)]
        return {"lines": lines, "count": len(lines), "mode": mode, "persona": persona}

    def set_vj(self, payload: dict[str, Any], elapsed: float, source: str) -> dict[str, Any]:
        mode, persona, system = MODES[payload["mode"] == "chat"]
        self.vj = {**payload, "director_mode": payload["mode"], "mode": mode, "persona": persona,
                   "system_prompt": system, "dialogue": payload["spoken_line"], "generated_at": time.time(),
                   "elapsed_sec": round(elapsed, 2), "source": f"ChatVJDirector via {source}",
                   "busy": False, "error": None, "rendered": None}
        self.health["ChatVJDirector"] = "ok"
        return self.vj

    def vj_state(self, chat: dict[str, Any]) -> dict[str, Any]:
        if self.vj.get("scene"):
            return self.vj
        mode, persona, system = MODES[chat["count"] > 0]
        return {**self.vj, "mode": mode, "persona": persona, "system_prompt": system}

    def direct(self, lines: list[Any]) -> dict[str, Any]:
        """Synchronously synthesize a fresh scene, spoken line and H3 prompt from the given chat lines."""
        with self.lock:
            if self.vj.get("busy"):
                raise ConsoleError(409, "The Chat VJ is already synthesizing a prompt")
            self.vj.update(busy=True, error=None)
        t0 = time.time()
        try:
            result, source = run_gpu_job("vj", {"lines": lines}, self.local_job)
            if result.get("error"):
                raise RuntimeError(result["error"])
            return self.set_vj(result["vj"], time.time() - t0, source)
        except Exception as exc:
            self.health["ChatVJDirector"] = self.vj["error"] = f"Chat VJ failed: {exc}"[:300]
            raise ConsoleError(502, self.vj["error"]) from exc
        finally:
            self.vj["busy"] = False

    def start_render(self, seed: int | None, fresh_prompt: bool) -> dict[str, Any]:
        with self.lock:
            if self.clip_job["busy"]:
                raise ConsoleError(409, "A 10 s clip is already rendering")
            stage_msg = "rendering on resident dual GPUs (ARBOR-Zero + ARBOR-Aleph Quotient Uncollapse)"
            self.clip_job.update(busy=True, stage=stage_msg, error=None)
        seed = secrets.randbelow(2 ** 31 - 1) if seed is None else seed
        prompt = None if fresh_prompt or self.vj.get("rendered") else self.vj.get("prompt")
        self.renders.submit(self.render, seed, prompt, self.poller.get_last_10_chat_lines())
        source = "the latest Chat VJ prompt" if prompt else "a fresh Chat VJ prompt"
        return {"message": f"Rendering a 10 s clip (seed {seed}) from {source}", "seed": seed}

    def render(self, seed: int, prompt: str | None, lines: list[Any]) -> None:
        seq = self.clips_rendered + 1
        out = ARTIFACTS_DIR / f"thereisnospoon_{time.strftime('%Y%m%d_%H%M%S')}_{seq:04d}_s{seed}.mp4"
        request = {"prompt": prompt, "lines": lines, "out_mp4": str(out), "seed": seed}
        t0 = time.time()
        try:
            result, source = run_gpu_job("render", request, self.local_job)
            elapsed = round(float(result.get("render_sec") or (time.time() - t0)), 2)
            if result.get("vj"):
                self.set_vj(result["vj"], float(result["vj"].get("elapsed_ms", 0.0)) / 1000.0, source)
            if result.get("error"):
                raise RuntimeError(result["error"])
            clip = Path(result["clip"])
            self.clips_rendered += 1
            rt_x = round(10.0 / max(elapsed, 0.05), 2)
            self.clip_job.update(clips_rendered=self.clips_rendered, last_render_sec=elapsed, realtime_factor_x=rt_x)
            self.health["Clean10sClipGenerator"] = "ok"
            if self.vj.get("prompt") == (prompt or result["vj"]["prompt"]):
                self.vj["rendered"] = clip.name
            self._evict_old_clips()
        except Exception as exc:
            self.health["Clean10sClipGenerator"] = self.clip_job["error"] = f"Render failed: {exc}"[:300]
            return
        finally:
            self.clip_job.update(busy=False, stage="idle")
        self.schedule_quality(clip)
        if self.broadcaster.is_streaming:
            try:
                self.broadcaster.enqueue_clip(clip)
            except (BufferError, ValueError, RuntimeError) as exc:
                self.clip_job["error"] = f"Rendered {clip.name} but the RTMP enqueue failed: {exc}"

    def measure(self, clip: Path | None) -> None:
        """Refresh the Study v2 baseline, then compare one clip and write its receipt and contact sheet."""
        key = signature(clip) if clip else "baseline"
        try:
            self.baseline = base = study_v2_baseline(self.baseline, self.evaluate)
            if clip is not None:
                refs = [STUDY_V2_DIR / base["clips"][0], STUDY_V2_DIR / base["clips"][-1]]
                sheet = sheet_path(clip)
                report = finite(self.compare(clip, refs, sheet)["clean"])
                write_json(receipt_path(clip), {"clip": clip.name, "signature": key, "measured_at": time.time(),
                                                "clean": report, "study_v2_mean": base["mean"],
                                                "study_v2_clips": base["clips"], "contact_sheet": sheet.name})
            self.health["evaluate_clip_quality"] = "ok"
            self.quality.update(error=None, failed=None)
        except Exception as exc:
            self.health["evaluate_clip_quality"] = _short(exc)
            self.quality.update(error=f"Quality measurement failed: {exc}"[:300], failed=key)
        finally:
            self.quality["busy"] = False

    def schedule_quality(self, clip: Path | None, force: bool = False) -> bool:
        """Queue a measurement when the baseline is missing or the clip has no matching receipt."""
        with self.lock:
            if self.quality["busy"]:
                return False
            try:
                key = signature(clip) if clip else "baseline"
                stale = clip is not None and (read_json(receipt_path(clip)) or {}).get("signature") != key
            except OSError:
                return False
            if not force and (self.quality["failed"] == key or not (stale or self.baseline is None)):
                return False
            self.quality["busy"] = True
        self.measurer.submit(self.measure, clip)
        return True

    def configure_stream(self, action: str, rtmp_url: str | None, stream_key: str | None,
                         autoloop: bool | None = None) -> dict[str, Any]:
        """Set the RTMP target and start or stop the broadcaster. The unmasked target never leaves this object."""
        with self.stream_lock:
            try:
                target = stream_target(rtmp_url, stream_key)
                if action == "stop":
                    self.broadcaster.stop()
                if target is not None and target != self.rtmp_target:
                    if self.broadcaster.is_streaming:
                        raise ConsoleError(409, "Stop the broadcast before changing the RTMP target")
                    self.broadcaster, self.rtmp_target = self.broadcaster_factory(output_url=target), target
                if action == "start" and not self.broadcaster.is_streaming:
                    clip = latest_clip()
                    if clip is None:
                        raise ConsoleError(409, "Render a 10 s clip before starting the broadcast")
                    self.broadcaster.start(initial_clip=clip)
                    self.autoloop = True if autoloop is None else bool(autoloop)
                    rtmp, deadline = self.broadcaster, time.monotonic() + 3.0
                    while not (rtmp.is_streaming or rtmp.last_error) and time.monotonic() < deadline:
                        time.sleep(0.1)
                elif autoloop is not None:
                    self.autoloop = bool(autoloop)
            except (ValueError, BufferError) as exc:
                raise ConsoleError(400, str(exc)) from exc
            except (RuntimeError, TimeoutError) as exc:
                raise ConsoleError(503, str(exc)) from exc
            return {"action": action, "autoloop": self.autoloop, "telemetry": self.broadcaster.get_telemetry()}

    def components(self) -> dict[str, str]:
        poller, rtmp = self.poller, self.broadcaster
        return {"YouTubeChatPoller": poller.last_error or ("ok" if self.polling else "not loaded"), **self.health,
                "YouTubeRTMPBroadcaster": rtmp.last_error or ("ok" if rtmp.is_streaming else "not loaded"),
                "run_job.sh": "ok" if RUN_JOB.exists() else "missing, GPU jobs run in-process"}

    def status(self) -> dict[str, Any]:
        chat, clip = self.chat(), latest_clip()
        self.schedule_quality(clip)
        quality = {**quality_view(clip, self.baseline), "busy": self.quality["busy"], "error": self.quality["error"]}
        return finite({
            "channel": {"handle": CHANNEL_HANDLE, "id": CHANNEL_ID, "url": CHANNEL_URL, "live_url": CHANNEL_LIVE_URL,
                        "studio_url": STUDIO_CHANNEL_URL, "live": bool(self.poller.is_live),
                        "video_id": self.poller.video_id or None},
            "chat": chat, "vj": self.vj_state(chat), "clip": {**clip_view(clip), **self.clip_job}, "quality": quality,
            "rtmp": {**self.broadcaster.get_telemetry(), "autoloop": self.autoloop}, "components": self.components(),
            "server": {"uptime_sec": time.time() - self.started, "version": __version__},
        })

    def shutdown(self) -> None:
        self._stop_loop.set()
        self.autoloop = False
        stop_resident_worker()
        for stop in (self.poller.stop, self.broadcaster.stop):
            with contextlib.suppress(Exception):
                stop()
        self.renders.shutdown(wait=False, cancel_futures=True)
        self.measurer.shutdown(wait=False, cancel_futures=True)
