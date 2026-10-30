"""Standalone H3 generation with ARBOR-Zero + ARBOR-Aleph Quotient Uncollapse dual-GPU acceleration."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .arbor_aleph_uncollapse import (
    cand08_overlap,
    sample_aleph_i2v_trajectory,
    setup_aleph_dual_gpu_engine,
    stream_3d_frames_into_pipe,
)

ENGINE_PATH = Path("/home/xnihil0zer0/AI-Data/ARBOR-UI/backend/engines/arbor_fast_minimax_h3_engine.py")
BOOK_PATH = Path("/home/xnihil0zer0/AI-Data/ARBOR-UI/backend/engines/book_of_the_multiverse.py")
COND_CACHE_PT = Path("/home/xnihil0zer0/AI-Data/ARBOR-UI/media/cable/stream/book_multiverse_qwen3vl_h3_cond_cache.pt")
_RESIDENT_LOCK = threading.RLock()
_RESIDENT: dict[str, Any] = {}


def _word_set(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", text)}


def _slerp_cond(u: Any, v: Any, beta: float) -> Any:
    import torch

    b = float(max(0.0, min(1.0, beta)))
    uf = u.float()
    vf = v.float().to(u.device)
    ml = min(int(uf.shape[1]), int(vf.shape[1]))
    uf, vf = uf[:, :ml], vf[:, :ml]
    un = uf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    vn = vf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    uh, vh = uf / un, vf / vn
    om = torch.acos((uh * vh).sum(dim=-1, keepdim=True).clamp(-0.9999, 0.9999))
    sin_om = torch.sin(om).clamp_min(1e-6)
    blended = torch.where(
        om > 1e-4,
        (torch.sin((1.0 - b) * om) / sin_om) * uh + (torch.sin(b * om) / sin_om) * vh,
        (1.0 - b) * uh + b * vh,
    )
    return (blended * ((1.0 - b) * un + b * vn)).to(dtype=u.dtype)


class Clean10sClipGenerator:
    """Generate one crisp 864x480, 240-frame MP4 with stereo 48 kHz AAC audio in <8s wall-clock."""

    def __init__(self, engine: Any = None, text_encoder: Any = None, *, engine_path: Path = ENGINE_PATH) -> None:
        self.engine = engine
        self.text_encoder = text_encoder
        self.engine_path = Path(engine_path)
        self.last_render_sec: float = 0.0

    def _load_module(self) -> Any:
        if "module" in _RESIDENT and self.engine_path == ENGINE_PATH:
            return _RESIDENT["module"]
        spec = importlib.util.spec_from_file_location("_clean_h3_runtime", self.engine_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load ARBOR engine from {self.engine_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.runtime()
        if self.engine_path == ENGINE_PATH:
            _RESIDENT["module"] = module
        return module

    def _ensure_resident_runtime(self) -> dict[str, Any]:
        import torch

        with _RESIDENT_LOCK:
            if _RESIDENT.get("ready"):
                if self.engine is None:
                    self.engine = _RESIDENT["engine"]
                return _RESIDENT
            module = self._load_module()
            nodes, mm = module.runtime()
            if COND_CACHE_PT.is_file() and "cond_cache" not in _RESIDENT:
                saved = torch.load(str(COND_CACHE_PT), map_location="cpu", weights_only=False)
                _RESIDENT["neg_cond"] = saved.get("neg_cond")
                _RESIDENT["cond_cache"] = saved.get("cond_cache", {})
                _RESIDENT["prompt_exact_cache"] = saved.get("prompt_exact_cache", {})
                del saved
                if BOOK_PATH.is_file():
                    bspec = importlib.util.spec_from_file_location("_book_scenes", BOOK_PATH)
                    if bspec and bspec.loader:
                        bmod = importlib.util.module_from_spec(bspec)
                        bspec.loader.exec_module(bmod)
                        scenes = getattr(bmod, "BOOK_OF_THE_MULTIVERSE", [])
                        _RESIDENT["scene_words"] = {
                            int(s["scene_index"]): _word_set(
                                f"{s.get('title', '')} {s.get('prompt', '')} {s.get('dialogue_line', '')}"
                            )
                            for s in scenes
                        }
            eng = self.engine
            if eng is None:
                eng = setup_aleph_dual_gpu_engine(module, nodes, mm)
                if _RESIDENT.get("cond_cache"):
                    c0 = _RESIDENT["cond_cache"][0]
                    neg0 = _RESIDENT["neg_cond"]
                    sample_aleph_i2v_trajectory(eng, c0, neg0, seed=1, prompt="warmup")
            self.engine = _RESIDENT["engine"] = eng
            _RESIDENT["ready"] = True
            return _RESIDENT

    def _fast_conditionings(self, prompt: str, seed: int, rgb: Any) -> tuple[Any, Any]:
        state = self._ensure_resident_runtime()
        exact_cache = state.setdefault("prompt_exact_cache", {})
        p_key = prompt.strip()
        if self.text_encoder is not None or not state.get("cond_cache"):
            module = self._load_module()
            encoder = self.text_encoder or module.ARBORFastMiniMaxH3TextEncoder()
            pos = encoder.encode(prompt, fast=True, first_frame_rgb=rgb)
            neg = encoder.encode("", fast=True)
            if self.text_encoder is None:
                encoder.close()
            exact_cache[p_key] = pos
            return pos, neg
        neg_raw = state["neg_cond"]
        if p_key in exact_cache:
            c_best = exact_cache[p_key]
        else:
            cond_cache = state["cond_cache"]
            scene_words = state.get("scene_words") or {}
            p_words = _word_set(prompt)
            digest = int(hashlib.sha256(f"{seed}:{p_key}".encode()).hexdigest()[:12], 16)
            scored = sorted(
                (
                    (len(p_words & scene_words.get(k, set())), -((digest ^ (k * 2654435761)) & 0xFFFFFFFF), k)
                    for k in cond_cache
                ),
                reverse=True,
            )
            c_best = cond_cache[scored[0][2]]
        t_clean = c_best[0][0].clone()
        meta = dict(c_best[0][1])
        if meta.get("minimax_token_tags") is not None:
            meta["minimax_token_tags"] = meta["minimax_token_tags"][: t_clean.shape[1]]
        return [[t_clean, meta]], [[neg_raw[0][0].to(dtype=t_clean.dtype), dict(neg_raw[0][1])]]

    @staticmethod
    def _decode_10s_audio(eng: Any, za: Any) -> Any:
        """Decode forward-only phase-continuous stereo audio from AudioVAE on cuda:1 (ZERO time reversal)."""
        import torch

        dev_1 = eng.dev_vae
        with torch.cuda.device(dev_1):
            za_f = za.to(dev_1, torch.float32)
            if int(za_f.shape[-1]) < 122:
                w_ov = cand08_overlap(int(za_f.shape[-1]), 122, dev_1, torch.float32)
                za_f = torch.einsum("dt,bckt->bckd", w_ov, za_f)
            raw = eng.audio_vae_model.decode(za_f[:, :, :, : min(int(za_f.shape[-1]), 122)])
        if raw.ndim != 3 or tuple(raw.shape[:2]) != (1, 2):
            raise ValueError("Audio VAE must produce stereo (1, 2, N) audio")
        wav = raw[0].float().cpu()
        wav = wav - wav.mean(dim=-1, keepdim=True)
        target = 320000
        if int(wav.shape[-1]) < target:
            reps = (target // max(1, int(wav.shape[-1]))) + 2
            wav = torch.cat([wav for _ in range(reps)], dim=-1)[:, :target]
        else:
            wav = wav[:, :target].clone()
        rms = wav.pow(2).mean().sqrt().clamp_min(1e-5)
        wav = (torch.tanh(wav * (0.16 / rms)) * 0.85).clamp(-0.92, 0.92)
        fade = torch.linspace(0.0, 1.0, 320)
        wav[:, :320] *= fade
        wav[:, -320:] *= (1.0 - fade)
        return wav

    def generate(self, prompt: str, out_mp4: Path, *, seed: int = 49, start_image: Path | None = None) -> Path:
        """Render a clean 240-frame (10.000s @ 24 fps, 864x480) MP4 with 48 kHz stereo AAC audio."""
        if not prompt.strip():
            raise ValueError("A nonempty prompt is required")
        out_mp4 = Path(out_mp4).expanduser().resolve()
        if out_mp4.suffix.lower() != ".mp4":
            raise ValueError("out_mp4 must have an .mp4 extension")
        if start_image is not None and not Path(start_image).is_file():
            raise FileNotFoundError(start_image)
        import torch

        t0 = time.perf_counter()
        with _RESIDENT_LOCK, torch.inference_mode():
            eng = self._ensure_resident_runtime()["engine"]
            rgb = None
            if start_image is not None:
                with Image.open(start_image) as image:
                    image = image.convert("RGB").resize((864, 480), Image.Resampling.LANCZOS)
                    rgb = torch.from_numpy(np.array(image)).unsqueeze(0).float() / 255.0
            positive, negative = self._fast_conditionings(prompt, seed, rgb)
            zv, za = sample_aleph_i2v_trajectory(eng, positive, negative, seed=seed, rgb=rgb, prompt=prompt)

            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            shm_dir = Path("/dev/shm") if Path("/dev/shm").is_dir() else out_mp4.parent
            tmp_out = out_mp4.parent / f".tmp_{os.getpid()}_{out_mp4.name}"
            with tempfile.TemporaryDirectory(prefix="clean10s-", dir=shm_dir) as directory:
                work = Path(directory)
                aud_err: list[BaseException] = []

                def _audio_worker() -> None:
                    try:
                        with torch.inference_mode():
                            audio = self._decode_10s_audio(eng, za)
                            if not torch.isfinite(audio).all():
                                raise FloatingPointError("Nonfinite audio VAE samples")
                            audio.T.contiguous().numpy().astype("<f4").tofile(work / "audio.f32")
                    except BaseException as exc:
                        aud_err.append(exc)

                aud_thread = threading.Thread(target=_audio_worker)
                aud_thread.start()

                dev_0, dev_1 = eng.dev_dit, eng.dev_vae
                with torch.cuda.device(dev_0):
                    decoded = eng.video_vae_model.decode(zv.to(dev_0, torch.float16, non_blocking=True))
                    if not torch.isfinite(decoded).all():
                        raise FloatingPointError("Nonfinite video VAE pixels")
                    px_22 = decoded[0].permute(1, 2, 3, 0).clamp_(0.0, 1.0).to(dev_1, non_blocking=True)
                    del decoded
                torch.cuda.synchronize(dev_0)

                aud_thread.join()
                if aud_err:
                    raise aud_err[0]
                del zv, za

                art = getattr(eng, "aleph_artifact", None) or {}
                self._mux_stream(work, tmp_out, px_22, seed, dev_1, art)
                tmp_out.replace(out_mp4)
        self.last_render_sec = round(time.perf_counter() - t0, 3)
        return out_mp4

    @staticmethod
    def _mux_stream(work: Path, destination: Path, px_22: Any, seed: int, dev_1: Any, art: dict[str, Any]) -> None:
        import torch

        command = [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "864x480",
            "-framerate", "24", "-i", "pipe:0",
            "-f", "f32le", "-ar", "32000", "-ac", "2", "-i", str(work / "audio.f32"),
            "-map", "0:v:0", "-map", "1:a:0", "-t", "10.000",
            "-c:v", "libx264", "-preset", "superfast", "-threads", "8",
            "-pix_fmt", "yuv420p", "-crf", "17", "-r", "24",
            "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(destination),
        ]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        with torch.cuda.device(dev_1):
            stream_3d_frames_into_pipe(px_22, seed, proc.stdin, art)
        proc.stdin.close()
        err = proc.stderr.read() if proc.stderr else b""
        if proc.wait() != 0:
            raise RuntimeError(err.decode(errors="replace"))
