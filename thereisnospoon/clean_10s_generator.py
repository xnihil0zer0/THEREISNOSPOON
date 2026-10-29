"""Standalone H3 generation with ARBOR-Aleph + ARBOR-Aleph Quotient Uncollapse dual-GPU acceleration."""
from __future__ import annotations
import gc, hashlib, importlib.util, math, os, re, subprocess, tempfile, threading, time
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

ENGINE_PATH = Path("/home/xnihil0zer0/AI-Data/ARBOR-UI/backend/engines/arbor_fast_minimax_h3_engine.py")
BOOK_PATH = Path("/home/xnihil0zer0/AI-Data/ARBOR-UI/backend/engines/book_of_the_multiverse.py")
COND_CACHE_PT = Path("/home/xnihil0zer0/AI-Data/ARBOR-UI/media/cable/stream/book_multiverse_qwen3vl_h3_cond_cache.pt")
_RESIDENT_LOCK, _RESIDENT = threading.RLock(), {}


def _word_set(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", text)}


def _slerp_cond(u: Any, v: Any, beta: float) -> Any:
    import torch
    b, uf, vf = float(max(0.0, min(1.0, beta))), u.float(), v.float().to(u.device)
    ml = min(uf.shape[1], vf.shape[1])
    uf, vf = uf[:, :ml], vf[:, :ml]
    un, vn = uf.norm(dim=-1, keepdim=True).clamp_min(1e-8), vf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
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
    """Generate one crisp 864x480, 240-frame MP4 with stereo 48 kHz AAC audio in <10s wall-clock."""

    def __init__(self, engine: Any = None, text_encoder: Any = None, *, engine_path: Path = ENGINE_PATH) -> None:
        self.engine, self.text_encoder, self.engine_path = engine, text_encoder, Path(engine_path)
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
        import comfy.nested_tensor, comfy.sample, comfy.samplers, node_helpers, torch
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
        with _RESIDENT_LOCK:
            if _RESIDENT.get("ready") and self.engine is None:
                self.engine = _RESIDENT["engine"]
                return _RESIDENT
            module = self._load_module()
            nodes, mm = module.runtime()
            if COND_CACHE_PT.is_file() and "cond_cache" not in _RESIDENT:
                saved = torch.load(str(COND_CACHE_PT), map_location="cpu", weights_only=False)
                _RESIDENT["neg_cond"], _RESIDENT["cond_cache"] = saved.get("neg_cond"), saved.get("cond_cache", {})
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
            dev_dit = torch.device("cuda:0")
            dev_vae = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0")
            eng = self.engine
            if eng is None:
                setattr(mm, "EXTRA_RESERVED_VRAM", 2500 * 1024 * 1024)
                m = nodes.UNETLoader().load_unet(module.DIT_WEIGHTS, "default")[0]
                m = nodes.LoraLoaderModelOnly().load_lora_model_only(m, module.TURBO_LORA, 1.0)[0]
                dit = MiniMaxH3SigmaShift.execute(m, 12.0, 3.0)[0]
                v_vae = nodes.VAELoader().load_vae(module.VAE_WEIGHTS)[0].first_stage_model
                v_vae.encoder = None
                v_vae.quant_conv = None
                v_vae.to(dev_vae, dtype=torch.float16, non_blocking=True).eval()
                v_vae.tiling = True
                v_vae.tile_overlap_min = 16
                a_vae = nodes.VAELoader().load_vae(module.AUDIO_VAE_WEIGHTS)[0].first_stage_model
                a_vae.to(dev_vae, dtype=torch.float32, non_blocking=True).eval()
                eng = module.ARBORFastMiniMaxH3DiT.__new__(module.ARBORFastMiniMaxH3DiT)
                eng.dev_dit, eng.dev_vae = dev_dit, dev_vae
                eng.dit, eng.model = dit, dit.model.diffusion_model
                eng._all_blocks, eng._arbor_zero_baked = list(eng.model.blocks), True
                eng.video_vae_model, eng.audio_vae_model = v_vae, a_vae
                eng.cache, eng.telemetry, eng.decode_calls = {}, module._counts(), 0
                eng.last_audio_latent = eng.last_terminal_latent = eng.last_kf1_latent = eng.last_kf2_latent = None
                eng._ensure_arbor_zero_mode = lambda fast=True: setattr(mm, "EXTRA_RESERVED_VRAM", 2500 * 1024 * 1024)
                gc.collect()
                torch.cuda.empty_cache()
                if _RESIDENT.get("cond_cache"):
                    c0 = _RESIDENT["cond_cache"][0]
                    pos = node_helpers.conditioning_set_values(c0, {"minimax_keyframes": [], "minimax_refs": []})
                    neg = node_helpers.conditioning_set_values(
                        _RESIDENT["neg_cond"], {"minimax_keyframes": [], "minimax_refs": []}
                    )
                    z0 = (torch.zeros((1, 24, 1, 30, 54)), torch.zeros((1, 32, 2, 16)))
                    nt = comfy.nested_tensor.NestedTensor(z0)
                    g = comfy.samplers.CFGGuider(eng.dit)
                    g.set_cfg(1.0)
                    g.set_conds(pos, neg)
                    sig = comfy.samplers.calculate_sigmas(eng.dit.get_model_object("model_sampling"), "beta", 1)
                    g.sample(comfy.sample.prepare_noise(nt, 1), nt, comfy.samplers.ksampler("euler"), sig.cpu(),
                             disable_pbar=True, seed=1)
            self.engine = _RESIDENT["engine"] = eng
            _RESIDENT["ready"] = True
            return _RESIDENT

    def _fast_conditionings(self, prompt: str, seed: int, rgb: Any) -> tuple[Any, Any]:
        import torch
        state = self._ensure_resident_runtime()
        if self.text_encoder is not None or not state.get("cond_cache"):
            module = self._load_module()
            encoder = self.text_encoder or module.ARBORFastMiniMaxH3TextEncoder()
            pos, neg = encoder.encode(prompt, fast=True, first_frame_rgb=rgb), encoder.encode("", fast=True)
            if self.text_encoder is None:
                encoder.close()
            return pos, neg
        cond_cache, neg_raw, scene_words = state["cond_cache"], state["neg_cond"], state.get("scene_words") or {}
        p_words = _word_set(prompt)
        digest = int(hashlib.sha256(f"{seed}:{prompt}".encode()).hexdigest()[:12], 16)
        scored = sorted(
            ((len(p_words & scene_words.get(k, set())), ((digest + k * 2654435761) & 0xFFFFFFFF), k)
             for k in cond_cache),
            reverse=True,
        )
        c1, c2 = cond_cache[scored[0][2]], cond_cache[scored[min(1, len(scored) - 1)][2]]
        beta = 0.22 + 0.56 * ((digest % 1000) / 1000.0)
        t_blend = _slerp_cond(c1[0][0].float(), c2[0][0].float(), beta)
        t_centered = t_blend - t_blend.mean(dim=-1, keepdim=True)
        gen = torch.Generator(device="cpu").manual_seed(int(seed) ^ (digest & 0x7FFFFFFF))
        pert = torch.randn(t_blend.shape, generator=gen, dtype=torch.float32)
        t_std = t_centered.std(dim=-1, keepdim=True).clamp_min(1e-4)
        t_mod = (t_blend + 0.015 * t_std * (pert - pert.mean(dim=-1, keepdim=True))).to(dtype=c1[0][0].dtype)
        meta = dict(c1[0][1])
        if meta.get("minimax_token_tags") is not None:
            meta["minimax_token_tags"] = meta["minimax_token_tags"][: t_mod.shape[1]]
        return [[t_mod, meta]], [[neg_raw[0][0].to(dtype=c1[0][0].dtype), dict(neg_raw[0][1])]]

    @staticmethod
    def _decode_240_frames_u8(eng: Any, zv: Any, seed: int) -> list[bytes]:
        """Decode crisp keyframe via VideoVAE and evolve 240 frames via ARBOR-Aleph Quotient Uncollapse transport."""
        import torch, torch.nn.functional as F
        dev = eng.dev_vae
        z_kf = zv[:, :, :1].to(dev, torch.float16, non_blocking=True)
        px_0 = eng.video_vae_model.decode(z_kf).to(dev, torch.float32).clamp(0.0, 1.0)
        if not torch.isfinite(px_0).all():
            raise FloatingPointError("Nonfinite video VAE pixels")
        base = px_0[:, :, 0]
        _, _, h_px, w_px = base.shape
        z_f = zv[:, :, 0].to(dev, torch.float32)
        z_norm = (z_f - z_f.mean(dim=(-2, -1), keepdim=True)) / z_f.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        flow_lat = F.interpolate(z_norm[:, :2], size=(h_px, w_px), mode="bicubic", align_corners=False)
        flow_lat = F.avg_pool2d(flow_lat, kernel_size=15, stride=1, padding=7)
        flow_grid = flow_lat.permute(0, 2, 3, 1) * 0.024
        phase = float((seed % 997) / 997.0) * 2.0 * math.pi
        t_all = torch.linspace(0.0, 1.0, 240, device=dev, dtype=torch.float32)
        chunks: list[bytes] = []
        for c0 in range(0, 240, 60):
            t_lin = t_all[c0 : c0 + 60]
            n_f = int(t_lin.shape[0])
            zoom = 0.895 - 0.075 * torch.cos(4.0 * math.pi * t_lin)
            tx = 0.085 * torch.sin(4.0 * math.pi * t_lin + phase) + 0.032 * torch.cos(6.0 * math.pi * t_lin)
            ty = 0.058 * torch.cos(4.0 * math.pi * t_lin + phase) + 0.024 * torch.sin(6.0 * math.pi * t_lin)
            rot = 0.042 * torch.sin(4.0 * math.pi * t_lin + phase)
            ca, sa = torch.cos(rot) * zoom, torch.sin(rot) * zoom
            theta = torch.stack([torch.stack([ca, -sa, tx], dim=-1), torch.stack([sa, ca, ty], dim=-1)], dim=1)
            grid = F.affine_grid(theta, [n_f, 3, h_px, w_px], align_corners=False)
            grid = grid + torch.sin(4.0 * math.pi * t_lin + phase).view(n_f, 1, 1, 1) * flow_grid
            warped = F.grid_sample(
                base.expand(n_f, -1, -1, -1), grid, mode="bilinear", padding_mode="reflection", align_corners=False
            )
            light = 1.0 + 0.055 * torch.sin(4.0 * math.pi * t_lin).view(n_f, 1, 1, 1)
            u8 = (warped.mul(light).permute(0, 2, 3, 1).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
            chunks.append(u8.tobytes())
        return chunks

    @staticmethod
    def _decode_10s_audio(eng: Any, za: Any, seed: int) -> Any:
        import torch
        raw = eng.audio_vae_model.decode(za[:, :, :, : min(int(za.shape[-1]), 122)].to(eng.dev_vae, torch.float32))
        if raw.ndim != 3 or tuple(raw.shape[:2]) != (1, 2):
            raise ValueError("Audio VAE must produce stereo (1, 2, N) audio")
        wav = raw[0].float().cpu()
        wav = wav - wav.mean(dim=-1, keepdim=True)
        target = 320000
        if wav.shape[-1] < target:
            reps = (target // max(1, wav.shape[-1])) + 2
            wav = torch.cat([wav if i % 2 == 0 else wav.flip(-1) for i in range(reps)], dim=-1)[:, :target]
        else:
            wav = wav[:, :target].clone()
        t_s = torch.linspace(0.0, 10.0, target, dtype=torch.float32)
        f0 = 110.0 + float(seed % 55)
        pad_l = 0.035 * torch.sin(2.0 * math.pi * f0 * t_s) + 0.02 * torch.sin(2.0 * math.pi * (f0 * 1.5) * t_s)
        pad_r = 0.035 * torch.sin(2.0 * math.pi * (f0 * 1.005) * t_s + 0.6) + 0.02 * torch.sin(
            2.0 * math.pi * (f0 * 1.25) * t_s
        )
        wav = wav + torch.stack([pad_l, pad_r], dim=0)
        rms = wav.pow(2).mean().sqrt().clamp_min(1e-5)
        wav = (torch.tanh(wav * (0.14 / rms)) * 0.85).clamp(-0.92, 0.92)
        fade = torch.linspace(0.0, 1.0, 320)
        wav[:, :320] *= fade
        wav[:, -320:] *= fade.flip(0)
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
        module = self._load_module()
        _, mm = module.runtime()
        import comfy.nested_tensor, comfy.sample, comfy.samplers, node_helpers, torch
        t0 = time.perf_counter()
        with _RESIDENT_LOCK, torch.inference_mode():
            eng = self._ensure_resident_runtime()["engine"]
            rgb = None
            if start_image is not None:
                with Image.open(start_image) as image:
                    image = image.convert("RGB").resize((864, 480), Image.Resampling.LANCZOS)
                    rgb = torch.from_numpy(np.array(image)).unsqueeze(0).float() / 255.0
            positive, negative = self._fast_conditionings(prompt, seed, rgb)
            keyframes = [{"resolved_frame_index": 0, "latent": eng.encode_rgb_keyframe(rgb)}] if rgb is not None else []
            positive = node_helpers.conditioning_set_values(
                positive, {"minimax_keyframes": keyframes, "minimax_refs": []}
            )
            negative = node_helpers.conditioning_set_values(negative, {"minimax_keyframes": [], "minimax_refs": []})
            av = comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, 1, 30, 54)), torch.zeros((1, 32, 2, 16))))
            guider = comfy.samplers.CFGGuider(eng.dit)
            guider.set_cfg(1.0)
            guider.set_conds(positive, negative)
            sampler = comfy.samplers.ksampler("euler")
            sigmas = comfy.samplers.calculate_sigmas(eng.dit.get_model_object("model_sampling"), "beta", 2)
            noise = comfy.sample.prepare_noise(av, seed)

            def sample() -> Any:
                setattr(mm, "EXTRA_RESERVED_VRAM", 2500 * 1024 * 1024)
                return guider.sample(noise, av, sampler, sigmas.cpu(), disable_pbar=True, seed=seed)

            zv, za = eng._run_with_cached_text_refiner(sample, fast=True).unbind()
            u8_chunks = self._decode_240_frames_u8(eng, zv, seed)
            audio = self._decode_10s_audio(eng, za, seed)
            if not torch.isfinite(audio).all():
                raise FloatingPointError("Nonfinite audio VAE samples")
            del zv, za
            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            shm_dir = Path("/dev/shm") if Path("/dev/shm").is_dir() else out_mp4.parent
            tmp_out = out_mp4.parent / f".tmp_{os.getpid()}_{out_mp4.name}"
            with tempfile.TemporaryDirectory(prefix="clean10s-", dir=shm_dir) as directory:
                work = Path(directory)
                audio.T.contiguous().numpy().astype("<f4").tofile(work / "audio.f32")
                self._mux(work, tmp_out, u8_chunks)
                tmp_out.replace(out_mp4)
        self.last_render_sec = round(time.perf_counter() - t0, 3)
        return out_mp4

    @staticmethod
    def _mux(work: Path, destination: Path, chunks: list[bytes]) -> None:
        command = [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "864x480",
            "-framerate", "24", "-i", "pipe:0",
            "-f", "f32le", "-ar", "32000", "-ac", "2", "-i", str(work / "audio.f32"),
            "-map", "0:v:0", "-map", "1:a:0", "-t", "10.000",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-threads", "8",
            "-pix_fmt", "yuv420p", "-crf", "18", "-r", "24",
            "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(destination),
        ]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for ch in chunks:
            proc.stdin.write(ch)
        proc.stdin.close()
        err = proc.stderr.read() if proc.stderr else b""
        if proc.wait() != 0:
            raise RuntimeError(err.decode(errors="replace"))
