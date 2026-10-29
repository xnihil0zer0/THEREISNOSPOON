"""Standalone H3 generation with native VAE decoding and an exact ten-second export.

Models load only when generate() is called. Importing or constructing this class
does not start inference. A supplied start_image anchors frame zero exclusively.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ENGINE_PATH = Path(
    "/home/xnihil0zer0/AI-Data/ARBOR-UI/backend/engines/arbor_fast_minimax_h3_engine.py"
)


class Clean10sClipGenerator:
    """Generate one 864x480, 240-frame MP4 with stereo 48 kHz AAC audio.

    Example: Clean10sClipGenerator().generate("A dancer spins", Path("dance.mp4"))
    Optional engine and text_encoder arguments accept already loaded ARBOR models.
    There are no sequence joins, spatial masks, or output illumination corrections.
    """

    def __init__(
        self, engine: Any = None, text_encoder: Any = None, *, engine_path: Path = ENGINE_PATH
    ) -> None:
        self.engine = engine
        self.text_encoder = text_encoder
        self.engine_path = Path(engine_path)

    def _load_module(self) -> Any:
        spec = importlib.util.spec_from_file_location("_clean_h3_runtime", self.engine_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load ARBOR engine from {self.engine_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.runtime()
        return module

    def _runtime(self) -> tuple[Any, Any]:
        module = self._load_module()
        if self.text_encoder is None:
            self.text_encoder = module.ARBORFastMiniMaxH3TextEncoder()
        if self.engine is None:
            self.engine = module.ARBORFastMiniMaxH3DiT()
        return self.engine, self.text_encoder

    def generate(
        self,
        prompt: str,
        out_mp4: Path,
        *,
        seed: int = 49,
        start_image: Path | None = None,
    ) -> Path:
        """Sample 243 frames once, retain frames 0..239, and export 10.000 seconds.

        A starting image is resized to the canvas and encoded as one keyframe.
        Frames 1..242 receive no keyframe constraints. No terminal anchor is made.
        """
        if not prompt.strip():
            raise ValueError("A nonempty prompt is required")
        out_mp4 = Path(out_mp4).expanduser().resolve()
        if out_mp4.suffix.lower() != ".mp4":
            raise ValueError("out_mp4 must have an .mp4 extension")
        if start_image is not None and not Path(start_image).is_file():
            raise FileNotFoundError(start_image)
        import gc
        import torch
        module = self._load_module()
        _, mm = module.runtime()
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
        import comfy.sample
        import comfy.samplers
        import node_helpers
        from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV

        with torch.inference_mode():
            rgb = None
            if start_image is not None:
                with Image.open(start_image) as image:
                    image = image.convert("RGB").resize((864, 480), Image.Resampling.LANCZOS)
                    rgb = torch.from_numpy(np.array(image)).unsqueeze(0).float() / 255.0
            encoder = self.text_encoder or module.ARBORFastMiniMaxH3TextEncoder()
            positive = encoder.encode(prompt, fast=True, first_frame_rgb=rgb)
            negative = encoder.encode("", fast=True)
            if self.text_encoder is None:
                encoder.close()
                del encoder
            mm.unload_all_models()
            gc.collect()
            torch.cuda.empty_cache()
            eng = self.engine or module.ARBORFastMiniMaxH3DiT()
            self.engine = eng
            vae = eng.video_vae_model
            native_chunks = (vae.tokens_chunk_size, vae.token_overlap)
            if native_chunks != (5, 2):
                raise ValueError(f"Expected native VAE chunk/overlap (5, 2), got {native_chunks}")
            keyframes = []
            if rgb is not None:
                keyframes = [{"resolved_frame_index": 0, "latent": eng.encode_rgb_keyframe(rgb)}]
            positive = node_helpers.conditioning_set_values(
                positive, {"minimax_keyframes": keyframes, "minimax_refs": []}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"minimax_keyframes": [], "minimax_refs": []}
            )
            av = EmptyMiniMaxH3LatentAV.execute(864, 480, 243)[0]["samples"]
            expected = ((1, 24, 72, 30, 54), (1, 32, 2, 405))
            if tuple(tuple(z.shape) for z in av.unbind()) != expected:
                raise ValueError("The H3 node returned an unexpected AV latent layout")
            guider = comfy.samplers.CFGGuider(eng.dit)
            guider.set_cfg(1.0)
            guider.set_conds(positive, negative)
            sampler = comfy.samplers.ksampler("euler")
            sigmas = comfy.samplers.calculate_sigmas(eng.dit.get_model_object("model_sampling"), "beta", 6)
            noise = comfy.sample.prepare_noise(av, seed)

            def sample() -> Any:
                setattr(mm, "EXTRA_RESERVED_VRAM", 7000 * 1024 * 1024)
                return guider.sample(noise, av, sampler, sigmas.cpu(), disable_pbar=True, seed=seed)

            zv, za = eng._run_with_cached_text_refiner(sample, fast=True).unbind()
            if (tuple(zv.shape), tuple(za.shape)) != expected:
                raise ValueError("Sampling changed the requested AV latent layout")
            decoded = eng.video_vae_model.decode(zv.to(eng.dev_vae, torch.float16))
            if tuple(decoded.shape) != (1, 3, 243, 480, 864):
                raise ValueError(f"Unexpected decoded video shape {tuple(decoded.shape)}")
            audio = eng.audio_vae_model.decode(za.to(eng.dev_vae, torch.float32))
            if audio.ndim != 3 or tuple(audio.shape[:2]) != (1, 2) or audio.shape[-1] < 320000:
                raise ValueError("Audio VAE must produce at least 320000 samples of 32 kHz stereo")
            audio = audio[0, :, :320000].float().cpu().clone()
            if not torch.isfinite(audio).all():
                raise FloatingPointError("Nonfinite audio VAE samples")
            fade = torch.linspace(0.0, 1.0, 320)
            audio[:, :320] *= fade
            audio[:, -320:] *= fade.flip(0)
            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="clean10s-", dir=out_mp4.parent) as directory:
                work = Path(directory)
                audio.T.contiguous().numpy().astype("<f4").tofile(work / "audio.f32")
                with (work / "video.rgb").open("wb") as stream:
                    for frame in decoded[0, :, :240].unbind(dim=1):
                        if not torch.isfinite(frame).all():
                            raise FloatingPointError("Nonfinite video VAE pixels")
                        pixels = frame.permute(1, 2, 0).float().clamp(0.0, 1.0)
                        stream.write((pixels * 255.0).round().to(torch.uint8).cpu().numpy().tobytes())
                self._mux(work, work / "clip.mp4")
                (work / "clip.mp4").replace(out_mp4)
        return out_mp4

    @staticmethod
    def _mux(work: Path, destination: Path) -> None:
        command = [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "864x480",
            "-framerate", "24", "-i", str(work / "video.rgb"),
            "-f", "f32le", "-ar", "32000", "-ac", "2", "-i", str(work / "audio.f32"),
            "-map", "0:v:0", "-map", "1:a:0", "-t", "10.000",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", "24",
            "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(destination),
        ]
        subprocess.run(command, check=True, capture_output=True)
