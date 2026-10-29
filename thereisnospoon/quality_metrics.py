"""CPU-only format checks, artifact diagnostics, and aligned comparison sheets.

Motion uses RGB values in [0, 1], with active transitions exceeding 1/255 mean L1.
Acceleration is mean pixel second-difference L1 divided by mean motion L1.
The 17-frame pulse score compares the strongest motion phase to all other phases.
Sharpness is mean variance of the four-neighbor Laplacian of 8-bit luminance.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SHEET_FRAMES = (0, 48, 96, 120, 144, 192, 239)


def _number(value: Any) -> float | None:
    try:
        return float(Fraction(str(value)))
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _ratio(numerator: float, denominator: float, neutral: float = 1.0) -> float:
    if denominator > 1e-12:
        return float(numerator / denominator)
    return neutral if numerator <= 1e-12 else float("inf")


def _probe(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    command = ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json"]
    probe = json.loads(subprocess.run(command + [str(path)], check=True, capture_output=True).stdout)
    streams = probe.get("streams", [])
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")
    observed = {
        "width": video.get("width"), "height": video.get("height"),
        "nb_frames": _number(video.get("nb_frames")), "fps": _number(video.get("avg_frame_rate")),
        "duration": _number(probe.get("format", {}).get("duration")),
        "audio_sample_rate": _number((audio or {}).get("sample_rate")),
        "audio_channels": (audio or {}).get("channels"), "audio_codec": (audio or {}).get("codec_name"),
    }
    expected = dict(width=864, height=480, nb_frames=240, fps=24.0, duration=10.0,
                    audio_sample_rate=48000, audio_channels=2, audio_codec="aac")
    checks = {key: observed[key] == value for key, value in expected.items()}
    checks.update(
        video_duration=_number(video.get("duration")) == 10.0,
        audio_duration=_number((audio or {}).get("duration")) == 10.0,
        nominal_fps=_number(video.get("r_frame_rate")) == 24.0,
        counted_frames=_number(video.get("nb_read_frames")) == 240,
    )
    return {"path": str(path), **observed, "format_checks": checks}, audio


def _frames(path: Path, width: int, height: int) -> Iterator[np.ndarray]:
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-hwaccel", "none", "-noautorotate",
        "-threads", "1", "-i", str(path), "-map", "0:v:0", "-an", "-vsync", "0",
        "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    size = width * height * 3
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            if process.stdout is None:
                raise RuntimeError("ffmpeg did not open its video output")
            while True:
                data = process.stdout.read(size)
                if not data:
                    break
                if len(data) != size:
                    raise ValueError(f"Truncated RGB frame in {path}")
                yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace"))
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()


def _gradient(pixels: np.ndarray) -> np.ndarray:
    """Mean absolute RGB difference of left/up neighbors at coordinates x,y >= 1."""
    dx = np.abs(pixels[1:, 1:] - pixels[1:, :-1]).mean(axis=-1)
    dy = np.abs(pixels[1:, 1:] - pixels[:-1, 1:]).mean(axis=-1)
    return (dx + dy) * 0.5


def _analyse(path: Path, keep_frames: bool = False) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    path = Path(path).expanduser().resolve()
    report, audio = _probe(path)
    width, height = report["width"], report["height"]
    if min(width, height) < 3:
        raise ValueError("Quality metrics require at least 3x3 pixels")
    y, x = np.ogrid[1:height, 1:width]
    grid = (x % 16 == 0) | (y % 16 == 0)
    if not grid.any() or grid.all():
        raise ValueError("Quality metrics need both 16-pixel boundaries and interior pixels")
    boundary, interior, sharpness, acceleration = [], [], [], []
    motions, selected = [], {}
    previous = previous_delta = None
    count = 0
    for index, frame in enumerate(_frames(path, width, height)):
        count += 1
        if keep_frames and index in SHEET_FRAMES:
            selected[index] = frame.copy()
        pixels = frame.astype(np.float32) / 255.0
        gradient = _gradient(pixels)
        boundary.append(float(gradient[grid].mean()))
        interior.append(float(gradient[~grid].mean()))
        gray = (pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)) * 255.0
        lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        lap -= 4.0 * gray[1:-1, 1:-1]
        sharpness.append(float(lap.var(dtype=np.float64)))
        if previous is not None:
            delta = pixels - previous
            motions.append(float(np.abs(delta).mean()))
            if previous_delta is not None:
                acceleration.append(float(np.abs(delta - previous_delta).mean()))
            previous_delta = delta
        previous = pixels
    if count == 0:
        raise ValueError(f"No decoded frames in {path}")
    motion = np.asarray(motions, dtype=np.float64)
    mean_motion = float(motion.mean()) if motion.size else 0.0
    phases = np.arange(1, count) % 17
    phase_means = [float(motion[phases == p].mean()) if np.any(phases == p) else 0.0 for p in range(17)]
    strongest = int(np.argmax(phase_means))
    others = motion[phases != strongest]
    pulse = _ratio(phase_means[strongest], float(others.mean())) if others.size and count >= 35 else 1.0
    rms_dbfs = None
    if audio is not None:
        command = [
            "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
            "-map", "0:a:0", "-vn", "-ar", "48000", "-ac", "2",
        ]
        duration = _number(audio.get("duration"))
        if duration is not None and duration > 0:
            command += ["-t", str(duration)]
        raw = subprocess.run(command + ["-f", "f32le", "pipe:1"], check=True, capture_output=True).stdout
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        if not samples.size or not np.isfinite(samples).all():
            raise ValueError("Audio decoding produced empty or nonfinite samples")
        power = float(np.mean(samples * samples))
        rms_dbfs = float(10.0 * np.log10(power)) if power > 0 else float("-inf")
    report.update(
        decoded_frames=count, grid_16x16_ratio=_ratio(float(np.mean(boundary)), float(np.mean(interior))),
        temporal_motion_mean_l1=mean_motion,
        active_motion_frame_fraction=float(np.mean(motion > 1.0 / 255.0)) if motion.size else 0.0,
        temporal_accel_ratio=_ratio(float(np.mean(acceleration)) if acceleration else 0.0, mean_motion, 0.0),
        vae_17f_pulse_ratio=pulse, mean_laplacian_sharpness=float(np.mean(sharpness)), audio_rms_dbfs=rms_dbfs,
    )
    report["format_checks"]["decoded_frames"] = count == 240
    report["format_valid"] = all(report["format_checks"].values())
    report["grid_16x16_pass"] = report["grid_16x16_ratio"] < 1.15
    report["quality_pass"] = report["format_valid"] and report["grid_16x16_pass"]
    return report, selected


def evaluate_clip_quality(mp4_path: Path) -> dict[str, Any]:
    """Return observed metadata, per-field format checks, metrics, and pass flags.

    Format mismatches are reported as false checks, so Study v2 remains measurable.
    Grid gradients use x % 16 == 0 OR y % 16 == 0, with both nonzero for interior.
    A zero/zero grid or pulse ratio is neutral (1). Missing audio has RMS None.
    quality_pass requires exact format compliance and grid_16x16_ratio < 1.15.
    """
    return _analyse(mp4_path)[0]


_REF_CACHE: dict[tuple[str, int, int], tuple[dict[str, Any], dict[int, np.ndarray]]] = {}


def compare_against_study_v2(
    clean_mp4: Path, study_v2_mp4s: Sequence[Path], out_sheet_png: Path
) -> dict[str, Any]:
    """Measure each clip and save aligned frames plus shared-gain gradient crops.

    Each column uses the exact requested frame index. Short clips visibly mark
    unavailable frames. Gradients show a grid-aligned central 64x64 crop at 3x
    magnification with a fixed gain of 8 for every clip, without normalization.
    """
    if not study_v2_mp4s:
        raise ValueError("Provide at least one Study v2 clip")
    destination = Path(out_sheet_png).expanduser().resolve()
    paths = [Path(clean_mp4), *(Path(p) for p in study_v2_mp4s)]
    sheet = Image.new("RGB", (1552, 54 + 396 * len(paths)), "#10151c")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 10), "10s comparison | frames at 24 fps | center pixel gradients: gain 8, 16px cells", fill="white")
    reports = []
    for row, path in enumerate(paths):
        if row > 0:
            st = path.stat()
            ck = (str(path), st.st_size, st.st_mtime_ns)
            if ck not in _REF_CACHE:
                _REF_CACHE[ck] = _analyse(path, keep_frames=True)
            report, frames = _REF_CACHE[ck]
        else:
            report, frames = _analyse(path, keep_frames=True)
        reports.append(report)
        top = 54 + row * 396
        label = "Clean 10s" if row == 0 else f"Study v2 #{row}"
        summary = f"{label}: {path.name} | grid {report['grid_16x16_ratio']:.3f} | format {report['format_valid']}"
        draw.text((20, top), summary[:190], fill="#a7dfed")
        for column, index in enumerate(SHEET_FRAMES):
            left = 20 + column * 216
            draw.text((left, top + 24), f"f={index}  t={index / 24:.3f}s", fill="white")
            frame = frames.get(index)
            if frame is None:
                draw.text((left, top + 75), "Frame unavailable", fill="#f2b0aa")
                continue
            thumbnail = Image.fromarray(frame).resize((208, 116), Image.Resampling.LANCZOS)
            sheet.paste(thumbnail, (left, top + 44))
            gradient = _gradient(frame.astype(np.float32) / 255.0)
            height, width = frame.shape[:2]
            x0 = max(1, ((width - 64) // 2 // 16) * 16)
            y0 = max(1, ((height - 64) // 2 // 16) * 16)
            crop = gradient[y0 - 1:y0 + 63, x0 - 1:x0 + 63]
            pixels = np.clip(crop * (255.0 * 8.0), 0, 255).astype(np.uint8)
            zoom = Image.fromarray(pixels).resize((192, 192), Image.Resampling.NEAREST).convert("RGB")
            draw.text((left, top + 167), "64px gradient crop x3", fill="#c6cbd1")
            sheet.paste(zoom, (left, top + 187))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")
    return {"clean": reports[0], "study_v2": reports[1:], "contact_sheet": str(destination)}
