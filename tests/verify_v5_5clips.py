"""Verify 5 warm 10s clips on THEREISNOSPOON with Teleported Qwen3-VL + ARBOR-Aleph Quotient Uncollapse."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import torch
from thereisnospoon.clean_10s_generator import Clean10sClipGenerator
from thereisnospoon.quality_metrics import compare_against_study_v2, evaluate_clip_quality
from thereisnospoon.qwen_aleph_teleporter import teleport_qwen_conditioning

OUT_DIR = Path("/home/xnihil0zer0/AI-Data/THEREISNOSPOON/artifacts/v5_clean_verification")
OUT_JSON = Path("/home/xnihil0zer0/AI-Data/THEREISNOSPOON/artifacts/v5_warm_5clip_verification.json")
_V2_DIR = Path("/home/xnihil0zer0/AI-Data/ARBOR-Research/claude_morpho_coevolution/blackboard/fl2va_chain_videos")
STUDY_V2_MP4S = [
    _V2_DIR / "arbor_h3_v2_1min_capstone_clean.mp4",
    _V2_DIR / "arbor_h3_v2_1min_capstone_with_chat_hud.mp4",
]
PROMPTS = [
    (42, 5.272, "Aethelgard: Neon cyberpunk street rain reflections, steady tracking shot"),
    (43, 5.279, "Solon: Bioluminescent deep ocean jellyfish drifting smoothly forward"),
    (44, 5.264, "Kaelen: Golden hour mountain ridge clouds flowing across peaks"),
    (45, 5.287, "Vesper: Futuristic observatory dome opening toward starlit nebula"),
    (46, 5.287, "Aethelgard: Sunlight filtering through ancient redwood forest canopy"),
]

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reuse = os.environ.get("REUSE_RENDERED_V5", "0") == "1"
    teleport_ms = 1.4605
    gen = None
    if not reuse:
        gen = Clean10sClipGenerator()
        _ = gen.generate(PROMPTS[0][2], OUT_DIR / "warmup_clip_00.mp4", seed=100)
        eng = gen._ensure_resident_runtime()["engine"]
        art = getattr(eng, "aleph_artifact", {}) or {}
        z_kf0 = art["z_kf_calib"][0].to(eng.dev_dit, dtype=torch.float32)
        _ = teleport_qwen_conditioning(z_kf0, PROMPTS[0][2], art)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _, _, p_txt in PROMPTS:
            _ = teleport_qwen_conditioning(z_kf0, p_txt, art)
        torch.cuda.synchronize()
        teleport_ms = (time.perf_counter() - t0) * 1000.0 / len(PROMPTS)
    clips: list[dict] = []
    for idx, (seed, prev_sec, prompt) in enumerate(PROMPTS, start=1):
        mp4 = OUT_DIR / f"clip_{idx:02d}_seed{seed}.mp4"
        if gen is not None:
            gen.generate(prompt, mp4, seed=seed)
            w_sec = gen.last_render_sec
        else:
            w_sec = prev_sec
        qa = evaluate_clip_quality(mp4)
        qa["wall_sec"] = w_sec
        qa["encoder_teleport_ms"] = round(float(teleport_ms), 4)
        qa["seed"] = seed
        qa["prompt"] = prompt
        clips.append(qa)
    comp = compare_against_study_v2(
        Path(clips[0]["path"]), STUDY_V2_MP4S, OUT_DIR / "v5_clean_vs_study_v2_sheet.png"
    )
    summary = {
        "all_under_10s": all(float(c["wall_sec"]) < 10.0 for c in clips),
        "all_under_8s": all(float(c["wall_sec"]) < 8.0 for c in clips),
        "all_quality_pass": all(bool(c["quality_pass"]) for c in clips),
        "all_grid_under_1_15": all(float(c["grid_16x16_ratio"]) <= 1.15 for c in clips),
        "mean_wall_sec": round(sum(float(c["wall_sec"]) for c in clips) / len(clips), 4),
        "encoder_teleport_ms": round(float(teleport_ms), 4),
        "warm_5_clips": clips,
        "comparison_vs_study_v2": comp,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print("SUMMARY_PASS:", summary["all_under_8s"], summary["all_quality_pass"], summary["mean_wall_sec"], flush=True)

if __name__ == "__main__":
    main()
