#!/usr/bin/env python3
"""First End-to-End Verification Test for THEREISNOSPOON: Chat VJ (0-Chat & 10-Chat) + Clean 10s Clip vs Study v2."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thereisnospoon.broadcast_support import (
    ARTIFACTS_DIR,
    BASELINE_JSON,
    STUDY_V2_DIR,
    finite,
    read_json,
    receipt_path,
    sheet_path,
    signature,
    study_v2_baseline,
    write_json,
)
from thereisnospoon.chat_vj_director import (
    CHAT_VJ_SYSTEM_PROMPT_NO_CHAT,
    CHAT_VJ_SYSTEM_PROMPT_WITH_CHAT,
    ChatVJDirector,
)
from thereisnospoon.clean_10s_generator import Clean10sClipGenerator
from thereisnospoon.quality_metrics import compare_against_study_v2, evaluate_clip_quality
from thereisnospoon.youtube_chat_poller import YouTubeChatPoller

SAMPLE_10_CHAT_LINES = [
    {"author": "@neon_moth", "message": "VJ, make the moon a disco ball and let it drip onto the city"},
    {"author": "@quietkettle", "message": "first time here, is this whole thing generated live??"},
    {"author": "@pixelgoblin", "message": "I want a cat captain steering a brass airship through neon clouds"},
    {"author": "@delta_rune", "message": "can it be raining tiny glowing paper lanterns please"},
    {"author": "@sk8rgrl", "message": "the energy in here is electric tonight, turn up the bass!"},
    {"author": "@orbitalfern", "message": "show a rooftop DJ booth overlooking a floating bioluminescent harbor"},
    {"author": "@crt_ghost", "message": "give the host a holographic visor and a silver jacket"},
    {"author": "@marmalade", "message": "wait did someone say there is no spoon?? bend the skyline!"},
    {"author": "@vortexvera", "message": "pan the camera around the turntables while sparks fly"},
    {"author": "@byte_bard", "message": "shout out the midnight crew watching live on THEREISNOSPOON!"},
]


def main() -> int:
    t_start = time.perf_counter()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    poller = YouTubeChatPoller()

    # 1. Test Mode A: Empty chat (0 lines -> "Creative & Strange" persona)
    poller.clear_chat()
    empty_lines = poller.get_last_10_chat_lines()
    assert len(empty_lines) == 0, f"Expected 0 chat lines, got {len(empty_lines)}"

    director = ChatVJDirector(seed=20261029)
    print("[1/4] Synthesizing Mode A (0 chat lines -> Creative & Strange mode) via Qwen3-VL-4B...")
    strange_scene = director.direct(empty_lines)
    print(f"  Mode A ({strange_scene.mode}) elapsed={strange_scene.elapsed_ms}ms")
    print(f"  Scene: {strange_scene.scene}")
    print(f"  Spoken Dialogue: {strange_scene.dialogue_tag}")

    # 2. Test Mode B: Last 10 chat lines (10 lines -> "Chat VJ" persona)
    poller.inject_chat_batch(SAMPLE_10_CHAT_LINES)
    chat_10_lines = poller.get_last_10_chat_lines()
    assert len(chat_10_lines) == 10, f"Expected 10 chat lines, got {len(chat_10_lines)}"

    print("[2/4] Synthesizing Mode B (10 chat lines -> Live Chat VJ mode) via Qwen3-VL-4B...")
    vj_scene = director.direct(chat_10_lines)
    print(f"  Mode B ({vj_scene.mode}) elapsed={vj_scene.elapsed_ms}ms")
    print(f"  Scene: {vj_scene.scene}")
    print(f"  Spoken Dialogue: {vj_scene.dialogue_tag}")

    # Release the 4B text generator before loading the 32B text encoder + H3 DiT + VAEs
    director.release_generator()
    director.close()

    # 3. Generate Clean 10-Second (240-Frame = 10.000s) Audio-Video Clip (Zero Blending / Zero Stitching)
    out_mp4 = ARTIFACTS_DIR / "clean_10s_chat_vj_test.mp4"
    print(f"[3/4] Generating Clean 10-Second (240-frame) AV clip -> {out_mp4} ...")
    t_gen0 = time.perf_counter()
    generator = Clean10sClipGenerator()
    generator.generate(vj_scene.prompt, out_mp4, seed=42)
    gen_elapsed_sec = round(time.perf_counter() - t_gen0, 2)
    print(f"  Clean 10s clip generated in {gen_elapsed_sec}s: {out_mp4}")

    # Free GPU models before running CPU ffprobe & comparison analysis
    del generator
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Evaluate Quality & Compare Against Running Studies (Study v2)
    print("[4/4] Auditing clip via ffprobe, 16x16 grid ratio, motion fluidity & comparing vs Study v2...")
    base = study_v2_baseline(read_json(BASELINE_JSON), evaluate_clip_quality)
    study_v2_refs = [
        STUDY_V2_DIR / "Run09_L9_A3_B3_C2_D1.mp4",
        STUDY_V2_DIR / "Run10_Confirm_A3_B3_C3_D3.mp4",
        STUDY_V2_DIR / "Arm00_Naive_Unedited_Baseline.mp4",
    ]
    out_sheet = sheet_path(out_mp4)
    comp = compare_against_study_v2(out_mp4, study_v2_refs, out_sheet)
    clean_metrics = finite(comp["clean"])

    receipt_payload = {
        "clip": out_mp4.name,
        "signature": signature(out_mp4),
        "measured_at": time.time(),
        "clean": clean_metrics,
        "study_v2_mean": base["mean"],
        "study_v2_clips": base["clips"],
        "contact_sheet": out_sheet.name,
    }
    write_json(receipt_path(out_mp4), receipt_payload)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_sec": round(time.perf_counter() - t_start, 2),
        "generation_elapsed_sec": gen_elapsed_sec,
        "mode_a_no_chat_creative_and_strange": {
            "mode": strange_scene.mode,
            "chat_count": len(strange_scene.chat_lines),
            "system_prompt": CHAT_VJ_SYSTEM_PROMPT_NO_CHAT,
            "scene": strange_scene.scene,
            "spoken_line": strange_scene.spoken_line,
            "dialogue_tag": strange_scene.dialogue_tag,
            "soundscape": strange_scene.soundscape,
            "music": strange_scene.music,
            "full_prompt": strange_scene.prompt,
            "elapsed_ms": strange_scene.elapsed_ms,
        },
        "mode_b_last_10_chat_vj": {
            "mode": vj_scene.mode,
            "chat_count": len(vj_scene.chat_lines),
            "chat_lines": list(vj_scene.chat_lines),
            "system_prompt": CHAT_VJ_SYSTEM_PROMPT_WITH_CHAT,
            "scene": vj_scene.scene,
            "spoken_line": vj_scene.spoken_line,
            "dialogue_tag": vj_scene.dialogue_tag,
            "soundscape": vj_scene.soundscape,
            "music": vj_scene.music,
            "full_prompt": vj_scene.prompt,
            "elapsed_ms": vj_scene.elapsed_ms,
        },
        "clean_10s_clip_metrics": clean_metrics,
        "study_v2_comparison": {
            "study_v2_mean": base["mean"],
            "study_v2_reference_reports": finite(comp["study_v2"]),
            "contact_sheet_png": str(out_sheet),
        },
    }
    report_path = ARTIFACTS_DIR / "first_verification_test_report.json"
    write_json(report_path, report)
    print("VERIFICATION SUMMARY:")
    print(json.dumps(finite(clean_metrics), indent=2))
    assert clean_metrics["format_valid"], f"Format validation failed: {clean_metrics['format_checks']}"
    assert clean_metrics["grid_16x16_pass"], f"16x16 grid check failed: {clean_metrics['grid_16x16_ratio']}"
    assert clean_metrics["quality_pass"], "Quality pass failed"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
