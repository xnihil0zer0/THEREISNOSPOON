"""Comprehensive CPU test suite for THEREISNOSPOON modules."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from thereisnospoon import (
    CHANNEL_HANDLE,
    CHANNEL_ID,
    CHANNEL_URL,
    CHAT_VJ_SYSTEM_PROMPT_NO_CHAT,
    CHAT_VJ_SYSTEM_PROMPT_WITH_CHAT,
    ChatVJDirector,
    YouTubeChatPoller,
    YouTubeRTMPBroadcaster,
    normalize_chat_lines,
)
from thereisnospoon.arbor_aleph_uncollapse import (
    ARTIFACT_PT,
    FIXED_TEXT_LEN,
    apply_quotient_uncollapse,
    cand08_overlap,
    normalize_text_len,
    stream_3d_frames_into_pipe,
    uncollapse_16x16_seams,
)
from thereisnospoon.broadcast_console import BroadcastConsole
from thereisnospoon.broadcast_support import ARTIFACTS_DIR, finite, normalize_lines, resolve_artifact, stream_target
from thereisnospoon.chat_vj_director import _clean_line, _parse_scene, _parse_sound
from thereisnospoon.clean_10s_generator import Clean10sClipGenerator, _slerp_cond, _word_set
from thereisnospoon.quality_metrics import _number, _ratio


def test_channel_constants_and_system_prompts() -> None:
    assert CHANNEL_HANDLE == "@THEREISNOSPOON-t4j"
    assert CHANNEL_ID == "UC9ryqiyaNstEvvsw9EKpccA"
    assert CHANNEL_URL == "https://www.youtube.com/@THEREISNOSPOON-t4j"
    assert "last 10 lines" in CHAT_VJ_SYSTEM_PROMPT_WITH_CHAT
    assert "<d>[English]" in CHAT_VJ_SYSTEM_PROMPT_WITH_CHAT
    assert "creative" in CHAT_VJ_SYSTEM_PROMPT_NO_CHAT.lower()
    assert "strange" in CHAT_VJ_SYSTEM_PROMPT_NO_CHAT.lower()
    assert "<d>[English]" in CHAT_VJ_SYSTEM_PROMPT_NO_CHAT


def test_normalize_chat_lines_truncates_to_last_10() -> None:
    raw = [{"author": f"@u{i}", "message": f"line {i}"} for i in range(15)]
    lines = normalize_chat_lines(raw)
    assert len(lines) == 10
    assert lines[0] == "@u5: line 5" and lines[-1] == "@u14: line 14"
    assert normalize_chat_lines([]) == []


def test_chat_vj_parsers_and_prompt_builder() -> None:
    scene_raw = (
        "Scene: A neon observatory floats above a bioluminescent sea while a silver-jacketed VJ "
        "spins a holographic deck as the camera orbits smoothly around the glowing glass dome "
        "and paper lanterns drift into the stars."
    )
    parsed_scene = _parse_scene(scene_raw)
    assert parsed_scene is not None and "neon observatory" in parsed_scene
    line = _clean_line('<d>[English] "Welcome to the midnight sky where every beat bends gravity!"</d>')
    assert line == "Welcome to the midnight sky where every beat bends gravity!"
    snd1 = _parse_sound("SOUNDSCAPE: Deep ocean waves and crisp vinyl crackle.\nMUSIC: Warm pulsing analog synth bass.")
    assert snd1 == ("Deep ocean waves and crisp vinyl crackle.", "Warm pulsing analog synth bass.")
    snd2 = _parse_sound("Deep ocean waves and crisp vinyl crackle.\nWarm pulsing analog synth bass.")
    assert snd2 == snd1
    prompt = ChatVJDirector.build_prompt(parsed_scene, line, snd1[0], snd1[1])
    assert '<d>[English] "Welcome to the midnight sky where every beat bends gravity!"</d>' in prompt
    assert "integrated_multimodal_description:" in prompt
    assert "overall_soundscape:" in prompt and "non_diegetic_music:" in prompt


def test_youtube_chat_poller_rolling_window_and_clear() -> None:
    poller = YouTubeChatPoller()
    poller.clear_chat()
    assert poller.get_last_10_chat_lines() == []
    for idx in range(12):
        assert bool(poller.inject_chat_message(f"@viewer_{idx}", f"Hello broadcast #{idx}")) is True
    last_10 = poller.get_last_10_chat_lines()
    assert len(last_10) == 10
    assert last_10[0]["author"] == "@viewer_2" and last_10[-1]["author"] == "@viewer_11"
    poller.clear_chat()
    assert poller.get_last_10_chat_lines() == []


def test_stream_target_and_broadcaster_masking() -> None:
    target = stream_target(None, "abcd-1234-efgh-5678")
    assert target == "rtmp://a.rtmp.youtube.com/live2/abcd-1234-efgh-5678"
    broadcaster = YouTubeRTMPBroadcaster(stream_key="abcd-1234-efgh-5678")
    tel = broadcaster.get_telemetry()
    assert "abcd-1234-efgh-5678" not in tel["rtmp_url_masked"]
    assert "[redacted]" in tel["rtmp_url_masked"] and tel["is_streaming"] is False


def test_finite_and_normalize_lines() -> None:
    payload = {"a": float("nan"), "b": float("inf"), "c": Path("/tmp/x.mp4"), "d": [1.5, float("-inf")]}
    assert finite(payload) == {"a": None, "b": None, "c": "/tmp/x.mp4", "d": [1.5, None]}
    rows = normalize_lines(["hello world", {"author": "@alice", "message": "drop the beat"}])
    assert rows == [
        {"author": "@console_viewer", "message": "hello world"},
        {"author": "@alice", "message": "drop the beat"},
    ]


def test_fast_resident_vj_director_both_modes() -> None:
    director = ChatVJDirector(seed=42)
    s_empty = director.direct([])
    assert s_empty.mode == "no_chat" and s_empty.elapsed_ms < 100.0
    assert '<d>[English] "' in s_empty.prompt
    sample_10 = [{"author": f"@u{i}", "message": f"cyber disco wave {i}"} for i in range(10)]
    s_chat = director.direct(sample_10)
    assert s_chat.mode == "chat" and len(s_chat.chat_lines) == 10 and s_chat.elapsed_ms < 100.0
    assert '<d>[English] "' in s_chat.prompt


def test_broadcast_console_and_api_modes(monkeypatch: Any) -> None:
    monkeypatch.setenv("THEREISNOSPOON_AUTOPOLL", "0")
    monkeypatch.setenv("THEREISNOSPOON_AUTOLOOP", "0")
    import thereisnospoon.broadcast_console as bc_mod
    import thereisnospoon.server as srv_mod

    srv_mod.console.shutdown()

    def fake_run_gpu_job(kind: str, request: dict[str, Any], _local: Any) -> tuple[dict[str, Any], str]:
        lines = request.get("lines") or []
        mode = "chat" if lines else "no_chat"
        spoken = "Chat mode rocks the neon deck tonight!" if lines else "Strange clocks melt across the silent sky."
        vj = {
            "mode": mode, "chat_lines": [str(x) for x in lines],
            "scene": "A glowing holographic stage floats over a surreal crystal sea.",
            "spoken_line": spoken, "dialogue_tag": f'<d>[English] "{spoken}"</d>',
            "soundscape": "Crisp synth hum and gentle ocean wind.", "music": "Deep warm bassline.",
            "prompt": f'integrated_multimodal_description: scene <d>[English] "{spoken}"</d>',
            "samples": [[42, 0.9]], "elapsed_ms": 12.5,
        }
        return {"vj": vj}, "fake_gpu"

    monkeypatch.setattr(bc_mod, "run_gpu_job", fake_run_gpu_job)
    poller = YouTubeChatPoller()
    console = BroadcastConsole(
        poller, YouTubeRTMPBroadcaster,
        lambda _p: {"grid_16x16_ratio": 1.01, "format_valid": True},
        lambda _c, _r, _s: {"clean": {"grid_16x16_ratio": 1.01, "format_valid": True}},
        lambda _k, _req: {},
    )
    monkeypatch.setattr(srv_mod, "console", console)
    client = TestClient(srv_mod.app)

    res0 = client.post("/api/chat/inject", json={"clear": True})
    assert res0.status_code == 200 and res0.json()["chat"]["mode"] == "creative_and_strange_no_chat"
    vj0 = client.post("/api/vj/preview-prompt", json={"use_chat": True})
    assert vj0.status_code == 200 and vj0.json()["director_mode"] == "no_chat"

    sample_lines = [{"author": f"@u{i}", "message": f"msg {i}"} for i in range(10)]
    res10 = client.post("/api/chat/inject", json={"clear": True, "lines": sample_lines})
    assert res10.status_code == 200 and res10.json()["chat"]["count"] == 10
    vj10 = client.post("/api/vj/preview-prompt", json={"use_chat": True})
    assert vj10.status_code == 200 and vj10.json()["director_mode"] == "chat"
    console.shutdown()


def test_arbor_aleph_quotient_uncollapse_operators_and_artifact() -> None:
    assert ARTIFACT_PT.is_file()
    art = torch.load(str(ARTIFACT_PT), map_location="cpu", weights_only=False)
    assert "ARBOR-Aleph" in art["architecture"]
    assert float(art["kf_uncollapse"]["gamma_loo"]) > 0.0 and float(art["i2v_uncollapse"]["gamma_loo"]) > 0.0
    w_ov = cand08_overlap(13, 50, torch.device("cpu"), torch.float32)
    assert w_ov.shape == (50, 13) and torch.allclose(w_ov.sum(dim=1), torch.ones(50), atol=1e-5)
    z = torch.randn(1, 24, 2, 8, 8)
    z_out = apply_quotient_uncollapse(z, art["i2v_uncollapse"])
    assert z_out.shape == z.shape and not torch.allclose(z_out, z)
    t_norm, meta_norm = normalize_text_len(torch.randn(1, 77, 3072), {"minimax_token_tags": torch.ones(77)})
    assert t_norm.shape == (1, FIXED_TEXT_LEN, 3072) and meta_norm["minimax_token_tags"] is None
    cleaned = uncollapse_16x16_seams(torch.rand(4, 48, 64, 3), beta=0.20)
    assert cleaned.shape == (4, 48, 64, 3) and torch.all((cleaned >= 0.0) & (cleaned <= 1.0))


def test_no_affine_grid_sample_or_sine_wave_audio_in_generator() -> None:
    root = Path(__file__).resolve().parent.parent / "thereisnospoon"
    for name in ("clean_10s_generator.py", "arbor_aleph_uncollapse.py"):
        src = (root / name).read_text()
        assert "affine_grid" not in src and "grid_sample" not in src and "0.035 * torch.sin" not in src
        assert ".flip(" not in src and "0.042 * torch.sin" not in src


def test_autoloop_defaults_to_true_on_startup(monkeypatch: Any) -> None:
    monkeypatch.delenv("THEREISNOSPOON_AUTOPOLL", raising=False)
    monkeypatch.delenv("THEREISNOSPOON_AUTOLOOP", raising=False)
    console = BroadcastConsole(
        YouTubeChatPoller(), YouTubeRTMPBroadcaster,
        lambda _p: {"grid_16x16_ratio": 1.01, "format_valid": True},
        lambda _c, _r, _s: {"clean": {"grid_16x16_ratio": 1.01, "format_valid": True}},
        lambda _k, _req: {},
    )
    try:
        assert console.autoloop is True
    finally:
        console.shutdown()


def test_single_first_frame_i2v_conditioning_contract() -> None:
    root = Path(__file__).resolve().parent.parent / "thereisnospoon"
    aleph_src = (root / "arbor_aleph_uncollapse.py").read_text()
    assert '{"resolved_frame_index": 0, "latent": z_kf0}' in aleph_src and '"minimax_refs": []' in aleph_src


def test_clean_10s_generator_validation_errors(tmp_path: Path) -> None:
    gen = Clean10sClipGenerator()
    with pytest.raises(ValueError, match="nonempty prompt"):
        gen.generate("   ", tmp_path / "out.mp4")
    with pytest.raises(ValueError, match=".mp4 extension"):
        gen.generate("valid prompt", tmp_path / "out.avi")
    with pytest.raises(FileNotFoundError):
        gen.generate("valid prompt", tmp_path / "out.mp4", start_image=tmp_path / "missing.png")


def test_slerp_cond_and_word_set_helpers() -> None:
    words = _word_set("Cyberpunk Neon Observatory! 123 go")
    assert "cyberpunk" in words and "neon" in words and "123" in words and "go" not in words
    blended = _slerp_cond(torch.randn(1, 16, 64), torch.randn(1, 20, 64), 0.5)
    assert blended.shape == (1, 16, 64) and torch.isfinite(blended).all()


def test_stream_3d_frames_into_pipe_produces_exact_240_frames() -> None:
    buf = io.BytesIO()
    stream_3d_frames_into_pipe(torch.rand(22, 32, 48, 3, dtype=torch.float32), seed=42, stdin_pipe=buf, art={})
    assert len(buf.getvalue()) == 240 * 32 * 48 * 3


def test_cand08_overlap_identity_and_non_square() -> None:
    assert torch.allclose(cand08_overlap(7, 7, torch.device("cpu"), torch.float32), torch.eye(7))
    w = cand08_overlap(7, 22, torch.device("cpu"), torch.float32)
    assert w.shape == (22, 7) and torch.allclose(w.sum(dim=1), torch.ones(22), atol=1e-5)


def test_apply_quotient_uncollapse_noop_when_empty() -> None:
    z = torch.randn(1, 24, 1, 4, 4)
    assert torch.equal(apply_quotient_uncollapse(z, None), z)
    empty_fit = {"gamma_loo": 0.0, "w_hat": torch.eye(24), "b_hat": torch.zeros(24)}
    assert torch.equal(apply_quotient_uncollapse(z, empty_fit), z)


def test_normalize_text_len_truncation() -> None:
    t_out, meta = normalize_text_len(torch.randn(1, 300, 128), {"minimax_token_tags": torch.ones(300)}, fixed_len=256)
    assert t_out.shape == (1, 256, 128) and meta["minimax_token_tags"] is None


def test_resolve_artifact_security_and_types() -> None:
    assert resolve_artifact("../server.py") is None and resolve_artifact(".hidden.mp4") is None
    if (ARTIFACTS_DIR / "arbor_aleph_quotient_uncollapse_h3.pt").exists():
        assert resolve_artifact("arbor_aleph_quotient_uncollapse_h3.pt") is None


def test_quality_metrics_ratio_and_number_helpers() -> None:
    assert _number("24/1") == 24.0 and _number("invalid") is None
    assert _ratio(0.0, 0.0, neutral=1.0) == 1.0 and _ratio(2.0, 4.0) == 0.5
