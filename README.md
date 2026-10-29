# THEREISNOSPOON (`@THEREISNOSPOON-t4j`)

**Autonomous Live YouTube Chat VJ & Clean 10-Second Neural Audio-Video Broadcast Engine**

- **YouTube Channel**: [`https://www.youtube.com/@THEREISNOSPOON-t4j`](https://www.youtube.com/@THEREISNOSPOON-t4j) (`UC9ryqiyaNstEvvsw9EKpccA`)
- **YouTube Live Stream URL**: [`https://www.youtube.com/channel/UC9ryqiyaNstEvvsw9EKpccA/live`](https://www.youtube.com/channel/UC9ryqiyaNstEvvsw9EKpccA/live)
- **Neural Audio-Video Engine**: `ARBOR-Zero` `MiniMax-H3` (`ARBORFastMiniMaxH3DiT` + `MiniMaxH3VideoVAE` + `MiniMaxH3AudioVAE` + `Qwen3-VL`)
- **Clip Specification**: Standalone, unblended, unstitched **10.0-second (`240`-frame @ `24 fps`, `864x480`, `48000 Hz` stereo AAC) `.mp4` clips**

---

## 1. Architecture Overview

`THEREISNOSPOON` is an autonomous interactive broadcast system that turns live YouTube viewer chat into a continuous sequence of newly generated 10-second neural audio-video scenes with synchronized spoken dialogue:

1. **Live YouTube Chat Poller (`thereisnospoon/youtube_chat_poller.py`)**:
   - Connects to `@THEREISNOSPOON-t4j` (`UC9ryqiyaNstEvvsw9EKpccA`) via YouTube InnerTube `live_chat/get_live_chat` continuation polling (plus optional YouTube Data API v3 and live dashboard chat injection).
   - Maintains a thread-safe rolling window of the **last 10 chat lines**.
2. **Open-Ended Chat VJ Director (`thereisnospoon/chat_vj_director.py`)**:
   - **Active Chat Mode (`1..10` Chat Lines)**: Acts as a live **Chat VJ**, open-endedly interpreting the collective humor, questions, requests, and energy across the last 10 lines of chat to conjure something genuinely fun, vivid, and entertaining for the chat to watch, including a rich visual scene description and synchronized spoken dialogue (`<d>[English] "..."</d>`).
   - **No-Chat Mode (`0` Chat Lines)**: When no chat messages have arrived yet, the expanded system prompt instructs the Chat VJ to be **wildly creative and strange** — inventing surreal, bizarre, visually mesmerizing scenes and intriguing spoken monologue so the broadcast is never static or repetitive.
   - Uses the local `Qwen3-VL` text encoder (`CLIPLoader` `qwen3vl_4b_bf16.safetensors` `clip.tokenize -> clip.generate -> clip.decode` + `ARBORFastMiniMaxH3TextEncoder`) for 100% local open-ended prompt synthesis and conditioning.
3. **Clean 10-Second Audio-Video Clip Generator (`thereisnospoon/clean_10s_generator.py`)**:
   - Generates a single continuous `243`-frame (`72` video latent tokens, `405` audio latent tokens, `17 * 14 + 5 = 243` per Lean 4 Category 26 Theorem 194) MiniMax-H3 clip and trims cleanly to **`240` frames (`10.000s` @ `24 fps`)** with **`48000 Hz` stereo AAC audio**.
   - **Zero Seam Blending & Zero Latent Stitching**: No multi-chunk latent splicing or cross-fading.
   - **Zero `16x16` ViT Patch Grid Artifacts**: Always decodes via native `vae.decode(zv)` (`latent_T = 7`, `tokens_chunk_size = 5`, `token_overlap = 2`), never `_adaptive_decode` with `latent_T = 15/20`, and never applies `30x54` spatial latent masks (`bg_vault_mask`, `fg_corridor`, `T_lf2ref`).
   - **Natural Unclamped Motion**: Never pins the terminal frame (`frame 120` or `242`) to a static keyframe, allowing natural camera and character motion across the full 10-second duration.
4. **Continuous RTMP Broadcaster & Web Console (`thereisnospoon/rtmp_broadcaster.py` & `thereisnospoon/server.py`)**:
   - Streams generated 10-second clips to `rtmp://a.rtmp.youtube.com/live2/<STREAM_KEY>` via FFmpeg while serving a real-time web control console (`http://127.0.0.1:8090`).
5. **Quantitative & Visual Quality Verifier (`thereisnospoon/quality_metrics.py`)**:
   - Audits generated 10-second clips via `ffprobe`, `grid_16x16_ratio`, temporal motion fluidity, `vae_17f_pulse_ratio`, Laplacian sharpness, and audio RMS against Study v2 clips.

---

## 2. Running Verification & Broadcast Server

All GPU generation jobs are dispatched through the hardware resource queue wrapper:

```bash
/home/xnihil0zer0/AI-Data/ARBOR-Research/adversarial_audit/tools/run_job.sh both 600 \
  /home/xnihil0zer0/miniconda3/bin/python3 scripts/run_first_verification_test.py
```

To launch the local Web Broadcast Console:

```bash
/home/xnihil0zer0/miniconda3/bin/uvicorn thereisnospoon.server:app --host 127.0.0.1 --port 8090
```
