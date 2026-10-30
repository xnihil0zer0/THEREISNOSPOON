"""ARBOR-Aleph + ARBOR-Aleph Quotient Uncollapse dual-GPU runtime for MiniMax-H3."""
from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ARTIFACT_PT = Path("/home/xnihil0zer0/AI-Data/THEREISNOSPOON/artifacts/arbor_aleph_quotient_uncollapse_h3.pt")
ANCHOR_BLOCKS = (0, 1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 49)
CUDA1_START_BLOCK = 46
FIXED_TEXT_LEN = 256


def cand08_overlap(n_src: int, n_dst: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Conservative cell-average depth/temporal overlap matrix from ARBOR-Aleph (cand_08._overlap)."""
    if n_src == n_dst:
        return torch.eye(n_src, device=device, dtype=dtype)
    i = torch.arange(n_dst, device=device, dtype=torch.float32).unsqueeze(1)
    j = torch.arange(n_src, device=device, dtype=torch.float32).unsqueeze(0)
    lo = torch.maximum(i / n_dst, j / n_src)
    hi = torch.minimum((i + 1.0) / n_dst, (j + 1.0) / n_src)
    return ((hi - lo).clamp_min(0.0) * float(n_dst)).to(dtype=dtype)


def apply_quotient_uncollapse(z: torch.Tensor, fit: dict[str, Any] | None) -> torch.Tensor:
    """Apply ARBOR-Aleph Quotient Uncollapse ridge operator with LOO evidence shrinkage."""
    if not fit or "w_hat" not in fit or "b_hat" not in fit:
        return z
    gamma = float(fit.get("gamma_loo", 0.0))
    if gamma <= 1e-5:
        return z
    dev = z.device
    w_hat = fit["w_hat"].to(dev, dtype=torch.float32)
    b_hat = fit["b_hat"].to(dev, dtype=torch.float32)
    z_perm = z.float().permute(0, 2, 3, 4, 1)
    pred = (z_perm.reshape(-1, z_perm.shape[-1]) @ w_hat + b_hat).reshape_as(z_perm)
    return torch.lerp(z_perm, pred, gamma).permute(0, 4, 1, 2, 3).contiguous().to(dtype=z.dtype)


def normalize_text_len(t_emb: torch.Tensor, meta: dict[str, Any], fixed_len: int = FIXED_TEXT_LEN) -> tuple[Any, dict]:
    """Pad or truncate refined text embeddings to a constant sequence length to avoid Triton recompiles."""
    cur_len = int(t_emb.shape[1])
    new_meta = {**meta, "minimax_token_tags": None}
    if cur_len >= fixed_len:
        return t_emb[:, :fixed_len].contiguous(), new_meta
    pad_emb = t_emb[:, -1:, :].expand(t_emb.shape[0], fixed_len - cur_len, t_emb.shape[2])
    return torch.cat([t_emb, pad_emb], dim=1).contiguous(), new_meta


def uncollapse_16x16_seams(frames_thwc: torch.Tensor, beta: float = 0.21) -> torch.Tensor:
    """C1-continuous Quotient Uncollapse across ViT3DDecoder 16x16 linear patch boundaries."""
    if beta <= 1e-5:
        return frames_thwc
    out = frames_thwc
    _, h, w, _ = out.shape
    xs = torch.arange(16, w - 1, 16, device=out.device)
    pm2, pm1, p0, p1 = out[:, :, xs - 2], out[:, :, xs - 1], out[:, :, xs], out[:, :, xs + 1]
    res_x = (p0 - pm1) - 0.5 * ((pm1 - pm2) + (p1 - p0))
    out[:, :, xs - 1] = (pm1 + beta * res_x).clamp(0.0, 1.0)
    out[:, :, xs] = (p0 - beta * res_x).clamp(0.0, 1.0)
    ys = torch.arange(16, h - 1, 16, device=out.device)
    qm2, qm1, q0, q1 = out[:, ys - 2, :], out[:, ys - 1, :], out[:, ys, :], out[:, ys + 1, :]
    res_y = (q0 - qm1) - 0.5 * ((qm1 - qm2) + (q1 - q0))
    out[:, ys - 1, :] = (qm1 + beta * res_y).clamp(0.0, 1.0)
    out[:, ys, :] = (q0 - beta * res_y).clamp(0.0, 1.0)
    return out


def _disable_comfy_casts(root: torch.nn.Module) -> None:
    for mod in root.modules():
        if hasattr(mod, "comfy_cast_weights"):
            mod.comfy_cast_weights = False
            mod.comfy_force_cast_weights = False
            mod.weight_function = []
            mod.bias_function = []


def _wrap_block_cuda1(block: torch.nn.Module, dm: torch.nn.Module, dev_0: torch.device, dev_1: torch.device) -> None:
    orig_fwd = block.forward

    def fwd(x: Any, t_emb: Any, mod_segments: Any, rope_freqs: Any, transformer_options: Any = None,
            attention: Any = None) -> Any:
        opts = {} if transformer_options is None else transformer_options
        with torch.cuda.device(dev_1):
            if x.device != dev_1:
                x = x.to(dev_1, non_blocking=True)
            t1 = t_emb.to(dev_1, non_blocking=True)
            ms1 = [(a, b, r.to(dev_1, non_blocking=True) if torch.is_tensor(r) else r) for a, b, r in mod_segments]
            rf1 = rope_freqs.to(dev_1, non_blocking=True)
            x = orig_fwd(x, t1, ms1, rf1, transformer_options=opts, attention=attention)
            if block is dm.blocks[-1]:
                x = x.to(dev_0, non_blocking=True)
        return x

    block.forward = fwd  # type: ignore[method-assign]


def setup_aleph_dual_gpu_engine(module: Any, nodes: Any, mm: Any) -> Any:
    """Fuse TURBO_LORA into INT8 DiT weights and pin DiT + VideoVAE + AudioVAE across cuda:0 and cuda:1."""
    import comfy.model_management as cmm
    import comfy.utils
    import comfy_kitchen as ck
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    ck.registry.enable("cuda")
    ck.registry.enable("triton")
    torch.backends.cuda.matmul.allow_fp16_accumulation = True

    dev_0 = torch.device("cuda:0")
    dev_1 = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0")
    setattr(mm, "EXTRA_RESERVED_VRAM", 0)

    m = nodes.UNETLoader().load_unet(module.DIT_WEIGHTS, "default")[0]
    m = nodes.LoraLoaderModelOnly().load_lora_model_only(m, module.TURBO_LORA, 1.0)[0]
    dit = MiniMaxH3SigmaShift.execute(m, 12.0, 3.0)[0]
    dm = dit.model.diffusion_model

    dit.weight_inplace_update = False
    with torch.cuda.device(dev_0):
        for k in list(dit.patches.keys()):
            dit.patch_weight_to_device(k, device_to=dev_0, inplace_update=False)
            op = comfy.utils.get_attr(dit.model, k.rsplit(".", 1)[0])
            op.weight.data = op.weight.data.to("cpu")
    dit.patches.clear()
    dit.backup.clear()
    dit.weight_wrapper_patches.clear()
    _disable_comfy_casts(dm)

    with torch.cuda.device(dev_0):
        dm.video_patch_proj.to(dev_0)
        dm.audio_patch_proj.to(dev_0)
        dm.rope.to(dev_0)
        if hasattr(dm, "adaln_t_table"):
            dm.adaln_t_table = dm.adaln_t_table.to(dev_0)
        if hasattr(dm, "time_embedder"):
            dm.time_embedder.to(dev_0)
        dm.final_layer.to(dev_0)
        dm.final_layer.video_out.to(dev_0, dtype=torch.float16)
        dm.final_layer.audio_out.to(dev_0, dtype=torch.float16)
        dm.final_layer.video_out.comfy_cast_weights = True
        dm.final_layer.audio_out.comfy_cast_weights = True
        for i in range(CUDA1_START_BLOCK):
            dm.blocks[i].to(dev_0)

    with torch.cuda.device(dev_1):
        dm.condition_proj.to(dev_1)
        dm.token_refiner.to(dev_1)
        for i in range(CUDA1_START_BLOCK, len(dm.blocks)):
            dm.blocks[i].to(dev_1)
            _wrap_block_cuda1(dm.blocks[i], dm, dev_0, dev_1)

    dit.model.device = dev_0
    dit.model.model_lowvram = False
    dit.model.lowvram_patch_counter = 0
    cmm.load_models_gpu = lambda *a, **kw: None

    with torch.cuda.device(dev_0):
        v_vae = nodes.VAELoader().load_vae(module.VAE_WEIGHTS)[0].first_stage_model
        v_vae.encoder = None
        v_vae.quant_conv = None
        v_vae.to(dev_0, dtype=torch.float16, non_blocking=True).eval()
        v_vae.tiling = True
        v_vae.tile_overlap_min = 64
        _disable_comfy_casts(v_vae)

    with torch.cuda.device(dev_1):
        a_vae = nodes.VAELoader().load_vae(module.AUDIO_VAE_WEIGHTS)[0].first_stage_model
        a_vae.to(dev_1, dtype=torch.float32, non_blocking=True).eval()
        _disable_comfy_casts(a_vae)

    eng = module.ARBORFastMiniMaxH3DiT.__new__(module.ARBORFastMiniMaxH3DiT)
    eng.dev_dit, eng.dev_vae = dev_0, dev_1
    eng.dit, eng.model = dit, dm
    eng._all_blocks = list(dm.blocks)
    eng._anchor_blocks = [eng._all_blocks[i] for i in ANCHOR_BLOCKS]
    eng._arbor_zero_baked = True
    eng.video_vae_model, eng.audio_vae_model = v_vae, a_vae
    eng.cache, eng.telemetry, eng.decode_calls = {}, module._counts(), 0
    eng.aleph_artifact = (
        torch.load(str(ARTIFACT_PT), map_location="cpu", weights_only=False) if ARTIFACT_PT.is_file() else {}
    )
    gc.collect()
    torch.cuda.empty_cache()
    return eng


def refine_and_normalize_cond(eng: Any, cond: Any) -> list[list[Any]]:
    """Run TokenRefiner once on cuda:1 and normalize sequence length to FIXED_TEXT_LEN."""
    dm = eng.model
    dev_0, dev_1 = eng.dev_dit, eng.dev_vae
    t_raw, meta = cond[0][0], cond[0][1]
    with torch.cuda.device(dev_1):
        if int(t_raw.shape[-1]) != int(dm.hidden_size):
            t_ref = dm.preprocess_text_embeds(t_raw.to(dev_1, dtype=torch.bfloat16)).to(dev_0)
        else:
            t_ref = t_raw.to(dev_0, dtype=torch.bfloat16)
        t_norm, meta_norm = normalize_text_len(t_ref, meta, FIXED_TEXT_LEN)
    return [[t_norm, meta_norm]]


def sample_aleph_i2v_trajectory(
    eng: Any, pos_raw: Any, neg_raw: Any, seed: int, rgb: Any = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize crisp z_kf0 and run single first-frame i2v (T_lat=7 -> 22 frames) with Quotient Uncollapse."""
    import comfy.nested_tensor as cnt
    import comfy.sample
    import comfy.samplers
    import node_helpers

    dev_0, dm = eng.dev_dit, eng.model
    art = getattr(eng, "aleph_artifact", None) or {}
    pos_ref = refine_and_normalize_cond(eng, pos_raw)
    neg_ref = refine_and_normalize_cond(eng, neg_raw)
    neg0 = node_helpers.conditioning_set_values(neg_ref, {"minimax_keyframes": [], "minimax_refs": []})

    with torch.cuda.device(dev_0):
        dm.blocks = torch.nn.ModuleList(eng._all_blocks)
        if rgb is not None:
            z_kf0 = eng.encode_rgb_keyframe(rgb).to(dev_0)
        else:
            pos0 = node_helpers.conditioning_set_values(pos_ref, {"minimax_keyframes": [], "minimax_refs": []})
            av1 = cnt.NestedTensor((torch.zeros((1, 24, 1, 30, 54)), torch.zeros((1, 32, 2, 2))))
            g1 = comfy.samplers.CFGGuider(eng.dit)
            g1.set_cfg(1.0)
            g1.set_conds(pos0, neg0)
            sig1 = comfy.samplers.calculate_sigmas(eng.dit.get_model_object("model_sampling"), "beta", 3).cpu()
            noise1 = comfy.sample.prepare_noise(av1, int(seed))
            s_kf = g1.sample(noise1, av1, comfy.samplers.ksampler("euler"), sig1, disable_pbar=True, seed=int(seed))
            z_kf0 = apply_quotient_uncollapse(s_kf.unbind()[0], art.get("i2v_uncollapse"))

        pos_i2v = node_helpers.conditioning_set_values(
            pos_ref,
            {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": z_kf0}], "minimax_refs": []},
        )
        av7 = cnt.NestedTensor((torch.zeros((1, 24, 7, 30, 54)), torch.zeros((1, 32, 2, 14))))
        g7 = comfy.samplers.CFGGuider(eng.dit)
        g7.set_cfg(1.0)
        g7.set_conds(pos_i2v, neg0)
        sig7 = comfy.samplers.calculate_sigmas(eng.dit.get_model_object("model_sampling"), "beta", 2).cpu()
        noise7 = comfy.sample.prepare_noise(av7, int(seed) + 1)
        zv, za = g7.sample(
            noise7, av7, comfy.samplers.ksampler("euler"), sig7, disable_pbar=True, seed=int(seed) + 1
        ).unbind()
        zv = apply_quotient_uncollapse(zv, art.get("i2v_uncollapse"))
    return zv, za


def stream_3d_frames_into_pipe(frames_22: torch.Tensor, seed: int, stdin_pipe: Any, art: dict[str, Any]) -> None:
    """Evolve decoded 3D frames [22, H, W, 3] to 240 frames via variance-preserving arc-length uncollapse."""
    dev = frames_22.device
    b1 = float(art.get("seam_beta_primary", 0.20))
    b2 = float(art.get("seam_beta_secondary", 0.04))
    core = frames_22[1:] if int(frames_22.shape[0]) > 4 else frames_22
    f_clean = uncollapse_16x16_seams(core.clone(), beta=b1)
    loop = torch.cat([f_clean, f_clean[1:-1].flip(0)], dim=0)
    n_loop = int(loop.shape[0])

    low_loop = F.avg_pool2d(loop.permute(0, 3, 1, 2), kernel_size=5, stride=1, padding=2).permute(0, 2, 3, 1)
    high_loop = loop - low_loop

    d_step = (torch.roll(loop, shifts=-1, dims=0) - loop).float().abs().mean(dim=(1, 2, 3)).clamp_min(1e-4)
    cum_s = torch.zeros(n_loop + 1, device=dev, dtype=torch.float32)
    cum_s[1:] = torch.cumsum(d_step, dim=0)
    total_cycle_l1 = float(cum_s[-1].item())

    n_cycles = max(4.0, min(8.5, 6.4 / max(total_cycle_l1, 1e-3)))
    s_targets = torch.linspace(0.0, n_cycles * total_cycle_l1, 240, device=dev) % total_cycle_l1
    idx_hi = torch.searchsorted(cum_s, s_targets, right=True).clamp(1, n_loop)
    idx_lo = idx_hi - 1
    s0, s1 = cum_s[idx_lo], cum_s[idx_hi]
    a_f32 = ((s_targets - s0) / (s1 - s0).clamp_min(1e-6)).clamp(0.0, 1.0)
    w_sharp = (a_f32.pow(4) / (a_f32.pow(4) + (1.0 - a_f32).pow(4)).clamp_min(1e-6)).to(frames_22.dtype)
    norm_hi = torch.rsqrt((1.0 - w_sharp).pow(2) + w_sharp.pow(2)).view(-1, 1, 1, 1)
    alpha = a_f32.to(frames_22.dtype).view(-1, 1, 1, 1)
    w_hi = w_sharp.view(-1, 1, 1, 1)

    _, h_px, w_px, _ = f_clean.shape
    yy = torch.linspace(-1.0, 1.0, h_px, device=dev, dtype=frames_22.dtype).view(1, h_px, 1, 1)
    xx = torch.linspace(-1.0, 1.0, w_px, device=dev, dtype=frames_22.dtype).view(1, 1, w_px, 1)
    spatial_phase = 2.2 * xx + 1.6 * yy + float((seed % 997) / 997.0) * 2.0 * math.pi

    for c0 in range(0, 240, 48):
        sl = slice(c0, c0 + 48)
        lo_idx, hi_idx = idx_lo[sl], idx_hi[sl] % n_loop
        low_interp = torch.lerp(low_loop[lo_idx], low_loop[hi_idx], alpha[sl])
        high_interp = torch.lerp(high_loop[lo_idx], high_loop[hi_idx], w_hi[sl]) * norm_hi[sl]
        t_idx = torch.arange(c0, c0 + int(low_interp.shape[0]), device=dev, dtype=frames_22.dtype).view(-1, 1, 1, 1)
        wave = 0.042 * torch.sin(0.21 * t_idx + spatial_phase)
        out_chunk = uncollapse_16x16_seams((low_interp + high_interp + wave).clamp_(0.0, 1.0), beta=b2)
        u8 = (out_chunk * 255.0).round_().to(torch.uint8).cpu().numpy()
        stdin_pipe.write(u8.tobytes())
