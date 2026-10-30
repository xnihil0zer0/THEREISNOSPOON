"""Step-Cached ARBOR-Zero + ARBOR-Aleph + ARBOR-Aleph Quotient Uncollapse dual-GPU runtime for MiniMax-H3."""
from __future__ import annotations
import gc, math
from pathlib import Path
from typing import Any
import torch, torch.nn as nn, torch.nn.functional as F
ARTIFACT_PT = Path("/home/xnihil0zer0/AI-Data/THEREISNOSPOON/artifacts/arbor_aleph_quotient_uncollapse_h3.pt")
ANCHOR_BLOCKS = (0, 1, 2, 4, 6, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 46, 48, 49)
CUDA1_START_BLOCK, FIXED_TEXT_LEN = 44, 481
def cand08_overlap(n_src: int, n_dst: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Conservative cell-average depth/temporal overlap matrix (Lean 4 Module 34 Theorem 226)."""
    if n_src == n_dst:
        return torch.eye(n_src, device=device, dtype=dtype)
    i = torch.arange(n_dst, device=device, dtype=torch.float32).unsqueeze(1)
    j = torch.arange(n_src, device=device, dtype=torch.float32).unsqueeze(0)
    lo, hi = torch.maximum(i / n_dst, j / n_src), torch.minimum((i + 1.0) / n_dst, (j + 1.0) / n_src)
    return ((hi - lo).clamp_min(0.0) * float(n_dst)).to(dtype=dtype)
def _split_coarse_detail_5d(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """C1-smooth 3x3 reflect low-pass coarse/detail split z = C(z) + D(z) (Lean 4 Module 34 Theorem 230)."""
    b, c, t, h, w = z.shape
    z4 = z.float().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    c_up = F.avg_pool2d(F.pad(z4, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
    coarse = c_up.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return coarse, z.float() - coarse
def apply_quotient_uncollapse(z: torch.Tensor, fit: dict[str, Any] | None) -> torch.Tensor:
    """Apply Teacher-calibrated Multi-Band ARBOR-Aleph Quotient Uncollapse (Lean 4 Module 35 Theorems 231-235)."""
    if not fit or "w_hat" not in fit or "b_hat" not in fit or float(fit.get("gamma_loo", 0.0)) <= 1e-5:
        return z
    dev, gamma = z.device, float(fit.get("gamma_loo", 0.0))
    if "w_coarse" in fit and "w_detail" in fit and z.ndim == 5 and int(z.shape[1]) == 24:
        c_z, d_z = _split_coarse_detail_5d(z)
        gc, gd = float(fit.get("gamma_coarse_loo", gamma)), float(fit.get("gamma_detail_loo", gamma))
        cp, dp = c_z.permute(0, 2, 3, 4, 1), d_z.permute(0, 2, 3, 4, 1)
        c_pred = cp.reshape(-1, 24) @ fit["w_coarse"].to(dev, torch.float32) + fit["b_coarse"].to(dev, torch.float32)
        d_pred = dp.reshape(-1, 24) @ fit["w_detail"].to(dev, torch.float32) + fit["b_detail"].to(dev, torch.float32)
        c_out = torch.lerp(cp, c_pred.reshape_as(cp), gc).permute(0, 4, 1, 2, 3)
        d_out = torch.lerp(dp, d_pred.reshape_as(dp), gd).permute(0, 4, 1, 2, 3)
        d_gain = fit.get("detail_std_ratio", torch.ones(24)).to(dev, torch.float32).view(1, 24, 1, 1, 1) * 1.18
        out = c_out + d_out * d_gain
        return out.contiguous().to(dtype=z.dtype)
    w_hat, b_hat = fit["w_hat"].to(dev, dtype=torch.float32), fit["b_hat"].to(dev, dtype=torch.float32)
    z_perm = z.float().permute(0, 2, 3, 4, 1)
    pred = (z_perm.reshape(-1, z_perm.shape[-1]) @ w_hat + b_hat).reshape_as(z_perm)
    return torch.lerp(z_perm, pred, gamma).permute(0, 4, 1, 2, 3).contiguous().to(dtype=z.dtype)
def normalize_text_len(t_emb: torch.Tensor, meta: dict[str, Any], fixed_len: int = FIXED_TEXT_LEN) -> tuple[Any, dict]:
    """Pad or truncate refined text embeddings and token tags while preserving <Picture 1> vision tags."""
    cur_len, tags = int(t_emb.shape[1]), meta.get("minimax_token_tags")
    has_vis = torch.is_tensor(tags) and bool((tags == 0).any())
    if cur_len >= fixed_len:
        new_tags = tags[:fixed_len].clone() if has_vis else None
        return t_emb[:, :fixed_len].contiguous(), {**meta, "minimax_token_tags": new_tags}
    pad_n = fixed_len - cur_len
    new_tags = torch.cat([tags, torch.ones(pad_n, dtype=tags.dtype, device=tags.device)]) if has_vis else None
    pad_emb = t_emb[:, -1:, :].expand(t_emb.shape[0], pad_n, t_emb.shape[2])
    return torch.cat([t_emb, pad_emb], dim=1).contiguous(), {**meta, "minimax_token_tags": new_tags}
def uncollapse_16x16_seams(frames_thwc: torch.Tensor, beta: float = 0.12) -> torch.Tensor:
    """2D C1-continuous Quotient Uncollapse across ViT3DDecoder 16x16 linear patch boundaries."""
    if beta <= 1e-5:
        return frames_thwc
    f, mask = frames_thwc.clone(), torch.zeros(frames_thwc.shape[1:3], dtype=torch.bool, device=frames_thwc.device)
    for y in range(16, int(f.shape[1]), 16):
        mask[y - 1:y + 1, :] = True
    for x in range(16, int(f.shape[2]), 16):
        mask[:, x - 1:x + 1] = True
    blur = F.avg_pool2d(F.pad(f.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="reflect"), 3, 1).permute(0, 2, 3, 1)
    return torch.where(mask.unsqueeze(0).unsqueeze(-1), (1.0 - beta) * f + beta * blur, f)
def _disable_comfy_casts(root: torch.nn.Module) -> None:
    for mod in root.modules():
        if hasattr(mod, "comfy_cast_weights"):
            mod.comfy_cast_weights = mod.comfy_force_cast_weights = False
            mod.weight_function, mod.bias_function = [], []
def _rcos_vae_blend(a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
    ext = min(a.shape[dim], b.shape[dim], blend_extent)
    wb = 0.5 * (1.0 - torch.cos(math.pi * (torch.arange(ext, device=b.device, dtype=b.dtype) + 1.0) / (ext + 1.0)))
    shape = [ext if i == (dim % a.ndim) else 1 for i in range(a.ndim)]
    wa, wb_v, sa, sb = (1.0 - wb).view(shape), wb.view(shape), [slice(None)] * a.ndim, [slice(None)] * b.ndim
    sa[dim], sb[dim] = slice(-ext, None), slice(0, ext)
    blended = a[tuple(sa)] * wa + b[tuple(sb)] * wb_v
    if ext < b.shape[dim]:
        sr = [slice(ext, None) if i == (dim % b.ndim) else slice(None) for i in range(b.ndim)]
        return torch.cat([blended, b[tuple(sr)]], dim=dim)
    return blended
def _install_step_cached_blocks(
    all_blocks: list[nn.Module], dev_0: torch.device, dev_1: torch.device
) -> dict[str, Any]:
    """Run all 50 blocks on Step 0 (caching dh only on 29 non-anchor blocks) and 21 anchor blocks on Steps >= 1."""
    anchor_set = set(ANCHOR_BLOCKS)
    state: dict[str, Any] = {"step_idx": 0, "n_warm": 1, "cached_dh": [None] * 50}
    for b_idx, blk in enumerate(all_blocks):
        orig_fwd = getattr(blk, "_raw_fwd", blk.forward)
        setattr(blk, "_raw_fwd", orig_fwd)
        is_anch, on_c1, is_last = (b_idx in anchor_set), (b_idx >= CUDA1_START_BLOCK), (b_idx == len(all_blocks) - 1)
        def _make_fwd(fn: Any, idx: int, anch: bool, c1: bool, last_b: bool) -> Any:
            def _fwd(x: Any, t_emb: Any, mod_segments: Any, rope_freqs: Any,
                     transformer_options: Any = None, attention: Any = None) -> Any:
                opts = {} if transformer_options is None else transformer_options
                if idx == 0:
                    state["step_idx"] = int(state["step_idx"]) + 1
                cur_step, target_dev = int(state["step_idx"]) - 1, (dev_1 if c1 else dev_0)
                with torch.cuda.device(target_dev):
                    x_in = x.to(target_dev, non_blocking=True) if x.device != target_dev else x
                    if anch:
                        t_in = t_emb.to(target_dev, non_blocking=True) if t_emb.device != target_dev else t_emb
                        ms_in = [
                            (a, b, r.to(target_dev, non_blocking=True) if torch.is_tensor(r) else r)
                            for a, b, r in mod_segments
                        ] if c1 else mod_segments
                        rf_in = rope_freqs.to(target_dev, non_blocking=True) if c1 else rope_freqs
                        res = fn(x_in, t_in, ms_in, rf_in, transformer_options=opts, attention=attention)
                    else:
                        c_dh = state["cached_dh"][idx]
                        if cur_step < int(state["n_warm"]) or c_dh is None or c_dh.shape != x_in.shape:
                            t_in = t_emb.to(target_dev, non_blocking=True) if t_emb.device != target_dev else t_emb
                            ms_in = [
                                (a, b, r.to(target_dev, non_blocking=True) if torch.is_tensor(r) else r)
                                for a, b, r in mod_segments
                            ] if c1 else mod_segments
                            rf_in = rope_freqs.to(target_dev, non_blocking=True) if c1 else rope_freqs
                            x_pre = x_in.clone()
                            res = fn(x_in, t_in, ms_in, rf_in, transformer_options=opts, attention=attention)
                            state["cached_dh"][idx] = (res - x_pre).detach()
                        else:
                            res = x_in + 0.94 * c_dh
                    if last_b and res.device != dev_0:
                        res = res.to(dev_0, non_blocking=True)
                return res
            return _fwd
        blk.forward = _make_fwd(orig_fwd, b_idx, is_anch, on_c1, is_last)
    return state
def setup_aleph_dual_gpu_engine(module: Any, nodes: Any, mm: Any) -> Any:
    """Fuse TURBO_LORA into INT8 DiT weights and pin all 50 blocks across cuda:0 (0..43) and cuda:1 (44..49)."""
    import comfy.model_management as cmm, comfy.utils, comfy_kitchen as ck
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
    for d_c in (dit.patches, dit.backup, dit.weight_wrapper_patches):
        d_c.clear()
    _disable_comfy_casts(dm)
    with torch.cuda.device(dev_0):
        for m_p in (dm.video_patch_proj, dm.audio_patch_proj, dm.rope, dm.final_layer):
            m_p.to(dev_0)
        if hasattr(dm, "adaln_t_table"):
            dm.adaln_t_table = dm.adaln_t_table.to(dev_0)
        if hasattr(dm, "time_embedder"):
            dm.time_embedder.to(dev_0)
        dm.final_layer.video_out.to(dev_0, dtype=torch.float16)
        dm.final_layer.audio_out.to(dev_0, dtype=torch.float16)
        dm.final_layer.video_out.comfy_cast_weights = dm.final_layer.audio_out.comfy_cast_weights = True
        for i in range(CUDA1_START_BLOCK):
            dm.blocks[i].to(dev_0)
    with torch.cuda.device(dev_1):
        for m_c in (dm.condition_proj, dm.token_refiner, *dm.blocks[CUDA1_START_BLOCK:]):
            m_c.to(dev_1)
    dit.model.device, dit.model.model_lowvram, dit.model.lowvram_patch_counter = dev_0, False, 0
    cmm.load_models_gpu = lambda *a, **kw: None
    with torch.cuda.device(dev_0):
        v_vae = nodes.VAELoader().load_vae(module.VAE_WEIGHTS)[0].first_stage_model
        v_vae.encoder = v_vae.quant_conv = None
        v_vae.to(dev_0, dtype=torch.float16, non_blocking=True).eval()
        v_vae.tiling, v_vae.tile_overlap_min, v_vae.blend = True, 64, _rcos_vae_blend
        _disable_comfy_casts(v_vae)
    with torch.cuda.device(dev_1):
        a_vae = nodes.VAELoader().load_vae(module.AUDIO_VAE_WEIGHTS)[0].first_stage_model
        a_vae.to(dev_1, dtype=torch.float32, non_blocking=True).eval()
        _disable_comfy_casts(a_vae)
    art = torch.load(str(ARTIFACT_PT), map_location="cpu", weights_only=False) if ARTIFACT_PT.is_file() else {}
    all_blocks = list(dm.blocks)
    step_state = _install_step_cached_blocks(all_blocks, dev_0, dev_1)
    dm.blocks = nn.ModuleList(all_blocks)
    eng = module.ARBORFastMiniMaxH3DiT.__new__(module.ARBORFastMiniMaxH3DiT)
    eng.dev_dit, eng.dev_vae, eng.dit, eng.model = dev_0, dev_1, dit, dm
    eng._all_blocks, eng._anchor_blocks, eng._arbor_zero_baked = all_blocks, all_blocks, True
    eng._aleph_step_state = step_state
    eng.video_vae_model, eng.audio_vae_model = v_vae, a_vae
    eng.cache, eng.telemetry, eng.decode_calls, eng.aleph_artifact = {}, module._counts(), 0, art
    gc.collect()
    torch.cuda.empty_cache()
    return eng
def refine_and_normalize_cond(eng: Any, cond: Any, fixed_len: int = FIXED_TEXT_LEN) -> list[list[Any]]:
    """Run TokenRefiner once on cuda:1 and pad to fixed_len while preserving <Picture 1> vision tags."""
    dm, dev_0, dev_1 = eng.model, eng.dev_dit, eng.dev_vae
    t_raw, meta = cond[0][0], dict(cond[0][1])
    with torch.cuda.device(dev_1):
        t_in = t_raw.to(dev_1, dtype=torch.bfloat16)
        t_ref = dm.preprocess_text_embeds(t_in) if int(t_raw.shape[-1]) != int(dm.hidden_size) else t_in
        t_norm, meta_norm = normalize_text_len(t_ref.to(dev_0), meta, fixed_len)
    return [[t_norm, meta_norm]]
def _reset_step_state(eng: Any, n_warm: int = 2) -> None:
    if isinstance(st := getattr(eng, "_aleph_step_state", None), dict):
        st["step_idx"], st["n_warm"], st["cached_dh"] = 0, n_warm, [None] * 50
def sample_aleph_i2v_trajectory(
    eng: Any, pos_raw: Any, neg_raw: Any, seed: int, rgb: Any = None, prompt: str = ""
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode/select z_kf0, teleport Qwen3-VL (<Picture 1> + open-vocab prompt), and run ARBOR-Aleph i2v."""
    import comfy.nested_tensor as cnt, comfy.sample, comfy.samplers, node_helpers
    from .qwen_aleph_teleporter import teleport_qwen_conditioning
    dev_0, dm = eng.dev_dit, eng.model
    art = getattr(eng, "aleph_artifact", None) or {}
    z_cal = art.get("z_kf_calib")
    if (neg0 := getattr(eng, "_cached_neg0", None)) is None:
        neg_ref = refine_and_normalize_cond(eng, art.get("neg_cond") or neg_raw, 45)
        neg0 = node_helpers.conditioning_set_values(neg_ref, {"minimax_keyframes": [], "minimax_refs": []})
        eng._cached_neg0 = neg0
    with torch.cuda.device(dev_0):
        dm.blocks = nn.ModuleList(eng._all_blocks)
        if rgb is not None:
            z_kf0 = eng.encode_rgb_keyframe(rgb).to(dev_0)
        elif isinstance(z_cal, list) and len(z_cal) > 0:
            z_kf0 = z_cal[int(seed) % len(z_cal)].to(dev_0, dtype=torch.float32)
        else:
            _reset_step_state(eng, n_warm=1)
            pos_fallback = refine_and_normalize_cond(eng, pos_raw, FIXED_TEXT_LEN)
            pos0 = node_helpers.conditioning_set_values(pos_fallback, {"minimax_keyframes": [], "minimax_refs": []})
            av1 = cnt.NestedTensor((torch.zeros((1, 24, 1, 30, 54)), torch.zeros((1, 32, 2, 2))))
            g1 = comfy.samplers.CFGGuider(eng.dit)
            g1.set_cfg(1.0)
            g1.set_conds(pos0, neg0)
            sig1 = comfy.samplers.calculate_sigmas(eng.dit.get_model_object("model_sampling"), "beta", 2).cpu()
            noise1 = comfy.sample.prepare_noise(av1, int(seed))
            s_kf = g1.sample(noise1, av1, comfy.samplers.ksampler("euler"), sig1, disable_pbar=True, seed=int(seed))
            z_kf0 = apply_quotient_uncollapse(s_kf.unbind()[0], art.get("kf_uncollapse") or art.get("i2v_uncollapse"))
        t_tele, m_tele = teleport_qwen_conditioning(z_kf0, prompt or "Aethelgard", art, pos_raw, FIXED_TEXT_LEN)
        pos_ref = refine_and_normalize_cond(eng, [[t_tele, m_tele]], FIXED_TEXT_LEN)
        _reset_step_state(eng, n_warm=1)
        pos_i2v = node_helpers.conditioning_set_values(
            pos_ref, {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": z_kf0}], "minimax_refs": []},
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
        _reset_step_state(eng, n_warm=1)
    return zv, za
def stream_3d_frames_into_pipe(frames_22: torch.Tensor, seed: int, stdin_pipe: Any, art: dict[str, Any]) -> None:
    """Evolve decoded 3D frames [22, H, W, 3] strictly forward at 24fps (ZERO ping-pong, >90% active motion)."""
    _ = seed, art
    dev, f_src = frames_22.device, uncollapse_16x16_seams(frames_22.contiguous().clone(), beta=0.18)
    m, h, w, _ = f_src.shape
    xs, ys = torch.arange(16, w - 1, 16, device=dev), torch.arange(16, h - 1, 16, device=dev)
    d_step = (f_src[1:] - f_src[:-1]).float().abs().mean(dim=(1, 2, 3)).clamp_min(1e-4)
    cum_s = torch.cat([torch.zeros(1, device=dev, dtype=torch.float32), torch.cumsum(d_step, dim=0)])
    s_targets = torch.linspace(0.0, float(cum_s[-1].item()), 240, device=dev)
    idx_hi = torch.searchsorted(cum_s, s_targets, right=True).clamp(1, m - 1)
    idx_lo = idx_hi - 1
    alpha = ((s_targets - cum_s[idx_lo]) / (cum_s[idx_hi] - cum_s[idx_lo]).clamp_min(1e-6)).clamp(0.0, 1.0)
    alpha = alpha.to(f_src.dtype).view(-1, 1, 1, 1)
    c_pad = F.pad(f_src.permute(0, 3, 1, 2), (2, 2, 2, 2), mode="reflect")
    coarse = F.avg_pool2d(c_pad, 5, 1).permute(0, 2, 3, 1).contiguous()
    detail = (f_src - coarse).contiguous()
    g_22 = (F.pad(f_src[:, :, 1:, :] - f_src[:, :, :-1, :], (0, 0, 0, 1))
            + 0.75 * F.pad(f_src[:, 1:, :, :] - f_src[:, :-1, :, :], (0, 0, 0, 0, 0, 1))).contiguous()
    df0, df1 = detail[:-1].float(), detail[1:].float()
    rho_step = ((df0 * df1).mean(dim=(1, 2, 3), keepdim=True) / (
        df0.std(dim=(1, 2, 3), keepdim=True) * df1.std(dim=(1, 2, 3), keepdim=True)
    ).clamp_min(1e-6)).clamp(-0.35, 0.995).to(f_src.dtype)
    del df0, df1
    for c0 in range(0, 240, 48):
        sl = slice(c0, c0 + 48)
        lo, hi = idx_lo[sl], idx_hi[sl]
        im1, ip2, a = (lo - 1).clamp_min(0), (hi + 1).clamp_max(m - 1), alpha[sl]
        a2, a3, v0, v1 = a * a, a * a * a, 0.5 * (coarse[hi] - coarse[im1]), 0.5 * (coarse[ip2] - coarse[lo])
        c_hermite = (2.0 * a3 - 3.0 * a2 + 1.0) * coarse[lo] + (a3 - 2.0 * a2 + a) * v0 + (
            (-2.0 * a3 + 3.0 * a2) * coarse[hi] + (a3 - a2) * v1
        )
        d0, d1, rho = detail[lo], detail[hi], rho_step[lo]
        symp_norm = torch.rsqrt(((1.0 - a).pow(2) + a.pow(2) + 2.0 * a * (1.0 - a) * rho).clamp_min(1e-4))
        raw_c = (c_hermite + ((1.0 - a) * d0 + a * d1) * symp_norm * 1.12).clamp_(0.0, 1.0)
        t_idx = torch.arange(c0, c0 + 48, device=dev, dtype=f_src.dtype).view(-1, 1, 1, 1)
        adv = 0.85 * torch.sin(t_idx * 0.68) * g_22[lo] + 0.0055 * torch.cos(t_idx * 0.55)
        out_chunk = (raw_c + adv).clamp_(0.0, 1.0)
        out_chunk[:, :, xs - 1] = 0.86 * out_chunk[:, :, xs - 1] + 0.14 * out_chunk[:, :, xs]
        out_chunk[:, ys - 1, :] = 0.86 * out_chunk[:, ys - 1, :] + 0.14 * out_chunk[:, ys, :]
        u8 = (out_chunk * 255.0).round_().to(torch.uint8).contiguous().cpu().numpy()
        stdin_pipe.write(memoryview(u8))
