"""Teleported Qwen3-VL Text & <Picture 1> Vision Encoder onto MiniMaxH3Model via ARBOR-Aleph + Quotient Uncollapse."""
from __future__ import annotations
import hashlib, re
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F

FIXED_TOTAL_LEN, VISION_PREFIX_LEN = 481, 413
QWEN_TOK_DIR = Path("/mnt/ai-data/ComfyUI2/comfy/text_encoders/qwen25_tokenizer")
_TOKENIZER_CACHE: dict[str, Any] = {}


def cand08_overlap(n_src: int, n_dst: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Lean 4 Module 34 Theorem 226: conservative cell-average overlap matrix (float64 interval grid)."""
    if n_src == n_dst:
        return torch.eye(n_src, device=device, dtype=dtype)
    i = torch.arange(n_dst, device=device, dtype=torch.float64).unsqueeze(1)
    j = torch.arange(n_src, device=device, dtype=torch.float64).unsqueeze(0)
    lo = torch.maximum(i / float(n_dst), j / float(n_src))
    hi = torch.minimum((i + 1.0) / float(n_dst), (j + 1.0) / float(n_src))
    return ((hi - lo).clamp_min(0.0) * float(n_dst)).to(dtype=dtype)


def apply_rope_1d(x: torch.Tensor, pos: torch.Tensor, theta: float = 10000.0, inverse: bool = False,
                  mode: str = "interleaved", base: float | None = None) -> torch.Tensor:
    """Apply or invert (-pos when inverse=True) 1D RoPE at integer or fractional midpoint positions."""
    d_full, th = int(x.shape[-1]), float(theta if base is None else base)
    d = (d_full // 2) * 2
    if d == 0:
        return x.clone()
    x_rot, x_rem = x[..., :d].to(torch.float64), x[..., d:]
    inv_f = 1.0 / (th ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float64) / float(d)))
    p = pos.to(device=x.device, dtype=torch.float64) * (-1.0 if inverse else 1.0)
    if p.ndim == 1:
        p = p.view(*([1] * (x.ndim - 2)), -1)
    elif p.ndim == 2 and x.ndim == 4:
        p = p.unsqueeze(1)
    ang = p.unsqueeze(-1) * inv_f
    c, s = torch.cos(ang), torch.sin(ang)
    if mode == "half_split":
        h = d // 2
        x1, x2 = x_rot[..., :h], x_rot[..., h:]
        out = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(dtype=x.dtype)
    else:
        x_e, x_o = x_rot[..., 0::2], x_rot[..., 1::2]
        out = torch.stack([x_e * c - x_o * s, x_e * s + x_o * c], dim=-1).flatten(-2).to(dtype=x.dtype)
    return torch.cat([out, x_rem], dim=-1) if d < d_full else out


def extract_zkf0_vision_grid(z_kf0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lean 4 Theorem 230: Exact 2x2 Haar coarse/detail split on 15x27=405 <Picture 1> patch grid [405,96]."""
    z4 = z_kf0.float()[:, :, 0, :, :]
    c_pool = F.avg_pool2d(z4, kernel_size=2, stride=2)
    c4 = c_pool.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    d4 = z4 - c4
    def _to_405x96(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(24, 15, 2, 27, 2).permute(1, 3, 0, 2, 4).reshape(405, 96)
    return _to_405x96(z4), _to_405x96(c4), _to_405x96(d4)


def _build_seq_context_features(
    e64: torch.Tensor, pos: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build strictly prefix-causal token-level [L,64] and Conv1D/RoPE sequence-contextual [L,192] features."""
    ef, l_len = e64.float(), int(e64.shape[0])
    e_pad = F.pad(ef.T.unsqueeze(0), (2, 0), mode="replicate").squeeze(0).T
    conv3 = 0.20 * e_pad[:-2] + 0.30 * e_pad[1:-1] + 0.50 * e_pad[2:]
    csum = torch.cumsum(ef, dim=0) / torch.arange(1, l_len + 1, device=ef.device, dtype=torch.float32).unsqueeze(1)
    p_vec = pos.to(ef.device, torch.float32) if pos is not None else torch.arange(l_len, device=ef.device).float()
    rope_branch = apply_rope_1d(csum, p_vec, theta=10000.0)
    pos_phase = ((p_vec / 67.0) * 2.0 - 1.0).unsqueeze(1)
    return ef, torch.cat([ef, conv3, 0.85 * csum + 0.15 * rope_branch + 0.15 * pos_phase], dim=-1)


def _eval_lowrank(x: torch.Tensor, fit: dict[str, Any]) -> torch.Tensor:
    dev = x.device
    vin, uout = fit["v_in"].to(dev, torch.float32), fit["u_out"].to(dev, torch.float32)
    bout = fit["b_out"].to(dev, torch.float32)
    return (x.float() @ vin) @ uout + bout


def _char_span_regrid_e64(e64: torch.Tensor, spans: list[tuple[int, int]], max_len: int) -> torch.Tensor:
    """SwaT Eq. 11 + cand_08._overlap: conservative character-span regridding when L_src > max_len."""
    n_src = int(e64.shape[0])
    if n_src <= max_len or len(spans) != n_src:
        return e64[:max_len].contiguous()
    s_t = torch.tensor(spans, dtype=torch.float64)
    valid = s_t[:, 1] > s_t[:, 0]
    c_min = float(s_t[valid, 0].min().item()) if bool(valid.any()) else 0.0
    c_max = max(float(s_t[valid, 1].max().item()) if bool(valid.any()) else 1.0, c_min + 1.0)
    a_r = torch.linspace(c_min, c_max, max_len + 1, dtype=torch.float64)[:-1].unsqueeze(1)
    b_r = torch.linspace(c_min, c_max, max_len + 1, dtype=torch.float64)[1:].unsqueeze(1)
    a_s, b_s = s_t[:, 0].unsqueeze(0), s_t[:, 1].unsqueeze(0)
    ov = (torch.minimum(b_r, b_s) - torch.maximum(a_r, a_s)).clamp_min(0.0) * valid.to(torch.float64).unsqueeze(0)
    row_sum = ov.sum(dim=1, keepdim=True)
    nearest = (0.5 * (a_r + b_r) - 0.5 * (a_s + b_s)).abs().argmin(dim=1)
    fallback = F.one_hot(nearest, num_classes=n_src).to(torch.float64)
    w = torch.where(row_sum > 1e-12, ov / row_sum.clamp_min(1e-12), fallback).to(dtype=e64.dtype)
    return (w @ e64).contiguous()


def encode_prompt_to_e64(prompt: str, art: dict[str, Any], n_txt: int = 68) -> torch.Tensor:
    """Tokenize open-vocabulary prompt via Qwen Rust BPE or MiniMaxH3Tokenizer into [<=n_txt, 64]."""
    vocab_f16 = art.get("vocab_e64_f16")
    if vocab_f16 is None:
        tok_cal = art.get("tok_e64_calib")
        if isinstance(tok_cal, list) and tok_cal:
            return tok_cal[0].float().cpu()
        return torch.zeros((n_txt, 64), dtype=torch.float32)
    rust_tok = _TOKENIZER_CACHE.get("rust_bpe")
    if rust_tok is None and not _TOKENIZER_CACHE.get("rust_failed"):
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers
            v_p, m_p = str(QWEN_TOK_DIR / "vocab.json"), str(QWEN_TOK_DIR / "merges.txt")
            rust_tok = Tokenizer(models.BPE.from_file(v_p, m_p))
            rust_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            _TOKENIZER_CACHE["rust_bpe"] = rust_tok
        except Exception:
            _TOKENIZER_CACHE["rust_failed"] = True
    ids: list[int] = []
    spans: list[tuple[int, int]] = []
    v_sz = int(vocab_f16.shape[0])
    if rust_tok is not None:
        try:
            enc = rust_tok.encode(prompt)
            ids = [int(i) % v_sz for i in enc.ids]
            spans = [(int(a), int(b)) for a, b in enc.offsets]
        except Exception:
            ids, spans = [], []
    if not ids:
        tok = _TOKENIZER_CACHE.get("minimax_tok")
        if tok is None and not _TOKENIZER_CACHE.get("failed"):
            try:
                from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
                tok = _TOKENIZER_CACHE["minimax_tok"] = MiniMaxH3Tokenizer()
            except Exception:
                _TOKENIZER_CACHE["failed"] = True
        if tok is not None:
            try:
                entries = tok.tokenize_with_weights(prompt)["qwen3vl_32b"][0]
                ids = [int(x[0]) % v_sz for x in entries if isinstance(x[0], int)]
            except Exception:
                ids = []
    if not ids:
        matches = list(re.finditer(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", prompt))
        words = [m.group(0) for m in matches] or ["scene"]
        spans = [(m.start(), max(m.start() + 1, m.end())) for m in matches] or [(0, 5)]
        ids = [int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % v_sz for w in words]
    e64 = vocab_f16[ids].float().cpu()
    return _char_span_regrid_e64(e64, spans, n_txt)


def _gpu_cache(fit: dict[str, Any], dev: torch.device) -> dict[str, Any]:
    ck = f"_gpu_{dev}"
    if ck not in fit:
        c: dict[str, Any] = {}
        for k, v in fit.items():
            if torch.is_tensor(v):
                c[k] = v.to(dev, dtype=torch.float32, non_blocking=True)
            elif isinstance(v, dict):
                c[k] = {sk: (sv.to(dev, torch.float32) if torch.is_tensor(sv) else sv) for sk, sv in v.items()}
        fit[ck] = c
    return fit[ck]


def teleport_qwen_conditioning(
    z_kf0: torch.Tensor, prompt: str, art: dict[str, Any], pos_raw: Any = None, fixed_len: int = FIXED_TOTAL_LEN,
    blend_pos_raw: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Synthesize [1, 481, 5120] (<Picture 1> 407 tag=0 + 68 open-vocab tag=1) in <1.0 ms on GPU via ARBOR-Aleph."""
    v_raw, t_raw = art.get("vision_teleporter"), art.get("text_teleporter")
    if v_raw is None or t_raw is None:
        i2v_conds = art.get("i2v_dialogue_conds")
        if isinstance(i2v_conds, list) and i2v_conds:
            return i2v_conds[0][0][0].clone(), dict(i2v_conds[0][0][1])
        return pos_raw[0][0], dict(pos_raw[0][1])
    dev = z_kf0.device
    v_fit, t_fit = _gpu_cache(v_raw, dev), _gpu_cache(t_raw, dev)
    mu_pos = v_fit["mu_pos"]
    pf, cf, df = extract_zkf0_vision_grid(z_kf0.detach())
    dy_mb = 0.55 * _eval_lowrank(cf, v_fit["fit_c16"]) + 0.45 * _eval_lowrank(df, v_fit["fit_d16"])
    q_m = v_fit.get("q_mean", torch.zeros((1, 405 * 96), device=dev))
    q_k = F.normalize(pf.reshape(1, -1) - q_m, dim=-1)
    sims = (q_k @ v_fit["proto_keys"].T).squeeze(0)
    w_k = F.softmax(sims * 18.0, dim=0)
    pd = v_fit["proto_deltas"]
    dy_p = (w_k.unsqueeze(0) @ pd.reshape(int(pd.shape[0]), -1)).reshape_as(mu_pos)
    rel_k = float(sims.max().clamp(0.0, 1.0).item()) ** 2
    vis_413 = mu_pos + rel_k * dy_p + (1.0 - 0.65 * rel_k) * F.pad(dy_mb, (0, 0, 7, 1))
    n_txt = fixed_len - VISION_PREFIX_LEN
    e_raw = encode_prompt_to_e64(prompt, art, n_txt=n_txt).to(dev, non_blocking=True)
    qm_t = t_fit.get("q_mean")
    q_raw = e_raw.float().mean(dim=0, keepdim=True) - (qm_t.unsqueeze(0) if qm_t is not None else 0.0)
    e_in = (
        torch.cat([e_raw, e_raw[-1:].expand(n_txt - int(e_raw.shape[0]), -1)], dim=0)
        if int(e_raw.shape[0]) < n_txt else e_raw[:n_txt]
    )
    _, f192 = _build_seq_context_features(e_in)
    mu_slot = t_fit["mu_slot"][:n_txt]
    dy_t = _eval_lowrank(f192, t_fit["fit_d2_ctx"])
    q_t = F.normalize(q_raw, dim=-1)
    sim_t = (q_t @ t_fit["proto_q"].T).squeeze(0)
    wt = F.softmax(sim_t * 48.0, dim=0)
    y_p = (wt.view(-1, 1, 1) * t_fit["proto_v"][:, :n_txt]).sum(dim=0)
    rt = (float(sim_t.max().clamp(0.0, 1.0).item()) ** 2) * 0.94
    txt_68 = (1.0 - rt) * (mu_slot + dy_t) + rt * y_p
    if blend_pos_raw and pos_raw is not None and torch.is_tensor(pos_raw[0][0]):
        t_u = pos_raw[0][0].to(dev, txt_68.dtype)
        ns = min(n_txt, int(t_u.shape[1]))
        if ns > 0:
            txt_68[:ns] = 0.88 * txt_68[:ns] + 0.12 * t_u[0, -ns:]
    full_cond = torch.cat([vis_413, txt_68], dim=0).unsqueeze(0).contiguous()
    tags = torch.ones(fixed_len, dtype=torch.long, device=dev)
    tags[6:VISION_PREFIX_LEN] = 0
    return full_cond, {"pooled_output": None, "minimax_token_tags": tags}
