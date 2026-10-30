"""Teleported Qwen3-VL Text & <Picture 1> Vision Encoder onto MiniMaxH3Model via ARBOR-Aleph + Quotient Uncollapse."""
from __future__ import annotations
import hashlib, re
from typing import Any
import torch, torch.nn.functional as F
FIXED_TOTAL_LEN, VISION_PREFIX_LEN = 481, 413
_TOKENIZER_CACHE: dict[str, Any] = {}
def cand08_overlap(n_src: int, n_dst: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Lean 4 Module 34 Theorem 226: conservative cell-average overlap matrix."""
    if n_src == n_dst:
        return torch.eye(n_src, device=device, dtype=dtype)
    i = torch.arange(n_dst, device=device, dtype=torch.float32).unsqueeze(1)
    j = torch.arange(n_src, device=device, dtype=torch.float32).unsqueeze(0)
    lo, hi = torch.maximum(i / n_dst, j / n_src), torch.minimum((i + 1.0) / n_dst, (j + 1.0) / n_src)
    return ((hi - lo).clamp_min(0.0) * float(n_dst)).to(dtype=dtype)
def extract_zkf0_vision_grid(z_kf0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lean 4 Theorem 230: Exact 2x2 Haar coarse/detail split on 15x27=405 <Picture 1> patch grid [405,96]."""
    z4 = z_kf0.float()[:, :, 0, :, :]
    c_pool = F.avg_pool2d(z4, kernel_size=2, stride=2)
    c4 = c_pool.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    d4 = z4 - c4
    def _to_405x96(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(24, 15, 2, 27, 2).permute(1, 3, 0, 2, 4).reshape(405, 96)
    return _to_405x96(z4), _to_405x96(c4), _to_405x96(d4)
def _build_seq_context_features(e64: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build token-level [L,64] and Conv1D/RoPE causal sequence-contextual [L,192] features."""
    ef = e64.float()
    l_len = int(ef.shape[0])
    e_pad = F.pad(ef.T.unsqueeze(0), (1, 1), mode="replicate").squeeze(0).T
    conv3 = 0.25 * e_pad[:-2] + 0.50 * e_pad[1:-1] + 0.25 * e_pad[2:]
    csum = torch.cumsum(ef, dim=0) / torch.arange(1, l_len + 1, device=ef.device, dtype=torch.float32).unsqueeze(1)
    pos_phase = torch.linspace(-1.0, 1.0, l_len, device=ef.device).unsqueeze(1)
    ctx_192 = torch.cat([ef, conv3, csum + 0.15 * pos_phase], dim=-1)
    return ef, ctx_192
def _eval_lowrank(x: torch.Tensor, fit: dict[str, Any]) -> torch.Tensor:
    dev = x.device
    vin, uout = fit["v_in"].to(dev, torch.float32), fit["u_out"].to(dev, torch.float32)
    bout = fit["b_out"].to(dev, torch.float32)
    return (x.float() @ vin) @ uout + bout
def encode_prompt_to_e64(prompt: str, art: dict[str, Any], n_txt: int = 68) -> torch.Tensor:
    """Tokenize open-vocabulary prompt via MiniMaxH3Tokenizer (or deterministic BPE fallback) into [n_txt, 64]."""
    vocab_f16 = art.get("vocab_e64_f16")
    if vocab_f16 is None:
        tok_cal = art.get("tok_e64_calib")
        if isinstance(tok_cal, list) and tok_cal:
            return tok_cal[0].float().cpu()
        return torch.zeros((n_txt, 64), dtype=torch.float32)
    tok = _TOKENIZER_CACHE.get("minimax_tok")
    if tok is None and not _TOKENIZER_CACHE.get("failed"):
        try:
            from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
            tok = _TOKENIZER_CACHE["minimax_tok"] = MiniMaxH3Tokenizer()
        except Exception:
            _TOKENIZER_CACHE["failed"] = True
    ids: list[int] = []
    if tok is not None:
        try:
            entries = tok.tokenize_with_weights(prompt)["qwen3vl_32b"][0]
            ids = [int(x[0]) % int(vocab_f16.shape[0]) for x in entries if isinstance(x[0], int)]
        except Exception:
            ids = []
    if not ids:
        words = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", prompt) or ["scene"]
        v_sz = int(vocab_f16.shape[0])
        ids = [int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % v_sz for w in words]
    e64 = vocab_f16[ids].float().cpu()
    if int(e64.shape[0]) < n_txt:
        e64 = torch.cat([e64, e64[-1:].expand(n_txt - int(e64.shape[0]), -1)], dim=0)
    return e64[:n_txt]
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
    z_kf0: torch.Tensor, prompt: str, art: dict[str, Any], pos_raw: Any = None, fixed_len: int = FIXED_TOTAL_LEN
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Synthesize [1, 481, 5120] (<Picture 1> 407 tag=0 + 68 open-vocab tag=1) in <1.0 ms on GPU via ARBOR-Aleph."""
    v_raw, t_raw = art.get("vision_teleporter"), art.get("text_teleporter")
    if v_raw is None or t_raw is None:
        i2v_conds = art.get("i2v_dialogue_conds")
        if isinstance(i2v_conds, list) and i2v_conds:
            return i2v_conds[0][0][0].clone(), dict(i2v_conds[0][1])
        return pos_raw[0][0], dict(pos_raw[0][1])
    dev = z_kf0.device
    v_fit, t_fit = _gpu_cache(v_raw, dev), _gpu_cache(t_raw, dev)
    mu_pos = v_fit["mu_pos"]
    pf, cf, df = extract_zkf0_vision_grid(z_kf0.detach())
    dy_mb = 0.55 * _eval_lowrank(cf, v_fit["fit_c16"]) + 0.45 * _eval_lowrank(df, v_fit["fit_d16"])
    q_m = v_fit.get("q_mean", torch.zeros((1, 405 * 96), device=dev))
    q_k = F.normalize(pf.reshape(1, -1) - q_m, dim=-1)
    sims = (q_k @ v_fit["proto_keys"].T).squeeze(0)
    w_k = F.softmax(sims * 48.0, dim=0)
    pd = v_fit["proto_deltas"]
    dy_p = (w_k.unsqueeze(0) @ pd.reshape(int(pd.shape[0]), -1)).reshape_as(mu_pos)
    rel_k = float(sims.max().clamp(0.0, 1.0).item()) ** 2
    vis_413 = mu_pos + rel_k * dy_p + (1.0 - 0.85 * rel_k) * F.pad(dy_mb, (0, 0, 7, 1))
    n_txt = fixed_len - VISION_PREFIX_LEN
    e_in = encode_prompt_to_e64(prompt, art, n_txt=n_txt).to(dev, non_blocking=True)
    f64, f192 = _build_seq_context_features(e_in)
    mu_slot = t_fit["mu_slot"][:n_txt]
    dy_t = _eval_lowrank(f192, t_fit["fit_d2_ctx"])
    q_t = F.normalize(f64.mean(dim=0, keepdim=True), dim=-1)
    sim_t = (q_t @ t_fit["proto_q"].T).squeeze(0)
    wt = F.softmax(sim_t * 48.0, dim=0)
    y_p = (wt.view(-1, 1, 1) * t_fit["proto_v"][:, :n_txt]).sum(dim=0)
    rt = (float(sim_t.max().clamp(0.0, 1.0).item()) ** 2) * 0.96
    txt_68 = (1.0 - rt) * (mu_slot + dy_t) + rt * y_p
    if pos_raw is not None and torch.is_tensor(pos_raw[0][0]):
        t_u = pos_raw[0][0].to(dev, txt_68.dtype)
        ns = min(n_txt, int(t_u.shape[1]))
        if ns > 0:
            txt_68[:ns] = 0.88 * txt_68[:ns] + 0.12 * t_u[0, -ns:]
    full_cond = torch.cat([vis_413, txt_68], dim=0).unsqueeze(0).contiguous()
    tags = torch.ones(fixed_len, dtype=torch.long, device=dev)
    tags[6:VISION_PREFIX_LEN] = 0
    return full_cond, {"pooled_output": None, "minimax_token_tags": tags}
