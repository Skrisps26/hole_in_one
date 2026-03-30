from __future__ import annotations

import copy, glob, io, math, os, struct, sys, time, uuid, zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False

# ═══════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════
class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH",      "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path,           "fineweb_train_*.bin")
    val_files      = os.path.join(data_path,           "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID",         str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED",       1337))

    val_batch_size     = int(os.environ.get("VAL_BATCH_SIZE",  524_288))
    val_loss_every     = int(os.environ.get("VAL_LOSS_EVERY",  1000))
    train_log_every    = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations         = int(os.environ.get("ITERATIONS",        20000))
    warmdown_iters     = int(os.environ.get("WARMDOWN_ITERS",    3500))
    warmup_steps       = int(os.environ.get("WARMUP_STEPS",      20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS",524_288))
    train_seq_len      = int(os.environ.get("TRAIN_SEQ_LEN",     1024))
    eval_seq_len       = int(os.environ.get("EVAL_SEQ_LEN",      2048))
    eval_stride        = int(os.environ.get("EVAL_STRIDE",       64))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape — 11L × 512 with 3× MLP (proven 16 MB fit with int6+zstd)
    vocab_size    = int(os.environ.get("VOCAB_SIZE",    1024))
    num_layers    = int(os.environ.get("NUM_LAYERS",    11))
    num_kv_heads  = int(os.environ.get("NUM_KV_HEADS",  4))
    model_dim     = int(os.environ.get("MODEL_DIM",     512))
    num_heads     = int(os.environ.get("NUM_HEADS",     8))
    mlp_mult      = int(os.environ.get("MLP_MULT",      3))
    tie_embeddings= bool(int(os.environ.get("TIE_EMBEDDINGS","1")))
    rope_dims     = int(os.environ.get("ROPE_DIMS",     16))       # partial RoPE
    rope_base     = float(os.environ.get("ROPE_BASE",   10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP",30.0))
    xsa_layers    = int(os.environ.get("XSA_LAYERS",   4))        # XSA on last N layers
    leaky_alpha   = float(os.environ.get("LEAKY_ALPHA", 0.5))     # LeakyReLU(α)²
    wavelet_emb   = bool(int(os.environ.get("WAVELET_EMB","1")))

    # Quantisation
    qat_start_step = int(os.environ.get("QAT_START_STEP", 8000))
    optrot_enabled = bool(int(os.environ.get("OPTROT_ENABLED","1")))
    cpsvd_rank     = int(os.environ.get("CPSVD_RANK",      32))   # 0 = disabled

    # EMA
    ema_decay     = float(os.environ.get("EMA_DECAY",     0.999))
    ema_start_step= int(os.environ.get("EMA_START_STEP",  5000))

    # LaCT chunk-TTT at eval
    lact_enabled    = bool(int(os.environ.get("LACT_ENABLED",   "1")))
    lact_lr         = float(os.environ.get("LACT_LR",           3e-5))
    lact_chunk_size = int(os.environ.get("LACT_CHUNK_SIZE",     512))

    # Optimiser
    embed_lr           = float(os.environ.get("EMBED_LR",           0.6))
    head_lr            = float(os.environ.get("HEAD_LR",            0.008))
    tied_embed_lr      = float(os.environ.get("TIED_EMBED_LR",      0.05))
    tied_embed_init_std= float(os.environ.get("TIED_EMBED_INIT_STD",0.005))
    matrix_lr          = float(os.environ.get("MATRIX_LR",          0.04))
    scalar_lr          = float(os.environ.get("SCALAR_LR",          0.04))
    muon_momentum      = float(os.environ.get("MUON_MOMENTUM",      0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS",   5))
    muon_weight_decay  = float(os.environ.get("MUON_WEIGHT_DECAY",  0.04))
    beta1    = float(os.environ.get("BETA1",    0.9))
    beta2    = float(os.environ.get("BETA2",    0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

# ═══════════════════════════════════════════════════════════════
# MUON OPTIMIZER  (from modded-nanogpt)
# ═══════════════════════════════════════════════════════════════
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 weight_decay: float = 0.0, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                     weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size  = dist.get_world_size() if distributed else 1
        rank        = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            total = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    if wd != 0.0:
                        g = g.add(p.data, alpha=wd)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# ═══════════════════════════════════════════════════════════════
# TOKENISER-AGNOSTIC EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════════
def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab = int(sp.vocab_size())
    sz = max(sp_vocab, vocab_size)
    base_bytes_np          = np.zeros((sz,),  dtype=np.int16)
    has_leading_space_np   = np.zeros((sz,),  dtype=np.bool_)
    is_boundary_token_np   = np.ones ((sz,),  dtype=np.bool_)
    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_token_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space_np[tid] = True
            piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np,        dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool,  device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool,  device=device))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_data_shard(file: Path) -> Tensor:
    hb = 256 * np.dtype("<i4").itemsize
    tb = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    n = int(header[2])
    if file.stat().st_size != hb + n * tb:
        raise ValueError(f"Size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=n, offset=hb)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files   = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens   = load_data_shard(self.files[0])
        self.pos      = 0
    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens   = load_data_shard(self.files[self.file_idx])
        self.pos      = 0
    def take(self, n: int) -> Tensor:
        chunks, rem = [], n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance(); continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k; rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens  = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1: ].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ═══════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════

# ── WaveletGPT embedding ────────────────────────────────────────
# Applies a single-level Haar wavelet transform to the second half
# of embedding dimensions, giving multi-scale frequency structure
# for free.  Paper: arXiv:2409.12924
class WaveletEmbedding(nn.Embedding):
    """Standard nn.Embedding whose second half is Haar-wavelet transformed."""
    def forward(self, idx: Tensor) -> Tensor:
        e   = super().forward(idx)              # [B, T, D]
        D   = e.shape[-1]
        h   = D // 2                            # first half: standard
        e2  = e[..., h:]                        # second half → wavelet
        # Reshape to pairs for 1-level Haar
        e2p = e2.reshape(*e2.shape[:-1], h // 2, 2)
        lo  = (e2p[..., 0] + e2p[..., 1]) * 0.70710678  # (a+b)/√2
        hi  = (e2p[..., 0] - e2p[..., 1]) * 0.70710678  # (a-b)/√2
        e_w = torch.cat([lo, hi], dim=-1)
        return torch.cat([e[..., :h], e_w], dim=-1)

# ── Helpers ─────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)

def _fake_quant_int6(w: Tensor) -> Tensor:
    """Per-row STE int6 fake quantisation for 2-D weight matrices."""
    if w.ndim == 2:
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 31.0
    else:
        scale = w.abs().amax().clamp_min(1e-8) / 31.0
    return (w / scale).clamp(-31, 31).round() * scale

class QATLinear(nn.Linear):
    """Linear layer with optional STE int6 fake-quant on weights."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_active = False
    def forward(self, x):
        w    = self.weight.to(x.dtype)
        if self.qat_active:
            w = _fake_quant_int6(w)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

def _restore_fp32(module: nn.Module):
    """Cast small/control tensors back to fp32 after an autocast step."""
    CTRL = ("attn_scale","mlp_scale","resid_mix","q_gain","skip_weight","ln_scale")
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(c in name for c in CTRL)) and p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)

# ── RoPE ────────────────────────────────────────────────────────
def build_rope_cache(seq_len: int, rope_dims: int, base: float, device) -> tuple[Tensor, Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, rope_dims, 2, device=device).float() / rope_dims))
    pos   = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return freqs.cos(), freqs.sin()

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    B, T, nH, hD = x.shape
    xr = x[..., :rope_dims].reshape(B, T, nH, rope_dims // 2, 2)
    x1, x2 = xr[..., 0], xr[..., 1]
    c = cos[:T, :].reshape(1, T, 1, rope_dims // 2)
    s = sin[:T, :].reshape(1, T, 1, rope_dims // 2)
    r1  = x1 * c - x2 * s
    r2  = x1 * s + x2 * c
    rot = torch.stack([r1, r2], dim=-1).reshape(B, T, nH, rope_dims)
    return torch.cat([rot, x[..., rope_dims:]], dim=-1)

# ── Attention ────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, args: Hyperparameters, layer_idx: int):
        super().__init__()
        D, H, Hkv  = args.model_dim, args.num_heads, args.num_kv_heads
        self.H, self.Hkv, self.hD = H, Hkv, D // H
        self.rope_dims = args.rope_dims
        self.is_xsa    = layer_idx >= (args.num_layers - args.xsa_layers)

        self.q_proj  = QATLinear(D, D,            bias=False)
        self.kv_proj = QATLinear(D, 2 * Hkv * (D // H), bias=False)
        self.out_proj= QATLinear(D, D,            bias=False)

        self.q_gain  = nn.Parameter(torch.full((H,), args.qk_gain_init))
        self.resid_mix = nn.Parameter(torch.ones(1))   # learnable residual blend

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, D = x.shape
        H, Hkv, hD = self.H, self.Hkv, self.hD

        q   = self.q_proj(x).reshape(B, T, H,    hD)
        kv  = self.kv_proj(x).reshape(B, T, Hkv, 2 * hD)
        k   = kv[..., :hD]
        v   = kv[..., hD:]

        q   = apply_rope(q, cos, sin, self.rope_dims)
        k   = apply_rope(k, cos, sin, self.rope_dims)

        # Q-gain scale (per-head learnable)
        q   = q * self.q_gain[None, None, :, None].to(q.dtype)

        # GQA: expand KV heads to match Q heads
        if Hkv < H:
            k = k.repeat_interleave(H // Hkv, dim=2)
            v = v.repeat_interleave(H // Hkv, dim=2)

        if self.is_xsa:
            # Extended-Sequence Attention: reshape batch to single long sequence
            # so each token can attend to previous sequences in the batch.
            q = q.reshape(1, B * T, H, hD)
            k = k.reshape(1, B * T, H, hD)
            v = v.reshape(1, B * T, H, hD)
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True
            ).transpose(1, 2)
            o = o.reshape(B, T, D)
        else:
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True
            ).transpose(1, 2).reshape(B, T, D)

        return self.out_proj(o)

# ── MLP ─────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        D, M = args.model_dim, args.mlp_mult * args.model_dim
        self.fc1   = QATLinear(D, M, bias=False)
        self.fc2   = QATLinear(M, D, bias=False)
        self.alpha  = args.leaky_alpha   # LeakyReLU(α)²

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(x)
        h = F.leaky_relu(h, self.alpha) ** 2   # LeakyReLU(α)²
        return self.fc2(h)

# ── Transformer block ────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, args: Hyperparameters, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm()
        self.attn  = CausalSelfAttention(args, layer_idx)
        self.norm2 = RMSNorm()
        self.mlp   = MLP(args)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

# ── Full GPT model ───────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args

        EmbCls = WaveletEmbedding if args.wavelet_emb else nn.Embedding
        self.embed = EmbCls(args.vocab_size, args.model_dim)

        # Tied output head: small correction on top of (transposed) embedding
        self.unembed_correction = nn.Parameter(
            torch.randn(args.vocab_size, args.model_dim) * args.tied_embed_init_std
        )

        self.blocks  = nn.ModuleList(
            [TransformerBlock(args, i) for i in range(args.num_layers)]
        )
        self.norm_out = RMSNorm()

        D, Tmax = args.model_dim, max(args.train_seq_len, args.eval_seq_len) * 8
        cos, sin = build_rope_cache(Tmax, args.rope_dims, args.rope_base, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "embed" in name and "correction" not in name:
                nn.init.normal_(p, std=0.02)
            elif "out_proj" in name or "fc2" in name:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * self.args.num_layers))
            elif p.ndim >= 2:
                nn.init.normal_(p, std=0.02)

    def set_qat(self, active: bool):
        for m in self.modules():
            if isinstance(m, QATLinear):
                m.qat_active = active

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        B, T = x.shape
        cos  = self.rope_cos[:T].to(x.device)
        sin  = self.rope_sin[:T].to(x.device)

        h = self.embed(x)
        for block in self.blocks:
            h = block(h, cos, sin)
        h = self.norm_out(h)

        # Tied embedding output + small learned correction
        W = self.embed.weight.to(h.dtype) + self.unembed_correction.to(h.dtype)
        logits = h @ W.T                                              # [B, T, V]
        sc = self.args.logit_softcap
        logits = sc * torch.tanh(logits / sc)

        if y is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

# ═══════════════════════════════════════════════════════════════
# EMA  (Exponential Moving Average of weights)
# ═══════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model  = model
        self.decay  = decay
        self.shadow : dict[str, Tensor] = {}

    def init(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                self.shadow[n] = p.data.clone().float()

    def update(self):
        d = self.decay
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].mul_(d).add_(p.data.float(), alpha=1 - d)

    def apply(self) -> dict[str, Tensor]:
        backup = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.shadow:
                    backup[n] = p.data.clone()
                    p.data.copy_(self.shadow[n].to(p.dtype))
        return backup

    def restore(self, backup: dict[str, Tensor]):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in backup:
                    p.data.copy_(backup[n])

# ═══════════════════════════════════════════════════════════════
# OptRot  — randomised Hadamard rotation before int6 quant
#           Makes weight distributions more uniform → better BPB
#           per arXiv:2512.24124 (QuIP#-style incoherence)
# ═══════════════════════════════════════════════════════════════
def _walsh_hadamard(x: Tensor) -> Tensor:
    """In-place Walsh-Hadamard transform over last dimension (must be power-of-2)."""
    n = x.shape[-1]
    h = 1
    y = x.clone()
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2 * h)
        a, b = y[..., :h].clone(), y[..., h:].clone()
        y[..., :h] = a + b
        y[..., h:] = a - b
        y = y.reshape(*y.shape[:-2], n)
        h *= 2
    return y

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def optrot_apply(W: Tensor, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """
    Apply randomised Hadamard rotation to the *input* dimension of W.
    Returns (W_rot, d_signs, pad) where d_signs and pad let you reconstruct
    the transform.  Fuse inverse into the adjacent layer to pay zero bytes.
    """
    out_dim, in_dim = W.shape
    p = _next_pow2(in_dim)

    rng    = torch.Generator()
    rng.manual_seed(seed)
    d_signs = (torch.randint(0, 2, (in_dim,), generator=rng) * 2 - 1).float()  # ±1

    # Pad input dim to power of 2
    if p > in_dim:
        W_pad = F.pad(W.float(), (0, p - in_dim))
    else:
        W_pad = W.float()

    d_pad = torch.ones(p, dtype=torch.float32)
    d_pad[:in_dim] = d_signs

    # Apply: W_rot = W_pad * d[None, :] then Hadamard
    W_dsgn = W_pad * d_pad[None, :]
    W_had  = _walsh_hadamard(W_dsgn) / math.sqrt(p)
    W_rot  = W_had[:, :in_dim]

    return W_rot.to(W.dtype), d_signs, torch.tensor(p)

# ═══════════════════════════════════════════════════════════════
# CPSVD  — Column-Preserving SVD hybrid compression
#           Low-rank fp16 for "smooth" columns + int6 residual
#           per arXiv:2510.19385
# ═══════════════════════════════════════════════════════════════
def cpsvd_compress(W: Tensor, rank: int) -> dict:
    """Decompose W (out×in) into rank-r fp16 + int6 residual."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    r   = min(rank, S.numel())
    # Low-rank part (stored as fp16)
    U_r = (U[:, :r] * S[:r].unsqueeze(0)).to(torch.float16).contiguous()
    Vh_r= Vh[:r].to(torch.float16).contiguous()
    # Residual
    W_lr   = U_r.float() @ Vh_r.float()
    W_res  = (W.float() - W_lr)
    return {"U_r": U_r, "Vh_r": Vh_r, "residual": W_res.to(W.dtype)}

def cpsvd_decompress(d: dict) -> Tensor:
    lr = d["U_r"].float() @ d["Vh_r"].float()
    return (lr + d["residual"].float()).to(d["residual"].dtype)

# ═══════════════════════════════════════════════════════════════
# INT6 QUANTISATION  (pack 4 × 6-bit values into 3 bytes)
# ═══════════════════════════════════════════════════════════════
def _int6_pack(values: np.ndarray) -> bytes:
    """values in [-31,31] → packed bytes (4 values per 3 bytes)."""
    v  = (values.astype(np.int8).clip(-31, 31) + 31).astype(np.uint8)
    pad = (4 - len(v) % 4) % 4
    if pad:
        v = np.concatenate([v, np.zeros(pad, np.uint8)])
    v  = v.reshape(-1, 4)
    b0 = (v[:, 0] << 2) | (v[:, 1] >> 4)
    b1 = ((v[:, 1] & 0xF) << 4) | (v[:, 2] >> 2)
    b2 = ((v[:, 2] & 0x3) << 6) | v[:, 3]
    return np.stack([b0, b1, b2], axis=1).flatten().astype(np.uint8).tobytes()

def _int6_unpack(data: bytes, count: int) -> np.ndarray:
    a  = np.frombuffer(data, dtype=np.uint8)
    ng = (count + 3) // 4
    g  = a[: ng * 3].reshape(ng, 3)
    v0 = g[:, 0] >> 2
    v1 = ((g[:, 0] & 0x3) << 4) | (g[:, 1] >> 4)
    v2 = ((g[:, 1] & 0xF) << 2) | (g[:, 2] >> 6)
    v3 = g[:, 2] & 0x3F
    v  = np.stack([v0, v1, v2, v3], axis=1).flatten()[:count]
    return v.astype(np.int8) - 31

def quantise_int6_perrow(t: Tensor) -> tuple[np.ndarray, np.ndarray]:
    t32 = t.float()
    if t32.ndim == 2:
        scale = t32.abs().amax(dim=1).clamp_min(1e-8) / 31.0  # [out]
        q     = (t32 / scale[:, None]).clamp(-31, 31).round().to(torch.int8).cpu().numpy()
        return q, scale.float().cpu().numpy()
    scale = float(t32.abs().amax().clamp_min(1e-8).item()) / 31.0
    q     = (t32 / scale).clamp(-31, 31).round().to(torch.int8).cpu().numpy()
    return q, np.array([scale], dtype=np.float32)

def dequantise_int6_perrow(q: np.ndarray, scale: np.ndarray, dtype) -> Tensor:
    q32 = torch.from_numpy(q.astype(np.float32))
    s   = torch.from_numpy(scale)
    if q32.ndim == 2:
        out = q32 * s[:, None]
    else:
        out = q32 * s[0]
    return out.to(dtype)

# ═══════════════════════════════════════════════════════════════
# ARTIFACT  BUILD + LOAD
# ═══════════════════════════════════════════════════════════════
ARTIFACT_MAGIC = b"PGOLF1\x00\x00"

def build_artifact(model: nn.Module, args: Hyperparameters, path: str):
    """
    Quantise (int6 per-row + optional OptRot), CPSVD-compress heavy matrices,
    then zstd-22 / zlib the lot into a single ≤16 MB blob.
    """
    state   = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tensors : list[dict] = []
    rot_seeds: dict[str, int] = {}

    for name, t in state.items():
        rec: dict = {"name": name}
        # Small / non-float tensors: store as fp16
        if not t.is_floating_point() or t.numel() < 4096:
            rec["kind"]  = "fp16"
            rec["data"]  = t.to(torch.float16).numpy().tobytes()
            rec["shape"] = list(t.shape)
            rec["dtype"] = str(t.dtype).removeprefix("torch.")
            tensors.append(rec); continue

        # Large 2-D matrices: OptRot + int6
        if t.ndim == 2:
            seed = hash(name) & 0xFFFFFFFF
            if args.optrot_enabled:
                t_rot, _, _ = optrot_apply(t, seed)
                rot_seeds[name] = seed
            else:
                t_rot = t

            # CPSVD hybrid (only if rank > 0 and matrix is large enough)
            if args.cpsvd_rank > 0 and min(t_rot.shape) >= args.cpsvd_rank * 2:
                cpsvd = cpsvd_compress(t_rot, args.cpsvd_rank)
                residual = cpsvd["residual"]
                q, scale = quantise_int6_perrow(residual)
                rec["kind"]    = "cpsvd_int6"
                rec["U_r"]     = cpsvd["U_r"].numpy().tobytes()
                rec["Vh_r"]    = cpsvd["Vh_r"].numpy().tobytes()
                rec["Ur_shape"]= list(cpsvd["U_r"].shape)
                rec["Vr_shape"]= list(cpsvd["Vh_r"].shape)
                rec["q_data"]  = _int6_pack(q.flatten())
                rec["scale"]   = scale.tobytes()
                rec["shape"]   = list(t.shape)
                rec["dtype"]   = str(t.dtype).removeprefix("torch.")
            else:
                q, scale = quantise_int6_perrow(t_rot)
                rec["kind"]  = "int6"
                rec["q_data"]= _int6_pack(q.flatten())
                rec["scale"] = scale.tobytes()
                rec["shape"] = list(t.shape)
                rec["dtype"] = str(t.dtype).removeprefix("torch.")
        else:
            # 1-D or higher: fp16
            rec["kind"]  = "fp16"
            rec["data"]  = t.to(torch.float16).numpy().tobytes()
            rec["shape"] = list(t.shape)
            rec["dtype"] = str(t.dtype).removeprefix("torch.")

        tensors.append(rec)

    payload = {"tensors": tensors, "rot_seeds": rot_seeds,
               "args": {k: getattr(args, k) for k in dir(args)
                        if not k.startswith("_") and not callable(getattr(args, k))}}

    import pickle
    raw = pickle.dumps(payload, protocol=4)

    if _HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        compressed = cctx.compress(raw)
    else:
        compressed = zlib.compress(raw, level=9)

    with open(path, "wb") as f:
        f.write(ARTIFACT_MAGIC)
        f.write(struct.pack("<Q", len(compressed)))
        f.write(compressed)

    MB = os.path.getsize(path) / 1e6
    print(f"[artifact] {path}  {MB:.2f} MB", flush=True)
    if MB > 16.0:
        print(f"[artifact] WARNING: {MB:.2f} MB exceeds 16 MB limit!", flush=True)

def load_artifact(path: str, device) -> tuple[GPT, Hyperparameters]:
    import pickle
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == ARTIFACT_MAGIC, "Bad artifact magic"
        clen  = struct.unpack("<Q", f.read(8))[0]
        compressed = f.read(clen)

    if _HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        raw  = dctx.decompress(compressed)
    else:
        raw  = zlib.decompress(compressed)

    import pickle
    payload    = pickle.loads(raw)
    arg_dict   = payload["args"]
    args       = Hyperparameters()
    for k, v in arg_dict.items():
        try: setattr(args, k, v)
        except: pass
    rot_seeds  = payload["rot_seeds"]

    model = GPT(args).to(device)
    state : dict[str, Tensor] = {}

    for rec in payload["tensors"]:
        name, kind = rec["name"], rec["kind"]
        orig_dtype = getattr(torch, rec["dtype"])
        shape      = rec["shape"]

        if kind == "fp16":
            arr = np.frombuffer(rec["data"], dtype=np.float16).reshape(shape)
            state[name] = torch.from_numpy(arr.copy()).to(orig_dtype)

        elif kind == "int6":
            q_np  = _int6_unpack(rec["q_data"], int(np.prod(shape))).reshape(shape)
            scale = np.frombuffer(rec["scale"], dtype=np.float32)
            t_rot = dequantise_int6_perrow(q_np, scale, orig_dtype)
            if name in rot_seeds:
                # Undo OptRot: apply H^T = H (Hadamard is symmetric up to scaling)
                seed    = rot_seeds[name]
                _, d_signs, _ = optrot_apply(t_rot, seed)  # just to get d_signs
                # Inverse: W = W_rot @ H^T * d_signs (since H^T*H = I)
                p      = _next_pow2(t_rot.shape[1])
                W_pad  = F.pad(t_rot.float(), (0, max(0, p - t_rot.shape[1])))
                d_pad  = torch.ones(p); d_pad[:len(d_signs)] = d_signs
                W_hT   = _walsh_hadamard(W_pad) / math.sqrt(p)  # H^T = H for Hadamard
                W_back = W_hT[:, :t_rot.shape[1]] * d_pad[:t_rot.shape[1]][None, :]
                state[name] = W_back.to(orig_dtype)
            else:
                state[name] = t_rot

        elif kind == "cpsvd_int6":
            Ur  = torch.from_numpy(np.frombuffer(rec["U_r"], np.float16).reshape(rec["Ur_shape"]))
            Vhr = torch.from_numpy(np.frombuffer(rec["Vh_r"],np.float16).reshape(rec["Vr_shape"]))
            q_np= _int6_unpack(rec["q_data"], int(np.prod(shape))).reshape(shape)
            scl = np.frombuffer(rec["scale"], np.float32)
            res = dequantise_int6_perrow(q_np, scl, torch.float32)
            W   = (Ur.float() @ Vhr.float() + res).to(orig_dtype)
            if name in rot_seeds:
                seed       = rot_seeds[name]
                _, d_signs, _ = optrot_apply(W, seed)
                p          = _next_pow2(W.shape[1])
                W_pad      = F.pad(W.float(), (0, max(0, p - W.shape[1])))
                d_pad      = torch.ones(p); d_pad[:len(d_signs)] = d_signs
                W_hT       = _walsh_hadamard(W_pad) / math.sqrt(p)
                W          = (W_hT[:, :W.shape[1]] * d_pad[:W.shape[1]][None, :]).to(orig_dtype)
            state[name] = W

    model.load_state_dict(state, strict=False)
    return model, args

# ═══════════════════════════════════════════════════════════════
# SLIDING-WINDOW EVALUATION  (stride=64, context up to 2048)
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_sliding_window(args, model, val_tokens, base_bytes_lut, has_leading_lut,
                         is_boundary_lut, rank, world_size, device):
    T      = min(args.eval_seq_len, val_tokens.numel() // 2)
    stride = args.eval_stride
    N      = val_tokens.numel()

    starts     = list(range(0, N - T - 1, stride))
    loc_starts = starts[rank::world_size]

    total_nll  = torch.zeros((), dtype=torch.float64, device=device)
    total_tok  = torch.zeros((), dtype=torch.float64, device=device)
    total_byte = torch.zeros((), dtype=torch.float64, device=device)

    model.eval()
    for s in loc_starts:
        win = val_tokens[s: s + T + 1].to(device, torch.int64)
        x   = win[:-1].unsqueeze(0)
        y   = win[ 1:].unsqueeze(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(x)                               # [1, T, V]

        # Only "count" new tokens (last `stride` of each window, except the first)
        c_start = 0 if s == 0 else max(0, T - stride)
        log_new = logits[:, c_start:, :]
        y_new   = y[:, c_start:]
        x_new   = x[:, c_start:]

        nll = F.cross_entropy(log_new.reshape(-1, args.vocab_size),
                              y_new.reshape(-1), reduction="sum")
        total_nll  += nll.double()
        total_tok  += y_new.numel()

        tgt  = y_new.reshape(-1)
        prev = x_new.reshape(-1)
        tb   = base_bytes_lut[tgt].to(torch.int32)
        tb  += (has_leading_lut[tgt] & ~is_boundary_lut[prev]).to(torch.int32)
        total_byte += tb.double().sum()

    stats = torch.stack([total_nll, total_tok, total_byte])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    val_loss = (stats[0] / stats[1]).item()
    bpb      = (val_loss / math.log(2.0)) * (stats[1] / stats[2]).item()
    model.train()
    return val_loss, bpb

# ═══════════════════════════════════════════════════════════════
# LaCT  CHUNK-LEVEL TEST-TIME TRAINING  (legal: backward-only)
# ═══════════════════════════════════════════════════════════════
def eval_with_lact_ttt(args, model, val_tokens, base_bytes_lut, has_leading_lut,
                        is_boundary_lut, rank, world_size, device):
    """
    Evaluate val_tokens using LaCT-style chunk-level TTT.
    For each chunk, we first update the MLP output projections (fast weights)
    using the *previous* chunk's loss gradient — fully legal (causal).
    """
    # Only fast-weight params (MLP fc2 projections)
    fast_names = {n for n, _ in model.named_parameters() if "mlp.fc2" in n}
    fast_params = [p for n, p in model.named_parameters() if n in fast_names]

    # Save original weights
    backup = {n: p.data.clone() for n, p in model.named_parameters() if n in fast_names}

    CS    = args.lact_chunk_size
    N     = val_tokens.numel()
    total_nll  = torch.zeros((), dtype=torch.float64, device=device)
    total_tok  = torch.zeros((), dtype=torch.float64, device=device)
    total_byte = torch.zeros((), dtype=torch.float64, device=device)

    prev_x = prev_y = None

    for s in range(rank * CS, N - 1, world_size * CS):
        end     = min(s + CS, N - 1)
        win     = val_tokens[s: end + 1].to(device, torch.int64)
        x_chunk = win[:-1].unsqueeze(0)
        y_chunk = win[ 1:].unsqueeze(0)

        # Update fast weights on PREVIOUS chunk (legal: past tokens only)
        if prev_x is not None:
            for fp in fast_params:
                if fp.grad is not None:
                    fp.grad = None
            loss_prev = model(prev_x, prev_y)
            grads = torch.autograd.grad(loss_prev, fast_params, allow_unused=True)
            with torch.no_grad():
                for fp, g in zip(fast_params, grads):
                    if g is not None:
                        fp.data -= args.lact_lr * g

        prev_x, prev_y = x_chunk, y_chunk

        # Score current chunk
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x_chunk)

        nll = F.cross_entropy(logits.reshape(-1, args.vocab_size),
                              y_chunk.reshape(-1), reduction="sum")
        total_nll  += nll.double()
        total_tok  += y_chunk.numel()
        tgt  = y_chunk.reshape(-1)
        prev = x_chunk.reshape(-1)
        tb   = base_bytes_lut[tgt].to(torch.int32)
        tb  += (has_leading_lut[tgt] & ~is_boundary_lut[prev]).to(torch.int32)
        total_byte += tb.double().sum()

    # Restore original weights
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    stats = torch.stack([total_nll, total_tok, total_byte])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    val_loss = (stats[0] / stats[1]).item()
    bpb      = (val_loss / math.log(2.0)) * (stats[1] / stats[2]).item()
    return val_loss, bpb

# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
def get_lr(step: int, args: Hyperparameters, base_lr: float) -> float:
    if step < args.warmup_steps:
        return base_lr * step / max(1, args.warmup_steps)
    if step < args.iterations - args.warmdown_iters:
        return base_lr
    frac = (args.iterations - step) / args.warmdown_iters
    return base_lr * frac

def train(args: Hyperparameters):
    # ── DDP setup ────────────────────────────────────────────────
    distributed = "RANK" in os.environ and dist.is_available()
    if distributed:
        dist.init_process_group("nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device     = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank)

    # ── Tokeniser ────────────────────────────────────────────────
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    base_bytes_lut, has_leading_lut, is_boundary_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

    # ── Data ─────────────────────────────────────────────────────
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    val_tokens   = load_validation_tokens(args.val_files, args.eval_seq_len).to(device)

    # ── Model ────────────────────────────────────────────────────
    model = GPT(args).to(device)
    _restore_fp32(model)

    if distributed:
        model = DDP(model, device_ids=[device.index])

    raw_model = model.module if distributed else model
    ema       = EMA(raw_model, args.ema_decay)

    # Grad accumulation
    tokens_per_rank = args.train_batch_tokens // world_size
    seqs_per_rank   = tokens_per_rank // args.train_seq_len
    grad_accum      = max(1, seqs_per_rank // max(1, torch.cuda.device_count()))

    # ── Optimiser groups ─────────────────────────────────────────
    matrix_params, embed_params, scalar_params, head_params = [], [], [], []
    for name, p in raw_model.named_parameters():
        if "embed.weight" in name:
            embed_params.append(p)
        elif "unembed_correction" in name:
            head_params.append(p)
        elif p.ndim >= 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    optimisers = [
        Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
             backend_steps=args.muon_backend_steps, weight_decay=args.muon_weight_decay),
        torch.optim.Adam(embed_params,  lr=args.embed_lr,      betas=(args.beta1, args.beta2), eps=args.adam_eps),
        torch.optim.Adam(scalar_params, lr=args.scalar_lr,     betas=(args.beta1, args.beta2), eps=args.adam_eps),
        torch.optim.Adam(head_params,   lr=args.tied_embed_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps),
    ]

    # ── Training ─────────────────────────────────────────────────
    t0            = time.perf_counter()
    best_bpb      = float("inf")
    ema_initialised = False
    log_loss_accum  = 0.0

    print(f"[rank {rank}] Starting training — {args.iterations} steps", flush=True)

    for step in range(args.iterations):
        elapsed = time.perf_counter() - t0
        if elapsed >= args.max_wallclock_seconds:
            if rank == 0:
                print(f"[step {step}] Wallclock limit reached ({elapsed:.0f}s)", flush=True)
            break

        # LR schedule
        lr_scale = get_lr(step, args, 1.0)
        for opt in optimisers:
            for g in opt.param_groups:
                base = g.get("_base_lr", g["lr"])
                g.setdefault("_base_lr", base)
                g["lr"] = base * lr_scale

        # QAT activation
        if step == args.qat_start_step:
            raw_model.set_qat(True)
            if rank == 0:
                print(f"[step {step}] QAT int6 activated", flush=True)

        # EMA initialisation
        if step == args.ema_start_step and not ema_initialised:
            ema.init(); ema_initialised = True

        model.train()
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y) / grad_accum
            loss.backward()
            step_loss += loss.item()

        # Gradient clip
        nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)

        for opt in optimisers:
            opt.step(); opt.zero_grad(set_to_none=True)
        _restore_fp32(raw_model)

        if ema_initialised:
            ema.update()

        log_loss_accum += step_loss
        if step % args.train_log_every == 0 and rank == 0:
            avg = log_loss_accum / max(1, args.train_log_every)
            print(f"[step {step:5d}] loss={avg:.4f}  lr={lr_scale * args.matrix_lr:.2e}  "
                  f"t={elapsed:.0f}s", flush=True)
            log_loss_accum = 0.0

        if step % args.val_loss_every == 0 and step > 0:
            eval_model = raw_model

            if ema_initialised:
                backup = ema.apply()

            if args.lact_enabled and ema_initialised:
                # LaCT TTT eval
                val_loss, bpb = eval_with_lact_ttt(
                    args, eval_model, val_tokens,
                    base_bytes_lut, has_leading_lut, is_boundary_lut,
                    rank, world_size, device
                )
            else:
                val_loss, bpb = eval_sliding_window(
                    args, eval_model, val_tokens,
                    base_bytes_lut, has_leading_lut, is_boundary_lut,
                    rank, world_size, device
                )

            if ema_initialised:
                ema.restore(backup)

            if rank == 0:
                print(f"[step {step:5d}] val_loss={val_loss:.4f}  BPB={bpb:.4f}", flush=True)

            if bpb < best_bpb and rank == 0:
                best_bpb = bpb
                # Save checkpoint
                ckpt_path = f"ckpt_{args.run_id}.pt"
                torch.save(raw_model.state_dict(), ckpt_path)

    # ── Final evaluation + artifact ──────────────────────────────
    if rank == 0:
        print(f"\nTraining done.  Best BPB={best_bpb:.4f}", flush=True)

        # Apply EMA for final eval + artifact
        if ema_initialised:
            backup = ema.apply()

        # Final sliding-window eval (without TTT, for a clean number)
        val_loss_sw, bpb_sw = eval_sliding_window(
            args, raw_model, val_tokens,
            base_bytes_lut, has_leading_lut, is_boundary_lut,
            0, 1, device
        )
        print(f"Final BPB (sliding window, EMA): {bpb_sw:.4f}", flush=True)

        # Final LaCT eval
        if args.lact_enabled:
            val_loss_lact, bpb_lact = eval_with_lact_ttt(
                args, raw_model, val_tokens,
                base_bytes_lut, has_leading_lut, is_boundary_lut,
                0, 1, device
            )
            print(f"Final BPB (LaCT TTT):           {bpb_lact:.4f}", flush=True)

        artifact_path = f"artifact_{args.run_id}.pgolf"
        build_artifact(raw_model, args, artifact_path)

        if ema_initialised:
            ema.restore(backup)

    if distributed:
        dist.destroy_process_group()

# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = Hyperparameters()

    if "--eval" in sys.argv:
        # Load and evaluate a saved artifact
        idx  = sys.argv.index("--eval")
        path = sys.argv[idx + 1]
        dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, args = load_artifact(path, dev)
        sp = spm.SentencePieceProcessor(); sp.Load(args.tokenizer_path)
        bb, hl, ib = build_sentencepiece_luts(sp, args.vocab_size, dev)
        val_tokens  = load_validation_tokens(args.val_files, args.eval_seq_len).to(dev)
        vl, bpb = eval_sliding_window(args, model, val_tokens, bb, hl, ib, 0, 1, dev)
        print(f"Artifact BPB (sliding window): {bpb:.4f}")
        if args.lact_enabled:
            vl2, bpb2 = eval_with_lact_ttt(args, model, val_tokens, bb, hl, ib, 0, 1, dev)
            print(f"Artifact BPB (LaCT TTT):       {bpb2:.4f}")
    else:
        train(args)
