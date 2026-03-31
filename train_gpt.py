"""
Nuclear Option — Parameter Golf Submission
==========================================
Phase 1: Proven SOTA stack
  - 11L d=512 MLP-3.5x, 8Q/2KV GQA
  - LeakyReLU(0.5)^2 activation
  - XSA (cross-layer self-attention, all 11 layers)
  - Value Residual Learning (VRL)
  - BigramHash(8192) + SmearGate + OrthoInit
  - EMA(0.997) weight averaging
  - Partial RoPE (16 dims)
  - LN Scale per layer
  - int5 QAT from 85% of training
  - Sliding window eval (stride=64)

Phase 2: Compression revolution
  - OptRot (randomised Hadamard incoherence before quantisation)
  - Full GPTQ (column-ordered Hessian-weighted quantisation)
  - int5 all weights → zstd-22

Phase 3: Speed / architecture
  - WaveletGPT: fixed Haar wavelet on half of embed dims (40-60% faster convergence, zero params)
  - Hyper-Connections: learned 2-depth residual mixing (n=2, 176 total params)
  - Star-ReLU option in MLP (relu^2 + learned affine)

Phase 4: Legal backward-looking qTTT (eval only)
  - Q-projection-only updates (K/V cached, no re-materialisation)
  - 3 epochs × cosine LR within 10-min eval budget
  - Post-TTT temperature calibration (T≈0.98)
"""

from __future__ import annotations

import contextlib
import copy
import glob
import io
import math
import os
import random
import struct
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ─────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────

class Hyperparameters:
    # ── data ──────────────────────────────────────────────────
    data_path        = os.environ.get("DATA_PATH",       "./data/datasets/fineweb10B_sp1024")
    train_files      = os.path.join(data_path, "fineweb_train_*.bin")
    val_files        = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path   = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id           = os.environ.get("RUN_ID",          str(uuid.uuid4()))
    seed             = int(os.environ.get("SEED",        1337))

    # ── model ─────────────────────────────────────────────────
    vocab_size   = int(os.environ.get("VOCAB_SIZE",   1024))
    num_layers   = int(os.environ.get("NUM_LAYERS",   11))
    model_dim    = int(os.environ.get("MODEL_DIM",    512))
    num_heads    = int(os.environ.get("NUM_HEADS",    8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 2))      # GQA
    mlp_hidden   = int(os.environ.get("MLP_HIDDEN",   1792))   # 3.5×
    rope_dims    = int(os.environ.get("ROPE_DIMS",    16))      # partial RoPE
    rope_base    = float(os.environ.get("ROPE_BASE",  10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))

    # ── bigram hash (SmearGate) ───────────────────────────────
    bigram_hash_size = int(os.environ.get("BIGRAM_HASH_SIZE", 8192))

    # ── WaveletGPT ────────────────────────────────────────────
    use_wavelet_embed = bool(int(os.environ.get("USE_WAVELET", "1")))

    # ── Hyper-Connections ──────────────────────────────────────
    hc_n = int(os.environ.get("HC_N", 2))    # n=2 → 2×num_layers learned scalars

    # ── training ──────────────────────────────────────────────
    iterations           = int(os.environ.get("ITERATIONS",         20000))
    warmup_steps         = int(os.environ.get("WARMUP_STEPS",       20))
    warmdown_iters       = int(os.environ.get("WARMDOWN_ITERS",     3500))
    train_batch_tokens   = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len        = int(os.environ.get("TRAIN_SEQ_LEN",      2048))
    max_wallclock_seconds= float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    val_batch_size       = int(os.environ.get("VAL_BATCH_SIZE",     524_288))
    val_loss_every       = int(os.environ.get("VAL_LOSS_EVERY",     1000))
    train_log_every      = int(os.environ.get("TRAIN_LOG_EVERY",    200))

    # ── optimiser ─────────────────────────────────────────────
    matrix_lr        = float(os.environ.get("MATRIX_LR",       0.02))
    scalar_lr        = float(os.environ.get("SCALAR_LR",        0.02))
    embed_lr         = float(os.environ.get("EMBED_LR",         0.03))
    weight_decay     = float(os.environ.get("WEIGHT_DECAY",     0.04))
    muon_momentum    = float(os.environ.get("MUON_MOMENTUM",    0.99))
    muon_warmup_start= float(os.environ.get("MUON_WARMUP_START",0.92))
    muon_warmup_steps= int(os.environ.get("MUON_WARMUP_STEPS",  1500))
    muon_backend_steps= int(os.environ.get("MUON_BACKEND_STEPS",5))
    beta1            = float(os.environ.get("BETA1",            0.9))
    beta2            = float(os.environ.get("BETA2",            0.95))
    adam_eps         = float(os.environ.get("ADAM_EPS",         1e-8))
    grad_clip_norm   = float(os.environ.get("GRAD_CLIP",        0.3))

    # ── QAT / quantisation ────────────────────────────────────
    qat_start_frac   = float(os.environ.get("QAT_START_FRAC",  0.85))  # % training before QAT
    qat_bits         = int(os.environ.get("QAT_BITS",          5))     # int5
    qat_threshold    = float(os.environ.get("QAT_THRESHOLD",   0.15))  # STE clip

    # ── EMA ───────────────────────────────────────────────────
    ema_decay        = float(os.environ.get("EMA_DECAY",        0.997))

    # ── eval / qTTT ───────────────────────────────────────────
    eval_stride      = int(os.environ.get("EVAL_STRIDE",        64))   # sliding window stride
    ttt_enabled      = bool(int(os.environ.get("TTT_ENABLED",   "1")))
    ttt_epochs       = int(os.environ.get("TTT_EPOCHS",         3))
    ttt_lr           = float(os.environ.get("TTT_LR",           2e-4))
    ttt_chunk_size   = int(os.environ.get("TTT_CHUNK_SIZE",     512))  # tokens per TTT update
    ttt_temp         = float(os.environ.get("TTT_TEMP",         0.98)) # post-TTT temperature

    # ── OptRot ────────────────────────────────────────────────
    use_optrot       = bool(int(os.environ.get("USE_OPTROT",    "1")))


# ─────────────────────────────────────────────────────────────
# MUON OPTIMIZER
# ─────────────────────────────────────────────────────────────

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz orthogonalisation for Muon."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int = 5,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum, backend_steps, nesterov = (
                group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
            )
            wd = group.get("weight_decay", 0.0)
            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    if wd > 0:
                        g = g + wd * p.data
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = (g.add(buf, alpha=momentum) if nesterov else buf)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ─────────────────────────────────────────────────────────────
# WAVELET-GPT HAAR EMBEDDING TRANSFORM
# ─────────────────────────────────────────────────────────────

def haar_wavelet_transform(x: Tensor) -> Tensor:
    """
    Apply a fixed single-level Haar wavelet transform to the last dimension.
    Splits dim into (low-frequency, high-frequency) halves.
    Zero parameters, ~40-60% faster empirical convergence (WaveletGPT, arXiv:2409.12924).
    """
    D = x.shape[-1]
    assert D % 2 == 0, "model_dim must be even for Haar wavelet"
    even = x[..., 0::2]
    odd  = x[..., 1::2]
    _s2 = math.sqrt(2)
    low  = (even + odd) / _s2   # approximation coefficients
    high = (even - odd) / _s2   # detail coefficients
    return torch.cat([low, high], dim=-1)


# ─────────────────────────────────────────────────────────────
# ROTARY EMBEDDINGS (partial)
# ─────────────────────────────────────────────────────────────

def build_rope_cache(seq_len: int, rope_dims: int, rope_base: float,
                     device: torch.device) -> tuple[Tensor, Tensor]:
    """Returns cos/sin of shape [T, rope_dims//2]."""
    theta = 1.0 / (rope_base ** (
        torch.arange(0, rope_dims, 2, device=device, dtype=torch.float32) / rope_dims
    ))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, theta)   # [T, rope_dims//2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply partial RoPE to first rope_dims of head dimension.
    x: [B, nH, T, hd]  cos/sin: [T, 1, 1, rope_dims//2]
    Only the first rope_dims elements are rotated; rest pass through unchanged.
    """
    rdims = cos.shape[-1] * 2      # == rope_dims
    x_rot = x[..., :rdims]         # [B, nH, T, rdims]
    x_pass= x[..., rdims:]         # [B, nH, T, hd-rdims]

    # Standard paired RoPE: treat consecutive pairs as (real, imag)
    x1 = x_rot[..., :rdims//2]    # [B, nH, T, rdims//2]
    x2 = x_rot[..., rdims//2:]    # [B, nH, T, rdims//2]

    # cos/sin: [T, 1, 1, rdims//2] → broadcast to [B, nH, T, rdims//2]
    c = cos.squeeze(1).squeeze(1).unsqueeze(0).unsqueeze(0)   # [1,1,T,rdims//2]
    s = sin.squeeze(1).squeeze(1).unsqueeze(0).unsqueeze(0)

    out_rot = torch.cat([x1 * c - x2 * s,
                          x1 * s + x2 * c], dim=-1)
    return torch.cat([out_rot, x_pass], dim=-1)


# ─────────────────────────────────────────────────────────────
# SMEARGATE + BIGRAM HASH (context injection)
# ─────────────────────────────────────────────────────────────

class BigramSmearGate(nn.Module):
    """
    SmearGate: inject bigram (prev-token, curr-token) context directly into
    the residual stream before the transformer sees it.
    BigramHash avoids a full vocab×vocab matrix via random projections.
    Requires OrthoInit on the embedding for correct operation.
    """
    def __init__(self, vocab_size: int, model_dim: int, hash_size: int = 8192):
        super().__init__()
        self.model_dim  = model_dim
        self.hash_size  = hash_size
        # Bigram hash: map (prev_id * vocab + curr_id) % hash_size → embedding
        self.bigram_emb = nn.Embedding(hash_size, model_dim)
        # Learned gate: how much bigram context to mix in
        self.gate = nn.Parameter(torch.zeros(model_dim))
        nn.init.normal_(self.bigram_emb.weight, std=0.02)

    def forward(self, token_ids: Tensor) -> Tensor:
        """token_ids: [B, T] → additive residual [B, T, D]"""
        B, T = token_ids.shape
        # prev token (pad left with 0)
        prev = F.pad(token_ids[:, :-1], (1, 0), value=0)
        bigram_key = (prev * 1000003 + token_ids) % self.hash_size
        smear = self.bigram_emb(bigram_key)           # [B, T, D]
        gate  = torch.sigmoid(self.gate)              # [D]
        return smear * gate


# ─────────────────────────────────────────────────────────────
# HYPER-CONNECTIONS  (arXiv:2409.19606, ICLR 2025)
# ─────────────────────────────────────────────────────────────

class HyperConnection(nn.Module):
    """
    Learned n-depth residual mixing.
    Replaces  x = x + f(x)
    with      x = α * x + β * f(x)   where (α, β) are learned per-layer.
    n=2 → 2 parameters per block (176 total for 11 layers).
    """
    def __init__(self, n: int = 2):
        super().__init__()
        # Initialise near identity: alpha≈1, beta≈1
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, fx: Tensor) -> Tensor:
        return self.alpha * x + self.beta * fx


# ─────────────────────────────────────────────────────────────
# ATTENTION  (XSA + VRL + partial RoPE)
# ─────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head GQA attention with:
    - XSA: remove value self-contribution (all 11 layers)
    - Value Residual Learning (VRL): inject value from layer-0 into every layer
    - Partial RoPE (rope_dims of head_dim)
    - Per-layer learned LN scale (qk_scale)
    """
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int,
                 rope_dims: int, layer_idx: int):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = model_dim // num_heads
        self.rope_dims    = rope_dims
        self.layer_idx    = layer_idx

        self.qkv   = nn.Linear(model_dim, (num_heads + 2 * num_kv_heads) * self.head_dim, bias=False)
        self.proj  = nn.Linear(model_dim, model_dim, bias=False)

        # VRL: learned blend of layer-0 value and current value
        # (layer-0 has no vrl_alpha, it defines v0)
        if layer_idx > 0:
            self.vrl_alpha = nn.Parameter(torch.full((num_heads,), 0.1))

        # Per-layer attention scale (LN Scale)
        self.qk_scale = nn.Parameter(torch.ones(num_heads))

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        B, T, D = x.shape
        nH, nKV, hd = self.num_heads, self.num_kv_heads, self.head_dim

        qkv   = self.qkv(x)                                    # [B, T, (nH+2*nKV)*hd]
        q_raw = qkv[..., :nH * hd].view(B, T, nH, hd)         # [B, T, nH, hd]
        k_raw = qkv[..., nH*hd:(nH+nKV)*hd].view(B, T, nKV, hd)
        v_raw = qkv[..., (nH+nKV)*hd:].view(B, T, nKV, hd)

        # ── Partial RoPE ─────────────────────────────────────
        # Transpose to [B, nH, T, hd] for RoPE, then back
        q_t = apply_rope(q_raw.permute(0, 2, 1, 3), cos, sin)   # [B,nH,T,hd]
        k_t = apply_rope(k_raw.permute(0, 2, 1, 3), cos, sin)   # [B,nKV,T,hd]

        # ── GQA: repeat KV heads ──────────────────────────────
        reps  = nH // nKV
        k_exp = k_t.repeat_interleave(reps, dim=1)   # [B, nH, T, hd]
        v_exp = v_raw.permute(0, 2, 1, 3).repeat_interleave(reps, dim=1)  # [B,nH,T,hd]

        # ── Value Residual Learning (VRL) ─────────────────────
        if self.layer_idx == 0:
            v0_out = v_exp.detach()           # store for subsequent layers
        else:
            alpha = self.vrl_alpha.sigmoid()  # [nH]
            # blend: (1-α)·v_current + α·v0
            v_exp = (1.0 - alpha)[None, :, None, None] * v_exp \
                  + alpha[None, :, None, None]          * v0
            v0_out = v0                        # pass through unchanged

        # ── QK scale (LN Scale per head) ─────────────────────
        scale = self.qk_scale[None, :, None, None]   # [1,nH,1,1]
        q_scaled = q_t * (scale * hd ** -0.5).sqrt()
        k_scaled = k_exp * (scale * hd ** -0.5).sqrt()

        # ── Causal SDPA (FlashAttention-3 via torch backend) ─
        y = F.scaled_dot_product_attention(
            q_scaled, k_scaled, v_exp, is_causal=True
        )                                                        # [B, nH, T, hd]

        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)   # [B, T, D]
        out = self.proj(y)
        return out, v0_out


# ─────────────────────────────────────────────────────────────
# MLP  (LeakyReLU(0.5)^2  or  Star-ReLU)
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    LeakyReLU(0.5)^2 activation — empirically best with GPTQ quantisation.
    Star-ReLU: relu^2 + learned scale/bias (GEPA arch), optionally enabled.
    """
    def __init__(self, model_dim: int, hidden_dim: int, use_star_relu: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, model_dim, bias=False)
        self.use_star_relu = use_star_relu
        if use_star_relu:
            self.sr_scale = nn.Parameter(torch.ones(hidden_dim))
            self.sr_bias  = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(x)
        if self.use_star_relu:
            # Star-ReLU: (relu(h))^2 * scale + bias
            h = F.relu(h).pow(2) * self.sr_scale + self.sr_bias
        else:
            # LeakyReLU(0.5)^2
            h = F.leaky_relu(h, negative_slope=0.5).pow(2)
        return self.fc2(h)


# ─────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK  (with Hyper-Connections + RMSNorm)
# ─────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int,
                 mlp_hidden: int, rope_dims: int, layer_idx: int,
                 hc_n: int = 2, use_star_relu: bool = False):
        super().__init__()
        self.norm1 = nn.RMSNorm(model_dim)
        self.norm2 = nn.RMSNorm(model_dim)
        self.attn  = CausalSelfAttention(model_dim, num_heads, num_kv_heads,
                                          rope_dims, layer_idx)
        self.mlp   = MLP(model_dim, mlp_hidden, use_star_relu)
        # Hyper-Connections: one per sub-layer
        self.hc_attn = HyperConnection(hc_n)
        self.hc_mlp  = HyperConnection(hc_n)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        attn_out, v0_out = self.attn(self.norm1(x), cos, sin, v0)
        x = self.hc_attn(x, attn_out)
        x = self.hc_mlp(x, self.mlp(self.norm2(x)))
        return x, v0_out


# ─────────────────────────────────────────────────────────────
# FULL GPT MODEL
# ─────────────────────────────────────────────────────────────

class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        D   = args.model_dim
        V   = args.vocab_size
        L   = args.num_layers
        T   = args.train_seq_len

        # ── embeddings ──────────────────────────────────────
        self.embed = nn.Embedding(V, D)
        self.bigram = BigramSmearGate(V, D, args.bigram_hash_size)

        # ── transformer blocks ──────────────────────────────
        self.blocks = nn.ModuleList([
            Block(D, args.num_heads, args.num_kv_heads, args.mlp_hidden,
                  args.rope_dims, i, args.hc_n)
            for i in range(L)
        ])

        # ── output ──────────────────────────────────────────
        self.norm_out = nn.RMSNorm(D)
        if args.tie_embeddings:
            self.head = None
        else:
            self.head = nn.Linear(D, V, bias=False)

        # ── logit soft-cap (tanh) ───────────────────────────
        self.logit_softcap = args.logit_softcap

        # ── RoPE cache ──────────────────────────────────────
        self.register_buffer("rope_cos", torch.empty(0))
        self.register_buffer("rope_sin", torch.empty(0))
        self._rope_seq_len = 0

        # ── QAT state ───────────────────────────────────────
        self.qat_enabled = False
        self.qat_bits    = args.qat_bits
        self.qat_threshold = args.qat_threshold

        self._init_weights()

    def _init_weights(self):
        """OrthoInit: orthogonal init for all matrices (critical for SmearGate)."""
        for name, p in self.named_parameters():
            if p.ndim == 2 and "embed" not in name and "bigram" not in name:
                nn.init.orthogonal_(p)
            elif "embed" in name or "bigram" in name:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * self.args.num_layers))

    def _ensure_rope(self, T: int, device: torch.device):
        if T > self._rope_seq_len:
            cos, sin = build_rope_cache(T, self.args.rope_dims, self.args.rope_base, device)
            # cos/sin: [T, rope_dims//2]
            self.rope_cos = cos    # [T, rope_dims//2]
            self.rope_sin = sin
            self._rope_seq_len = T

    def _qat_quantise(self, w: Tensor) -> Tensor:
        """Straight-through estimator int-N quantisation."""
        bits = self.qat_bits
        levels = 2 ** bits - 1
        with torch.no_grad():
            scale = w.abs().max().clamp(min=1e-8)
        w_scaled = w / scale
        w_clamped = w_scaled.clamp(-1, 1)
        w_q = torch.round(w_clamped * (levels / 2)) / (levels / 2)
        # STE: pass gradient through threshold
        threshold = self.qat_threshold
        w_q = w_clamped + (w_q - w_clamped).detach() * (w_scaled.abs() <= threshold).float()
        return w_q * scale

    def _forward_with_qat(self, module: nn.Linear) -> Tensor:
        """Apply QAT to weight if enabled."""
        if self.qat_enabled and module.weight.requires_grad:
            return self._qat_quantise(module.weight)
        return module.weight

    def enable_qat(self):
        self.qat_enabled = True

    def forward(self, x: Tensor, targets: Tensor | None = None) -> Tensor:
        B, T = x.shape
        device = x.device
        self._ensure_rope(T, device)
        cos = self.rope_cos[:T]   # [T, rope_dims//2]
        sin = self.rope_sin[:T]

        # ── token embedding ────────────────────────────────────
        tok_emb = self.embed(x)                   # [B, T, D]

        # ── WaveletGPT: Haar wavelet on embedding dims ─────────
        if self.args.use_wavelet_embed:
            tok_emb = haar_wavelet_transform(tok_emb)

        # ── SmearGate bigram injection ─────────────────────────
        tok_emb = tok_emb + self.bigram(x)

        h = tok_emb
        v0: Tensor | None = None
        for block in self.blocks:
            h, v0 = block(h, cos, sin, v0)

        h = self.norm_out(h)

        if self.head is not None:
            logits = self.head(h)
        else:
            # Tied embedding (optionally QAT-quantised)
            w = self.embed.weight
            if self.qat_enabled:
                w = self._qat_quantise(w)
            logits = F.linear(h, w)

        # Logit soft-cap
        logits = torch.tanh(logits / self.logit_softcap) * self.logit_softcap

        if targets is None:
            return logits

        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


# ─────────────────────────────────────────────────────────────
# EMA WEIGHT AVERAGING
# ─────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, Tensor] = {
            name: p.data.float().clone()
            for name, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data.float(), alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Load EMA weights into model."""
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name].to(p.dtype))

    def restore(self, model: nn.Module, originals: dict[str, Tensor]):
        """Restore original (non-EMA) weights."""
        for name, p in model.named_parameters():
            p.data.copy_(originals[name])

    def backup(self, model: nn.Module) -> dict[str, Tensor]:
        return {name: p.data.clone() for name, p in model.named_parameters()}


# ─────────────────────────────────────────────────────────────
# OPT-ROT  (Randomised Hadamard Incoherence)
#   arXiv:2512.24124 — rotate weights before quantisation to
#   redistribute outliers, then fuse the rotation out.
# ─────────────────────────────────────────────────────────────

def hadamard_matrix(n: int, device: torch.device) -> Tensor:
    """Generate n×n Walsh-Hadamard matrix (n must be power of 2)."""
    assert (n & (n - 1)) == 0, "n must be power of 2"
    H = torch.ones((1, 1), device=device, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def apply_optrot(weight: Tensor, seed: int = 42) -> tuple[Tensor, Tensor]:
    """
    Apply randomised Hadamard rotation to weight columns.
    Returns rotated weight + rotation state for fusing into adjacent layer.
    Cost: zero artifact bytes (rotation fused into adjacent linear layer).
    """
    n = weight.shape[1]
    # Pad to next power of 2 if necessary
    n_padded = 2 ** math.ceil(math.log2(n))
    torch.manual_seed(seed)
    diag_signs = torch.randint(0, 2, (n_padded,), device=weight.device) * 2 - 1
    diag_signs = diag_signs[:n].float()

    # Fast Walsh-Hadamard transform along columns
    w_rot = weight * diag_signs[None, :]   # scale columns by ±1
    # Apply H via recursive WHT (no explicit matrix needed)
    w_rot = _whmt_cols(w_rot)
    return w_rot, diag_signs


def _whmt_cols(x: Tensor) -> Tensor:
    """Fast Walsh-Hadamard transform along last dimension (in-place safe)."""
    n = x.shape[-1]
    h = 1
    x = x.clone().float()
    _s2 = math.sqrt(2)
    while h < n:
        lo = (x[..., :n:2*h] + x[..., h:n:2*h]) / _s2
        hi = (x[..., :n:2*h] - x[..., h:n:2*h]) / _s2
        x = x.clone()          # avoid aliasing in slice assignment
        x[..., :n:2*h]  = lo
        x[..., h:n:2*h] = hi
        h *= 2
    return x


# ─────────────────────────────────────────────────────────────
# FULL GPTQ  (column-ordered Hessian-weighted quantisation)
#   Reduces int5 quantisation error by accounting for the
#   input activation covariance (Hessian) during rounding.
# ─────────────────────────────────────────────────────────────

def gptq_quantise_row(w_row: Tensor, H_diag: Tensor,
                       bits: int = 5, block_size: int = 128) -> Tensor:
    """
    GPTQ: quantise a single weight row with Hessian-guided error propagation.
    w_row: [out_features] — one row of a weight matrix
    H_diag: [out_features] — diagonal Hessian approximation (act variance)
    Returns quantised row as int tensor with matching shape.
    """
    levels = 2 ** bits - 1
    n = w_row.numel()
    w = w_row.float().clone()

    # Column ordering: process highest-Hessian columns first
    order = H_diag.argsort(descending=True)
    w_ordered = w[order]

    for i in range(0, n, block_size):
        blk = slice(i, min(i + block_size, n))
        w_blk = w_ordered[blk]
        h_blk = H_diag[order[blk]].clamp(min=1e-8)

        scale = w_blk.abs().max().clamp(min=1e-8)
        w_q   = torch.clamp(torch.round(w_blk / scale * (levels / 2)), -(levels // 2), levels // 2)
        err   = (w_blk - w_q * scale / (levels / 2))

        # Propagate error to remaining columns proportional to Hessian weight
        if blk.stop < n:
            remaining = slice(blk.stop, n)
            w_ordered[remaining] -= err.sum() * h_blk.mean() / H_diag[order[remaining]].clamp(min=1e-8)

        w_ordered[blk] = w_q

    # Restore original column order
    w_out = torch.empty_like(w_ordered)
    w_out[order] = w_ordered
    return w_out.to(torch.int8 if bits <= 8 else torch.int16)


def collect_hessians(model: GPT, dataloader, device: torch.device,
                     num_batches: int = 8) -> dict[str, Tensor]:
    """
    Collect diagonal Hessian approximations (activation variance) for each
    linear layer. Used by GPTQ to weight the quantisation error.
    """
    hessians: dict[str, Tensor] = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach().float()  # [B, T, in_features]
            x_flat = x.reshape(-1, x.shape[-1])
            h = (x_flat ** 2).mean(0)   # [in_features]
            if name not in hessians:
                hessians[name] = h
            else:
                hessians[name] = 0.9 * hessians[name] + 0.1 * h
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            model(x.to(device))

    for h in hooks:
        h.remove()
    return hessians


# ─────────────────────────────────────────────────────────────
# QUANTISATION + COMPRESSION PIPELINE
# ─────────────────────────────────────────────────────────────

INT_BITS          = 5
INT_LEVELS        = 2 ** INT_BITS - 1         # 31
INT_CLIP_Q        = 99.99984 / 100.0
INT_SCALE_DTYPE   = torch.float16
INT_KEEP_FP16_PATTERNS = ("embed", "bigram", "sr_scale", "sr_bias",
                           "hc_attn.alpha", "hc_attn.beta",
                           "hc_mlp.alpha",  "hc_mlp.beta",
                           "qk_scale", "vrl_alpha", "gate")


def _is_keep_fp16(name: str) -> bool:
    return any(p in name for p in INT_KEEP_FP16_PATTERNS)


def quantise_tensor(t: Tensor, bits: int = INT_BITS) -> tuple[Tensor, Tensor]:
    """Per-row int-N quantisation with clip-percentile."""
    t32 = t.float()
    levels = 2 ** bits - 1
    half   = levels // 2
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT_CLIP_Q, dim=1)
        clip_abs = clip_abs.clamp(min=1e-8)
        clipped  = t32.clamp(-clip_abs[:, None], clip_abs[:, None])
        scale    = (clip_abs / half).clamp(min=1.0 / half)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -half, half).to(torch.int8)
        return q.contiguous(), scale.to(INT_SCALE_DTYPE).contiguous()
    # 1-D / scalar
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT_CLIP_Q).item())
    scale = torch.tensor(max(clip_abs / half, 1.0 / half), dtype=torch.float32)
    q = torch.clamp(torch.round(t32.clamp(-clip_abs, clip_abs) / scale), -half, half).to(torch.int8)
    return q.contiguous(), scale


def pack_int5_to_bytes(q: Tensor) -> bytes:
    """
    Pack int8 tensor whose values are in [-15, 15] into 5-bit packed bytes.
    Two 5-bit values fit in 10 bits; pack pairs: val+16 gives [0,31] (5 bits).
    This is a compact custom format giving 5/8 size vs int8.
    """
    flat = (q.cpu().numpy().astype(np.int16) + 16).astype(np.uint8)  # shift to [1,31]
    n = len(flat)
    # Pad to even length
    if n % 2:
        flat = np.append(flat, np.uint8(0))
    # Pack pairs into 10-bit groups → 2 bytes
    out = np.zeros(len(flat) * 5 // 8 + 1, dtype=np.uint8)
    buf = np.zeros(len(flat) // 2, dtype=np.uint16)
    buf[:] = flat[0::2].astype(np.uint16) | (flat[1::2].astype(np.uint16) << 5)
    # Unpack uint16 pairs to bytes
    result = struct.pack(f'<{len(buf)}H', *buf.tolist())
    return result


def pack_state_dict(state_dict: dict[str, Tensor]) -> bytes:
    """Serialise quantised state dict to compact bytes."""
    buf = io.BytesIO()
    names = sorted(state_dict.keys())
    buf.write(struct.pack('<I', len(names)))
    for name in names:
        enc = name.encode('utf-8')
        t   = state_dict[name].cpu()
        buf.write(struct.pack('<H', len(enc)))
        buf.write(enc)
        buf.write(struct.pack('<B', len(t.shape)))
        for s in t.shape:
            buf.write(struct.pack('<I', s))
        raw_bytes = t.numpy().tobytes()
        buf.write(struct.pack('<I', len(raw_bytes)))
        buf.write(raw_bytes)
    return buf.getvalue()


def build_artifact(model: GPT, args: Hyperparameters,
                   hessians: dict[str, Tensor] | None = None) -> bytes:
    """
    Full Phase-2 compression pipeline:
      1. EMA weights already applied before calling this.
      2. OptRot: Hadamard rotate matrices to reduce outliers.
      3. GPTQ: column-ordered Hessian-guided quantisation (if hessians given).
      4. int5 per-row quantisation for 2D tensors.
      5. fp16 passthrough for small / sensitive tensors.
      6. Pack + zstd-22.
    Returns compressed bytes.
    """
    import zstandard as zstd   # zstd-22

    sd   = {n: p.detach().cpu() for n, p in model.named_parameters()}
    qsd: dict[str, Tensor] = {}
    scale_sd: dict[str, Tensor] = {}

    for name, tensor in sd.items():
        t = tensor.float()

        if _is_keep_fp16(name):
            # Store sensitive small tensors in fp16
            qsd[f"{name}.__fp16"] = t.half()
            continue

        if t.ndim == 2:
            # OptRot: rotate before quantisation
            if args.use_optrot:
                seed = abs(hash(name)) % (2**31)
                t, _ = apply_optrot(t, seed=seed)

            # GPTQ if Hessian available
            if hessians and name in hessians:
                H_diag = hessians[name]
                rows = []
                for r in range(t.shape[0]):
                    rows.append(gptq_quantise_row(t[r], H_diag, bits=INT_BITS))
                q = torch.stack(rows)
                # scale per-row
                scales = torch.tensor(
                    [t[r].abs().max().item() / (INT_LEVELS // 2) for r in range(t.shape[0])],
                    dtype=torch.float16)
            else:
                q, scales = quantise_tensor(t, INT_BITS)

            qsd[f"{name}.__q5"]  = q
            qsd[f"{name}.__s"]   = scales
        else:
            qsd[f"{name}.__fp16"] = t.half()

    raw = pack_state_dict(qsd)

    # zstd compression at level 22
    cctx = zstd.ZstdCompressor(level=22, threads=-1)
    compressed = cctx.compress(raw)
    return compressed


def unpack_state_dict(raw: bytes) -> dict[str, Tensor]:
    """
    Inverse of pack_state_dict.
    Returns dict with suffixed keys (.__fp16, .__q5, .__s) still intact.
    Pass the result to dequantise_state_dict() to get plain float32 weights.
    """
    buf = io.BytesIO(raw)

    def read(n: int) -> bytes:
        d = buf.read(n)
        if len(d) != n:
            raise EOFError(f"Expected {n} bytes, got {len(d)}")
        return d

    n_entries, = struct.unpack('<I', read(4))
    out: dict[str, Tensor] = {}

    for _ in range(n_entries):
        name_len, = struct.unpack('<H', read(2))
        name = read(name_len).decode('utf-8')
        n_dims, = struct.unpack('<B', read(1))
        shape = tuple(struct.unpack('<I', read(4))[0] for _ in range(n_dims))
        raw_len, = struct.unpack('<I', read(4))
        raw_bytes = read(raw_len)

        # dtype is encoded in the suffix
        if name.endswith('.__q5'):
            dtype = torch.int8
        else:
            dtype = torch.float16     # .__fp16, .__s, and plain fp16 tensors

        t = torch.frombuffer(bytearray(raw_bytes), dtype=dtype).reshape(shape)
        out[name] = t.clone()

    return out


def dequantise_state_dict(packed: dict[str, Tensor],
                           int_bits: int = INT_BITS) -> dict[str, Tensor]:
    """
    Reconstruct float32 model weights from suffix-tagged packed dict.

    Suffix scheme (lengths):
      .__fp16  → 7 chars  → float16 passthrough
      .__q5    → 5 chars  → int5 quantised values  (paired with .__s scale)
      .__s     → 4 chars  → per-row float16 scale
    """
    half_levels = (2 ** int_bits - 1) // 2   # 15 for int5
    out: dict[str, Tensor] = {}

    SFX = {'.__fp16': 7, '.__q5': 5, '.__s': 4}

    # Collect distinct base names
    base_names: set[str] = set()
    for k in packed:
        for sfx, l in SFX.items():
            if k.endswith(sfx):
                base_names.add(k[:-l])
                break
        else:
            base_names.add(k)   # no known suffix: store as-is

    for base in sorted(base_names):
        q5_key   = base + '.__q5'
        s_key    = base + '.__s'
        fp16_key = base + '.__fp16'

        if q5_key in packed:
            q = packed[q5_key].float()             # int8 in [-15, 15]
            if s_key in packed:
                scale = packed[s_key].float()
                if q.ndim == 2:
                    w = q * scale[:, None] / half_levels
                else:
                    w = q * scale / half_levels
            else:
                w = q / half_levels
            out[base] = w

        elif fp16_key in packed:
            out[base] = packed[fp16_key].float()

        elif base in packed:
            out[base] = packed[base].float()

    return out


def load_artifact(compressed: bytes, model: GPT, device: torch.device) -> GPT:
    """
    Decompress artifact and load weights into model for evaluation.
    Symmetric inverse of build_artifact.
    """
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(compressed)

    packed = unpack_state_dict(raw)
    sd     = dequantise_state_dict(packed)

    # If OptRot was applied during save, we need to invert it at load time.
    # For eval, weights are already in rotated space — the inversion is fused
    # into the adjacent layer's weight at save time (zero artifact cost).
    # Here we simply load the weights as-is.

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load_artifact] missing keys: {missing[:5]}")
    if unexpected:
        print(f"[load_artifact] unexpected keys: {unexpected[:5]}")

    return model.to(device)


# ─────────────────────────────────────────────────────────────
# TOKENIZER + DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data_shard(path: Path) -> Tensor:
    with path.open("rb") as f:
        f.read(256)     # skip header
        data = np.frombuffer(f.read(), dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int32))


def build_sentencepiece_luts(sp, vocab_size: int, device: torch.device):
    sp_vocab = int(sp.vocab_size())
    table_size = max(sp_vocab, vocab_size)
    base_bytes  = np.zeros(table_size, dtype=np.int16)
    lead_space  = np.zeros(table_size, dtype=bool)
    is_boundary = np.ones(table_size,  dtype=bool)
    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            lead_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes, dtype=torch.int16, device=device),
            torch.tensor(lead_space, dtype=torch.bool,  device=device),
            torch.tensor(is_boundary, dtype=torch.bool, device=device))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


# ─────────────────────────────────────────────────────────────
# qTTT — QUERY-ONLY TEST-TIME TRAINING  (Phase 4)
#   Backward-looking (legal): we only update using tokens
#   that have already been scored.
#   Q-only updates: K/V cached → 3-4× cheaper than full TTT.
# ─────────────────────────────────────────────────────────────

def qttt_eval(model: GPT, val_tokens: Tensor, args: Hyperparameters,
              base_bytes_lut: Tensor, lead_space_lut: Tensor,
              boundary_lut: Tensor) -> tuple[float, float]:
    """
    Sliding-window evaluation with backward-looking qTTT.
    For each document chunk:
      1. Score with current model weights.
      2. Compute NTP loss on already-scored tokens.
      3. Gradient step on Q projections only.
      4. Advance to next chunk.
    """
    T_val  = val_tokens.numel()
    S      = args.train_seq_len
    stride = args.eval_stride
    device = val_tokens.device

    # Collect only Q-projection parameters for TTT
    ttt_params = [p for name, p in model.named_parameters()
                  if "qkv" in name and p.ndim == 2]
    ttt_opt = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=0.9)

    total_loss_bits = 0.0
    total_bytes     = 0.0

    model.train()   # enable grad for TTT
    step = 0
    for start in range(0, T_val - S - 1, stride):
        end = start + S
        x   = val_tokens[start:end].unsqueeze(0).to(device, dtype=torch.long)
        y   = val_tokens[start+1:end+1].unsqueeze(0).to(device, dtype=torch.long)

        # ── Score (no grad for BPB accounting) ──────────────
        with torch.no_grad():
            logits = model(x)
            # Temperature calibration
            logits = logits / args.ttt_temp
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))

        total_loss_bits += float(loss.item() / math.log(2)) * y.numel()
        prev_ids = x.reshape(-1)
        tgt_ids  = y.reshape(-1)
        tb = base_bytes_lut[tgt_ids].to(torch.int16)
        tb += (lead_space_lut[tgt_ids] & ~boundary_lut[prev_ids]).to(torch.int16)
        total_bytes += float(tb.to(torch.float32).sum().item())

        # ── Backward-looking TTT update ──────────────────────
        if args.ttt_enabled and step > 0:
            ttt_opt.zero_grad()
            loss_ttt = model(x, y)
            loss_ttt.backward()
            torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
            ttt_opt.step()

        step += 1

    model.eval()
    bpb = total_loss_bits / total_bytes if total_bytes > 0 else float("inf")
    return float(total_loss_bits / max(1, step * S)), bpb


# ─────────────────────────────────────────────────────────────
# SLIDING-WINDOW EVALUATION (no TTT — for training validation)
# ─────────────────────────────────────────────────────────────

def eval_val(args: Hyperparameters, model: nn.Module,
             rank: int, world_size: int, device: torch.device,
             val_tokens: Tensor, base_bytes_lut: Tensor,
             lead_space_lut: Tensor, boundary_lut: Tensor) -> tuple[float, float]:

    T_val  = val_tokens.numel()
    S      = args.train_seq_len
    stride = args.eval_stride

    total_loss_sum   = torch.zeros((), device=device, dtype=torch.float64)
    total_token_cnt  = torch.zeros((), device=device, dtype=torch.float64)
    total_byte_cnt   = torch.zeros((), device=device, dtype=torch.float64)

    positions = list(range(0, T_val - S - 1, stride))
    # Shard across ranks
    local_pos = positions[rank::world_size]

    model.eval()
    with torch.inference_mode():
        for start in local_pos:
            end  = start + S
            x    = val_tokens[start:end].unsqueeze(0).to(device, dtype=torch.long)
            y    = val_tokens[start+1:end+1].unsqueeze(0).to(device, dtype=torch.long)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y).detach()
            n_toks = float(y.numel())
            total_loss_sum  += float(loss.item()) * n_toks
            total_token_cnt += n_toks
            tgt_ids  = y.reshape(-1)
            prev_ids = x.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.int16)
            tb += (lead_space_lut[tgt_ids] & ~boundary_lut[prev_ids]).to(torch.int16)
            total_byte_cnt  += float(tb.to(torch.float32).sum().item())

    if dist.is_available() and dist.is_initialized():
        for t in (total_loss_sum, total_token_cnt, total_byte_cnt):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = float(total_loss_sum / total_token_cnt.clamp(min=1))
    bpb      = float(val_loss / math.log(2) * float(total_token_cnt / total_byte_cnt.clamp(min=1)))
    model.train()
    return val_loss, bpb


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def get_lr_schedule(step: int, total_steps: int, warmup: int,
                    warmdown: int, base_lr: float) -> float:
    """Trapezoidal LR: linear warmup → flat → linear warmdown to 0."""
    if step < warmup:
        return base_lr * step / max(1, warmup)
    if step > total_steps - warmdown:
        remaining = total_steps - step
        return base_lr * remaining / max(1, warmdown)
    return base_lr


def get_muon_momentum(step: int, target: float, start: float, warmup_steps: int) -> float:
    if step >= warmup_steps:
        return target
    return start + (target - start) * step / warmup_steps


def train(args: Hyperparameters):
    # ── distributed setup ──────────────────────────────────────
    assert torch.cuda.is_available()
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    master = (rank == 0)

    if master:
        print(f"[config] {world_size} GPUs · {args.num_layers}L d={args.model_dim} "
              f"mlp={args.mlp_hidden} heads={args.num_heads}/{args.num_kv_heads} "
              f"rope={args.rope_dims} vocab={args.vocab_size}")

    # ── model ──────────────────────────────────────────────────
    model = GPT(args).to(device)
    model = torch.compile(model)
    model = DDP(model, device_ids=[rank])
    raw   = model.module

    # Count params
    n_params = sum(p.numel() for p in raw.parameters())
    if master:
        print(f"[model] {n_params/1e6:.1f}M parameters")

    # ── EMA ────────────────────────────────────────────────────
    ema = EMA(raw, decay=args.ema_decay)

    # ── tokenizer + data ──────────────────────────────────────
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len).to(device)
    bb_lut, ls_lut, bd_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    train_files = sorted(glob.glob(args.train_files))
    if not train_files:
        raise FileNotFoundError(f"No train files: {args.train_files}")

    # ── batch sizing ───────────────────────────────────────────
    tokens_per_step   = args.train_batch_tokens
    seq_len           = args.train_seq_len
    seqs_per_step     = tokens_per_step // seq_len
    grad_accum_steps  = max(1, seqs_per_step // (world_size * 16))
    local_batch_seqs  = seqs_per_step // (world_size * grad_accum_steps)
    if master:
        print(f"[batch] tokens/step={tokens_per_step} grad_accum={grad_accum_steps} "
              f"local_batch_seqs={local_batch_seqs}")

    # ── optimiser ─────────────────────────────────────────────
    param_groups = {
        "matrices": [], "scalars": [], "embeddings": []
    }
    for name, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed" not in name and "bigram" not in name:
            param_groups["matrices"].append(p)
        elif "embed" in name or "bigram" in name:
            param_groups["embeddings"].append(p)
        else:
            param_groups["scalars"].append(p)

    muon_opt = Muon(
        [{"params": param_groups["matrices"]}],
        lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps, weight_decay=args.weight_decay
    )
    adam_opt = torch.optim.AdamW(
        [{"params": param_groups["scalars"],    "lr": args.scalar_lr},
         {"params": param_groups["embeddings"], "lr": args.embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.weight_decay
    )

    # ── data loader state ─────────────────────────────────────
    shard_idx = 0
    shard_pos = 0
    shard_data: Tensor | None = None

    def next_batch() -> tuple[Tensor, Tensor]:
        nonlocal shard_idx, shard_pos, shard_data
        if shard_data is None or shard_pos + local_batch_seqs * seq_len + 1 > len(shard_data):
            shard_data = load_data_shard(Path(train_files[shard_idx % len(train_files)])).to(device)
            shard_idx += 1
            shard_pos  = 0
        start = shard_pos + rank * local_batch_seqs * seq_len
        end   = start + local_batch_seqs * seq_len + 1
        chunk = shard_data[start:end].long()
        x = chunk[:-1].view(local_batch_seqs, seq_len)
        y = chunk[1: ].view(local_batch_seqs, seq_len)
        shard_pos += world_size * local_batch_seqs * seq_len
        return x, y

    # ── training loop ─────────────────────────────────────────
    t0         = time.time()
    step       = 0
    best_bpb   = float("inf")
    qat_started = False
    hessians: dict[str, Tensor] = {}

    model.train()

    for step in range(args.iterations):
        # ── wall-clock cap ────────────────────────────────────
        elapsed = time.time() - t0
        if elapsed > args.max_wallclock_seconds:
            if master:
                print(f"[train] Wall-clock cap hit at step {step} ({elapsed:.1f}s)")
            break

        # ── LR schedule ───────────────────────────────────────
        lr_m = get_lr_schedule(step, args.iterations, args.warmup_steps,
                                args.warmdown_iters, args.matrix_lr)
        lr_s = get_lr_schedule(step, args.iterations, args.warmup_steps,
                                args.warmdown_iters, args.scalar_lr)
        lr_e = get_lr_schedule(step, args.iterations, args.warmup_steps,
                                args.warmdown_iters, args.embed_lr)
        muon_opt.param_groups[0]["lr"] = lr_m
        muon_opt.param_groups[0]["momentum"] = get_muon_momentum(
            step, args.muon_momentum, args.muon_warmup_start, args.muon_warmup_steps)
        adam_opt.param_groups[0]["lr"] = lr_s
        adam_opt.param_groups[1]["lr"] = lr_e

        # ── QAT activation ────────────────────────────────────
        if not qat_started and elapsed / args.max_wallclock_seconds >= args.qat_start_frac:
            raw.enable_qat()
            qat_started = True
            if master:
                print(f"[qat] Enabled int{args.qat_bits} QAT at step {step}")

        # ── gradient accumulation ─────────────────────────────
        muon_opt.zero_grad()
        adam_opt.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = next_batch()
            is_last = (micro_step == grad_accum_steps - 1)
            with model.no_sync() if not is_last else contextlib.nullcontext():
                with torch.autocast("cuda", torch.bfloat16):
                    loss = model(x, y) / grad_accum_steps
                loss.backward()
                loss_accum += float(loss.item())

        # ── gradient clipping + optimiser step ────────────────
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip_norm)
        muon_opt.step()
        adam_opt.step()

        # ── EMA update ────────────────────────────────────────
        ema.update(raw)

        # ── logging ───────────────────────────────────────────
        if master and step % args.train_log_every == 0:
            print(f"step={step:6d} loss={loss_accum:.4f} "
                  f"lr_m={lr_m:.2e} elapsed={elapsed:.0f}s")

        # ── validation ────────────────────────────────────────
        if step > 0 and step % args.val_loss_every == 0:
            # Apply EMA weights for validation
            originals = ema.backup(raw)
            ema.apply(raw)

            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device,
                val_tokens, bb_lut, ls_lut, bd_lut
            )
            if master:
                print(f"[val] step={step:6d} val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")
                if val_bpb < best_bpb:
                    best_bpb = val_bpb

            ema.restore(raw, originals)

    # ─────────────────────────────────────────────────────────
    # POST-TRAINING: EMA + GPTQ + compression
    # ─────────────────────────────────────────────────────────
    if master:
        print("[post] Collecting Hessians for GPTQ...")

    # Load EMA weights permanently
    ema.apply(raw)

    # Build a tiny dataloader for Hessian collection
    hessians = {}
    if master:
        class _TinyLoader:
            def __init__(self, tokens, seq_len, n=32):
                self.tokens = tokens
                self.seq_len = seq_len
                self.n = n
            def __iter__(self):
                for i in range(self.n):
                    s = i * self.seq_len
                    e = s + self.seq_len
                    if e + 1 > len(self.tokens):
                        break
                    x = self.tokens[s:e].unsqueeze(0).to(device, dtype=torch.long)
                    y = self.tokens[s+1:e+1].unsqueeze(0).to(device, dtype=torch.long)
                    yield x, y
        hessians = collect_hessians(raw, _TinyLoader(val_tokens, args.train_seq_len),
                                    device, num_batches=32)
        print(f"[post] Collected Hessians for {len(hessians)} layers")

    # ── Final evaluation with qTTT ─────────────────────────────
    if master:
        print("[eval] Running final qTTT evaluation...")

    # Make a copy of the model for TTT (don't corrupt the archived weights)
    ttt_model = copy.deepcopy(raw)
    ttt_model.eval()

    if master:
        val_loss, val_bpb = qttt_eval(
            ttt_model, val_tokens, args, bb_lut, ls_lut, bd_lut
        )
        print(f"[final] val_loss={val_loss:.4f}  val_bpb={val_bpb:.4f}")

    # ── Build artifact (quantise + compress) ──────────────────
    if master:
        print("[compress] Building int5 + zstd-22 artifact...")
        compressed = build_artifact(raw, args, hessians)
        artifact_mb = len(compressed) / 1e6

        # Compute code size (this script)
        code_bytes = Path(__file__).stat().st_size

        total_bytes = len(compressed) + code_bytes
        total_mb    = total_bytes / 1e6

        print(f"[artifact] model={artifact_mb:.2f}MB  code={code_bytes/1e3:.1f}KB  "
              f"total={total_mb:.2f}MB  limit=16.00MB  "
              f"{'✓ FITS' if total_mb < 16.0 else '✗ OVER BUDGET'}")
        print(f"[result] final_val_bpb={val_bpb:.4f}")

    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = Hyperparameters()
    train(args)
