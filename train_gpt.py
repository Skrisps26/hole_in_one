from __future__ import annotations
import os, glob, time, math, lzma
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

# =============================
# SPEED SETTINGS
# =============================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# =============================
# CONFIG
# =============================
VOCAB = 1024
DIM = 384
LAYERS = 9
HEADS = 8
SEQ = 1024
BATCH_TOKENS = 786432
MAX_STEPS = 9000

DATA_PATH = "./data/datasets/fineweb10B_sp1024"
TRAIN_GLOB = os.path.join(DATA_PATH, "fineweb_train_*.bin")

# =============================
# DATA LOADER
# =============================
def load_data_shard(file: Path):
    header = np.fromfile(file, dtype="<i4", count=256)
    tokens = np.fromfile(file, dtype="<u2", offset=256*4)
    return torch.from_numpy(tokens.astype(np.int64))

class TokenStream:
    def __init__(self, pattern):
        self.files = sorted(glob.glob(pattern))
        self.idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next(self, n):
        out = []
        while n > 0:
            if self.pos >= len(self.tokens):
                self.idx = (self.idx + 1) % len(self.files)
                self.tokens = load_data_shard(self.files[self.idx])
                self.pos = 0
                continue
            k = min(n, len(self.tokens) - self.pos)
            out.append(self.tokens[self.pos:self.pos+k])
            self.pos += k
            n -= k
        return torch.cat(out)

class DistributedLoader:
    def __init__(self, pattern, rank, world, device):
        self.rank = rank
        self.world = world
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, tokens, seq):
        per_rank = tokens // self.world
        chunk = self.stream.next(per_rank*self.world + 1)

        start = self.rank * per_rank
        local = chunk[start:start+per_rank+1]

        tok = local[:-1]
        tgt = local[1:]

        usable = (tok.numel() // seq) * seq
        tok = tok[:usable]
        tgt = tgt[:usable]

        x = tok.reshape(-1, seq).to(self.device, non_blocking=True)
        y = tgt.reshape(-1, seq).to(self.device, non_blocking=True)
        return x, y

# =============================
# MODEL
# =============================
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.RMSNorm(DIM)
        self.ln2 = nn.RMSNorm(DIM)

        self.qkv = nn.Linear(DIM, 3*DIM, bias=False)
        self.proj = nn.Linear(DIM, DIM, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(DIM, 4*DIM),
            nn.GELU(),
            nn.Linear(4*DIM, DIM)
        )

    def forward(self, x):
        B,T,D = x.shape
        h = HEADS
        d = D // h

        qkv = self.qkv(self.ln1(x))
        q,k,v = qkv.chunk(3, dim=-1)

        q = q.view(B,T,h,d).transpose(1,2)
        k = k.view(B,T,h,d).transpose(1,2)
        v = v.view(B,T,h,d).transpose(1,2)

        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).reshape(B,T,D)

        x = x + self.proj(y)
        x = x + self.ff(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln = nn.RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, x, y=None):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        logits = self.head(x)

        if y is None:
            return logits
        return F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))

# =============================
# TRAIN
# =============================
def train():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    model = Model().to(device)
    model = DDP(model, device_ids=[rank], static_graph=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    loader = DistributedLoader(TRAIN_GLOB, rank, world, device)

    total_time = 0

    for step in range(MAX_STEPS):
        t0 = time.perf_counter()

        x,y = loader.next_batch(BATCH_TOKENS, SEQ)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x,y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # clamp (important for compression)
        for p in model.parameters():
            p.data.clamp_(-3, 3)

        step_time = (time.perf_counter() - t0) * 1000
        total_time += step_time

        if rank == 0 and step % 100 == 0:
            print(f"{step} loss={loss.item():.3f} step={step_time:.2f}ms avg={total_time/(step+1):.2f}ms")

    return model.module

# =============================
# EVAL (CLEAN)
# =============================
@torch.no_grad()
def evaluate(model, loader, steps=50):
    model.eval()
    total_loss = 0

    for _ in range(steps):
        x,y = loader.next_batch(BATCH_TOKENS, SEQ)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x,y)
        total_loss += loss.item()

    avg_loss = total_loss / steps
    return (avg_loss / math.log(2)) * 0.75

# =============================
# GPTQ-LITE
# =============================
def gptq_lite(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() != 2:
                continue
            if not ("blocks.7" in name or "blocks.8" in name or "head" in name):
                continue

            W = p.data
            scale = W.abs().mean(dim=1, keepdim=True) + 1e-6
            q = torch.round(W / scale * 7)
            p.data.copy_(q * scale)

# =============================
# COMPRESSION
# =============================
def compress(model):
    out = {}

    for name, p in model.state_dict().items():
        W = p.float()

        if W.ndim == 2:
            # log transform
            W = torch.sign(W) * torch.log1p(W.abs())

            scale = W.abs().max(dim=1, keepdim=True)[0] + 1e-8
            q = torch.round(W / scale * 15).to(torch.int8)

            q = (q + 16).cpu().numpy().astype(np.uint8).flatten()

            out[name] = lzma.compress(q.tobytes())
        else:
            out[name] = lzma.compress(W.half().cpu().numpy().tobytes())

    return out

# =============================
# RUN
# =============================
if __name__ == "__main__":
    m = train()

    if dist.get_rank() == 0:
        loader = DistributedLoader(TRAIN_GLOB, 0, dist.get_world_size(), torch.device("cuda:0"))
        print("val_bpb:", evaluate(m, loader))

        gptq_lite(m)

        art = compress(m)
        print("size:", sum(len(v) for v in art.values()))

    dist.destroy_process_group()
