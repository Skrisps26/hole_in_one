from __future__ import annotations
import os, glob, math, time, io, zlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

# =============================
# SPEED
# =============================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# =============================
# CONFIG
# =============================
VOCAB = 1024
DIM = 448
LAYERS = 10
HEADS = 8
KV_HEADS = 2
SEQ = 1024
BATCH_TOKENS = 1_000_000
STEPS = 2000

DATA_PATH = "./data/datasets/fineweb10B_sp1024"
TRAIN_GLOB = os.path.join(DATA_PATH, "fineweb_train_*.bin")

# =============================
# DATA LOADER (CORRECT)
# =============================
def load_data_shard(file: Path):
    header = np.fromfile(file, dtype="<i4", count=256)
    assert header[0] == 20240520
    num_tokens = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256*4)
    return torch.from_numpy(tokens.astype(np.int64))


class TokenStream:
    def __init__(self, pattern):
        self.files = sorted(glob.glob(pattern))
        assert len(self.files) > 0
        self.idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next(self, n):
        out = []
        while n > 0:
            avail = len(self.tokens) - self.pos
            if avail == 0:
                self.idx = (self.idx + 1) % len(self.files)
                self.tokens = load_data_shard(self.files[self.idx])
                self.pos = 0
                continue
            k = min(n, avail)
            out.append(self.tokens[self.pos:self.pos+k])
            self.pos += k
            n -= k
        return torch.cat(out)

@torch.no_grad()
def evaluate(model, loader, steps=50):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for _ in range(steps):
        x, y = loader.next_batch(BATCH_TOKENS, SEQ)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    model.train()

    avg_loss = total_loss / total_tokens

    # convert CE → bits per token
    bpt = avg_loss / math.log(2)

    # dataset specific (fineweb ~0.75 bytes/token)
    bpb = bpt * 0.75

    return bpb
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

        tokens = local[:-1]
        targets = local[1:]
        
        # trim to multiple of seq
        usable = (tokens.numel() // seq) * seq
        
        tokens = tokens[:usable]
        targets = targets[:usable]
        
        x = tokens.reshape(-1, seq).to(self.device, non_blocking=True)
        y = targets.reshape(-1, seq).to(self.device, non_blocking=True)
        
        return x, y

# =============================
# MODEL
# =============================
class Smear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb = nn.Embedding(8192, dim)
        self.g = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        prev = F.pad(x[:, :-1], (1,0))
        h = (prev * 1315423911 + x) % 8192
        return self.emb(h) * torch.sigmoid(self.g)


class GQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(DIM, DIM, bias=False)
        self.k = nn.Linear(DIM, KV_HEADS*(DIM//HEADS), bias=False)
        self.v = nn.Linear(DIM, KV_HEADS*(DIM//HEADS), bias=False)
        self.o = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x):
        B,T,D = x.shape
        h = HEADS
        d = D//h

        q = self.q(x).view(B,T,h,d).transpose(1,2)
        k = self.k(x).view(B,T,KV_HEADS,d).transpose(1,2)
        v = self.v(x).view(B,T,KV_HEADS,d).transpose(1,2)

        k = k.repeat_interleave(h//KV_HEADS, dim=1)
        v = v.repeat_interleave(h//KV_HEADS, dim=1)

        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).reshape(B,T,D)
        return self.o(y)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.RMSNorm(DIM)
        self.n2 = nn.RMSNorm(DIM)
        self.attn = GQA()

        self.ff = nn.Sequential(
            nn.Linear(DIM, int(3.2*DIM)),
            nn.SiLU(),
            nn.Linear(int(3.2*DIM), DIM)
        )

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.smear = Smear(DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln = nn.RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, x, y=None):
        x = self.emb(x) + self.smear(x)

        for b in self.blocks:
            x = b(x)

        x = self.ln(x)
        logits = self.head(x)

        if y is None:
            return logits

        return F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))

# =============================
# EMA
# =============================
class EMA:
    def __init__(self, model, decay=0.995):
        self.shadow = {k: v.clone() for k,v in model.state_dict().items()}
        self.decay = decay

    def update(self, model):
        for k,v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1-self.decay)

    def apply(self, model):
        model.load_state_dict(self.shadow)

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

    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, betas=(0.9,0.95))
    ema = EMA(model.module)

    loader = DistributedLoader(TRAIN_GLOB, rank, world, device)
    start = time.time()
    step = 0
    while time.time() - start < 540:
        step += 1
        x, y = loader.next_batch(BATCH_TOKENS, SEQ)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x,y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model.module)

        if rank == 0 and step % 100 == 0:
            print(step, loss.item())

    ema.apply(model.module)
    if rank == 0:
        val_loader = DistributedLoader(TRAIN_GLOB, rank, world, device)
        val_bpb = evaluate(model.module, val_loader)
        print("val_bpb:", val_bpb)
    return model.module

# =============================
# AR CALIBRATION
# =============================
@torch.no_grad()
def generate(model, length=512, temp=0.8):
    device = next(model.parameters()).device
    x = torch.zeros((1,1), dtype=torch.long, device=device)

    for _ in range(length):
        logits = model(x)[:,-1,:] / temp
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs,1)
        x = torch.cat([x,nxt],dim=1)

    return x[:,1:]


def build_calib(model):
    data = []
    for _ in range(32):  # reduced for speed
        data.append(generate(model))
    return torch.cat(data, dim=0)

# =============================
# GPTQ
# =============================
def gptq(W, X, block=64):
    W = W.clone()
    H = X.T @ X

    for i in range(0, W.shape[1], block):
        j = min(i+block, W.shape[1])

        Hb = H[i:j,i:j] + 1e-4*torch.eye(j-i, device=W.device)
        Wb = W[:,i:j]

        L = torch.linalg.cholesky(Hb)

        scale = Wb.abs().max(dim=1, keepdim=True)[0] + 1e-8
        Q = torch.round(Wb/scale*15)

        E = Wb - Q*scale
        C = torch.cholesky_solve(E.T, L).T

        W[:,i:j] = Q*scale + C

    return W

def apply_gptq(model):
    calib = build_calib(model)

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if not ("blocks.8" in name or "blocks.9" in name or "head" in name):
                continue

            X = calib.reshape(-1, calib.size(-1))
            W = m.weight.data
            try:
                m.weight.data = gptq(W, X)
                print("GPTQ:", name)
            except:
                pass

# =============================
# COMPRESS
# =============================
def compress(model):
    out = {}
    for n,p in model.state_dict().items():
        W = p.float()

        if W.ndim == 2:
            scale = W.abs().max(dim=1, keepdim=True)[0] + 1e-8
            q = torch.round(W/scale*15).to(torch.int8)

            q = (q+16).cpu().numpy().astype(np.uint8)
            q = q.flatten()
            if len(q)%2: q = np.append(q,0)

            packed = (q[0::2] | (q[1::2]<<5)).astype(np.uint16)
            compressed = zlib.compress(packed.tobytes(), level=9)
            out[n] = compressed
        else:
            out[n] = W.half().cpu().numpy().tobytes()

    return out

# =============================
# RUN
# =============================
if __name__ == "__main__":
    m = train()
    apply_gptq(m)
    art = compress(m)

    if dist.get_rank() == 0:
        print("size:", sum(len(v) for v in art.values()))
