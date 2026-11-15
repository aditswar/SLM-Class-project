"""Step 2: train a ~14M Llama-style model from scratch.
  python train.py                      # train (auto-resumes from ckpt.pt)
  python train.py --sample "Once upon" # generate from the latest checkpoint
Needs: pip install torch numpy tokenizers
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- config -------------------------
VOCAB, DIM, LAYERS, HEADS, FF, CTX = 8192, 384, 6, 6, 1024, 512
MICRO_BS, ACCUM = 16, 16            # 16*512*16 = 131k tokens/step. If OOM: MICRO_BS=8, ACCUM=32
MAX_STEPS, WARMUP = 7600, 200       # ~1B tokens
LR, MIN_LR, WD = 1e-3, 1e-4, 0.1
EVAL_EVERY, SAMPLE_EVERY, EVAL_ITERS = 250, 500, 50
COMPILE = False                     # torch.compile needs Triton (Linux/WSL). Try True there.
CKPT = "ckpt.pt"

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32


# ------------------------- model -------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.w).type_as(x)


def apply_rope(x, cos, sin):  # x: (B, H, T, hd)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(DIM, 3 * DIM, bias=False)
        self.proj = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, HEADS, DIM // HEADS).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, DIM))


class MLP(nn.Module):  # SwiGLU
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(DIM, FF, bias=False)
        self.up = nn.Linear(DIM, FF, bias=False)
        self.down = nn.Linear(FF, DIM, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.attn = RMSNorm(DIM), Attention()
        self.n2, self.mlp = RMSNorm(DIM), MLP()

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)
        self.head.weight = self.emb.weight  # tied embeddings
        hd = DIM // HEADS
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        f = torch.outer(torch.arange(CTX).float(), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():  # scale residual projections
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * LAYERS))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.emb(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, VOCAB), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n=200, temp=0.8, top_k=50):
        for _ in range(n):
            logits, _ = self(idx[:, -CTX:])
            logits = logits[:, -1].float() / temp
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx


# ------------------------- data -------------------------
data = {}


def get_batch(split):
    if split not in data:
        data[split] = np.memmap(f"data/{split}.bin", dtype=np.uint16, mode="r")
    d = data[split]
    ix = np.random.randint(0, len(d) - CTX - 1, MICRO_BS)
    x = torch.stack([torch.from_numpy(d[i : i + CTX].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1 : i + 1 + CTX].astype(np.int64)) for i in ix])
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)


# ------------------------- helpers -------------------------
def get_lr(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "validation"):
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
                losses.append(model(x, y)[1].item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def sample(model, tok, prompt="Once upon a time", n=200):
    model.eval()
    ids = torch.tensor([tok.encode(prompt).ids], device=dev)
    with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
        out = model.generate(ids, n=n)
    model.train()
    return tok.decode(out[0].tolist())


# ------------------------- main -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=str, default=None)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("data/tokenizer.json")
    model = GPT().to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M on {dev}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95), fused=dev == "cuda",
    )

    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print("resumed from step", start)

    if args.sample is not None:
        print(sample(model, tok, args.sample))
        raise SystemExit

    cmodel = torch.compile(model) if COMPILE else model
    model.train()
    t0 = time.time()
    for step in range(start, MAX_STEPS):
        for g in opt.param_groups:
            g["lr"] = get_lr(step)

        for _ in range(ACCUM):
            x, y = get_batch("train")
            with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
                _, loss = cmodel(x, y)
            (loss / ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 20 == 0:
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            t0 = time.time()
            tps = MICRO_BS * CTX * ACCUM * (20 if step > start else 1) / dt
            print(f"step {step} | loss {loss.item():.3f} | lr {get_lr(step):.2e} | {tps:,.0f} tok/s")

        if step > start and step % EVAL_EVERY == 0:
            print("EVAL", estimate_loss(model))
        if step > start and step % SAMPLE_EVERY == 0:
            print("--- sample ---\n" + sample(model, tok) + "\n--------------")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1}, CKPT)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": MAX_STEPS}, CKPT)
    print("done")
