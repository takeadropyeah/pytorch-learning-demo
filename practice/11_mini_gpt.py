"""11 —— 从零训练一个 char-level mini GPT（莎士比亚）。

零件全部来自前面几个文件：
    Embedding + 位置编码 → N × TransformerBlock(因果掩码) → ln_f → unembedding → 交叉熵
权重绑定（weight tying）：输出层直接复用词嵌入矩阵，省掉一半嵌入参数（GPT-2 的做法）。

用法：
    python .\\11_mini_gpt.py --steps 3000 --tag gpt_small
"""

import argparse
import importlib.util
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Windows 控制台默认 GBK，打印 ² 之类的字符会 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- 复用 08/09/10
def _load_py_module(mod_name, filename):
    path = pathlib.Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TransformerBlock = _load_py_module("block09", "09_transformer_block.py").TransformerBlock
causal_mask = _load_py_module("posmask10", "10_pos_and_mask.py").causal_mask


# ---------------------------------------------------------------- 模型
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d=128, h=4, n_layer=4, block_size=64, mult=4, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(block_size, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, h, mult) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight        # ← 权重绑定

        # 因果掩码只跟 block_size 有关，算一次存起来
        self.register_buffer("mask", causal_mask(block_size), persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度 {T} 超过 block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = self.mask[:T, :T]
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ---------------------------------------------------------------- 数据
def load_corpus(path):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return text, data, stoi, itos


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step, max_steps, base_lr, warmup=100, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * ratio)))


# ---------------------------------------------------------------- 主流程
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=r"D:\pytorch-learning\data\shakespeare.txt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--h", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--tag", default="gpt_small")
    args = p.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    text, data, stoi, itos = load_corpus(args.data)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    vocab = len(stoi)
    print(f"语料 {len(text):,} 字符 | 词表 {vocab} | 训练 {len(train_data):,} / 验证 {len(val_data):,}")

    model = MiniGPT(vocab, d=args.d, h=args.h, n_layer=args.layers,
                    block_size=args.block).to(device)
    total = sum(p.numel() for p in model.parameters())
    emb = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
    print(f"参数 {total:,}（其中嵌入 {emb:,}，因权重绑定未重复计算）")
    print(f"每层理论值 12·d² = {12 * args.d ** 2:,} × {args.layers} 层 = "
          f"{12 * args.d ** 2 * args.layers:,}")

    decode = lambda ids: "".join(itos[i] for i in ids)
    start_ids = torch.tensor([[stoi[c] for c in "ROMEO:"]], device=device)

    print("\n--- 训练前（随机初始化）---")
    print(repr(decode(model.generate(start_ids, 120, temperature=1.0)[0].tolist())))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        lr = get_lr(step, args.steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.block, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 100 == 0 or step == 0:
            print(f"step {step + 1:5d}/{args.steps}  lr={lr:.2e}  "
                  f"loss={loss.item():.4f}  {time.time() - t0:6.1f}s")

        if (step + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                losses = {}
                for name, d_ in (("train", train_data), ("val", val_data)):
                    acc = []
                    for _ in range(20):
                        xb, yb = get_batch(d_, args.block, args.batch, device)
                        _, l = model(xb, yb)
                        acc.append(l.item())
                    losses[name] = sum(acc) / len(acc)
            print(f"  [eval] train={losses['train']:.4f}  val={losses['val']:.4f}  "
                  f"val_ppl={math.exp(losses['val']):.1f}")
            model.train()

    print(f"\n训练完成，用时 {time.time() - t0:.1f}s")
    print("\n--- 训练后（temperature=0.8, top_k=40）---")
    print(model.generate(start_ids, 400, temperature=0.8, top_k=40)[0].tolist().__str__()
          if False else decode(model.generate(start_ids, 400, temperature=0.8, top_k=40)[0].tolist()))

    ckpt_dir = pathlib.Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / f"{args.tag}.pth"
    torch.save({"model": model.state_dict(), "args": vars(args), "stoi": stoi,
                "itos": itos, "vocab": vocab}, ckpt)
    print(f"\ncheckpoint -> {ckpt}")


if __name__ == "__main__":
    main()
