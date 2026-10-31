"""把 11_mini_gpt.py 训好的模型拆开，逐个打印每个张量到底长什么样。

用法：python .\\tools\\inspect_minigpt.py
"""
import importlib.util
import pathlib
import sys

import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mini = load("mini11", "11_mini_gpt.py")
ck = torch.load(ROOT / "checkpoints" / "gpt_shakespeare.pth", map_location="cpu", weights_only=False)

args, stoi, itos = ck["args"], ck["stoi"], ck["itos"]
model = mini.MiniGPT(ck["vocab"], d=args["d"], h=args["h"],
                     n_layer=args["layers"], block_size=args["block"])
model.load_state_dict(ck["model"])
model.eval()

V, d, T = ck["vocab"], args["d"], args["block"]
print("=" * 70)
print(f"超参：vocab_size={V}  d={d}  h={args['h']}  n_layer={args['layers']}  block_size={T}")
print("=" * 70)

print("\n【每个属性是什么】")
for name, mod in model.named_children():
    n = sum(p.numel() for p in mod.parameters())
    first = next(mod.parameters(), None) if hasattr(mod, "parameters") else None
    shape = tuple(mod.weight.shape) if hasattr(mod, "weight") else "(无 trainable 权重)"
    print(f"  {name:10s} 参数 {n:>8,}   weight 形状 {shape}")

prompt = "ROMEO:"
ids = [stoi[c] for c in prompt]
print(f"\n【把 '{prompt}' 变成数字】")
for c, i in zip(prompt, ids):
    print(f"  '{c}' -> id {i:2d}", end="")
print()

print(f"\n【tok_emb：{V}×{d} 的查表，每行是一个字的向量】")
for c in "RO3":
    row = model.tok_emb.weight[stoi[c]]
    print(f"  '{c}' 的向量前 6 维: {[round(float(v), 4) for v in row[:6].tolist()]}")

print(f"\n【pos_emb：{T}×{d} 的查表，每行是一个位置的向量（与内容是啥无关）】")
for p in (0, 1, 2):
    row = model.pos_emb.weight[p]
    print(f"  位置 {p} 的向量前 6 维: {[round(float(v), 4) for v in row[:6].tolist()]}")

idx = torch.tensor([ids])
print(f"\n【forward 里的形状演变】")
print(f"  idx                     {tuple(idx.shape)}   {ids}")
tok = model.tok_emb(idx)
print(f"  tok_emb(idx)            {tuple(tok.shape)}   ← 每个 id 换成 {d} 维向量")
pos = torch.arange(len(ids))
pe = model.pos_emb(pos)
print(f"  pos_emb(arange({len(ids)}))      {tuple(pe.shape)}   ← 位置 0..5 各一个向量")
x = model.drop(tok + pe)
print(f"  tok + pe                {tuple(x.shape)}   ← 广播相加，逐元素")

mask = model.mask[: len(ids), : len(ids)]
print(f"  mask[:6,:6]             {tuple(mask.shape)}  bool，True=屏蔽")
for r in mask.int().tolist():
    print("       ", r)

for i, blk in enumerate(model.blocks):
    x = blk(x, attn_mask=mask)
    print(f"  第 {i} 个 block 之后         {tuple(x.shape)}   ← 形状永远不变")

xn = model.ln_f(x)
print(f"  ln_f(x)                 {tuple(xn.shape)}   ← 最后再归一化一次")
logits = model.head(xn)
print(f"  head(xn)                {tuple(logits.shape)}   ← 最后一维变成词表大小 {V}")

last = logits[0, -1]
top = torch.topk(last, 5)
print(f"\n【第 6 个位置（'ROMEO:' 之后）的预测】")
for score, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  '{itos[i]}'  logit={score:7.3f}  概率={F.softmax(last, -1)[i]:.3f}")

print(f"\n【权重绑定验证】")
print(f"  head.weight 与 tok_emb.weight 是同一个对象? "
      f"{model.head.weight is model.tok_emb.weight}")
print(f"  head.weight.data_ptr() == tok_emb.weight.data_ptr()? "
      f"{model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()}")
print(f"  参数总数（自动去重后）: {sum(p.numel() for p in model.parameters()):,}")
print(f"  若不绑定，则会多出 {V * d:,} 个")
