"""验证 11_mini_gpt.py 训出来的模型：它到底学到了什么，学得对不对。

四层检查：
    ① 数值基线 —— 和「瞎猜」「只看字符频率」「只看前一个字符」比
    ② 因果性   —— 改动未来，过去必须纹丝不动（证明掩码没写错）
    ③ 生成质量 —— 不同温度下长什么样
    ④ 一致性   —— 同一个 seed 两次生成必须逐位相同

用法：python .\\tools\\verify_minigpt.py
"""
import importlib.util
import math
import pathlib
import sys

import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("mini11", ROOT / "11_mini_gpt.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ck = torch.load(ROOT / "checkpoints" / "gpt_shakespeare.pth", map_location="cpu", weights_only=False)
args, stoi, itos = ck["args"], ck["stoi"], ck["itos"]
model = m.MiniGPT(ck["vocab"], d=args["d"], h=args["h"],
                  n_layer=args["layers"], block_size=args["block"])
model.load_state_dict(ck["model"])
model.eval()

text, data, _, _ = m.load_corpus(args["data"])
n = int(0.9 * len(data))
train_d, val_d = data[:n], data[n:]
V, T = ck["vocab"], args["block"]
BLOCK, BATCH = args["block"], 64


def mean_loss(dataset, rounds=40):
    tot = 0.0
    with torch.no_grad():
        for _ in range(rounds):
            x, y = m.get_batch(dataset, BLOCK, BATCH, "cpu")
            _, l = model(x, y)
            tot += l.item()
    return tot / rounds


print("=" * 64)
print("① 四层基线：模型到底比「瞎猜」好多少")
print("=" * 64)
val_loss = mean_loss(val_d)
train_loss = mean_loss(train_d)
print(f"  均匀分布（65 个字等概率）  ln(65) = {math.log(V):.4f}")

# unigram：只统计每个字出现频率，不看上下文
cnt = torch.bincount(train_d, minlength=V).float()
p_uni = (cnt + 1) / (cnt.sum() + V)          # 加 1 平滑
uni = -(p_uni * p_uni.log()).sum().item()
print(f"  unigram（只看字符频率）           = {uni:.4f}")

# bigram：只看前一个字符
big = torch.ones(V, V)
for a, b in zip(train_d[:-1].tolist(), train_d[1:].tolist()):
    big[a, b] += 1
p_bi = big / big.sum(dim=1, keepdim=True)
bi = 0.0
for a, b in zip(val_d[:-1].tolist(), val_d[1:].tolist()):
    bi -= math.log(p_bi[a, b].item())
bi /= (len(val_d) - 1)
print(f"  bigram（只看前 1 个字符）         = {bi:.4f}")
print(f"  本模型 val                        = {val_loss:.4f}   ← 比 bigram 低 {bi - val_loss:.4f}")
print(f"  本模型 train                      = {train_loss:.4f}   gap={val_loss - train_loss:.4f}")
print(f"  困惑度 perplexity = exp(val)      = {math.exp(val_loss):.2f}")

print()
print("=" * 64)
print("② 因果性：改动未来，过去必须不变")
print("=" * 64)
prompt = "ROMEO:"
x = torch.tensor([[stoi[c] for c in prompt]])
x2 = x.clone()
x2[0, -1] = stoi["?"]                        # 把最后一个字符换掉
with torch.no_grad():
    l1, _ = model(x)
    l2, _ = model(x2)
diff = (l1[0] - l2[0]).abs().amax(dim=1)
print(f"  原文 {prompt!r}  vs  末字改成 '?'")
for i, v in enumerate(diff.tolist()):
    tag = "  ← 只有这里该变" if i == len(prompt) - 1 else ""
    print(f"    位置 {i}: logits 最大变化 {v:.6f}{tag}")
ok = diff[:-1].max().item() < 1e-5
print(f"  结论：前面 {len(prompt) - 1} 个位置完全没变 -> 因果掩码{'正确' if ok else '有问题！'}")

print()
print("=" * 64)
print("③ 不同温度下生成 200 字符")
print("=" * 64)
start = torch.tensor([[stoi[c] for c in "ROMEO:"]])
for temp in (0.5, 0.8, 1.2):
    torch.manual_seed(1234)
    out = model.generate(start, 200, temperature=temp, top_k=40)
    s = "".join(itos[i] for i in out[0].tolist())[len("ROMEO:"):]
    print(f"\n--- temperature={temp} ---")
    print(s.replace("\n", "\n    "))

print()
print("=" * 64)
print("④ 可复现性：同 seed 两次生成")
print("=" * 64)
torch.manual_seed(7)
a = model.generate(start, 120, temperature=0.8, top_k=40)
torch.manual_seed(7)
b = model.generate(start, 120, temperature=0.8, top_k=40)
print(f"  两次结果完全相同: {torch.equal(a, b)}")
