"""温度到底在改什么：拿同一个位置的 logits，除以不同温度看分布变化。"""
import importlib.util
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
model = m.MiniGPT(ck["vocab"], d=args["d"], h=args["h"], n_layer=args["layers"], block_size=args["block"])
model.load_state_dict(ck["model"])
model.eval()


def show(c):
    return "\\n" if c == "\n" else ("' '" if c == " " else c)


prompt = "Why should "
idx = torch.tensor([[stoi[c] for c in prompt]])
with torch.no_grad():
    logits, _ = model(idx)
last = logits[0, -1]                      # 最后一个位置的 65 个分数

print(f"prompt = {prompt!r}")
print(f"logits（未归一化）形状 = {tuple(last.shape)}")
top = torch.topk(last, 5)
print(f"\n原始 logits 前 5 名：")
for s, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"   {show(itos[i]):>4s}  logit = {s:8.3f}")

print(f"\n{'温度':>6} | " + " | ".join(f"{show(itos[i]):>5s} 概率" for i in top.indices.tolist()) + " | 前3名合计")
print("-" * 78)
for temp in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0):
    p = F.softmax(last / temp, dim=-1)
    row = " | ".join(f"{p[i].item():9.4f}" for i in top.indices.tolist())
    print(f"{temp:>6.1f} | {row} | {p.sort(descending=True).values[:3].sum().item():.4f}")

print("\n注：温度越小，原本的领先者被放得越大；温度越大，分布越平。")
print("    温度 -> 0 就退化成 argmax（永远取最高分，容易重复）")
print("    温度 = 1 是模型本来的分布")
