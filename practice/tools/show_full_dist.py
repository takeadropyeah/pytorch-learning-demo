"""查看模型对 65 个字符的完整输出分布。"""
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

prompt = "ROMEO:"
idx = torch.tensor([[stoi[c] for c in prompt]])
with torch.no_grad():
    logits, _ = model(idx)


def show(c):
    return "\\n" if c == "\n" else ("' '" if c == " " else c)


print(f"logits 形状 = {tuple(logits.shape)}  → 6 个位置 × 65 个字，一个不少\n")

for t, label in ((0, "位置0 只看到 'R'"), (5, "位置5 看到 'ROMEO:'")):
    p = F.softmax(logits[0, t], -1)
    vals, ids = torch.sort(p, descending=True)
    print(f"=== {label} ===")
    print("  全部 65 个概率降序：")
    for k in range(12):
        print(f"    {k + 1:2d}. {show(itos[ids[k].item()]):>4s}  {vals[k].item():.6f}")
    print(f"    ... 剩余 {65 - 12} 个加起来 = {vals[12:].sum().item():.6f}")
    print(f"  累计占比：前3名 {vals[:3].sum().item():.4f} | 前10名 {vals[:10].sum().item():.4f} | 全部 {vals.sum().item():.4f}")
    print()

p = F.softmax(logits[0, 5], -1)
vals, _ = torch.sort(p, descending=True)
kept, cut = vals[:40].sum().item(), vals[40:].sum().item()
print("=== top_k=40 砍掉了什么（位置5）===")
print(f"  前 40 名占总概率 {kept:.6f}")
print(f"  被砍掉的 25 个占  {cut:.6f}")
print(f"  被砍掉的最大一个  {vals[40].item():.3e}")
