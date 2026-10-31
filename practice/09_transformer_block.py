"""09 —— 拼出一个完整的 Transformer block，并与官方 nn.TransformerEncoderLayer 逐位对齐。

结构（Pre-LN）：
    x → LN1 → MHA → (+) → LN2 → FFN → (+) → out
    其中 (+) 加的是进入该子层之前的那一份 x。

验收：norm_first=True、dropout=0、activation="gelu" 的官方层，权重抄过去后输出必须逐位一致。
"""

import importlib.util
import pathlib

import torch
import torch.nn as nn


# ---------------------------------------------------------------- 复用 08 的实现
def _load_py_module(mod_name, filename):
    """按文件名加载同目录下的 .py（文件名以数字开头，不能直接 import）。"""
    path = pathlib.Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mha = _load_py_module("mha08", "08_mha.py")
MultiHeadSelfAttention = _mha.MultiHeadSelfAttention


# ---------------------------------------------------------------- FFN
class FeedForward(nn.Module):
    """逐位置的两层 MLP：d → mult*d → d。token 之间不交流。"""

    def __init__(self, d, mult=4):
        super().__init__()
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------- Block
class TransformerBlock(nn.Module):
    """Pre-LN Transformer block。

    关键：x 在每个子层被用了两次，但不是同一个版本——
        送进子层的是 LN(x)，参与相加的是原来的 x。
    """

    def __init__(self, d, h, mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = FeedForward(d, mult)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)   # 通信：token 之间交换信息
        x = x + self.ffn(self.ln2(x))                        # 加工：每个 token 自己非线性变换
        return x


# ---------------------------------------------------------------- 验收
def main():
    torch.manual_seed(0)
    B, L, d, h, mult = 2, 6, 32, 4, 4

    x = torch.randn(B, L, d)
    blk = TransformerBlock(d, h, mult)
    out = blk(x)
    print(f"in {tuple(x.shape)} -> out {tuple(out.shape)}")

    # ---------- ① 与官方 Pre-LN 层逐位对齐 ----------
    ref = nn.TransformerEncoderLayer(
        d_model=d,
        nhead=h,
        dim_feedforward=mult * d,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,          # ← 这个布尔值决定 Pre-LN / Post-LN
    )
    with torch.no_grad():
        # 注意力：官方把 q/k/v 三块拼成一个 (3d, d)
        ref.self_attn.in_proj_weight.copy_(
            torch.cat([blk.attn.wq.weight, blk.attn.wk.weight, blk.attn.wv.weight])
        )
        ref.self_attn.in_proj_bias.copy_(
            torch.cat([blk.attn.wq.bias, blk.attn.wk.bias, blk.attn.wv.bias])
        )
        ref.self_attn.out_proj.weight.copy_(blk.attn.wo.weight)
        ref.self_attn.out_proj.bias.copy_(blk.attn.wo.bias)
        # 两个 LayerNorm
        ref.norm1.weight.copy_(blk.ln1.weight)
        ref.norm1.bias.copy_(blk.ln1.bias)
        ref.norm2.weight.copy_(blk.ln2.weight)
        ref.norm2.bias.copy_(blk.ln2.bias)
        # FFN
        ref.linear1.weight.copy_(blk.ffn.fc1.weight)
        ref.linear1.bias.copy_(blk.ffn.fc1.bias)
        ref.linear2.weight.copy_(blk.ffn.fc2.weight)
        ref.linear2.bias.copy_(blk.ffn.fc2.bias)
        ref.eval()
        ref_out = ref(x)

    print("Pre-LN 与官方一致:", torch.allclose(out, ref_out, atol=1e-6),
          "| 最大差:", float((out - ref_out).abs().max().detach()))

    # ---------- ② norm_first=False 会得到完全不同的一组数 ----------
    ref_post = nn.TransformerEncoderLayer(
        d_model=d, nhead=h, dim_feedforward=mult * d,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=False,
    )
    with torch.no_grad():
        ref_post.load_state_dict(ref.state_dict())   # 权重一模一样
        ref_post.eval()
        post_out = ref_post(x)
    print("Post-LN 与我们的输出差:",
          float((out - post_out).abs().max().detach()),
          "  ← 权重完全相同，只是 LN 位置不同")

    # ---------- ③ 参数构成 ----------
    total = sum(p.numel() for p in blk.parameters())
    parts = {
        "LN1": sum(p.numel() for p in blk.ln1.parameters()),
        "MHA": sum(p.numel() for p in blk.attn.parameters()),
        "LN2": sum(p.numel() for p in blk.ln2.parameters()),
        "FFN": sum(p.numel() for p in blk.ffn.parameters()),
    }
    print(f"\nblock 参数 {total}（d={d}, h={h}, mult={mult}）")
    for k, v in parts.items():
        print(f"  {k:4s} {v:6d}  {v / total * 100:5.1f}%")

    # ---------- ④ 残差确实是恒等通路 ----------
    with torch.no_grad():
        ln1_out = blk.ln1(x)
        attn_out = blk.attn(ln1_out)
        after_attn = x + attn_out
        print(f"\nLN1 后 std={ln1_out.std().item():.4f}"
              f"  attn 输出 std={attn_out.std().item():.4f}"
              f"  相加后 std={after_attn.std().item():.4f}")
        print(f"  输入的均值/方差: {x.mean().item():.6f} / {x.std().item():.4f}"
              f"  → LN 后: {ln1_out.mean().item():.6f} / {ln1_out.std().item():.4f}")


if __name__ == "__main__":
    main()
