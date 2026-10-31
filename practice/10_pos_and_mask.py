"""10 —— 位置编码与因果掩码：给 attention 补上「顺序」和「不能偷看未来」。

两个实验，对应两件事：
    ① attention 本身对 token 顺序完全无感（置换等变），必须外挂位置信息；
    ② 语言模型不能看见未来，必须在 softmax 之前把上三角屏蔽成 -inf。
"""

import importlib.util
import math
import pathlib

import torch
import torch.nn as nn


def _load_py_module(mod_name, filename):
    path = pathlib.Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mha = _load_py_module("mha08", "08_mha.py")
MultiHeadSelfAttention = _mha.MultiHeadSelfAttention


# ---------------------------------------------------------------- 位置编码
def sinusoidal_pe(L, d):
    """正弦位置编码，(L, d)。偶数维 sin、奇数维 cos。"""
    assert d % 2 == 0, "d 必须是偶数，才能一半 sin 一半 cos"
    pos = torch.arange(L, dtype=torch.float32).unsqueeze(1)      # (L, 1)
    i = torch.arange(0, d, 2, dtype=torch.float32)               # (d/2,)
    div = torch.exp(-math.log(10000.0) * i / d)                  # (d/2,) 频率衰减
    pe = torch.empty(L, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------- 因果掩码
def causal_mask(L):
    """(L, L) bool。True = 屏蔽。位置 i 只能看见 j <= i。"""
    return torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)


# ---------------------------------------------------------------- ① 置换等变
def part1_permutation():
    print("=== ① attention 没有位置概念 ===")
    torch.manual_seed(0)
    B, L, d, h = 2, 6, 16, 4
    x = torch.randn(B, L, d)
    mha = MultiHeadSelfAttention(d, h)
    mha.eval()

    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    print(f"  原始顺序: 0 1 2 3 4 5")
    print(f"  打乱顺序: {perm.tolist()}")

    with torch.no_grad():
        out = mha(x)
        out_perm = mha(x[:, perm, :])
        ok = torch.allclose(out_perm, out[:, perm, :], atol=1e-6)
        print(f"  打乱输入后，输出只是跟着重排?  {ok}"
              f"   最大差={float((out_perm - out[:, perm, :]).abs().max()):.2e}")

        pe = sinusoidal_pe(L, d)
        out_pe = mha(x + pe)
        out_pe_perm = mha(x[:, perm, :] + pe)
        ok2 = torch.allclose(out_pe_perm, out_pe[:, perm, :], atol=1e-6)
        print(f"  加上位置编码后，还只是重排吗?  {ok2}"
              f"   最大差={float((out_pe_perm - out_pe[:, perm, :]).abs().max()):.4f}")


# ---------------------------------------------------------------- ② PE 的性质
def part2_pe_structure():
    print("\n=== ② 正弦位置编码的数值性质 ===")
    L, d = 16, 32
    pe = sinusoidal_pe(L, d)
    pe_n = pe / pe.norm(dim=1, keepdim=True)
    sim = pe_n @ pe_n.T

    print("  位置 0 与其它位置的余弦相似度：")
    print("    距离 1~5  :", [round(float(sim[0, k]), 4) for k in range(1, 6)])
    print("    距离 10~15:", [round(float(sim[0, k]), 4) for k in range(10, 16)])

    print("\n  外推：训练时 L=16，推理时要看 64 个位置")
    pe64 = sinusoidal_pe(64, d)
    print(f"    正弦编码直接算得出来        形状 {tuple(pe64.shape)}"
          f"   第 40 位前 6 维 {[round(v, 3) for v in pe64[40, :6].tolist()]}")

    emb = nn.Embedding(16, d)
    try:
        emb(torch.arange(20))
        print("    可学习编码竟然没报错？")
    except Exception as e:
        print(f"    可学习编码 nn.Embedding(16, d) 直接崩: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- ③ 因果掩码
def part3_causal():
    print("\n=== ③ 因果掩码 ===")
    torch.manual_seed(0)
    B, L, d, h = 2, 6, 16, 4
    x = torch.randn(B, L, d)
    mha = MultiHeadSelfAttention(d, h)
    mha.eval()

    mask = causal_mask(L)
    print("  掩码矩阵（1 = 屏蔽）：")
    for row in mask.int().tolist():
        print("    ", row)

    with torch.no_grad():
        q = mha._split_heads(mha.wq(x))
        k = mha._split_heads(mha.wk(x))
        s = q @ k.transpose(-2, -1) / math.sqrt(mha.dh)
        w_full = torch.softmax(s, dim=-1)
        w_causal = torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1)

    r3 = lambda t: [round(float(v), 3) for v in t]
    print("\n  batch0 head0：")
    print("    全可见 第0行:", r3(w_full[0, 0, 0]))
    print("    因果   第0行:", r3(w_causal[0, 0, 0]), "  ← 只剩下位置 0")
    print("    因果   第3行:", r3(w_causal[0, 0, 3]), "  ← 只剩位置 0~3")

    # 真正的因果性检验：只改最后一个 token
    x2 = x.clone()
    x2[:, -1, :] += 100.0
    with torch.no_grad():
        o1 = mha(x, attn_mask=mask)
        o2 = mha(x2, attn_mask=mask)
    diff = (o2 - o1).abs().amax(dim=(0, 2))
    print("\n  只把最后一个 token 的值 +100，各位置输出变化量：")
    for i, v in enumerate(diff.tolist()):
        flag = "  ← 只有这里变了" if v > 1e-6 else ""
        print(f"    位置 {i}: {v:.6f}{flag}")


def main():
    part1_permutation()
    part2_pe_structure()
    part3_causal()
    print("\n提示：softmax 里若一整行全是 -inf（比如整句都被 padding 遮住），"
          "结果会是 NaN。真实实现要保证每行至少有一个可见位置。")


if __name__ == "__main__":
    main()
