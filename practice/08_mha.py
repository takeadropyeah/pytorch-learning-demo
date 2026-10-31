import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力：h 张独立的 (L,L) 注意力图，最后用 W_O 混合。"""

    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0, f"d={d} 必须能被 h={h} 整除"
        self.d = d
        self.h = h
        self.dh = d // h
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)          # W_O

    def _split_heads(self, t):
        """(B, L, d) -> (B, h, L, dh)"""
        B, L, _ = t.shape
        return t.view(B, L, self.h, self.dh).transpose(1, 2)

    def forward(self, x, attn_mask=None):
        """attn_mask: 可选，(L,L) 或 (B,1,L,L) 的 bool 张量；True 的位置被屏蔽成 -inf。"""
        B, L, _ = x.shape
        q = self._split_heads(self.wq(x))            # (B,h,L,dh)
        k = self._split_heads(self.wk(x))
        v = self._split_heads(self.wv(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)   # (B,h,L,L)
        if attn_mask is not None:
            # 必须在 softmax 之前：-inf 取指数是 0，被屏蔽位置的权重才会归零
            scores = scores.masked_fill(attn_mask, float("-inf"))
        w = torch.softmax(scores, dim=-1)                       # 每头每行和=1
        out = w @ v                                             # (B,h,L,dh)

        out = out.transpose(1, 2).reshape(B, L, self.d)         # (B,h,L,dh) -> (B,L,d)
        return self.wo(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, d, h = 2, 5, 16, 4
    x = torch.randn(B, L, d)

    mha = MultiHeadSelfAttention(d, h)
    out = mha(x)
    print(f"in {tuple(x.shape)} -> out {tuple(out.shape)}")

    # ① 与官方逐位对齐
    ref_mod = nn.MultiheadAttention(d, h, batch_first=True)
    with torch.no_grad():
        ref_mod.in_proj_weight.copy_(torch.cat([mha.wq.weight, mha.wk.weight, mha.wv.weight]))
        ref_mod.in_proj_bias.copy_(torch.cat([mha.wq.bias, mha.wk.bias, mha.wv.bias]))
        ref_mod.out_proj.weight.copy_(mha.wo.weight)
        ref_mod.out_proj.bias.copy_(mha.wo.bias)
        ref, _ = ref_mod(x, x, x, need_weights=False)
    print("与官方一致:", torch.allclose(out, ref, atol=1e-6),
          "| 最大差:", float((out - ref).abs().max().detach()))

    # ② 四个头确实在看不同的东西
    with torch.no_grad():
        q = mha._split_heads(mha.wq(x))
        k = mha._split_heads(mha.wk(x))
        w = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(mha.dh), dim=-1)
    print("\nbatch0 每个头的注意力集中度：")
    for hh in range(h):
        print(f"  head{hh}  行和={[round(v, 4) for v in w[0, hh].sum(-1).tolist()]}"
              f"  行最大={[round(v, 3) for v in w[0, hh].max(-1).values.tolist()]}")

    # ③ h 不影响参数量
    print()
    for hh in (1, 2, 4, 8, 16):
        m = MultiHeadSelfAttention(16, hh)
        print(f"  h={hh:2d}  dh={16 // hh:2d}  参数={sum(p.numel() for p in m.parameters())}")