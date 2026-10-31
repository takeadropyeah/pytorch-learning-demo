import math

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.d = d
        self.wq = nn.Linear(d,d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)

    def forward(self,x):
        q=self.wq(x)
        k=self.wk(x)
        v=self.wv(x)

        scores=q @ k.transpose(-2,-1) / math.sqrt(self.d)
        w=torch.softmax(scores,dim=-1)
        out=w @ v
        return out

if __name__=="__main__":
    torch.manual_seed(0)
    B,L,d=2,5,8
    x=torch.randn(B,L,d)

    sa=SelfAttention(d)
    out=sa(x)

    print("in ",tuple(x.shape),"-> out",tuple(out.shape))

    q,k=sa.wq(x),sa.wk(x)
    w=torch.softmax(q @ k.transpose(-2,-1)/math.sqrt(d),dim=-1)
    print("w shape", tuple(w.shape))  # (2,5,5)
    print("行和    ", w.sum(-1))  # 全 1.0
    print("列和    ", w.sum(-2))  # 不是 1 ← 证明 dim=-1 对了

    mha = nn.MultiheadAttention(d, num_heads=1, batch_first=True)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([sa.wq.weight, sa.wk.weight, sa.wv.weight]))
        mha.in_proj_bias.copy_(torch.cat([sa.wq.bias, sa.wk.bias, sa.wv.bias]))
        mha.out_proj.weight.copy_(torch.eye(d))  # ← 补这两行
        mha.out_proj.bias.zero_()
        ref, _ = mha(x, x, x, need_weights=False)
    print("与官方一致:", torch.allclose(out, ref, atol=1e-6))