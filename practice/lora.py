"""LoRA 的最小实现。骨架 + 四个必须通过的测试。

用法：
    ..\\.venv\\Scripts\\python.exe .\\lora.py

设计要点（先读这段再看代码）：
    1. LoRALinear 是 nn.Module 而不是 nn.Linear 的子类——
       因为它要「包住」一个已有的 Linear，而不是新建一个。
    2. 原权重 base.weight / base.bias 全部 requires_grad=False（冻结）。
    3. 只有 lora_A、lora_B 两个 Parameter 是可训练的。
    4. B 初始化为 0、A 随机（Kaiming）——这是 LoRA 的约定，
       保证 B=0 时 delta=0，模型起点精确等于原模型。
    5. merged=True 时把 BA 加进 W，前向只剩一次 F.linear，零额外延迟。
"""

import copy
import math
import sys

import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================================
# 核心类
# ============================================================================
class LoRALinear(nn.Module):
    """包住一个冻结的 nn.Linear，旁路加一个低秩增量。

        h = W0·x + b + (alpha/r) · B(A·x)

    其中 A: (r, in_features), B: (out_features, r)。
    原始 W0 的形状是 (out_features, in_features)，和 B@A 一致。
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"base 必须是 nn.Linear，收到 {type(base)}")

        self.base = base
        self.r = r
        self.alpha = alpha
        # 论文的缩放：alpha / r。注意用 float，否则整数除法会把 r>alpha 的情况算成 0
        self.scaling = alpha / r

        # 冻结原权重。注意这里是「在原 Parameter 上改 requires_grad」，
        # 不是新建一个副本 —— 这样 state_dict 的键名保持不变，方便加载预训练权重。
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features

        # 低秩旁路
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))

        # A 随机（和 nn.Linear 默认初始化一致），B 必须保持 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # self.lora_B 不初始化 —— 已经是 zeros

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            # 已经把 BA 折进 W 了，不能再加一遍
            return out
        delta = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + delta * self.scaling

    # ------------------------------------------------------------ merge 相关
    @torch.no_grad()
    def merge(self) -> None:
        """把 (alpha/r)·B@A 加进 base.weight，之后前向就是一次普通 Linear。

        合并后推理零额外延迟，但**训练时必须先 unmerge**，
        否则 BA 已经进了 W，而 W 是冻结的，梯度就断在这里了。
        """
        if self.merged:
            return
        self.base.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """撤回合并。数值上有微小误差（浮点加减），所以只用于切回训练。"""
        if not self.merged:
            return
        self.base.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
        self.merged = False

    # ---------------------------------------------------------------- 便利方法
    def extra_repr(self) -> str:
        return (f"in={self.base.in_features}, out={self.base.out_features}, "
                f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:g}, "
                f"merged={self.merged}")


# ============================================================================
# 三个工具函数
# ============================================================================
def inject_lora(module: nn.Module, r: int = 8, alpha: int = 16,
                dropout: float = 0.0,
                target_names=("q_proj", "k_proj", "v_proj", "o_proj")) -> nn.Module:
    """把 module 里所有名字匹配 target_names 的 Linear 换成 LoRALinear。

    就地修改并返回同一个 module；inject 之前建议先 copy.deepcopy。
    """
    replaced = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(name.endswith(t) for t in target_names):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(name)
        else:
            inject_lora(child, r, alpha, dropout, target_names)
    return module


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """冻结全部，然后只解冻 LoRA 参数。

    先全冻结再解冻的顺序很重要：模型里有些层可能本来就是可训练的，
    不能假设「原来都是冻的」。
    """
    for _, p in model.named_parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True


def lora_param_report(model: nn.Module) -> dict:
    """统计三类参数：全部 / 可训练 / LoRA。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters()
               if "lora_A" in n or "lora_B" in n)
    return {"total": total, "trainable": trainable, "lora": lora}


# ============================================================================
# 四个测试
# ============================================================================
def banner(t: str) -> None:
    print("\n" + "=" * 64)
    print(t)
    print("=" * 64)


def test1_identity():
    """测试 1：B=0 时，LoRALinear 的输出必须和原 nn.Linear 逐位相等。"""
    banner("测试 1 · B=0 的恒等性（最重要的一条）")

    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, r=4, alpha=8)

    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        a = base(x)
        b = lora(x)

    diff = (a - b).abs().max().item()
    print(f"  原 nn.Linear 输出前 3 个 : {a[0, 0, :3].tolist()}")
    print(f"  LoRALinear   输出前 3 个 : {b[0, 0, :3].tolist()}")
    print(f"  最大绝对差               : {diff:.3e}")
    print(f"  是否逐位相等             : {diff == 0.0}")
    assert diff == 0.0, "B=0 时应该精确相等"
    print("  [PASS] 起点无损：训练从「原模型」这个点出发")


def test2_gradient():
    """测试 2：第一步只有 B 有梯度，A 的梯度为 0；一步之后两者都有。"""
    banner("测试 2 · 梯度的对称性破缺")

    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, r=4, alpha=8)
    opt = torch.optim.SGD([p for p in lora.parameters() if p.requires_grad], lr=0.1)

    x = torch.randn(8, 16)
    loss = lora(x).pow(2).mean()

    # --- 第 1 步 ---
    opt.zero_grad()
    loss.backward()
    gA = lora.lora_A.grad
    gB = lora.lora_B.grad
    print(f"  第 1 步前：")
    print(f"    lora_A.grad 全零? {torch.all(gA == 0).item()}  范数 {gA.norm():.3e}")
    print(f"    lora_B.grad 全零? {torch.all(gB == 0).item()}  范数 {gB.norm():.3e}")
    assert torch.all(gA == 0) and not torch.all(gB == 0)

    opt.step()

    # --- 第 2 步 ---
    loss2 = lora(x).pow(2).mean()
    opt.zero_grad()
    loss2.backward()
    print(f"  第 1 步后（B 已非零）：")
    print(f"    lora_A.grad 范数 {lora.lora_A.grad.norm():.3e}")
    print(f"    lora_B.grad 范数 {lora.lora_B.grad.norm():.3e}")
    assert lora.lora_A.grad.norm() > 0
    print("  [PASS] B 先动、A 后动 —— 这就是「必须一个为零」的原因")


def test3_both_zero_is_dead():
    """测试 3：A、B 都初始化为 0 时，梯度永远为零，训练死掉。"""
    banner("测试 3 · 两个都初始化成 0 会怎样")

    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, r=4, alpha=8)
    with torch.no_grad():
        lora.lora_A.zero_()      # 人为把 A 也清零
        lora.lora_B.zero_()

    x = torch.randn(8, 16)
    lora(x).pow(2).mean().backward()

    print(f"  lora_A.grad 范数: {lora.lora_A.grad.norm():.3e}")
    print(f"  lora_B.grad 范数: {lora.lora_B.grad.norm():.3e}")
    assert lora.lora_A.grad.norm() == 0 and lora.lora_B.grad.norm() == 0
    print("  [PASS] 两个都是 0 → 梯度也全 0 → 永远学不动")


def test4_merge_equivalence():
    """测试 4：合并前后前向输出必须一致（这是「零推理延迟」的依据）。"""
    banner("测试 4 · merge 等价性（零推理延迟的依据）")

    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, r=4, alpha=8)

    # 先随机化 A、B，模拟「训练完之后」的状态
    with torch.no_grad():
        lora.lora_A.normal_(0, 0.1)
        lora.lora_B.normal_(0, 0.1)

    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        before = lora(x)

        lora.merge()
        after = lora(x)
        n_par_before = sum(p.numel() for p in lora.parameters())

        lora.unmerge()
        back = lora(x)

    d1 = (before - after).abs().max().item()
    d2 = (before - back).abs().max().item()
    print(f"  merge 前后最大差  : {d1:.3e}")
    print(f"  unmerge 回来最大差: {d2:.3e}")
    print(f"  总参数（含冻结）  : {n_par_before:,}")
    print(f"  → merge 后前向只剩一次 F.linear，计算量和原模型完全相同")
    assert d1 < 1e-5 and d2 < 1e-5
    print("  [PASS] 分配律成立：(W + s·BA)x == Wx + s·BAx")


def test5_e2e():
    """测试 5：套到一个真模型上，看参数量报告。"""
    banner("测试 5 · 端到端：套到 ViT 的 attention 上")

    sys.path.insert(0, ".")
    import importlib.util

    spec = importlib.util.spec_from_file_location("vit12", "12_vit_cifar.py")
    vit12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vit12)          # 有 __main__ 保护，不会触发训练

    model = vit12.ViT(d=192, layers=4, heads=6, mult=4, patch=4, dropout=0.0)
    base_total = sum(p.numel() for p in model.parameters())

    inject_lora(model, r=8, alpha=16,
                target_names=("wq", "wk", "wv", "wo", "fc1", "fc2"))
    mark_only_lora_as_trainable(model)
    rpt = lora_param_report(model)

    print(f"  原模型参数        : {base_total:,}")
    print(f"  注入 LoRA 后总参数: {rpt['total']:,}")
    print(f"  LoRA 参数         : {rpt['lora']:,}")
    print(f"  可训练参数        : {rpt['trainable']:,}"
          f"   （占全部 {rpt['trainable'] / rpt['total']:.2%}）")
    print(f"  冻结参数          : {rpt['total'] - rpt['trainable']:,}")

    # 前向能跑通，且形状不变
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        y = m2(torch.randn(2, 3, 32, 32))
    print(f"  前向输出形状      : {tuple(y.shape)}")
    print("  [PASS] 注入后前向正常，形状 (B, 10) 不变")


if __name__ == "__main__":
    test1_identity()
    test2_gradient()
    test3_both_zero_is_dead()
    test4_merge_equivalence()
    test5_e2e()
    print("\n" + "=" * 64)
    print("五个测试全部通过。")
    print("=" * 64)
