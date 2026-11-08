"""14 —— 给预训练 ViT-B/16 注入 LoRA，和 13 的全量微调比同一本账

    python 14_lora_vit.py --epochs 5 --lora-r 8 --tag lora_r8
    python 14_lora_vit.py --epochs 5 --lora-r 4 --tag lora_r4
    python 14_lora_vit.py --epochs 5 --lora-r 16 --tag lora_r16

要回答的问题只有一个：**能不能用 1% 的参数，拿到接近全量微调的精度？**
对照组是 13_vit_finetune.py 跑出来的三组（同一份 Flowers102、同一套增强、同样 5 轮）：

    线性探针（只训头 0.09%）   84.11%   0.59 GB
    全量微调（100%）           91.80%   5.10 GB
    从零训练（100%）           21.13%   5.10 GB

=============================================================================
为什么这次只加在 FFN 上，不碰 attention —— 这是踩出来的
=============================================================================
LoRA 论文的主实验加在 Wq、Wv 上（§4.2：只加这两个性价比最高）。
但在 torchvision 的 ViT 上，这条路走不通，有两个独立的障碍：

**障碍一：Q/K/V 被融合成一个 Parameter。**
    self_attention.in_proj_weight   shape (2304, 768)   ← Parameter，不是 Module
    self_attention.out_proj         shape (768, 768)    ← 这个才是 nn.Linear
LoRALinear 只能包 nn.Linear（它要读 in_features / out_features），
所以 in_proj_weight 抓不到。这就是为什么 HuggingFace 里叫 q_proj/k_proj/v_proj
（拆开的三个 Linear），而 torchvision 里根本没有这三个名字。

**障碍二：就算包住 out_proj，也会崩。**
nn.MultiheadAttention.forward 的快路径直接读 self.out_proj.weight，
而 LoRALinear 只有 .base.weight，于是报
`AttributeError: 'LoRALinear' object has no attribute 'weight'`。
更糟的是：如果给它加个 weight 属性让 MHA 继续跑，那条快路径会**绕过**
LoRA 的 forward，低秩增量被静默忽略——能跑、能收敛、但什么都没学到。

所以这次注入的目标收缩为「所有能安全包住的 Linear」：

    mlp.0      12 个   (768 → 3072)   第一层，扩维
    mlp.3      12 个   (3072 → 768)   第二层，降维

共 24 个。r=8 时 737,280 个 LoRA 参数 + 78,438 分类头 = 815,718，
占全模型 85,877,094 的 **0.95%**。

=============================================================================
后记：这两个障碍是可以拆掉的，加 --swap-attn
=============================================================================
上面两个障碍都不是「attention 不能加 LoRA」，而是「**torchvision 这个 MHA 的实现**
不让加」。拆法有两个，第二个才治本：

**治标**：只包 out_proj，并忍下 `AttributeError`。
没用——快路径绕过 forward 的那个坑还在。

**治本**：把 nn.MultiheadAttention 整个换成 08_mha.py 的手写实现
（`--swap-attn`）。手写版有 wq/wk/wv/wo **四个真 nn.Linear**，且 EncoderBlock
通过 `module(x)` 正常调用它们，两个判据都满足。

换的时候必须把权重搬过去并**验证前向一致**（实测 2.399e-06），否则就不是
「同一个起点 + LoRA」，而是「换了一个模型」。`swap_vit_attention()` 里那道
`assert diff < 1e-5` 就是干这个的。

顺带修好了 fast path 问题：手写实现是纯 PyTorch 算出来的，没有
`torch._native_multi_head_attention` 那条捷径，LoRA 的 delta 不可能被吞。

于是同一份代码可以跑四种对照：

    --targets mlp.0,mlp.3                      LoRA 落在 FFN     默认
    --swap-attn                                LoRA 落在 wq,wv   （论文 §4.2 的配置）
    --swap-attn --targets ...wq,...wv,mlp.0,mlp.3   两处都加
    --swap-attn --targets ...wq,...wk,...wv,...wo   四个投影全加

这也顺带说明一件更普遍的事：**「某方法在这个库上做不了」通常不等于「方法不行」，
而是「这个库的内部实现让包裹失效」**。同一个道理在量化、权重绑定、
torch.compile、flash-attention 融合 kernel 上都成立——凡是上层直接读参数而不
调 forward 的地方，包裹都会静默失效。

（顺带一提：FFN 的 LoRA 参数比 attention 大得多，因为中间维度 3072 是 768 的 4 倍。
在 HuggingFace 里如果只加 q_proj/v_proj，参数会明显少于这里。）
"""

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("TORCH_HOME", str(_REPO / ".cache" / "torch"))


def _load(name: str, filename: str):
    """按路径加载兄弟脚本。

    文件名以数字开头（13_vit_finetune.py）不能 import，所以用 importlib 按路径加载。
    这样 13 里的数据管道和训练循环只有一份，改了 LoRA 脚本不用跟着改。
    两个脚本都有 `if __name__ == "__main__"` 保护，所以加载时不会跑训练。

    **必须把模块登记进 sys.modules。** 否则 FlowersDataset 这个类的 __module__ 是
    "base13"，而 base13 不在 sys.modules 里，DataLoader 的多进程 worker 反序列化
    数据集时会报 `PicklingError: Can't pickle <class 'base13.FlowersDataset'>`。
    """
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # ← 少了这一行，多进程 DataLoader 会崩
    spec.loader.exec_module(mod)
    return mod


base13 = _load("base13", "13_vit_finetune.py")   # 数据管道 / 训练循环 / 模型构建
lora = _load("loramod", "lora.py")               # LoRALinear / 参数报告

CSV_PATH = _REPO / "practice" / "results_flowers.csv"
COLS = ["epochs", "batch", "lr", "mode", "params", "trainable",
        "train_acc", "val_acc", "test_acc", "peak_vram_gb", "sec_per_epoch", "tag"]


# ============================================================================
# 注入
# ============================================================================
def inject_lora_vit(model: nn.Module, r: int, alpha: float,
                    targets: tuple[str, ...], dropout: float = 0.0) -> list[str]:
    """把模型里匹配 targets 的 nn.Linear 换成 LoRALinear。

    用**完整路径**做 endswith 匹配，而不是只看叶子名。
    原因：torchvision 的 FFN 是 Sequential，两个 Linear 的叶子名分别是 "0" 和 "3"——
    只按叶子名匹配的话，"0" 这种名字太容易误伤别的模块。
    """
    # 先收集再替换：一边遍历 named_modules 一边 setattr 会改到正在遍历的字典
    names = [name for name, mod in model.named_modules()
             if isinstance(mod, nn.Linear) and any(name.endswith(t) for t in targets)]

    for name in names:
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path)          # torch>=1.9
        target = model.get_submodule(name)
        setattr(parent, leaf, lora.LoRALinear(target, r=r, alpha=alpha, dropout=dropout))
    return names


def freeze_all_but_lora_and_head(model: nn.Module) -> None:
    """冻结一切，然后只放开两组参数：LoRA 的 A/B，以及分类头。

    **分类头必须一起训。** 它是 nn.Linear(768, 102) 随机初始化的，
    不训它的话模型输出永远是噪声——这是「冻结一切」时最容易漏的一步。
    （13 里就踩过反向的坑：给优化器传了一个空 param_group，不报错但什么都不更新。）
    """
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
    for p in model.heads.head.parameters():
        p.requires_grad = True


def merge_check(model: nn.Module, sample: torch.Tensor) -> float:
    """验证「合并后 == 未合并」：把 BA 折进 W 之后，输出应该逐位相同。

    这不是形式检查——训练时分开算、推理时合并算，是 LoRA「零推理延迟」的全部依据。
    如果这两条路不等价，那句「不加推理延迟」就是假的。
    """
    wrappers = [m for m in model.modules() if isinstance(m, lora.LoRALinear)]
    unmerged = model(sample).clone()
    for m in wrappers:
        m.merge()
    merged = model(sample).clone()
    for m in wrappers:
        m.unmerge()                                        # 训练前必须还原
    return float((unmerged - merged).detach().abs().max())


# ============================================================================
# 主流程
# ============================================================================
# ============================================================================
# 把 torchvision 的 nn.MultiheadAttention 换成 08_mha.py 的手写实现
# ============================================================================
class ViTAttnAdapter(nn.Module):
    """让手写的 MultiHeadSelfAttention 冒充 nn.MultiheadAttention。

    为什么需要适配器：torchvision 的 EncoderBlock 是这么调的（vision_transformer.py:113）

        x, _ = self.self_attention(x, x, x, need_weights=False)

    它按 nn.MultiheadAttention 的三参 + 返回 tuple 签名调用。手写的那个是
    `forward(x, attn_mask=None)` 自注意力，签名对不上，所以包一层做翻译。

    包完之后 `wq/wk/wv/wo` 都是**真 nn.Linear**，且上层通过 `module(x)` 正常调用
    —— LoRA 包裹的两个判据都满足了，attention 终于可以被注入。

    顺带解决的还有 fast path 问题：nn.MultiheadAttention 在条件满足时会走
    torch._native_multi_head_attention，直接读 out_proj.weight 而不调 out_proj.forward，
    导致 LoRA 的 delta 被静默丢弃。换成显式调用的实现后，那条路径根本不存在。
    """

    def __init__(self, d: int, h: int, mha_cls):
        super().__init__()
        self.mha = mha_cls(d, h)
        self.embed_dim = d
        self.num_heads = h
        self.batch_first = True

    def forward(self, query, key, value, need_weights: bool = False,
                attn_mask=None, average_attn_weights: bool = True,
                is_causal: bool = False):
        # 只做自注意力：ViT 的 query/key/value 传进来是同一个张量
        out = self.mha(query, attn_mask=attn_mask)
        # torchvision 会解包 `x, _`，所以第二项必须给
        return out, None


@torch.no_grad()
def swap_vit_attention(model: nn.Module, device: str) -> None:
    """12 个 EncoderBlock 的 self_attention 全部换成手写实现，并搬运权重。

    搬运是必须的：换实现不等于重训，换完之后前向必须和原来**逐位接近**，
    否则后面所有实验都不再是「同一个起点 + LoRA」，而是「另一个模型」。
    """
    mha_cls = _load("mha08", "08_mha.py").MultiHeadSelfAttention
    for i, blk in enumerate(model.encoder.layers):
        old = blk.self_attention
        d, h = old.embed_dim, old.num_heads
        new = ViTAttnAdapter(d, h, mha_cls).to(device)

        # nn.MultiheadAttention 把 QKV 存成一个 (3d, d) 的大矩阵，
        # 手写实现是三个独立 Linear。按 [0:d] / [d:2d] / [2d:3d] 切开。
        W, b = old.in_proj_weight.detach(), old.in_proj_bias.detach()
        for k, lin in enumerate((new.mha.wq, new.mha.wk, new.mha.wv)):
            lin.weight.copy_(W[k * d:(k + 1) * d])
            lin.bias.copy_(b[k * d:(k + 1) * d])
        new.mha.wo.weight.copy_(old.out_proj.weight.detach())
        new.mha.wo.bias.copy_(old.out_proj.bias.detach())

        blk.self_attention = new
    return len(model.encoder.layers)


def main() -> None:
    p = argparse.ArgumentParser(description="给预训练 ViT-B/16 注入 LoRA 微调 Flowers102")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="LoRA 参数的学习率。LoRA 只训一小部分参数，"
                        "lr 通常比全量微调（1e-4）大一个量级")
    p.add_argument("--head-lr", type=float, default=None,
                   help="分类头单独的学习率（不给就同 --lr）")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="LoRA 论文对 LoRA 参数不加 weight decay")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--targets", type=str, default=None,
                   help="逗号分隔的模块名后缀。不给时：不加 --swap-attn 用 FFN "
                        "(mlp.0,mlp.3)，加了 --swap-attn 用 attention "
                        "(self_attention.mha.wq,wv)")
    p.add_argument("--swap-attn", action="store_true",
                   help="把手写 MultiHeadSelfAttention 换进来，让 LoRA 能落在 "
                        "attention 的 wq/wk/wv/wo 上（默认只能落在 FFN）")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--tag", type=str, default="lora")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if a.targets is not None:
        targets = tuple(t.strip() for t in a.targets.split(",") if t.strip())
    elif a.swap_attn:
        # wq/wv 是 LoRA 论文 §4.2 里性价比最高的两个（只加 q,v 得 22.1，全加得 23.0）
        targets = ("self_attention.mha.wq", "self_attention.mha.wv")
    else:
        targets = ("mlp.0", "mlp.3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(a.seed)

    print("=" * 68)
    print(f"ViT-B/16 + LoRA on Flowers102   r={a.lora_r} alpha={a.lora_alpha} "
          f"tag={a.tag}")
    print("=" * 68)
    print(f"  device = {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    print("\n[1/5] 加载数据")
    train_loader, val_loader, test_loader = base13.load_data(a.batch_size, a.workers)
    print(f"  每轮 batch 数 = {len(train_loader)}")

    print("\n[2/5] 构建模型 + 注入 LoRA")
    model = base13.build_model(scratch=False, head_only=False, device=device)
    if a.swap_attn:
        probe = next(iter(val_loader))[0][:8].to(device)
        model.eval()
        with torch.no_grad():
            ref = model(probe)
        n_layer = swap_vit_attention(model, device)
        with torch.no_grad():
            after = model(probe)
        diff = float((ref - after).abs().max())
        print(f"  self_attention 换成手写实现（{n_layer} 层），"
              f"换后前向最大差 = {diff:.3e}")
        # 这道断言是这次改动的全部意义：换实现不能换模型
        assert diff < 1e-5, f"换实现后输出不一致（{diff:.3e}），权重没搬对"
        model.train()
    frozen_before = sum(p.numel() for p in model.parameters())
    wrapped = inject_lora_vit(model, a.lora_r, a.lora_alpha, targets, a.lora_dropout)
    kinds = sorted({".".join(w.split(".")[-2:]) for w in wrapped})
    print(f"  注入 {len(wrapped)} 个 Linear，成员形如：{', '.join(kinds)}")

    freeze_all_but_lora_and_head(model)
    # 注入是在 build_model **之后**做的，所以新造的 lora_A / lora_B 在 CPU 上，
    # 而原模型已经在 GPU 了。必须再搬一次，否则前向时报
    # "Expected all tensors to be on the same device"。
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_cnt = sum(p.numel() for n, p in model.named_parameters()
                   if p.requires_grad and ("lora_A" in n or "lora_B" in n))
    head_cnt = sum(p.numel() for p in model.heads.head.parameters())
    print(f"  总参数   {total:,}（注入后比原来多 {total - frozen_before:,}）")
    print(f"  可训练   {trainable:,} ({trainable / total:.2%})")
    print(f"    ├ LoRA {lora_cnt:,} ({lora_cnt / total:.2%})")
    print(f"    └ 分类头 {head_cnt:,}（随机初始化，必须一起训）")
    assert lora_cnt > 0, "没注入成功"
    assert trainable == lora_cnt + head_cnt, "可训练集合和预期不符"

    # 合并等价性：训练前先验一次
    sample = next(iter(val_loader))[0][:4].to(device)
    print(f"  merge 等价性（训练前） 最大差 = {merge_check(model, sample):.3e}")

    print("\n[3/5] 优化器")
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
    head_params = [p for p in model.heads.head.parameters() if p.requires_grad]
    head_lr = a.head_lr if a.head_lr is not None else a.lr
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": a.lr},
         {"params": head_params, "lr": head_lr}],
        weight_decay=a.weight_decay, betas=(0.9, 0.999))
    total_steps = len(train_loader) * a.epochs
    print(f"  AdamW | LoRA lr={a.lr} | 头 lr={head_lr} | wd={a.weight_decay}")
    print(f"  warmup {a.warmup_steps} 步 + 余弦，总步数 ≈ {total_steps}")

    print(f"\n[4/5] 训练 {a.epochs} 轮")
    criterion = nn.CrossEntropyLoss()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_val, best_ep = 0.0, 0
    t_start = time.time()
    step = 0
    init_lrs = [g["lr"] for g in opt.param_groups]   # 各组的基准学习率，只记一次
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _train_epoch(model, train_loader, criterion, opt, device,
                                       a, total_steps, step, init_lrs)
        step += len(train_loader)
        va_loss, va_acc = base13.run_epoch(model, val_loader, criterion, device, opt=None)
        dt = time.time() - t0
        flag = ""
        if va_acc > best_val:
            best_val, best_ep = va_acc, ep
            flag = " *"
        print(f"  ep {ep:2d}  train {tr_loss:.4f}/{tr_acc:.2%}   "
              f"val {va_loss:.4f}/{va_acc:.2%}   {dt:.1f}s{flag}")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    else:
        peak = 0.0

    print("\n[5/5] 测试集（只报一次）")
    te_loss, te_acc = base13.run_epoch(model, test_loader, criterion, device, opt=None)
    print(f"  test  {te_loss:.4f}/{te_acc:.2%}")
    print(f"  merge 等价性（训练后） 最大差 = {merge_check(model, sample):.3e}")

    sec = (time.time() - t_start) / a.epochs
    print("\n" + "-" * 68)
    print("资源账")
    print("-" * 68)
    print(f"  总参数          {total:,}")
    print(f"  可训练参数      {trainable:,}   ({trainable / total:.2%})")
    print(f"  显存峰值        {peak:>9.2f} GB")
    print(f"  训练总耗时      {(time.time() - t_start):.1f} s   ({sec:.1f} s/轮)")
    print(f"  best val        {best_val:.2%}（ep{best_ep}）")

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / f"vit_flowers_lora_r{a.lora_r}_{a.tag}.pth"
    torch.save({"tag": a.tag, "lora_r": a.lora_r, "lora_alpha": a.lora_alpha,
                "targets": list(targets), "epochs": a.epochs, "lr": a.lr,
                "state_dict": model.state_dict()}, ckpt)
    print(f"\n  存档 → {ckpt}")

    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLS)
        w.writerow([a.epochs, a.batch_size, a.lr, f"lora_r{a.lora_r}",
                    total, trainable, f"{tr_acc:.4f}", f"{va_acc:.4f}",
                    f"{te_acc:.4f}", f"{peak:.2f}", f"{sec:.1f}", a.tag])
    print(f"  台账 → {CSV_PATH}")


def _train_epoch(model, loader, criterion, opt, device, a, total_steps, step0,
                 init_lrs):
    """一个训练轮，返回 (平均 loss, 准确率)。

    自己写而不复用 13 的 run_epoch，是因为这里要**按 step** 调学习率
    （warmup 是 step 级的），而 13 那个版本是按轮调余弦。

    init_lrs 在训练开始前记一次：如果每轮都从 param_groups 里读当前 lr，
    余弦就会被反复作用，变成指数衰减。
    """
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if a.max_steps and i >= a.max_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # 逐 step 设 lr
        s = step0 + i
        for g, base_lr in zip(opt.param_groups, init_lrs):
            if s < a.warmup_steps:
                g["lr"] = base_lr * (s + 1) / max(a.warmup_steps, 1)
            else:
                prog = (s - a.warmup_steps) / max(total_steps - a.warmup_steps, 1)
                g["lr"] = base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += x.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


if __name__ == "__main__":
    main()
