"""13 —— 在预训练 ViT-B/16 上做迁移学习（Flowers102）

=============================================================================
这次实验和 12_vit_cifar.py 的**根本区别只有一件事**：
    参数的起点不是随机初始化，而是别人（Google）在 ImageNet 上训出来的结果。
=============================================================================

三种模式，代码路径几乎相同，只差「起点是什么、哪些参数可训练」：

    python 13_vit_finetune.py --epochs 5                # 全量微调（86M 全部可训）
    python 13_vit_finetune.py --epochs 5 --head-only    # 线性探针（只训分类头 78K）
    python 13_vit_finetune.py --epochs 5 --scratch      # 从零训练（对照组）

三种模式共用同一个模型结构、同一份数据、同一套超参 —— 唯一变量是「起点」。
这样跑出来的差距才能归因给「预训练」，而不是别的东西。

数据集：Flowers102（102 类花），train 1,020 / val 1,020 / test 6,149 张。
训练集小到会过拟合 —— 这正是它适合当「微调」入门的原因：
    W3 全量微调 → 会看到过拟合
    W4 换成 LoRA → 看参数少了 99% 是不是反而更稳
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# torchvision 下载预训练权重时默认写到 ~/.cache/torch。
# 这台机器上那个目录建不了（WinError 5 拒绝访问），所以把缓存指到仓库内部：
#   .cache/torch/hub/checkpoints/vit_b_16-*.pth
# 好处不只是能跑——权重跟着仓库走，换台机器也不用重下。
_REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("TORCH_HOME", str(_REPO / ".cache" / "torch"))

DATA_DIR = _REPO / "data"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_CSV = Path(__file__).resolve().parent / "results_flowers.csv"

# 预训练权重期待的是 ImageNet 的统计量，不是 CIFAR 的。
# 用错会掉一大截精度，而且不会报错 —— 这是最隐蔽的一个坑。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_CLASSES = 102
IMG_SIZE = 224


# ============================================================================
# 数据
# ============================================================================
def build_transforms(train: bool):
    """训练和验证的预处理。

    注意 eval 侧不是「什么都不做」：
      Resize(256) + CenterCrop(224) 是为了让任意尺寸的图都变成 224×224，
      因为 ViT 的位置编码长度是固定的（197 个 token）。
      如果直接喂一张 500×500 的图，patch 数会变成 (500/16)^2 = 961，
      位置编码对不上，直接报错。
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FlowersDataset(Dataset):
    """最小数据集：从 jpg/ 读图，从 splits.json 读标签和划分。

    为什么自己写这 20 行而不是用 torchvision.datasets.Flowers102？
    因为后者 `from scipy.io import loadmat`，而我们不想为了两个小 .mat 文件
    往训练环境里塞 scipy。标签只解析一次，之后就是普通数据集。
    """

    def __init__(self, root: Path, meta_path: Path, split: str, transform=None):
        self.root = root
        self.transform = transform
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        # splits.json 里的 label 是 1..102（MATLAB 风格），转成 0..101 给 CrossEntropyLoss
        self.labels = [v - 1 for v in meta["labels"]]
        self.ids = meta[split]                      # 1-based 图像编号

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        path = self.root / "jpg" / f"image_{img_id:05d}.jpg"
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[img_id - 1]


def load_data(batch_size: int, workers: int):
    """构造三个 Dataset。

    这里没有用 torchvision.datasets.Flowers102，因为它依赖 scipy.io.loadmat
    去读 .mat 标签文件，而 venv 里没有 scipy（装不上）。

    替代方案：用一个一次性脚本把 imagelabels.mat / setid.mat 转成 splits.json，
    训练脚本只读 JSON。好处是训练环境彻底不依赖 scipy。
    """
    root = DATA_DIR / "flowers-102"
    meta_path = root / "splits.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"缺少 {meta_path}。\n"
            f"请先用带 scipy 的 Python 把 imagelabels.mat / setid.mat 转成 splits.json。")

    train_ds = FlowersDataset(root, meta_path, "train", build_transforms(True))
    val_ds = FlowersDataset(root, meta_path, "val", build_transforms(False))
    test_ds = FlowersDataset(root, meta_path, "test", build_transforms(False))

    common = dict(num_workers=workers, pin_memory=True,
                  persistent_workers=(workers > 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    print(f"  训练集 {len(train_ds):,} / 验证集 {len(val_ds):,} / 测试集 {len(test_ds):,}")
    return train_loader, val_loader, test_loader


# ============================================================================
# 模型
# ============================================================================
def build_model(scratch: bool, head_only: bool, device: str):
    """构建 ViT-B/16，并决定哪些参数可训练。

    torchvision.models.vit_b_16 的结构：
        model.patch_embed            Conv2d(3, 768, kernel=16, stride=16)
        model.class_token            可学习参数 (1,1,768)
        model.encoder                Transformer 主体，12 层
        model.heads.head             Linear(768, 1000)   ← 要换掉
    """
    if scratch:
        # 随机初始化，等价于 12_vit_cifar.py 那条路线
        model = models.vit_b_16(weights=None)
        print("  起点：随机初始化（对照组）")
    else:
        # IMAGENET1K_V1 = 先在 ImageNet-21k 预训练，再在 ImageNet-1k 微调
        # 权重约 330 MB，缓存在 ~/.cache/torch/hub/checkpoints/
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        print("  起点：ImageNet-21k 预训练 + ImageNet-1k 微调")

    # 换掉分类头：1000 类 → 102 类
    # 注意是替换，不是冻结 —— 新头必须是随机初始化且可训练的
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, NUM_CLASSES)
    print(f"  分类头：Linear({in_features}, {NUM_CLASSES})，"
          f"{in_features * NUM_CLASSES + NUM_CLASSES:,} 参数")

    if head_only:
        # 线性探针：主干全冻结，只训分类头
        # 这是「预训练特征本身有多好」的直接测量
        for p in model.parameters():
            p.requires_grad = False
        for p in model.heads.head.parameters():
            p.requires_grad = True
        print("  模式：线性探针（冻结主干，只训分类头）")

    return model.to(device)


def param_report(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (f"总参数 {total:,} | 可训练 {trainable:,} "
            f"({trainable / total:.2%})")


# ============================================================================
# 训练 / 评估
# ============================================================================
def run_epoch(model, loader, criterion, device, opt=None, max_steps=None):
    training = opt is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for step, (x, y) in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            out = model(x)
            loss = criterion(out, y)

            if training:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # 微调 pre-trained 模型时也可用梯度裁剪，防止偶尔的 loss 尖峰
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


def build_optimizer(model, lr: float, wd: float, head_lr: float | None = None):
    """只把 requires_grad=True 的参数交给优化器。

    这是 head_only 模式下必须的：如果传了冻结参数，
    AdamW 会为它们分配一阶/二阶动量，白白多吃一倍显存。

    head_lr 给分类头单独的学习率，理由是「谁从零开始，谁就需要更大的步长」：
      - 主干是预训练权重，只需要往任务上挪一点 → lr 1e-4 量级
      - 分类头是随机初始化、扔掉了 ImageNet 那 1000 类 → 要从零学会 102 类
    两者用同一个 lr 时，能跑通的是主干那条（微调），分类头那条会被饿死。
    实测：lr=1e-4 训 5 轮，head_only 只有 12.65%，而全量微调 91.80%
    —— 不是「预训练特征没用」，是「分类头没学会怎么用它们」。
    """
    head = {id(p) for p in model.heads.head.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head]
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head]

    # 两个组里任何一个为空都要跳过 —— 给 AdamW 传一个空 param_group
    # 不会报错，只是那组永远不更新；head_only 模式下主干全冻结，
    # 一旦把空组当成唯一组，整个优化器就什么都不做（loss 一条直线）。
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": lr})
    groups.append({"params": head_params,
                   "lr": head_lr if head_lr is not None else lr})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(0.9, 0.999))


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    """线性 warmup + 余弦退火。微调时 warmup 很短（几个 step 就够）。"""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


# ============================================================================
# 主流程
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="在预训练 ViT-B/16 上微调 Flowers102")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="主干的学习率（微调通常比从头训练小 10 倍）")
    p.add_argument("--head-lr", type=float, default=None,
                   help="分类头单独的学习率（不给就和 --lr 一样）。"
                        "分类头是随机初始化的，线性探针常用 1e-3 甚至 1e-2")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--scratch", action="store_true", help="不用预训练权重（对照组）")
    p.add_argument("--head-only", action="store_true", help="线性探针：只训分类头")
    p.add_argument("--max-steps", type=int, default=None,
                   help="每个 epoch 最多跑几个 batch（调试用）")
    p.add_argument("--tag", type=str, default="default")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(a.seed)

    mode = "scratch" if a.scratch else ("head_only" if a.head_only else "full_ft")

    print("=" * 68)
    print(f"ViT-B/16 on Flowers102   mode={mode}  tag={a.tag}")
    print("=" * 68)
    print(f"  device = {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    print("\n[1/4] 加载数据")
    train_loader, val_loader, test_loader = load_data(a.batch_size, a.workers)
    print(f"  每轮 batch 数 = {len(train_loader)}")

    print("\n[2/4] 构建模型")
    model = build_model(a.scratch, a.head_only, device)
    print(f"  {param_report(model)}")

    criterion = nn.CrossEntropyLoss()
    opt = build_optimizer(model, a.lr, a.weight_decay, a.head_lr)
    total_steps = a.epochs * len(train_loader)
    warmup = min(a.warmup_steps, total_steps // 10)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"\n[3/4] 训练 {a.epochs} 轮（lr={a.lr}, wd={a.weight_decay}, "
          f"warmup={warmup} 步, 总步数≈{total_steps}）")

    t0 = time.perf_counter()
    gstep = 0
    recs = []
    for ep in range(1, a.epochs + 1):
        ep_t0 = time.perf_counter()

        # 手动设 lr（LambdaLR 也可以，但手写更直观）
        for g in opt.param_groups:
            g["lr"] = lr_at(gstep, total_steps, a.lr, warmup)

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    opt, a.max_steps)
        gstep += min(len(train_loader), a.max_steps or len(train_loader))

        va_loss, va_acc = run_epoch(model, val_loader, criterion, device,
                                    None, a.max_steps)
        dt = time.perf_counter() - ep_t0
        recs.append((ep, tr_loss, tr_acc, va_loss, va_acc))

        print(f"  ep{ep:3d}  train {tr_loss:.4f}/{tr_acc:.2%}   "
              f"val {va_loss:.4f}/{va_acc:.2%}   {dt:.1f}s")

    train_time = time.perf_counter() - t0

    print("\n[4/4] 在测试集上评估（只报一次，不参与任何选择）")
    te_loss, te_acc = run_epoch(model, test_loader, criterion, device, None)
    print(f"  test  {te_loss:.4f}/{te_acc:.2%}")

    # ---------------------------------------------------------------- 资源账
    vram_gb = 0.0
    if device == "cuda":
        vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "-" * 68)
    print("资源账（这是本次实验最该记住的部分）")
    print("-" * 68)
    print(f"  总参数        {total_p:>12,}")
    print(f"  可训练参数    {train_p:>12,}   ({train_p / total_p:.2%})")
    print(f"  显存峰值      {vram_gb:>12.2f} GB")
    print(f"  训练总耗时    {train_time:>12.1f} s   "
          f"({train_time / a.epochs:.1f} s/轮)")
    print(f"  每 1000 张图耗时 {train_time / max(1, len(train_loader.dataset) * a.epochs / 1000):.1f} s")

    # ---------------------------------------------------------------- 落盘
    CKPT_DIR.mkdir(exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "config": vars(a),
        "epoch": a.epochs,
        "best_acc": te_acc,
        "val_acc": recs[-1][4],
    }
    ckpt_path = CKPT_DIR / f"vit_flowers_{mode}_{a.tag}.pth"
    torch.save(ckpt, ckpt_path)
    print(f"\n  存档 → {ckpt_path}")

    new_file = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epochs", "batch", "lr", "mode", "params", "trainable",
                        "train_acc", "val_acc", "test_acc", "peak_vram_gb",
                        "sec_per_epoch", "tag"])
        w.writerow([a.epochs, a.batch_size, a.lr, mode, total_p, train_p,
                    f"{recs[-1][2]:.4f}", f"{recs[-1][4]:.4f}", f"{te_acc:.4f}",
                    f"{vram_gb:.2f}", f"{train_time / a.epochs:.1f}", a.tag])
    print(f"  台账 → {LOG_CSV}")


if __name__ == "__main__":
    main()
