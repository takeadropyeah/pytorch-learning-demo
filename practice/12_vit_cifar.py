"""12 —— ViT on CIFAR-10（从零训练）

复现 Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021 的核心论点：

    缺少卷积归纳偏置的 Transformer，在小数据（5 万张 CIFAR-10）上
    打不过参数量相当的 ResNet。

对照基线（同一份数据管道、同样 40 轮、同样增强）：
    ResNet-18  11,173,962 参数  末轮 93.51% / 末 10 轮均值 93.25%   ← 实验 #17

本脚本的两个对照配置：
    --d 192 --layers 4   约 1.90M 参数（小，先跑通）
    --d 384 --layers 6   约 10.89M 参数（和 ResNet-18 参数匹配）

与语言模型的唯一区别在输入那一步（论文式 1）：
    文本：x = tok_emb(idx) + pos_emb(arange(T))
    图像：x = patch_embed(patches) + pos_emb(arange(N+1))，并在最前面拼一个 [CLS]

block 直接复用 `09_transformer_block.py` —— ViT 的 encoder 与 GPT 的逐字相同，
唯一的调用差异是 **不传 attn_mask**（图像分类是双向注意力，不需要因果掩码）。
"""

import argparse
import csv
import importlib.util
import math
import pathlib
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONFIG = None
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 复用 09 / 05
def _load_py_module(mod_name, filename):
    """按文件名加载同目录下的 .py（文件名以数字开头，不能直接 import）。"""
    path = pathlib.Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_blk09 = _load_py_module("blk09", "09_transformer_block.py")
_t05 = _load_py_module("train05", "05_resnet_train.py")
TransformerBlock = _blk09.TransformerBlock

# 混合增强（CutMix/MixUp）不在 transform 里，因为一次要同时改两张图的像素和标签，
# 而 transforms 一次只处理一张图。所以放在训练的 batch 循环里。
from augment_mix import apply_mix, mix_criterion  # noqa: E402  (需在同目录下运行)


# ---------------------------------------------------------------- 模型
class PatchEmbed(nn.Module):
    """把 (B,3,32,32) 切成 (B,N,P*P*3) 再线性投影到 d 维（论文式 1 的 E 矩阵）。

    等价写法是 nn.Conv2d(3, d, kernel_size=P, stride=P) ——
    卷积核等于步长时，每个输出位置只看一个 patch，就是逐 patch 的线性投影。
    这里用 unfold + Linear，是为了和论文式 1 的写法一一对应，方便对照。
    """

    def __init__(self, in_ch, patch, d):
        super().__init__()
        self.patch = patch
        self.in_ch = in_ch
        self.proj = nn.Linear(patch * patch * in_ch, d)

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        P = self.patch
        # (B,C,H,W) -> (B,C,H/P,W/P,P,P) -> (B,H/P,W/P,C,P,P) -> (B,N,P*P*C)
        p = x.unfold(2, P, P).unfold(3, P, P)
        p = p.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, P * P * C)
        return self.proj(p)


class ViT(nn.Module):
    """标准 Transformer encoder + [CLS] 分类头。没有任何视觉专用的东西。"""

    def __init__(self, img=32, patch=4, d=192, layers=4, heads=6,
                 mult=4, num_classes=10, dropout=0.1, in_ch=3):
        super().__init__()
        n = (img // patch) ** 2                      # patch 数（不含 CLS）
        self.patch_embed = PatchEmbed(in_ch, patch, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))       # 可学习的 [CLS]
        self.pos = nn.Parameter(torch.zeros(1, n + 1, d))   # 位置编码（含 CLS 那一位）
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, heads, mult) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(d)                  # 论文式 4
        self.head = nn.Linear(d, num_classes)
        self.n_tokens = n

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)                                  # (B,N,d)
        cls = self.cls.expand(x.size(0), -1, -1)                 # (B,1,d)
        x = torch.cat([cls, x], dim=1)                           # (B,N+1,d)
        x = self.drop(x + self.pos)                              # 广播相加
        for blk in self.blocks:
            x = blk(x)          # 注意：attn_mask=None —— 双向注意力，不遮任何位置
        return self.head(self.ln_f(x)[:, 0])                     # 只取 CLS 那一位


# ---------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="ViT on CIFAR-10（从零训练）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05,
                   help="ViT 不像 CNN 那样自带正则，wd 是关键；参考论文用 0.1")
    p.add_argument("--optim", choices=["adam", "adamw"], default="adamw")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patch", type=int, default=4, choices=[2, 4, 8],
                   help="patch 边长；32/4=8 -> 64 个 token，32/8=4 -> 16 个")
    p.add_argument("--d", type=int, default=192, help="隐藏维 d_model")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--mult", type=int, default=4, help="FFN 宽度倍数")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=0,
                   help="前 N 轮线性 warmup（0 = 不 warmup）。ViT 论文用了 warmup，"
                        "没有它的 Adam 前几步更新量恒等于 lr，大模型会抖")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--erase", type=float, default=0.5)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--mix", choices=["none", "cutmix", "mixup"], default="none",
                   help="batch 级混合增强；none 时行为与 #23-#26 完全一致")
    p.add_argument("--mix-alpha", type=float, default=1.0,
                   help="Beta(alpha,alpha) 参数；alpha 越小挖块大小越极端")
    p.add_argument("--mix-prob", type=float, default=0.5,
                   help="每个 batch 施加混合的概率（论文全混=1.0，DeiT 用 1.0）")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tag", type=str, default="default")
    return p.parse_args()


def build_config(a):
    return {
        "seed": a.seed, "batch_size": a.batch_size, "lr": a.lr, "epochs": a.epochs,
        "num_workers": a.workers, "weight_decay": a.weight_decay, "optim": a.optim,
        "no_augment": a.no_augment, "normalize": not a.no_normalize, "erase": a.erase,
        "patch": a.patch, "d": a.d, "layers": a.layers, "heads": a.heads,
        "mult": a.mult, "dropout": a.dropout, "warmup": a.warmup,
        "mix": a.mix, "mix_alpha": a.mix_alpha, "mix_prob": a.mix_prob,
        "resume": a.resume, "tag": a.tag,
    }


def aug_label(cfg):
    """写进 results.csv 的 augment 列。

    混合增强不进 transforms，所以必须单独记在这里，否则 #26（无）和 #27（有）
    在台账里 augment 列会一模一样。
    """
    s = "cifar-tf"
    if cfg.get("mix", "none") != "none":
        s += f"+{cfg['mix']}(a={cfg['mix_alpha']:g},p={cfg.get('mix_prob', 0.5):g})"
    return s


def model_name(cfg):
    # warmup 要进名字，否则 #24(无) 和 #25(有) 两行在 results.csv 里完全一样，
    # 只看表分不出哪次是 warmup 实验
    s = f"ViT-P{cfg['patch']}-d{cfg['d']}-L{cfg['layers']}"
    if cfg.get("warmup"):
        s += f"-w{cfg['warmup']}"
    return s


CKPT_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"
RESULTS = pathlib.Path(__file__).resolve().parent / "results.csv"


def main():
    print("=" * 52, flush=True)
    for k, v in CONFIG.items():
        print(f" {k:14s} = {v}", flush=True)
    print("=" * 52, flush=True)

    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    print(f"device={device} workers={CONFIG['num_workers']}", flush=True)

    train_tf = _t05.build_transforms(CONFIG, train=True)
    test_tf = _t05.build_transforms(CONFIG, train=False)
    print(f"transform(train) = {train_tf}", flush=True)

    data_dir = _t05.DATA_DIR
    train_set = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"],
                              persistent_workers=CONFIG["num_workers"] > 0,
                              pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=CONFIG["num_workers"],
                             persistent_workers=CONFIG["num_workers"] > 0,
                             pin_memory=(device == "cuda"))

    model = ViT(img=32, patch=CONFIG["patch"], d=CONFIG["d"], layers=CONFIG["layers"],
                heads=CONFIG["heads"], mult=CONFIG["mult"], dropout=CONFIG["dropout"]).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"模型: {model_name(CONFIG)}", flush=True)
    print(f"增强: {aug_label(CONFIG)}", flush=True)
    print(f"token 数: {model.n_tokens + 1}（{model.n_tokens} patch + 1 CLS）", flush=True)
    print(f"参数量: {n_par:,}   （ResNet-18 是 11,173,962）", flush=True)

    ckpt_path = CKPT_DIR / f"vit_{CONFIG['tag']}.pth"
    criterion = nn.CrossEntropyLoss()
    Opt = torch.optim.AdamW if CONFIG["optim"] == "adamw" else torch.optim.Adam
    optimizer = Opt(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    W, T = CONFIG["warmup"], CONFIG["epochs"]
    if W > 0:
        # 前 W 轮线性升到 1，之后余弦降到 0。
        # 为什么必须 warmup：Adam 第一步的 m̂/√v̂ = sign(g)，更新量恒等于 lr，
        # 随机初始化时方向是噪声，直接上大 lr 会把模型推坏。
        def _lr_lambda(ep):
            if ep < W:
                return (ep + 1) / W
            prog = (ep - W) / max(1, T - W)
            return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        print(f"lr 调度: 线性 warmup {W} 轮 → 余弦降到 0", flush=True)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
        print(f"lr 调度: 余弦 T_max={T}（无 warmup）", flush=True)

    start_epoch, best_acc = 1, 0.0
    if CONFIG["resume"] and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        old = ck.get("config", {})
        keys = ("epochs", "batch_size", "lr", "weight_decay", "optim", "no_augment",
                "normalize", "erase", "patch", "d", "layers", "heads", "mult", "dropout",
                "warmup", "mix", "mix_alpha", "mix_prob")
        bad = [f"{k}: 存档={old.get(k)!r} 当前={CONFIG.get(k)!r}"
               for k in keys if k in old and old.get(k) != CONFIG.get(k)]
        if bad:
            print("[resume] 配置和存档不一致，拒绝续跑：", flush=True)
            for b in bad:
                print("         " + b, flush=True)
            raise SystemExit(f"  -> 想重新训练请先删除 {ckpt_path}")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_acc = ck.get("best_acc", 0.0)
        print(f"[resume] 从第 {start_epoch} 轮继续（best={best_acc*100:.2f}%）", flush=True)

    if start_epoch > CONFIG["epochs"]:
        print(f"[resume] 已到第 {start_epoch-1} 轮，无需继续", flush=True)
        return

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        running = 0.0
        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            # 混合增强：mode='none' 时原样返回，走同一条 loss，行为与 #23-#26 一致
            if CONFIG["mix"] != "none" and torch.rand(1).item() < CONFIG["mix_prob"]:
                xb, ya, yb2, lam = apply_mix(CONFIG["mix"], xb, yb, CONFIG["mix_alpha"])
                loss = mix_criterion(model(xb), ya, yb2, lam)
            else:
                loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            if (i + 1) % 100 == 0:
                print(f"epoch {epoch} {i+1}/{len(train_loader)} loss={running/100:.4f}", flush=True)
                running = 0.0

        train_acc = _t05.evaluate(model, train_loader)
        test_acc = _t05.evaluate(model, test_loader)
        print(f"==>epoch {epoch} lr={optimizer.param_groups[0]['lr']:.6f} "
              f"train {train_acc*100:.2f}% test {test_acc*100:.2f}% "
              f"耗时{time.perf_counter()-t0:.1f}s", flush=True)
        scheduler.step()
        best_acc = max(best_acc, test_acc)

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best_acc": best_acc, "config": CONFIG}, ckpt_path)

    new_file = not RESULTS.exists()
    from tools.cost_lookup import COST_COLS, cost_columns
    name = model_name(CONFIG)
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epochs", "batch", "lr", "workers", "augment", "bn", "cosine",
                        "weight_decay", "model", "train_acc", "test_acc"] + COST_COLS)
        w.writerow([CONFIG["epochs"], CONFIG["batch_size"], CONFIG["lr"], CONFIG["num_workers"],
                    aug_label(CONFIG), False, True, CONFIG["weight_decay"], name,
                    f"{train_acc:.4f}", f"{test_acc:.4f}"] + cost_columns(name))
    print(f"\n结果已追加到 {RESULTS.name}（{name}）", flush=True)
    print(f"对照：ResNet-18 参数量 {11173962:,}，同管道 40 轮末轮 93.51%", flush=True)


if __name__ == "__main__":
    CONFIG = build_config(parse_args())
    main()
