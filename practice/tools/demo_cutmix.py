"""CutMix 参数演示：alpha 到底控制什么。

跑出来看两件事：
  1. 数值上——alpha 如何改变 lam 的分布（以及为什么实际 lam 的均值不是 0.5）
  2. 视觉上——同一个 lam 在图上到底挖掉多大一块

用法：
    python tools\\demo_cutmix.py
产出：
    tools/cutmix_demo.png
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# 从 tools/ 直接运行时要能 import 上级目录的 augment_mix
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augment_mix import cutmix_batch  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "cutmix_demo.png"
H = W = 32


# ---------------------------------------------------------------- 数值部分
def stats(alpha, n=20000, seed=0):
    """复刻 cutmix_batch 的几何逻辑，统计实际 lam 的分布。"""
    g = torch.Generator().manual_seed(seed)
    lams = torch.distributions.Beta(alpha, alpha).sample((n,), )
    actual = []
    for lam in lams.tolist():
        cut_rat = math.sqrt(1.0 - lam)
        cw, ch = int(W * cut_rat), int(H * cut_rat)
        cx = int(torch.randint(0, W, (1,), generator=g).item())
        cy = int(torch.randint(0, H, (1,), generator=g).item())
        x1 = min(max(cx - cw // 2, 0), W)
        x2 = min(max(cx + cw // 2, 0), W)
        y1 = min(max(cy - ch // 2, 0), H)
        y2 = min(max(cy + ch // 2, 0), H)
        actual.append(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
    return torch.tensor(actual)


def print_table():
    print(f"{'alpha':>6} {'采样lam均值':>12} {'实际lam均值':>12} "
          f"{'实际std':>9} {'平均挖掉面积':>13} {'大块占比':>9}")
    print("-" * 76)
    for a in (0.1, 0.5, 1.0, 4.0):
        s = stats(a)
        big = (s < 0.5).float().mean().item()  # 实际 lam<0.5 说明挖掉超过一半
        print(f"{a:>6.1f} {'0.500':>12} {s.mean():>12.3f} "
              f"{s.std():>9.3f} {1 - s.mean():>13.3f} {big:>9.1%}")
    print()
    print("注：'采样lam均值'恒为 0.5（Beta(a,a) 关于 0.5 对称），")
    print("    但'实际lam均值'大于它 —— 因为矩形被图像边界截断了，挖不满。")


# ---------------------------------------------------------------- 视觉部分
def make_pair():
    """优先用真实 CIFAR；读不到就退化成合成图案（沙箱可能读不了 data 目录）。"""
    try:
        from torchvision import datasets, transforms

        root = Path(__file__).resolve().parent.parent.parent / "data"
        ds = datasets.CIFAR10(root=str(root), train=True,
                              download=False, transform=transforms.ToTensor())
        a = ds[0][0].unsqueeze(0)
        b = ds[1][0].unsqueeze(0)
        return a, b, f"{ds.classes[ds[0][1]]} / {ds.classes[ds[1][1]]}"
    except Exception as e:  # noqa: BLE001
        print(f"[提示] 读不到 CIFAR（{type(e).__name__}），改用合成图案演示几何。")
        # A：横向红绿渐变；B：蓝色棋盘格。两者差异足够大，能看清挖了哪块。
        ramp = torch.linspace(0, 1, W).view(1, W).expand(H, W)
        a = torch.stack([ramp, ramp.flip(-1), torch.full((H, W), 0.15)], 0).unsqueeze(0)

        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        checker = ((yy // 4 + xx // 4) % 2).float()
        b = torch.stack([torch.full((H, W), 0.15), checker * 0.3,
                         torch.full((H, W), 0.85)], 0).unsqueeze(0)
        return a, b, "合成渐变 / 合成棋盘"


def find_lam(x, y, target, max_try=4000):
    """反复抽样，直到拿到接近 target 的实际 lam。用的是真函数，没复制逻辑。"""
    best = None
    for i in range(max_try):
        g = torch.Generator().manual_seed(i)
        xm, ya, yb, lam = cutmix_batch(x, y, alpha=1.0, generator=g)
        if best is None or abs(lam - target) < abs(best[3] - target):
            best = (xm, ya, yb, lam)
        if abs(lam - target) < 0.015:
            return best
    return best


def main():
    print_table()

    a, b, title = make_pair()
    # 把 b 放到 batch 的第 1 位，perm 才能取到它
    finder = torch.cat([a, b], 0)
    lab = torch.arange(2)

    targets = [0.95, 0.80, 0.60, 0.40, 0.20]

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[2.1, 1.6], hspace=0.32, wspace=0.18)

    def show(ax, img, caption):
        ax.imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(caption, fontsize=10, pad=5)

    ax = fig.add_subplot(gs[0, 0])
    show(ax, a[0], f"原图 A（λ=1.0）\n{title.split(' / ')[0]}")

    for i, t in enumerate(targets):
        xm, _, _, lam = find_lam(finder, lab, t)
        ax = fig.add_subplot(gs[0, i + 1])
        show(ax, xm[0], f"λ={lam:.2f}\n挖掉 {1 - lam:.0%}")

    # 第二行：三个 alpha 的实际 lam 分布
    for j, a_val in enumerate([0.1, 1.0, 4.0]):
        s = stats(a_val).numpy()
        ax = fig.add_subplot(gs[1, j * 2:j * 2 + 2])
        ax.hist(s, bins=40, range=(0, 1), color="#4a7fb5", edgecolor="none")
        ax.set_xlim(0, 1)
        ax.set_title(f"alpha={a_val} 时实际 λ 的分布（均值 {s.mean():.2f}）",
                     fontsize=10)
        ax.set_xlabel("λ（= 保留面积比例 = 标签 A 的权重）", fontsize=9)
        ax.set_ylabel("频次", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle("CutMix：λ 同时决定挖多大、留多少像素、标签怎么加权",
                 fontsize=13, y=0.97)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\n图已保存：{OUT}")


if __name__ == "__main__":
    main()
