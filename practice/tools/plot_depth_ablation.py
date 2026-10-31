r"""
复现 ResNet 论文 Figure 1（CIFAR-10 版）：深度增加时，plain 网络退化，残差网络不退化。

数据来源：logs/ 下四次 40 轮训练的日志
    ablation_resnet18.log   ResNet-18
    ablation_plain18.log    plain-18
    p4_resnet34.log         ResNet-34
    p4_plain34.log          plain-34

用法：
    python .\tools\plot_depth_ablation.py
产出：
    practice\depth_ablation.png
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE.parent / "logs"

# (日志文件, 显示名, 深度, 是否有残差)
RUNS = [
    ("ablation_plain18.log", "plain-18",  18, False),
    ("ablation_resnet18.log", "ResNet-18", 18, True),
    ("p4_plain34.log",       "plain-34",  34, False),
    ("p4_resnet34.log",      "ResNet-34", 34, True),
]

PAT = re.compile(r"^==>epoch (\d+) .*train ([0-9.]+)% test ([0-9.]+)%")


def read_log(name):
    """Tee-Object 写的是 UTF-16 LE，直接按 utf-16 读。"""
    path = LOGS / name
    for enc in ("utf-16", "utf-8", "gbk"):
        try:
            text = path.read_text(encoding=enc)
        except UnicodeError:
            continue
        epochs, test = [], []
        for line in text.splitlines():
            m = PAT.search(line)
            if m:
                epochs.append(int(m.group(1)))
                test.append(float(m.group(3)))
        if epochs:
            return epochs, test
    raise SystemExit(f"读不出 {path} 的逐轮数据")


def main():
    curves = {}
    for fname, label, depth, resid in RUNS:
        curves[label] = read_log(fname)

    color = {False: "#d62728", True: "#2a6fdb"}   # 红=plain，蓝=ResNet
    marker = {18: "o", 34: "s"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # ---- 左：末 10 轮均值 vs 深度 ----
    for resid in (False, True):
        xs, ys, labels = [], [], []
        for fname, label, depth, r in RUNS:
            if r != resid:
                continue
            test = curves[label][1][-10:]
            xs.append(depth)
            ys.append(sum(test) / len(test))
            labels.append(label)
        ax1.plot(xs, ys, marker="o", markersize=9, linewidth=2.4,
                 color=color[resid], label=("有残差" if resid else "无残差"))
        for x, y, lab in zip(xs, ys, labels):
            ax1.annotate(f"{lab}  {y:.2f}%", (x, y), textcoords="offset points",
                         xytext=(0, 13 if resid else -19), ha="center",
                         fontsize=9.5, color=color[resid], fontweight="bold")

    p18 = sum(curves["plain-18"][1][-10:]) / 10
    p34 = sum(curves["plain-34"][1][-10:]) / 10
    r18 = sum(curves["ResNet-18"][1][-10:]) / 10
    r34 = sum(curves["ResNet-34"][1][-10:]) / 10

    # 两个深度上的 gap 竖线
    for x, lo, hi, gap in ((18, p18, r18, r18 - p18), (34, p34, r34, r34 - p34)):
        ax1.annotate("", xy=(x, hi), xytext=(x, lo),
                     arrowprops=dict(arrowstyle="<->", color="#555", lw=1.3,
                                     shrinkA=8, shrinkB=8))
        ax1.text(x + 0.8, (lo + hi) / 2, f"gap\n{gap:.2f}", fontsize=10,
                 color="#555", va="center", fontweight="bold")

    ax1.set_xticks([18, 34])
    ax1.set_xlabel("网络深度")
    ax1.set_ylabel("末 10 轮 test 精度 (%)")
    ax1.set_title("深度增加的两种命运：plain 退化，ResNet 不退化")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower left", fontsize=10)
    ax1.set_xlim(14.5, 38.5)
    ax1.set_ylim(81.5, 95.2)

    # ---- 右：test 曲线 ----
    for fname, label, depth, resid in RUNS:
        epochs, test = curves[label]
        ax2.plot(epochs, test, linewidth=1.9, color=color[resid],
                 linestyle=("-" if resid else "--"), label=label)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test 精度 (%)")
    ax2.set_title("40 轮训练过程（同数据、同优化器、同超参）")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle("复现 ResNet 论文 Figure 1 · CIFAR-10 · 单变量：有/无 shortcut",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = BASE / "depth_ablation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"已保存 {out}")

    # ---- 顺带打印一张表 ----
    print("\n深度 | plain 末10 | ResNet 末10 | gap")
    for d in (18, 34):
        p = p18 if d == 18 else p34
        r = r18 if d == 18 else r34
        print(f"{d:4d} | {p:9.2f} | {r:11.2f} | {r - p:+.2f}")
    print(f"\nplain  18→34: {p34 - p18:+.2f} 点")
    print(f"ResNet 18→34: {r34 - r18:+.2f} 点")
    print(f"gap    18→34: {r18 - p18:.2f} → {r34 - p34:.2f}  （放大 {(r34 - p34) / (r18 - p18):.1f} 倍）")


if __name__ == "__main__":
    main()
