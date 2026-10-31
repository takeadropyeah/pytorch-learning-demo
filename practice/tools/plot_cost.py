r"""
A1 的画图部分：把 results.csv 的「参数量 / MACs / 延迟」和「精度」放在一张图上。

这是整个 A1 的产出——一条「精度-成本曲线」。

一条硬规则
----------
**只有同一个数据集的模型才能画在一张图上。**

LeNet5 是 MNIST 的 99.29%，ResNet18 是 CIFAR-10 的 92.53%。
把这两个点画在同一条曲线上，会得出「6 万参数的 LeNet 吊打 1100 万参数的 ResNet」
这种完全错误的结论——它们比的不是模型能力，是数据集难度。
所以脚本里按数据集分组，各画一张。

用法
----
    python .\tools\plot_cost.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # 不弹窗口，直接存 png
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results.csv"
OUT = BASE / "results_cost_accuracy.png"

# Windows 上 matplotlib 默认字体没有中文，会变成一串方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 模型 -> 训练用的数据集。新增模型时必须在这里登记，不然会报错
DATASET_OF = {
    "SmallCNN": "CIFAR-10",
    "Plain18": "CIFAR-10",
    "ResNet18": "CIFAR-10",
    "LeNet5-MNIST": "MNIST",
}


def load_best(path):
    """每个模型只取它最好的那次实验——不同轮数的实验不能混着比。"""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    best = {}
    for r in rows:
        name = r["model"]
        if name not in DATASET_OF:
            raise KeyError(f"results.csv 里的 {name!r} 没在 DATASET_OF 登记数据集")
        acc = float(r["test_acc"])
        cur = best.get(name)
        if cur is None or acc > cur["test_acc"]:
            best[name] = {
                "model": name,
                "dataset": DATASET_OF[name],
                "epochs": int(r["epochs"]),
                "test_acc": acc,
                "params": int(r["params"]),
                "macs_g": float(r["macs_g"]),
                "infer_ms": float(r["infer_ms"]),
            }
    return best


def human_num(n):
    """1200 -> 1.2K, 11173962 -> 11.17M"""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(n)


def plot(best):
    datasets = sorted({v["dataset"] for v in best.values()})
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 4.8),
                             squeeze=False)

    for ax, ds in zip(axes[0], datasets):
        pts = sorted([v for v in best.values() if v["dataset"] == ds],
                     key=lambda v: v["params"])
        xs = [p["params"] for p in pts]
        ys = [p["test_acc"] * 100 for p in pts]

        ax.plot(xs, ys, "o-", color="#3b82f6", lw=1.6, ms=9, zorder=3)

        for i, (p, x, y) in enumerate(zip(pts, xs, ys)):
            # 标注里必须带轮数：40 轮和 100 轮的实验不能混着比
            right_side = i != len(pts) - 1
            ax.annotate(f"{p['model']} · {p['epochs']}轮\n"
                        f"{y:.2f}%  {human_num(p['params'])} 参数\n"
                        f"{p['infer_ms']:.2f}ms",
                        (x, y), textcoords="offset points",
                        xytext=(12, 6) if right_side else (-12, 6),
                        ha="left" if right_side else "right",
                        fontsize=8.5, color="#1e3a8a")

        ax.set_xscale("log")
        ax.set_xlabel("参数量（对数轴）")
        ax.set_ylabel("test 准确率 (%)")
        ax.set_title(f"{ds}：精度 vs 成本（每模型取最好一次）")
        ax.grid(alpha=0.25, which="both")
        ax.margins(x=0.30, y=0.28)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"已保存 {OUT}")


if __name__ == "__main__":
    plot(load_best(RESULTS))
