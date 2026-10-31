"""演示：python / numpy / torch 是三个互相独立的随机数流。

跑两次这个脚本（两个全新进程），观察哪些数变了。

结论：
    torch.manual_seed(0) 只控制 torch 那一条流。
    python 的 random 和 numpy 的 np.random 各有自己的状态，
    在进程启动时由操作系统熵初始化 —— 所以每次跑都不一样。

这就是为什么 augment_mix.py 只允许用 torch 的随机数：
只要一条流，一个 seed 就能管住全部随机性。
"""

import random

import numpy as np
import torch


def show(tag):
    torch.manual_seed(0)  # 注意：这行只喂 torch，不喂 random / numpy
    print(f"[{tag}]")
    print("  torch :", [round(v, 4) for v in torch.rand(3).tolist()])
    print("  python:", [round(random.random(), 4) for _ in range(3)])
    print("  numpy :", np.round(np.random.rand(3), 4).tolist())


if __name__ == "__main__":
    show("只 seed 了 torch")
    print()

    # 三个都 seed：这才叫"可复现"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    print("[三个流都 seed]")
    print("  torch :", [round(v, 4) for v in torch.rand(3).tolist()])
    print("  python:", [round(random.random(), 4) for _ in range(3)])
    print("  numpy :", np.round(np.random.rand(3), 4).tolist())
