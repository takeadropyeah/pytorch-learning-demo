"""低秩近似演示：LoRA 为什么能成立，以及它赌的到底是什么。

用法：
    ..\\.venv\\Scripts\\python.exe .\\tools\\demo_lowrank.py

四个部分，逐段读注释：
    A. 一个满秩矩阵，低秩近似几乎什么都没近似掉  ← 所以不能直接分解 W
    B. 一个真正的低秩矩阵，r 够了误差就是 0      ← ΔW 的形状
    C. 低秩 + 噪声（最接近真实微调增量）
    D. 拿你训好的 mini GPT 里的真实权重，看它的谱长什么样
"""

import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

np.random.seed(0)


# ----------------------------------------------------------------------------
# 两个工具函数
# ----------------------------------------------------------------------------
def singular_values(M: np.ndarray) -> np.ndarray:
    """返回按从大到小排列的奇异值。数值上 = sqrt(eig(M^T M))。"""
    return np.linalg.svd(M, compute_uv=False)


def rel_error(M: np.ndarray, r: int) -> float:
    """用秩 r 近似 M 的相对误差 ||M - M_r||_F / ||M||_F。

    Eckart–Young 定理说：截断 SVD 是所有秩 r 矩阵里最优的那个近似。
    所以这个数就是「秩 r 能做到的最好结果」，不存在更好的低秩矩阵。
    """
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    M_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
    return float(np.linalg.norm(M - M_r) / np.linalg.norm(M))


def effective_rank(M: np.ndarray, energy: float = 0.99) -> int:
    """要保留多少能量（默认 99%）才能重建 M，就需要多少个奇异值。"""
    S = singular_values(M)
    keep = np.cumsum(S ** 2) / np.sum(S ** 2)
    return int(np.searchsorted(keep, energy) + 1)


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ----------------------------------------------------------------------------
# A. 满秩矩阵：低秩近似几乎无效
# ----------------------------------------------------------------------------
banner("A. 随机满秩矩阵 768×768（这就是「直接对 W 做低秩分解」的下场）")

W = np.random.randn(768, 768)
S_W = singular_values(W)

print(f"矩阵元素个数         : {W.size:,}  ({768 * 768:,})")
print(f"最大奇异值           : {S_W[0]:.1f}")
print(f"中位数奇异值         : {np.median(S_W):.1f}")
print(f"（最大 / 中位数只差 {S_W[0] / np.median(S_W):.1f} 倍 —— 谱几乎是平的，没有「几个大头」）")
print()
print("  秩 r  |  参数个数   |  相对误差")
print("  " + "-" * 38)
for r in (4, 16, 64, 256):
    n = 768 * r + r * 768
    print(f"  {r:5d} | {n:9,} | {rel_error(W, r):.1%}")

print()
print(f"要保留 99% 能量，需要 rank = {effective_rank(W)}（几乎等于满秩 768）")
print("→ 结论：**不能**直接拿 W 做低秩分解。省不下来。")


# ----------------------------------------------------------------------------
# B. 真正的低秩矩阵：r 够了就是 0 误差
# ----------------------------------------------------------------------------
banner("B. 真正的低秩矩阵 ΔW = B@A（r_true = 4）")

d, r_true = 768, 4
B = np.random.randn(d, r_true) * 0.01
A = np.random.randn(r_true, d) * 0.01
dW = B @ A          # 秩最多 4

S_dW = singular_values(dW)
print(f"元素个数         : {dW.size:,}")
print(f"非零奇异值个数   : {np.sum(S_dW > 1e-10)}  ← 只有 4 个")
print()
print("  秩 r  |  相对误差")
print("  " + "-" * 22)
for r in (1, 2, 4, 8):
    print(f"  {r:5d} | {rel_error(dW, r):.2e}")

print()
print("→ 结论：r=4 时误差是 0（到浮点精度）。**这就是 LoRA 的形状。**")


# ----------------------------------------------------------------------------
# C. 低秩 + 噪声：最接近真实的微调增量
# ----------------------------------------------------------------------------
banner("C. 低秩 + 独立噪声（更接近真实的 ΔW：有主导子空间，也有杂讯）")

# 噪声强度要小：信号 ||dW||_F ≈ 0.15，这里噪声 ≈ 0.015，约 10%
noise = np.random.randn(d, d) * 2e-5
dW_noisy = dW + noise

print(f"信号 ||ΔW||_F   : {np.linalg.norm(dW):.4f}")
print(f"噪声 ||noise||_F: {np.linalg.norm(noise):.4f}"
      f"   （{np.linalg.norm(noise) / np.linalg.norm(dW):.0%} of 信号）")
print()
print("  秩 r  |  相对误差")
print("  " + "-" * 22)
for r in (1, 2, 4, 8, 16, 64):
    print(f"  {r:5d} | {rel_error(dW_noisy, r):.2%}")

print()
print(f"保留 99% 能量需要 rank = {effective_rank(dW_noisy)}")
print("→ 前几个奇异值吃掉绝大部分能量，尾巴是噪声。r 稍大于 r_true 就够。")


# ----------------------------------------------------------------------------
# D. 你训练出来的真实权重：预训练权重是低秩的吗？
# ----------------------------------------------------------------------------
banner("D. 真实权重：从 checkpoints/gpt_shakespeare.pth 里取一个 Linear 层")

try:
    import torch

    ckpt = torch.load(
        "checkpoints/gpt_shakespeare.pth", map_location="cpu", weights_only=False
    )
    sd = ckpt.get("model", ckpt)

    # 找一个最大的 2D 权重（跳过嵌入层，它本身就有结构）
    cands = [
        (k, v) for k, v in sd.items()
        if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[0] == v.shape[1]
    ]
    if not cands:
        cands = [(k, v) for k, v in sd.items()
                 if isinstance(v, torch.Tensor) and v.dim() == 2]

    name, wt = max(cands, key=lambda kv: kv[1].numel())
    M = wt.float().numpy()
    print(f"取到的层        : {name}")
    print(f"形状            : {M.shape}   元素 {M.size:,}")

    S = singular_values(M)
    print(f"最大 / 最小奇异值: {S[0]:.4f} / {S[-1]:.6f}   比值 {S[0] / S[-1]:.1f}")
    print()
    print("  秩 r   |  相对误差")
    print("  " + "-" * 24)
    for r in (4, 8, 16, 64, 128):
        if r < min(M.shape):
            print(f"  {r:5d}  | {rel_error(M, r):.2%}")

    print()
    print(f"保留 99% 能量需要 rank = {effective_rank(M)} / 上限 {min(M.shape)}")

    print()
    print("→ 关键对照：这个权重是**训练好的 W**，不是 ΔW。")
    print("  它的谱有衰减但远不是「只有几个大头」——所以 W 本身不低秩。")
    print("  LoRA 赌的是**微调增量 ΔW**，不是 W。这两件事别混。")

except Exception as e:  # noqa: BLE001
    print(f"[跳过] 读不到 checkpoint：{type(e).__name__}: {e}")
    print("      先跑一次 `python 11_mini_gpt.py --steps 500 --tag quick` 生成一个。")


# ----------------------------------------------------------------------------
banner("一句话总结")
print("""A：满秩矩阵，r=4 只能省参数、不能近似 → 直接分解 W 行不通
B：纯低秩矩阵，r 够了误差就是 0     → ΔW 的形状
C：低秩+噪声，前几个奇异值吃掉大头 → 真实情形
D：训练好的 W 本身不低秩           → 所以只能适配「增量」

LoRA 的完整赌注：
    微调造成的 ΔW 有效秩很低（论文实测常常 < 8），
    所以用一个 4 倍的秩 r 就能近似它，
    而 W 保持不变、不占可训练参数。
""")
