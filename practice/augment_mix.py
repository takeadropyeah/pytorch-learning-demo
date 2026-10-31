"""07 · 混合增强：CutMix 与 MixUp

和 RandomErasing 的区别（这是第 7 课的核心）：
  - RandomErasing：把一块区域涂成噪声，**标签不变**。模型学到"局部缺失也能认"。
  - CutMix        ：把一块区域换成**另一张图**的同位置，**标签按面积混合**。
  - MixUp         ：整张图按比例叠加另一张图，**标签按比例混合**。

后两者都改变了监督信号本身，所以 train_acc 的读数含义也变了。

⚠️ 第 1 课的纪律：随机性有三个源头（python / numpy / torch），三者各自独立的种子流。
本模块**只用 torch 的随机数**，所以 torch.manual_seed(seed) 一种子管到底。
（最初版本用了 np.random.beta，结果同一个 torch seed 两次调用不可复现——自检脚本抓到了。）
"""

import math

import torch
import torch.nn.functional as F


def _sample_lam(alpha):
    """Beta(alpha, alpha) 采样，走 torch 全局 RNG，受 torch.manual_seed 控制。"""
    return float(torch.distributions.Beta(alpha, alpha).sample((1,)).item())


def mixup_batch(x, y, alpha=1.0, generator=None):
    """整图线性混合。

    返回 (x_mixed, y_a, y_b, lam)，loss 应为 lam*CE(pred,y_a)+(1-lam)*CE(pred,y_b)。
    注意 lam 恰好是 **y_a 的权重**，别和 cutmix 里"被挖掉的面积"搞反。
    """
    lam = _sample_lam(alpha)
    perm = torch.randperm(x.size(0), device=x.device, generator=generator)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    return x_mixed, y, y[perm], lam


def cutmix_batch(x, y, alpha=1.0, generator=None, recompute_lam=True):
    """按面积混合：从另一张图抠一块矩形贴过来。

    recompute_lam=False 是为了做一次"故意写错"的对照实验：
    用 Beta 采样出来的 lam 当标签权重，而不是用真实挖掉的面积。
    """
    lam = _sample_lam(alpha)
    perm = torch.randperm(x.size(0), device=x.device, generator=generator)
    x_mixed = x.clone()

    H, W = x.size(2), x.size(3)
    cut_rat = math.sqrt(1.0 - lam)          # 要挖掉的边长比例
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

    # 随机中心点（整个 batch 只切一刀，和原论文一致）
    cx = int(torch.randint(0, W, (1,)).item())
    cy = int(torch.randint(0, H, (1,)).item())
    x1 = int(min(max(cx - cut_w // 2, 0), W))
    x2 = int(min(max(cx + cut_w // 2, 0), W))
    y1 = int(min(max(cy - cut_h // 2, 0), H))
    y2 = int(min(max(cy + cut_h // 2, 0), H))

    x_mixed[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]

    # 关键一步：lam 要用**实际**挖掉的面积重算。
    # 因为 x1/x2 被边界截断过，Beta 采样出的 lam 和真实面积会对不上，
    # 直接用采样值会导致"标签权重"和"像素内容"不一致。
    if recompute_lam:
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x_mixed, y, y[perm], lam


def mix_criterion(pred, y_a, y_b, lam):
    """混合标签的 loss：两个 CE 按 lam 加权。"""
    return lam * F.cross_entropy(pred, y_a) + (1.0 - lam) * F.cross_entropy(pred, y_b)


MIXERS = {"none": None, "mixup": mixup_batch, "cutmix": cutmix_batch}


def apply_mix(mode, x, y, alpha, generator=None, recompute_lam=True):
    """统一入口：mode='none' 时原样返回、lam=1.0，保证不改变原有训练行为。"""
    fn = MIXERS[mode]
    if fn is None:
        return x, y, y, 1.0
    if mode == "cutmix":
        return fn(x, y, alpha=alpha, generator=generator, recompute_lam=recompute_lam)
    return fn(x, y, alpha=alpha, generator=generator)
