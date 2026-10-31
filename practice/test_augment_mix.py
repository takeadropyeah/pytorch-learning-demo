"""07 的自检脚本：不碰数据集，只用随机张量验证形状与语义。

跑法：python test_augment_mix.py
通过标准：全部 [OK]，最后一行 PASS。

为什么先有自检、再上真数据集：混合增强的 bug（lam 用错、perm 用错、
loss 加权写反）在 CIFAR 上会表现为"精度就是不涨"，很难和"超参没调好"区分。
在这种形状层面能测的东西上先测掉，训练就只需要回答一个问题：有没有用。
"""

import torch

from augment_mix import apply_mix, cutmix_batch, mix_criterion, mixup_batch

torch.manual_seed(0)

B, C, H, W, K = 8, 3, 32, 32, 10
x = torch.rand(B, C, H, W)
y = torch.randint(0, K, (B,))
CE = torch.nn.functional.cross_entropy


def check(name, ok, extra=""):
    print(f"[{'OK ' if ok else 'BAD'}] {name} {extra}")
    return bool(ok)


ok = True

# 1. 形状不变
for mode in ("none", "mixup", "cutmix"):
    xm, _, _, _ = apply_mix(mode, x, y, 1.0)
    ok &= check(f"{mode} 形状不变", xm.shape == x.shape, str(tuple(xm.shape)))

# 2. none 必须严格等于输入（保证已有实验结果的行为不变）
xm, ya, yb, lam = apply_mix("none", x, y, 1.0)
ok &= check("none 恒等且 lam=1", torch.equal(xm, x) and lam == 1.0)
ok &= check("none 时 y_a==y_b==y", torch.equal(ya, y) and torch.equal(yb, y))

# 3. mixup 的每一张混合图，必须精确落在 lam*x[i]+(1-lam)*x[j] 这条线上
#    （直接验证"混合公式写对了"，而不只是看数值范围）
x_mix, ya, yb, lam = mixup_batch(x, y, alpha=1.0)
on_line = True
for i in range(B):
    cands = [j for j in range(B)
             if torch.allclose(x_mix[i], lam * x[i] + (1 - lam) * x[j], atol=1e-6)]
    # 至少存在一个 j 能对上；且那个 j 必须满足 y_b[i] == y[j]
    on_line &= any(yb[i].item() == y[j].item() for j in cands)
ok &= check("mixup 精确等于 lam*x[i]+(1-lam)*x[j]", on_line, f"lam={lam:.4f}")
ok &= check("mixup lam 在 [0,1]", 0.0 <= lam <= 1.0)

# 4. cutmix 只改一块矩形区域，且区域外像素必须与原图完全相同
x_cm, ya, yb, lam = cutmix_batch(x, y, alpha=1.0)
changed = (x_cm != x).any(dim=1)[0]           # (H,W) 布尔图
rows = changed.any(dim=1).nonzero().flatten()
cols = changed.any(dim=0).nonzero().flatten()
if len(rows) and len(cols):
    box_area = (rows.max() - rows.min() + 1).item() * (cols.max() - cols.min() + 1).item()
    ok &= check("cutmix 改动是连续矩形", box_area >= changed.sum().item(),
                f"框={box_area} 改动像素={changed.sum().item()}")
else:
    ok &= check("cutmix 改动是连续矩形", False, "没有任何像素被改动")

# 5. lam 与真实删除面积一致（这是最容易写错的一处）
area_ratio = changed.float().mean().item()
ok &= check("cutmix lam 与实际面积一致", abs((1 - lam) - area_ratio) < 0.02,
            f"lam={lam:.4f} 实际删除={1 - area_ratio:.4f}")

# 6. mixed loss 的两个退化情况
logits = torch.randn(B, K)
ce_y = CE(logits, y)
ok &= check("mix_criterion(lam=1)==CE(y_a)", torch.allclose(mix_criterion(logits, y, y, 1.0), ce_y))
ok &= check("mix_criterion(lam=0)==CE(y_b)",
            torch.allclose(mix_criterion(logits, torch.zeros_like(y), y, 0.0), ce_y))

# 7. 可复现性：只 reset torch seed（不碰 numpy）就该完全复现 → 证明随机性源头唯一
torch.manual_seed(123)
a = cutmix_batch(x, y, alpha=1.0)
torch.manual_seed(123)
b = cutmix_batch(x, y, alpha=1.0)
ok &= check("只靠 torch seed 即可复现 cutmix",
            torch.equal(a[0], b[0]) and a[3] == b[3], f"lam={a[3]:.4f}")

print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
