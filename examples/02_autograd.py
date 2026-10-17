"""02 · 自动求导（autograd）

神经网络的训练本质：算出损失 -> 求每个参数的梯度 -> 沿梯度反方向更新参数。
PyTorch 帮你做中间那步。

这一课要彻底搞懂三件事：
  1. requires_grad  —— 谁需要被求导
  2. backward()     —— 从哪开始反向传播
  3. zero_grad()    —— 为什么每轮都要清零

运行：直接 Run。
"""

import torch

print("=" * 60)
print("1. 最小例子：y = x²，求 x=3 处的导数（应该是 6）")
print("=" * 60)

x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x.item()}, requires_grad = {x.requires_grad}")

y = x ** 2
print(f"y = x² = {y.item()}")
print(f"y.requires_grad = {y.requires_grad}  ← 沿计算图自动传播下来了")

y.backward()  # 反向传播，把 dy/dx 算出来存到 x.grad
print(f"y.backward() 之后，x.grad = {x.grad.item()}  ← 2x = 6 ✓")

print("\n" + "=" * 60)
print("2. 梯度会累加！这就是必须 zero_grad 的原因")
print("=" * 60)

x = torch.tensor(3.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第 {i + 1} 次 backward 后 x.grad = {x.grad.item()}")

print("""
看明白了吗？grad 是**累加**的：6 → 12 → 18。
真实训练里每轮都算了新梯度，如果不清零，旧梯度会一直叠加，
优化器就被污染了。所以每轮前必须 optimizer.zero_grad()。
""")

# 正确做法
x = torch.tensor(3.0, requires_grad=True)
for i in range(3):
    x.grad = None  # 等价于 zero_grad() 的简化写法
    (x ** 2).backward()
    print(f"清零后第 {i + 1} 次：x.grad = {x.grad.item()}  ← 每次都是 6 ✓")

print("\n" + "=" * 60)
print("3. 用 autograd 手工做一次梯度下降：拟合 y = 2x + 1")
print("=" * 60)

# 造数据：真实参数就是 w=2, b=1，看能不能学回来
torch.manual_seed(42)
xs = torch.linspace(-1, 1, 100)
ys = 2 * xs + 1 + 0.1 * torch.randn(100)  # 加一点噪声

# 待学习的参数（注意 requires_grad=True）
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(200):
    # 1. 前向：预测
    y_pred = w * xs + b
    # 2. 损失：均方误差
    loss = ((y_pred - ys) ** 2).mean()
    # 3. 清梯度
    w.grad = None
    b.grad = None
    # 4. 反向：算梯度
    loss.backward()
    # 5. 更新参数（要包在 no_grad 里，否则这一步也会被记进计算图）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    if step % 40 == 0:
        print(f"step {step:3d}  loss={loss.item():.4f}  w={w.item():.3f}  b={b.item():.3f}")

print(f"\n学到的 w={w.item():.3f} (真值 2), b={b.item():.3f} (真值 1)")

print("\n" + "=" * 60)
print("4. detach 与 no_grad：什么时候需要「断开」计算图")
print("=" * 60)

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
y_detached = y.detach()  # 共享数据，但不再追踪梯度
print(f"y.requires_grad       = {y.requires_grad}")
print(f"y.detach().requires_grad = {y_detached.requires_grad}")

with torch.no_grad():
    z = x * 3
print(f"no_grad 块里的 z.requires_grad = {z.requires_grad}")

print("""
经验：
  - 推理 / 评估 → 用 torch.no_grad()，省显存也更快
  - 手动改参数（像上面 w -= lr * grad）→ 必须 no_grad，否则报错
  - detach() → 只想拿数值、不想参与反向传播时用
""")

print("[OK] 第 2 课完成。现在你理解 optimizer 为什么长那样了。")
