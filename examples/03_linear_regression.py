"""03 · 完整训练循环（把这段背下来）

上一课用手工算梯度。这一课换成标准写法：nn.Module + DataLoader + optimizer。
所有 PyTorch 训练代码都是这个结构的变体。

任务：用两层小网络拟合一个非线性函数 y = sin(x) + 0.5x
运行：直接 Run，几秒钟出结果，CPU 够用。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------- 0. 设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ---------------------------------------------------------------- 1. 造数据
torch.manual_seed(0)
N = 2000
x = torch.linspace(-3, 3, N).unsqueeze(1)          # shape=(N, 1)  ← 特征
y = torch.sin(x) + 0.5 * x + 0.05 * torch.randn(N, 1)   # shape=(N, 1) ← 标签

# 划分训练集 / 验证集（8:2）—— 永远要分开，否则你不知道有没有过拟合
split = int(N * 0.8)
train_ds = TensorDataset(x[:split], y[:split])
val_ds = TensorDataset(x[split:], y[split:])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)   # 训练集要打乱
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)     # 验证集不用

print(f"训练样本 {len(train_ds)}，验证样本 {len(val_ds)}")

# ---------------------------------------------------------------- 2. 定义模型
class MLP(nn.Module):
    """两层全连接 + 激活函数。nn.Module 是所有模型的基类。"""

    def __init__(self):
        super().__init__()  # 必须调用，否则参数注册不上
        self.net = nn.Sequential(
            nn.Linear(1, 64),   # 输入 1 维 -> 隐藏 64 维
            nn.ReLU(),          # 非线性！去掉它这网络只能拟合直线
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),   # 输出 1 维
        )

    def forward(self, x):
        """前向传播。PyTorch 会自动调用它，不要自己手动调 forward()。"""
        return self.net(x)


model = MLP().to(device)   # ← 模型和数据都要搬到同一个设备
print(f"\n模型结构:\n{model}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------- 3. 损失 + 优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# ---------------------------------------------------------------- 4. 训练循环
EPOCHS = 100
print(f"\n{'epoch':>6} {'train_loss':>12} {'val_loss':>12}")

for epoch in range(1, EPOCHS + 1):
    # ---- 训练阶段
    model.train()                     # 告诉 PyTorch：现在在训练（影响 Dropout/BN）
    total = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)              # ① 前向
        loss = criterion(pred, yb)    # ② 算损失

        optimizer.zero_grad()         # ③ 清梯度（顺序很关键：在 backward 之前）
        loss.backward()               # ④ 反向传播
        optimizer.step()              # ⑤ 更新参数

        total += loss.item() * xb.size(0)
    train_loss = total / len(train_ds)

    # ---- 验证阶段
    model.eval()                      # 切换推理模式
    with torch.no_grad():             # 不用建计算图，省显存
        val_total = 0.0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_total += criterion(model(xb), yb).item() * xb.size(0)
    val_loss = val_total / len(val_ds)

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:6d} {train_loss:12.6f} {val_loss:12.6f}")

# ---------------------------------------------------------------- 5. 检查结果
model.eval()
with torch.no_grad():
    probe = torch.tensor([[-2.0], [0.0], [2.0]], device=device)
    pred = model(probe).cpu().squeeze().tolist()
    truth = (torch.sin(probe.cpu()) + 0.5 * probe.cpu()).squeeze().tolist()

print(f"\n{'x':>6} {'预测':>10} {'真值':>10}")
for px, pp, pt in zip([-2.0, 0.0, 2.0], pred, truth):
    print(f"{px:6.1f} {pp:10.4f} {pt:10.4f}")

print("""
这个循环记住了，以后换成图像、文本、音频，骨架完全不变，
变的只有三样东西：Dataset、模型结构、损失函数。
""")

print("[OK] 第 3 课完成。你已经会写训练循环了。")
