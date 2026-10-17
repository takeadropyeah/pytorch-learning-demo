"""04 · 第一个卷积神经网络（MNIST 手写数字）

从全连接到卷积。重点不是刷准确率，而是看懂这条数据流：
  图片(1,28,28) -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> 展平 -> Linear -> 10 类

首次运行会自动下载 MNIST（约 10 MB）到 ./data。
CPU 上 1 轮约 30 秒，GPU 上几秒。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------- 0. 设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
if device == "cuda":
    print(f"显卡: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------- 1. 数据
# ToTensor: PIL 图片 -> 张量，并把像素值从 [0,255] 归一化到 [0,1]
# Normalize: 再按 MNIST 的均值/标准差标准化，让训练更稳
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Windows 上 num_workers 用 0 最稳（多进程会有序列化问题）
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

print(f"训练集 {len(train_set)} 张，测试集 {len(test_set)} 张")
print(f"单张图片 shape: {train_set[0][0].shape}  ← (通道, 高, 宽)")


# ---------------------------------------------------------------- 2. 模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 输入 (1, 28, 28)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # -> (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (32, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),          # (32,7,7) -> 32*7*7 = 1568 维
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),       # 随机关掉 30% 神经元，抑制过拟合
            nn.Linear(128, 10),    # 10 个数字类别
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = CNN().to(device)
print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()   # 多分类标配；内部自带 softmax，别再加
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------------------------------------------- 3. 训练 + 评估
def evaluate() -> float:
    """在测试集上算准确率。"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)   # 取分数最高的那个类别
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total


EPOCHS = 2
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)

        loss = criterion(model(xb), yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()
        if (i + 1) % 100 == 0:
            print(f"epoch {epoch}  batch {i + 1:3d}/{len(train_loader)}  "
                  f"loss={running / 100:.4f}")
            running = 0.0

    acc = evaluate()
    print(f"==> epoch {epoch} 结束，测试集准确率 {acc * 100:.2f}%\n")

# ---------------------------------------------------------------- 4. 看一张预测
model.eval()
with torch.no_grad():
    xb, yb = next(iter(test_loader))
    xb, yb = xb.to(device), yb.to(device)
    probs = model(xb[:5]).softmax(dim=1)          # 转成概率
    preds = probs.argmax(dim=1)

print("前 5 张测试图的预测：")
for i in range(5):
    print(f"  真实 {yb[i].item()}  预测 {preds[i].item()}  "
          f"置信度 {probs[i, preds[i]].item() * 100:.1f}%")

print("""
下一步可以自己试：
  1. 把 EPOCHS 改成 5，看准确率还能涨多少
  2. 在 features 里再加一层 Conv，观察参数量变化
  3. 删掉 Dropout，对比训练集和测试集准确率的差距（这就是过拟合）
""")

print("[OK] 第 4 课完成。你已经跑通一个真正的 CNN 了。")
