"""01 · 张量基础

PyTorch 里的一切都是张量（Tensor）。这一课的目标只有一个：
看到任何一个张量，你都能立刻说出它的 shape 和 dtype。

运行：直接 Run 即可，不需要 GPU。
"""

import torch

print("=" * 60)
print("1. 创建张量")
print("=" * 60)

# 从 Python 列表创建
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"a = {a}")
print(f"  shape={a.shape}  dtype={a.dtype}  device={a.device}  ndim={a.ndim}")

# 常用工厂函数
print(f"全 0        : {torch.zeros(2, 3).tolist()}")
print(f"全 1        : {torch.ones(2, 3).tolist()}")
print(f"单位矩阵     : {torch.eye(3).tolist()}")
print(f"随机(0,1)    : shape={torch.rand(2, 3).shape}")
print(f"标准正态     : shape={torch.randn(2, 3).shape}")
print(f"等差序列     : {torch.arange(0, 10, 2).tolist()}")
print(f"未初始化     : shape={torch.empty(2, 3).shape}")

# 指定 dtype —— 深度学习里最常用的是 float32
print(f"\nfloat32 : {torch.tensor([1, 2], dtype=torch.float32).dtype}  ← 深度学习默认用这个")
print(f"float64 : {torch.tensor([1.0, 2.0], dtype=torch.float64).dtype}")
print(f"int64   : {torch.tensor([1, 2]).dtype}  ← 注意：整数默认是 int64")

print("\n" + "=" * 60)
print("2. 形状变换（最容易出错的地方，重点看）")
print("=" * 60)

x = torch.arange(24)
print(f"x                : shape={x.shape}")

y = x.view(4, 6)  # 把 24 个元素排成 4×6
print(f"x.view(4, 6)     : shape={y.shape}")

z = y.reshape(2, 3, 4)  # 继续重排成三维
print(f"y.reshape(2,3,4) : shape={z.shape}")

# view 要求内存连续，reshape 更宽容。不确定就用 reshape。
print(f"\nunsqueeze(0)     : {torch.tensor([1, 2, 3]).unsqueeze(0).shape}  ← 加一个「行」维度")
print(f"unsqueeze(1)     : {torch.tensor([1, 2, 3]).unsqueeze(1).shape}  ← 加一个「列」维度")
print(f"squeeze()        : {torch.zeros(1, 3, 1, 5).squeeze().shape}  ← 去掉所有大小为 1 的维度")

t = torch.zeros(2, 3, 4)
print(f"\n转置 t(2,3,4).transpose(1,2) -> {t.transpose(1, 2).shape}")
print(f"permute(2,0,1)               -> {t.permute(2, 0, 1).shape}")

print("\n" + "=" * 60)
print("3. 广播（broadcasting）")
print("=" * 60)
# 规则：从右往左对齐，维度相等 或 其中一个是 1 才能广播
m = torch.ones(3, 4)
v = torch.tensor([10.0, 20.0, 30.0, 40.0])  # shape=(4,)
print(f"m(3,4) + v(4,) = shape {(m + v).shape}")
print((m + v))  # 每一行都加上了 v

col = torch.tensor([[1.0], [2.0], [3.0]])  # shape=(3,1)
print(f"\ncol(3,1) + v(4,) = shape {(col + v).shape}")
print(col + v)

print("\n" + "=" * 60)
print("4. 索引与切片（和 NumPy 几乎一样）")
print("=" * 60)
t = torch.arange(12).reshape(3, 4)
print(t)
print(f"\nt[0]      -> {t[0].tolist()}         整行")
print(f"t[:, 1]   -> {t[:, 1].tolist()}      整列")
print(f"t[1:, 2:] -> \n{t[1:, 2:]}")
print(f"t[t > 5]  -> {t[t > 5].tolist()}     布尔索引")

print("\n" + "=" * 60)
print("5. 和 NumPy 互转")
print("=" * 60)
import numpy as np

arr = np.array([1.0, 2.0, 3.0])
tt = torch.from_numpy(arr)
print(f"numpy -> tensor : {tt}")
print(f"共享内存吗？改 arr 试试：", end="")
arr[0] = 999.0
print(f"tensor 现在是 {tt}")  # 会变！from_numpy 共享内存

print(f"tensor -> numpy : {tt.numpy()}")

print("\n" + "=" * 60)
print("6. 搬到 GPU")
print("=" * 60)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前可用设备：{device}")

a = torch.randn(3, 3)
print(f"在 CPU 上 : device={a.device}")
if torch.cuda.is_available():
    b = a.to("cuda")  # 常用写法；也可以用 a.cuda()
    print(f"搬到 GPU 后: device={b.device}")
    try:
        a + b  # CPU 和 GPU 的张量不能直接运算
    except RuntimeError as e:
        print(f"[!] 混合设备会报错：{str(e).splitlines()[0]}")
    print(f"搬回 CPU  : device={(b.to('cpu')).device}")

print("\n[OK] 第 1 课完成。现在你应该能说出任意张量的 shape。")
