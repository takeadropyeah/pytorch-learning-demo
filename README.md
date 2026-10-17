# PyTorch 深度学习入门 · 起点工程

这个目录是为「在 PyCharm 里从零学 PyTorch」准备的起点。环境已经装好，你打开就能写代码。

## 一、这台机器的环境（已配好）

| 项目 | 值 |
| --- | --- |
| Python | 3.13.2（`.venv` 虚拟环境，已隔离，不污染系统 Python） |
| PyTorch | 2.14.1 + CUDA 12.6（`torch-2.14.1+cu126`） |
| 显卡 | NVIDIA RTX 4060 Laptop，8 GB 显存 |
| 解释器路径 | `D:\pytorch-learning\.venv\Scripts\python.exe` |
| PyCharm | 2025.3（已安装） |

为什么是 CUDA 12.6 而不是 13.0：你的显卡驱动是 566.24，最高只支持到 CUDA 12.7。装 cu130 的轮子会直接报"驱动版本不足"。

### 装完后的实测结果（都是真跑出来的）

| 项目 | 结果 |
| --- | --- |
| `torch.cuda.is_available()` | `True` |
| 识别到的设备 | `NVIDIA GeForce RTX 4060 Laptop GPU`，8.0 GB，算力 sm_89 |
| 4096×4096 矩阵乘法 | CPU 1900 ms → GPU 44 ms，**43×** |
| `04_mnist_cnn.py` | 2 轮训练，测试集准确率 **98.62%** |
| MNIST 数据集 | 已下载到 `data\`（约 55 MB），不用再下一次 |

`examples\00`～`04` 五个脚本都已实际运行通过，无报错。

## 二、目录结构

```
D:\pytorch-learning\
├── .venv\                 # 虚拟环境，别删，别提交到 git
├── examples\              # 参考实现：按顺序读、按顺序跑，别往里写自己的代码
│   ├── 00_check_env.py    # 先跑这个，确认 GPU 可用
│   ├── 01_tensor_basics.py # 张量：PyTorch 的地基
│   ├── 02_autograd.py     # 自动求导：为什么要手动清零梯度
│   ├── 03_linear_regression.py  # 完整训练循环的骨架
│   └── 04_mnist_cnn.py    # 第一个真正的卷积网络
├── practice\              # ★ 你自己写的练习代码放这里（见 practice\README.md）
├── notebooks\
│   └── quickstart.ipynb   # Jupyter 版，边看结果边试
├── data\                  # 数据集缓存（MNIST 已下好）
├── requirements.txt
└── README.md
```

**新写的代码放哪**：自己在 `practice\` 下建文件，不要改动 `examples\`。`examples\` 是基准线——代码出问题时回去跑一遍，能跑通就说明环境没问题。具体规则见 `practice\README.md`。

**怎么把代码同步到 GitHub**：见 [GIT-WORKFLOW.md](GIT-WORKFLOW.md)，日常就是 `git add -A` → `git commit -m "..."` → `git push`。

## 三、在 PyCharm 里怎么起步

1. **打开项目**：PyCharm → `File` → `Open` → 选 `D:\pytorch-learning`。
2. **设解释器**（关键一步，别跳过）：`File` → `Settings` → `Project: pytorch-learning` → `Python Interpreter` → 齿轮 `Add Interpreter` → `Add Local Interpreter` → 选 `Existing`，路径填
   `D:\pytorch-learning\.venv\Scripts\python.exe`
   设好后右下角状态栏应显示 `Python 3.13 (pytorch-learning)`，并且包列表里能看到 torch 2.14.1。
3. **跑第一个文件**：右键 `examples\00_check_env.py` → `Run`。输出里要能看到 `CUDA 可用: True`。
4. **配 Jupyter**（要用 notebook 时）：打开 `notebooks\quickstart.ipynb`，右上角内核选 `Python 3 (ipykernel)`；如果列表里没有，点 `Select Interpreter` 选择同一个 `.venv` 解释器即可。

## 四、学习路线（建议按这个顺序，别跳）

**阶段 0 · 熟悉工具（1–2 天）**
跑通 `examples\00`，学会：打断点、逐行调试、看变量面板、在 PyCharm 里开 Terminal。调试能力比背 API 重要。

**阶段 1 · 张量（3–5 天）**
`01_tensor_basics.py`。必须手熟：`shape`、`view/reshape`、`squeeze/unsqueeze`、广播、`.to('cuda')`。**每个张量操作的 shape 变化都要能口算出来**，这是后面 debug 的唯一武器。

**阶段 2 · 自动求导（2–3 天）**
`02_autograd.py`。吃透三件事：`requires_grad`、`backward()`、`optimizer.zero_grad()`。三个里错一个，训练就不收敛。

**阶段 3 · 训练循环（1 周）**
`03_linear_regression.py`。把这段背下来，以后所有模型都是它的变形：

```
for epoch in range(epochs):
    for x, y in dataloader:
        pred = model(x)          # 前向
        loss = criterion(pred, y)  # 算损失
        optimizer.zero_grad()      # 清梯度（漏了必错）
        loss.backward()            # 反向
        optimizer.step()           # 更新
```

**阶段 4 · 第一个 CNN（1–2 周）**
`04_mnist_cnn.py`。重点是理解 `Conv2d → ReLU → MaxPool → 展平 → Linear` 这条链，以及训练集/测试集为什么要分开。

**阶段 5 · 系统补课**
到这里再去啃《动手学深度学习》(d2l.ai) 或官方 tutorials，你会读得飞快；顺序反了就是硬背公式。

## 五、几个必踩的坑（提前知道能省几小时）

- **`optimizer.zero_grad()` 忘了**：梯度会累加，loss 不降反涨。
- **`model.train()` / `model.eval()` 不分**：Dropout 和 BatchNorm 在推理时会出错。
- **`loss.backward()` 作用在非标量上**：报 `grad can be implicitly created only for scalar outputs`，要加 `.sum()` 或 `.mean()`。
- **数据没搬到 GPU**：`RuntimeError: Expected all tensors to be on the same device`。
- **显存不够（8 GB）**：先缩 `batch_size`，再考虑混合精度。
- **`pip install torch` 装成 CPU 版**：在 Windows 上直接装 PyPI 的 torch 就是 CPU 版，必须带 `--index-url https://download.pytorch.org/whl/cu126`。

## 六、以后重装/升级要注意

清华、阿里云都镜像了 PyTorch 的 wheel 目录，官方源慢的时候可以换：

```powershell
# 阿里云镜像（实测比官方源快，官方 ~0.8 MB/s，阿里云 ~3.3 MB/s）
pip install torch torchvision torchaudio --index-url https://mirrors.aliyun.com/pytorch-wheels/cu126
```

这台机器上装 torch 那一下约 2.6 GB，慢的话可能等半小时以上，属于正常。

## 七、装环境时如果出问题

整个环境是从零装起来的，重装命令：

```powershell
# 重建虚拟环境
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# GPU 版 PyTorch
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

# 其余工具
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
