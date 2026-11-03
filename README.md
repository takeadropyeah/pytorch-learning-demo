# 深度学习复现与消融研究

在单卡消费级 GPU（RTX 4060 Laptop，8 GB）上，从零手写模型并做**受控消融**。

核心工作是在 CIFAR-10 上复现 ResNet 论文（He et al., CVPR 2016）的**退化现象**，并用三条互相独立的读数把它与「过拟合」区分开——论文给出了结论，但把判定过程留给了读者。

- 研究简报（一页纸）：[`research-brief.md`](research-brief.md)
- 实验台账：[`practice/EXPERIMENTS.md`](practice/EXPERIMENTS.md) · [`practice/results.csv`](practice/results.csv)
- 中文笔记：[`practice/NOTES.md`](practice/NOTES.md)

---

## 一、核心结果：退化现象可以在 8 GB 笔记本上稳定复现

四个配置严格同构，唯一变量是**有没有残差连接**——同一个 `BasicBlock` 里一个 `plain` 开关控制：`plain=True` 时 `self.shortcut = None`，`forward` 直接返回 `self.relu(out)`。因此同深度两组的层数、通道、kernel 大小、BN 位置逐层一致。

40 轮，单卡 RTX 4060 Laptop 8 GB，同一份 CIFAR-10（5 万训练 / 1 万测试）、同一优化器（Adam）、同一初始学习率（1e-3）、同一余弦退火、同一 batch（128）。

| 配置 | 层数 | 残差 | 参数量 | 训练集 | 测试集 | train − test |
|---|---|---|---|---|---|---|
| plain-18 | 18 | 无 | 11,000,138 | 98.91% | 90.48% | 8.43 |
| **ResNet-18** | 18 | 有 | 11,173,962 | 99.88% | **92.53%** | 7.35 |
| **plain-34** | 34 | 无 | 21,108,298 | **85.72%** | **83.69%** | 2.03 |
| **ResNet-34** | 34 | 有 | 21,282,122 | 97.63% | **93.30%** | 4.33 |

> 口径：训练集取最后一轮，测试集取最后 10 轮均值（单轮波动约 ±0.3 点）。

**加深 16 层的结果是不对称的**：无残差 −6.79 点，有残差 +0.77 点。

而残差只引入 **173,824 个参数**（+1.58%），这个数字在 18 层和 34 层**完全相同**——因为它只来自三个 stage 通道数变化处的 1×1 投影，恒等分支本身零参数。这是「结构消融而非容量消融」的硬证据。

### 三条独立证据链：为什么是「优化失败」而不是「过拟合」

同一个现象可以由不同机制产生，必须能区分。过拟合与优化失败的解法完全相反（前者加正则，后者加结构）。

| # | 观察 | 过拟合会怎么预测 | 实测 |
|---|---|---|---|
| 1 | plain-34 训练集准确率只有 **85.72%** | 过拟合的模型应该把训练集拟合到接近 100% | 训练集都没拟合上 |
| 2 | plain-34 的 train − test 只有 **2.03**，比 ResNet-34（4.33）**更小** | 过拟合的标志是 gap 大 | gap 小恰恰说明不是泛化问题 |
| 3 | 第一个 epoch 就落后 **17.9–27.4 点** | 过拟合是「练久了才开始」 | 退化从训练一开始就存在 |

第 3 条做了 seed 加固：4 个 seed 下，plain-34 首轮测试精度区间 **20.60–25.98%**，ResNet-34 为 **43.86–47.95%**，两区间**不重叠**；最不利配对仍差 **17.88 点**。所以这不是初始化运气。

同配置同种子跑两遍，实测差 **0.06 点**——这是本项目的噪声底，也是判断任何增益是否可信的尺子。

---

## 二、仓库里有什么

### 1. 受控消融

- **结构消融**：`plain` 开关，18/34 层 × 有/无残差，四组对照（见上）
- **单变量消融链**：SmallCNN 84.53% → ResNet-18 92.53%，逐项定位 RandomErasing +0.45、CIFAR 归一化 +0.27、CutMix +0.32、延长至 100 轮 +1.50，累积到 **95.07%**
- **一个负结果**：参数匹配的 ViT 在同一数据管道下从头训练，200 轮后训练集追平 ResNet-18（98.51% vs 98.45%）但测试集差 **16.24 点**，补 CutMix 后压缩到 **10.38 点**——指向归纳偏置差异，而非训练预算不足

![深度消融](practice/depth_ablation.png)

### 2. 手写 Transformer 全栈

不调库，逐层实现，并与 PyTorch 官方实现**逐位对齐**：

| 文件 | 内容 | 与官方最大误差 |
|---|---|---|
| `07_attention.py` | 单头自注意力 | 1.19e-07 |
| `08_mha.py` | 多头自注意力 + 掩码 | 5.96e-08 |
| `09_transformer_block.py` | 完整 Pre-LN Block（MHA + FFN + 残差 + LN） | 4.77e-07 |
| `10_pos_and_mask.py` | 正弦位置编码 + 因果掩码 | 置换等变 / 因果性实证 |
| `11_mini_gpt.py` | char-level GPT，莎士比亚语料 | val loss 1.73（均匀基线 4.17，bigram 2.48）|

`12_vit_cifar.py` 直接复用 `09` 的 Block，一行未改——Transformer Block 与模态无关。

### 3. 自建评测工具

| 工具 | 作用 |
|---|---|
| `tools/model_stats.py` | 不依赖第三方库，用前向 hook 统计参数量 / MACs / CPU 推理延迟 |
| `tools/diff_config.py` | 对比两个 checkpoint 的完整 config，自动标出「实验变量」与「版本差异」，防止把两次不同条件的实验当成单变量对照 |
| `tools/plot_depth_ablation.py` | 复现论文 Figure 1 的深度消融图 |
| `tools/demo_*.py` | RNG / seed 隔离演示、注意力权重可视化、CutMix λ 分布 |
| `tools/verify_*.py` | mini GPT 的四层基线校验（均匀 / unigram / bigram / 本模型）|

`diff_config.py` 是这套流程里最实用的一件：每次训练把完整配置存进 checkpoint，对比时逐项 diff。

![精度-成本](practice/results_cost_accuracy.png)

---

## 三、复现核心结果

四个配置，各约 20 分钟（RTX 4060 Laptop）。

```powershell
cd practice

python .\05_resnet_train.py --epochs 40 --layers 18                --tag ablation_resnet18
python .\05_resnet_train.py --epochs 40 --layers 18 --plain        --tag ablation_plain18
python .\05_resnet_train.py --epochs 40 --layers 34                --tag p4_resnet34
python .\05_resnet_train.py --epochs 40 --layers 34 --plain        --tag p4_plain34
```

ViT 对照（各约 62 分钟）：

```powershell
python .\12_vit_cifar.py --epochs 200 --d 192 --layers 4 --heads 6 --tag vit_small_200ep
python .\12_vit_cifar.py --epochs 200 --d 192 --layers 4 --heads 6 `
    --mix cutmix --mix-alpha 1.0 --mix-prob 0.5 --tag p6_vit_cutmix_200ep
```

手写 Transformer 各组件的自检（不需要 GPU，几秒跑完）：

```powershell
python .\07_attention.py          # 与 nn.MultiheadAttention(d, 1) 逐位对齐
python .\08_mha.py                # 多头对齐（含总参数量不变性验证）
python .\09_transformer_block.py  # Pre-LN / Post-LN 对比
python .\10_pos_and_mask.py       # 置换等变、位置编码、因果性
```

---

## 四、实验台账怎么读

`practice/results.csv` 有 **29 条**记录，一行一次训练：

| 列 | 含义 |
|---|---|
| `epochs / batch / lr / workers` | 训练预算与并行度 |
| `augment` | 数据增强管线，取值见 `EXPERIMENTS.md` 开头的对照表 |
| `bn / cosine / weight_decay` | 归一化、调度、正则 |
| `model` | 模型标识（ViT 带 `-w5` 后缀表示 5 轮 warmup） |
| `train_acc / test_acc` | **最后一轮**准确率（研究简报中用最后 10 轮均值） |
| `params / macs_g / infer_ms` | 参数量 / 十亿次乘加 / CPU 单张推理延迟 |

`macs_g` 统计的是 **MACs 而非 FLOPs**（1 MAC = 2 FLOPs），且只计前向。**FLOPs 不等于实测延迟**——本仓库同时记录两者，正是因为二者会给出不同的结论。

---

## 五、已知局限

主动列出，因为「知道自己的实验哪里不够硬」比数字本身重要：

- **主体实验只有 1 个 seed**（同配置复跑实测差 0.06 点）。只有首轮读数做了 4 seed 加固。
- **跨深度对比不是严格单变量**：18 层两组用 `crop+flip`，34 层两组用 `crop+flip+erase+norm+cutmix`，增强同时变了。**同深度内部的有无残差对比是干净的**；跨深度只作趋势，不作因果。
  - 反向推断：34 层增强更强、按理只会提升泛化，它仍然掉了 6.79 点，所以退化的方向性结论只会被低估。
- **只跑了 40 轮**：plain-34 的训练集准确率在 40 轮时仍在缓慢上升，没有区分「优化不动」与「优化得慢」。
- **只有 CIFAR-10**，未在 ImageNet 尺度验证。
- **ViT 对照**同时改变了预训练数据、轮数、增强、优化器与 dropout，无法把差距单独归因于归纳偏置。

---

## 六、环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop，8 GB（sm_89）|
| PyTorch | 2.14.1 + CUDA 12.6 |
| Python | 3.13.2（`.venv`）|
| 驱动 | 566.24（最高支持 CUDA 12.7，故装 cu126 而非 cu130）|

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CIFAR-10 与 MNIST 首次运行时自动下载到 `data/`。

---

## 七、目录

```
.
├── research-brief.md          # 一页纸研究简报
├── practice/
│   ├── 01_baseline.py         # 可复现基线（配置抽离 + 固定 seed）
│   ├── 03_cnn_cifar.py        # SmallCNN on CIFAR-10
│   ├── resnet_model.py        # ResNet-18/34，含 --plain 消融开关
│   ├── 05_resnet_train.py     # 工程化训练脚本（argparse / checkpoint / 日志落盘）
│   ├── 06_lenet.py            # LeNet-5 on MNIST（99.29%）
│   ├── 07–11_*.py             # 手写 Transformer 全栈 + mini GPT
│   ├── 12_vit_cifar.py        # ViT on CIFAR-10（复用 09 的 Block）
│   ├── augment_mix.py         # CutMix / MixUp（按实际挖块面积重算 λ）
│   ├── tools/                 # 参数量/MACs/延迟、config diff、各类演示脚本
│   ├── EXPERIMENTS.md         # 29 组实验的技术参数与结论
│   ├── NOTES.md               # 13 课笔记（含论文精读）
│   └── results.csv            # 实验台账
└── examples/                  # 从零学 PyTorch 的参考实现（00–04）
```

`LEARNING-PLAN.md`、`DAILY-LOG.md`、`WSL-LINUX-TRAINING.md` 是学习过程记录，与上述研究内容无关。

---

## 八、作者

潘豪 · 电子科技大学 软件工程（2024 级）
181-5058-9223 · haopan1115@gmail.com

课题方向：图像复原与底层视觉。目前正在读 DnCNN / DPIR 系列，计划在消费级硬件上跑通 PnP-HQS 框架并做先验替换的对照实验。
