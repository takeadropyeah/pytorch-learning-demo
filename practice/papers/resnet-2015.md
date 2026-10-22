# ResNet（2015）· 阅读笔记

> 读了第一遍 + 第二遍前半。基于 arXiv 原文核实，未经核实的已标注。

## 元信息

| | |
|---|---|
| 标题 | Deep Residual Learning for Image Recognition |
| 作者 / 机构 | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun（微软亚洲研究院） |
| 年份 / 会议 | arXiv 2015-12-10；CVPR 2016 最佳论文 |
| 链接 | https://arxiv.org/abs/1512.03385 |
| 引用量 | 20 万+ |
| 读它的原因 | 手写第一个真正的深层网络；它是现代所有网络的骨架（Transformer 也是 `x + sublayer(x)`） |

---

## 第一遍 · 四要素

| 要素 | 一句话 |
|---|---|
| **要解决什么问题** | 深网络难训练：层数增加后**训练误差反而变高**（退化问题），不是过拟合 |
| **方法核心** | 把层改成学残差：`y = F(x) + x`，让"学恒等映射"变成"学零" |
| **效果** | 152 层，比 VGG 深 8 倍但复杂度更低（34 层基线只有 VGG-19 的 18% FLOPs）；ImageNet 集成错误率 3.57%，ILSVRC 2015 冠军 |
| **局限 / 没看懂的** | 退化现象的**根本原因**论文自己也没定论（原文："will be studied in the future"） |

### 摘要里三个要警惕的点

1. **"3.57%" 是 ensemble，不是单模型** —— 摘要里挑的是最好看的数字，单模型的条件在正文 Table 4。以后引用必须说清。
2. **"8× deeper but lower complexity"** —— 少的是**参数和 FLOPs**，因为 VGG 的参数量几乎全在最后三个全连接层（VGG-16 约 1.38 亿参数）。
3. **"easier to optimize" ≠ "higher accuracy"** —— 论文的主张是"容易优化"，准确率是结果，不是出发点。

### 必看的图

- **Figure 1**：CIFAR-10 上 plain-20 vs plain-56 的训练误差（左）和测试误差（右）。
  **核心信息：56 层在训练误差上就比 20 层更差。** 这一张图同时否掉了"过拟合"和"容量不足"两个解释——因为过拟合的话训练误差该更低，容量不足的话 56 层容量更大。
- **Figure 2**：一个 building block。就是我写的 `BasicBlock`：`x` 走 shortcut，`F(x)` 走两层卷积，在第二个 BN 之后相加，**相加后再过一次 ReLU**。
- **Figure 3**：VGG-19 / plain-34 / ResNet-34 三张架构图并排。虚线 shortcut = 需要改维度的那几个。
- **Figure 4**：ImageNet 上的同类对比，是 Figure 1 在另一数据集上的复现。

---

## 第二遍 · 论证链条（七步）

| 步骤 | 论文里的位置 |
|---|---|
| ① 观察现象 | Figure 1 / Section 1：56 层训练误差 > 20 层 |
| ② 排除其他解释 | 训练误差更高 → 不是过拟合；层更多 → 不是容量不足 |
| ③ 理论推导 | Section 1："深层网络至少应该和浅层一样好"（多出的层学成恒等即可） |
| ④ 定位病因 | 存在好解（恒等映射），但优化器找不到它 → **优化问题** |
| ⑤ 提出方法 | `y = F(x) + x`，把目标从"学恒等"变成"学零" |
| ⑥ 验证方法 | Table 2 / Figure 4：同深度同参数下，residual 版训练误差更低 |
| ⑦ 消融实验 | Table 3：三种 shortcut 形式对比 |

**这套七步是 CV 论文的标准模板。** 以后读任何论文先找这七步，缺哪步就是它最容易被攻击的地方。

---

## 关键实验表（数字已核实）

### Table 2 · ImageNet validation，top-1 错误率（%，10-crop）

| | plain（无残差） | ResNet |
|---|---|---|
| 18 层 | 27.94 | 27.88 |
| 34 层 | **28.54 ↑** | **25.03 ↓** |

**三个读法：**

1. plain 网络 18→34 层，错误率**上升**（27.94 → 28.54）→ 退化确实存在
2. ResNet 18→34 层，错误率**下降**（27.88 → 25.03）→ 退化被解决
3. 34 层 ResNet vs plain-34：**降低 3.5 个点**，而且参数量完全相同（这一版用零填充 shortcut，**零额外参数**）→ 最干净的控制变量对比

**一个反直觉的点**：18 层时 plain 和 ResNet 几乎一样（27.94 vs 27.88）。论文解释：网络"不够深"时普通网络照样能训好，**残差只在深到训不动时才显出价值**。

→ 这解释了我自己的经历：SmallCNN 只有 3 个卷积层，加不加残差都无所谓。

### Table 3 · shortcut 类型消融（top-1 / top-5）

| model | top-1 | top-5 |
|---|---|---|
| VGG-16 | 28.07 | 9.33 |
| plain-34 | 28.54 | 10.02 |
| ResNet-34 A（**零填充** shortcut） | 25.03 | 7.76 |
| ResNet-34 B（**1×1 投影** shortcut） | 24.52 | 7.46 |
| ResNet-34 C（**全部**用投影） | 未抓到（PDF 第 6 页） | 未抓到 |

**结论**：A 和 B 只差 **0.51 点**。论文说"the small differences between A/B/C indicate that **projection shortcuts are not essential**"。

> 真正起作用的是"**有**捷径"这件事，而不是"捷径怎么对齐形状"。

论文对 A 略差的解释：零填充补出来的那些通道 **"have no residual learning"**——shortcut 信号恒为 0，等于那部分只有主路径在学。

### 三种 shortcut 的准确定义

| 方案 | 形状变化时 | 参数量 |
|---|---|---|
| A | 零填充补通道，然后直接加 | 0 |
| B | 1×1 卷积投影 | 有 |
| C | 所有 shortcut 都用 1×1 投影 | 最多 |

**注意**：形状一变，"纯恒等映射"物理上就不存在了（64 通道加不到 128 通道上）。所以 A 不是"纯恒等"，而是"零填充 + 恒等"。

---

## 和我的代码的对应关系

| 论文 | 我的代码 |
|---|---|
| Figure 2 的 building block | `class BasicBlock` |
| Eqn (1) `y = F(x) + x` | `out = self.relu(out + self.shortcut(x))` |
| 恒等 shortcut（实线） | `self.shortcut = nn.Identity()` |
| 投影 shortcut（虚线，只用于改维度） | `nn.Sequential(Conv2d(in, out, 1, stride), BatchNorm2d(out))` |
| 选择方案 B（只在需要时投影） | 我写的 `if stride != 1 or in_ch != out_ch:` |
| BasicBlock（18/34 层用） | ✅ 已实现 |
| Bottleneck（50/101/152 层用） | ⬜ 还没写 |
| Global average pooling 替代大 FC | `nn.AdaptiveAvgPool2d(1)` + `nn.Linear(512, 10)` |

### 论文的训练配置（Section 3.4，已核实）

```
优化器     SGD, momentum 0.9
学习率     0.1 起步，误差平台时 ÷10
batch      256
weight decay 0.0001
迭代次数   最多 60×10⁴
正则       不用 dropout（用 BN 代替）
BN 位置    每个卷积之后、激活之前
初始化     何恺明初始化（He init）
```

⚠️ **和我现在的配置差别很大**：我用 Adam lr=1e-3 + 余弦退火。这是"小数据集 + 短训练"的现代常见做法，不是论文原配方。**如果哪天要严格复现论文数字，就得换成上面这套。**

---

## ⚠️ 二手解释 vs 原文（这篇最重要的收获）

| 解释 | 来源 | 和原文一致吗 |
|---|---|---|
| 恒等映射容易学：把"学 x"变成"学 0" | Section 3.1，**论文自己的核心论证** | ✅ 一致 |
| 残差连接给梯度修了高速路，解决梯度消失 | 业界流行说法 | ❌ **论文明确排除** |
| 等效于 2ⁿ 条路径的隐式集成 | Veit et al. 2016，后续工作 | 📖 不是原文观点 |

**关于"梯度消失"，论文原文（Section 4.1）写得非常明确：**

> "this optimization difficulty is **unlikely** to be caused by vanishing gradients. These plain networks are trained with BN, which ensures forward propagated signals to have non-zero variances. We also verify that the backward propagated gradients exhibit healthy norms with BN. **So neither forward nor backward signals vanish.**"

论文自己的猜想是 **"exponentially low convergence rates"（收敛速度指数级变慢）**，并且承认"原因有待未来研究"。

**这就是读一手论文最大的价值：你会发现"大家常说的解释"和"作者说的解释"不一致。**

---

## 我的疑问（待补）

- [ ] 既然 BN 让梯度健康，那"收敛速度指数级变慢"具体是怎么发生的？有没有更严格的解释？
- [ ] Table 4 的完整数据（ResNet-50/101/152 单模型 vs 集成）还没看
- [ ] Section 4.2 的 CIFAR-10 实验（20/32/44/56/110 层那张表）还没读——**那张表和我自己的实验直接对应**
- [ ] 为什么 1000 层的 ResNet 反而变差？（Section 4.2 有讨论）

---

## 一句话总结（自己的话）

> 深层网络的问题不是"装不下"，是"训不动"——存在一个好解（恒等映射），但优化器走不到。残差连接把"走到恒等映射"这件事从"要精确调出一堆权重"变成"把权重推向零"，于是路就通了。
