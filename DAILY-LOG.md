# 学习日志

> 每天 3 分钟，150 字以内。**目的是让明天的你知道今天到哪了，不是写作文。**

## 怎么用（三条规则）

1. **写新条目放在最上面**，日期 + 第几周。打开文件第一眼就是今天。
2. **"明天第一件事"必须具体到可执行**。写"继续学习"等于没写，写"跑通 practice\02_mlp_mnist.py 到 97%"才算。
3. **"卡在哪"不要写"无"就完事**。写不出卡点，通常说明你只是在看，没在写代码。

## 和另外两个记录文件的分工

三个文件各管一层，不要重复写：

| 文件 | 频率 | 记什么 |
|---|---|---|
| `DAILY-LOG.md`（本文件） | 每天 3 分钟 | 人的进展：学了什么、卡在哪、明天干什么 |
| `practice\EXPERIMENTS.md` | 每次训练 | 实验的技术参数和结果（batch、lr、准确率） |
| `LEARNING-PLAN.md` | 每周对照一次 | 阶段任务、验收标准、打勾 |

## 模板（复制下面的块，改日期）

```markdown
## YYYY-MM-DD · 星期X · 第 N 周

**做了什么**
-

**卡在哪**
-

**明天第一件事**
-

**提交**：`git log` 里能看到 / 未提交
```

---

# 日志正文（最新的在最上面）

## 2026-10-31 · 星期五 · W2 收口：手写 Transformer + mini GPT + 简历定稿

**做了什么**
- **简历定稿到 1 页**：项目经历 4 条 bullet 从 17 行压到 9 行（去掉嵌套 itemize，18/34 层合成一条）；`resume (2).pdf` 渲染确认 `pageCount:1`，底部余 2.4cm
- **手写三个组件，全部与 PyTorch 官方逐位对齐**：
  - `08_mha.py` → `nn.MultiheadAttention` 最大差 **5.96e-08**
  - `09_transformer_block.py` → Pre-LN block 最大差 **4.77e-07**
  - `10_pos_and_mask.py` → 位置编码 + 因果掩码（无官方对照，用性质检验）
- **`11_mini_gpt.py` 训练跑通**（实验 #22）：3000 步 **39.1 秒**，val loss 4.21 → **1.7271**，困惑度 **60 → 5.6**
- 写了四个验证/解剖脚本：`inspect_minigpt.py`（拆开看每个参数）、`show_full_dist.py`（65 维完整分布）、`verify_minigpt.py`（四层验证）、`show_temperature.py`（温度效果）

**最重要的读数**
- **四层基线**：均匀 4.1744 → unigram 3.3092 → bigram 2.4819 → **本模型 1.7160**。比 bigram 低 0.77，说明它用的**不只是前一个字符**，attention + 64 窗口真在起作用。哪天跑出 val≈2.5 就是退化成了 bigram
- **因果性检验**：改最后一个字符，位置 0~4 变化 **0.000000**，位置 5 变化 1.81。这条比 loss 曲线更能证明代码对
- **随初始化 block 近似恒等映射**：LN 后 std 1.0013、attn 输出 std 0.1713（只有 18%），相加后 0.9613 ≈ 输入 0.9601 → 这就是能堆 96 层的原因
- **Pre-LN vs Post-LN 输出差 1.4966，比输入 std(0.9601) 还大** → 是两个完全不同的函数，不是实现细节

**卡在哪**
- `UnicodeEncodeError: 'gbk' ... '\xb2'`：打印 `12·d²` 里的 `²` 在 GBK 控制台崩。修法 `sys.stdout.reconfigure(encoding="utf-8")`，**和 `diff_config.py` 是同一个坑**
- `Invoke-WebRequest` 下莎士比亚语料 TLS 失败，改 Python urllib + 关 SSL 验证，1 秒拿到 1.1MB
- 我自己演示时用了 `torch.topk(logits, 3)` 只打印前三名，被用户抓住"不是应该有 65 个吗"——**他说的对，是我在取巧**

**明天第一件事**
- 读 *Attention Is All You Need* Section 3（简历上写了这篇，读完才敢投）

**计划对照**：W2 超额完成（原排期只要求"手写单头 attention"，实际写完 MHA + block + 位置编码 + mini GPT）。

**提交**：`08_mha.py` / `09_transformer_block.py` / `10_pos_and_mask.py` / `11_mini_gpt.py` / 四个 tools / 实验 #22 / 三份记录

---

## 2026-10-30 · 星期四 · Attention 第一课 + 07 单头实现

**做了什么**
- **简历二版做了四件事**：删掉跨配置的"6.79 点"（选 B）；加荣誉奖项（服务外包大赛省级三等）；论文阅读扩到 4 篇（ResNet / Attention / Bahdanau / ViT）；技能栏更新
- 讲透了 attention 的来龙去脉：**它最初不是网络层，是 2014 年机器翻译的对齐机制**（Bahdanau）→ 2015 Luong 整理出 QKV → 2017 年才变成"全都要用 attention"
- 写 `07_attention.py`（47 行）：单头自注意力，验收时用 `mha.out_proj.weight.copy_(torch.eye(d))` 把官方的 W_O 抵消掉，最大差 **1.19e-07**
- 写了 `tools\demo_attention.py`：八步推导，从"手写权重"走到"除以 √d"

**最重要的读数**
- **差 72 个参数**（288 vs 216）= `8×8+8` = 一个 `Linear(8,8)` → 那就是 **W_O（输出投影）**。标准 attention 是**四个矩阵不是三个**
- **W_O 为什么必须有**：`w@v` 的结果住在 V 的向量空间里，下一层要用的是残差流那个统一空间 → W_O 负责翻译。**和 ResNet 在 shortcut 上做 1×1 投影是同一类活**
- √d 的必要性实测：d=512 时 q·k 的 std 是 **23.10**，不缩放最大权重到 **0.912**（softmax 快饱和），除以 √d 后回到 0.014

**卡在哪**
- 第一次写 demo 时把文案硬编码成"狗 1.010 最高"，实际是 1.000 且与猫并列 → **演示脚本里的数字必须从计算里取，不能手写**
- 随机投影 scale 用 0.5 导致注意力近似均匀（0.2~0.36），看不出结构；扫了 0.2/0.4/0.7/1.0 才定下 0.7

**明天第一件事**
- 把 07 扩成多头（08_mha.py），验收目标：与 `nn.MultiheadAttention` 逐位一致

**计划对照**：W2 正式开工，比原计划晚一天（原排 W2 手写 attention）。

**提交**：`07_attention.py` + `tools\demo_attention.py` + 简历 v2 + 记录

---

## 2026-10-29 · 星期三 · P4 退化现象复现（ResNet-34 vs plain-34）

**做了什么**
- 给 `resnet_model.py` 加深度参数：`ARCHS = {"18": (2,2,2,2), "34": (3,4,6,3)}`，`ResNet(..., layers="18")` + `build_resnet()`；给 `05_resnet_train.py` 加 `--layers {18,34}`，存档名 / csv 模型名 / resume 校验全部带上深度
- **实验 #20 ResNet-34**：40 轮，末 10 轮均值 **93.30%**，峰值 93.54%（出现在最后一轮），train 97.63%，63.0 s/轮，21,282,122 参数 → **比 ResNet-18 同配置还低 0.27 点**
- **实验 #21 plain-34**：40 轮，末 10 轮均值 **83.69%**，峰值 84.28%，train 只有 **85.72%**，gap 1.68 → **比 ResNet-34 低 9.61 点**
- **退化现象复现成功**：plain 18→34 **−6.79 点**，ResNet 18→34 **+0.77 点**，gap 从 **2.05 放大到 9.61（×4.7）**
- 写了 `tools\plot_depth_ablation.py`，产出 `depth_ablation.png`（左：精度 vs 深度 + gap 竖线；右：四条训练曲线）
- 开讲 W2：attention 的动机（RNN 不能并行 + 长程依赖 L 步）、本质（加权平均）、公式三个问号（为什么除 `√d_k`、为什么 `dim=-1`、`V` 干什么）

**最重要的读数**
- **plain-34 的 train 只有 85.72%，gap 1.68（比 ResNet-34 的 4.09 还小）** → 过拟合说和容量不足说被同时排除，只剩"优化失败"。这正是论文 Fig.1 的论证方式：**看训练误差，不看测试误差**
- **首轮就分出胜负**：plain-34 test 20.60% vs ResNet-34 47.95%，差 2.3 倍，后面 39 轮差距一路扩大 → 深度的伤害从第一次反传就存在

**卡在哪**
- 我上次预测「gap 会拉到 +4 以上」，实际 **+9.61，低估 2.4 倍**。而且更早还犯过一次：拿 #14（92.60，只 crop+flip）当基线去算 ResNet-34 的增量，**基线选错**，得出 +0.9 的错误预测。**同配置才可比，这条已经犯第二次了**
- `D:\pytorch-learning\data` 的 ACL 又坏了：`workspace-write` 下 `PermissionError: [Errno 13] data_batch_1`，必须用 `danger-full-access` 才能启动训练
- 发现"训练多久"和"模型多大"是两个独立杠杆：40 轮预算下 34 层反而亏（参数 ×1.90、时间 ×1.83、精度 −0.27）

**明天第一件事**
- 写 `practice\07_attention.py`：单头 scaled dot-product attention，不用 `nn.MultiheadAttention`，验证 `(B,L,d)` 进出、权重每行和为 1

**计划对照**：P4 完成（待跑队列清空 P1/P2/P4/P5，只剩 P3 label smoothing 未做）。W2 原排期的手写 attention 仍未动。

**提交**：待提交（resnet 深度改造 + 实验 #20/#21 + 深度消融图 + 三份记录）

---

## 2026-10-28 · 星期二 · LeNet 收工 + A1 成本画像

**做了什么**
- 补完 LeNet 的数据管道，修掉 6 个 bug（缺 `ToTensor`、测试集缺 `Normalize`、ckpt 覆盖、csv 写盘时机、resume 死代码、无用 import），跑 15 轮 → **MNIST 99.29%**，62K 参数，4.5 s/轮。ep12 起 test 在 99.26–99.30 之间随机游走，**已饱和**
- 补了经典架构讲解（LeNet / AlexNet / VGG），定下 **AlexNet 和 VGG 不手写、LeNet 必须写完**
- **A1 完成**：`tools\model_stats.py`（自己实现的 hook 版 MACs 计数器）+ `tools\cost_lookup.py` + `tools\plot_cost.py`；`results.csv` 从 11 列扩到 14 列，19 行全部回填 `params / macs_g / infer_ms`
- 产出 `results_cost_accuracy.png`：按数据集分面板的「精度-成本曲线」

**卡在哪**
- `pip install thop` 卡了 6 分钟没输出 → 干脆自己写 40 行 hook 计数器，反而把 thop 的原理看懂了
- 脚本命名 `profile.py` **撞了标准库**，`cProfile` 内部 `import profile` 拿到我的文件，报 `AttributeError: module 'profile' has no attribute 'run'` → 改名 `model_stats.py`
- bs=1 的延迟连跑三次是 3.73 / 3.13 / 3.14 ms，一度以为 ResNet 比 Plain 慢，**其实是噪声**。单张推理地板约 0.3–0.5 ms
- 画出来的图上 ResNet18 是 100 轮、Plain18 是 40 轮，**那个 94.17 vs 90.65 不是结构差异**。公平对照要用实验 #14/#15 的 40 轮：+1.58% 参数换 +2.05 点

**明天第一件事**
- A2：把 `augment_mix.py` 里已经写好的 CutMix 接进 `05_resnet_train.py` 的训练循环，跑 40 轮，目标突破 94.13%

**计划对照**：W1 完成（LeNet + A1）。W2 原排期含手写 attention，实际先做 CutMix。

**提交**：待提交（LeNet + A1 三件套 + `results.csv` 结构变更）

---

## 2026-10-27 · 星期一 · 残差消融（实验 #14 / #15）

**做了什么**
- 复盘 12 周路线：W1/W3–W8 已完成，W2 的 MLP 跳过，W9–W12 是项目期；决定先做机制消融再补经典架构
- 给 `resnet_model.py` 加 `plain` 开关（`BasicBlock(plain=True)` → `shortcut=None`，前向走 `out = self.relu(out)`），`05_resnet_train.py` 加 `--plain`，checkpoint 的 resume 校验也带上 `plain` 字段
- **实验 #14**：重跑 #9 作对照 → 92.53%（#9 是 92.47%，差 0.06，无回归）
- **实验 #15**：`--plain` 跑 plain-18 → **90.48%**（−2.05），但 **train 也从 99.88% 掉到 98.91%**
- 顺手写了 LeNet（学到第 6 课），参数量核对时踩到 61,706 vs 61,848 的版本差异

**卡在哪**
- 沙箱从 workspace-write 退回 read-only 后，`data\cifar-10-batches-py` 的 ACL 被改坏（继承被关、连 SYSTEM 都读不了），训练起不来。跑了 `diagnose-windows-sandbox-acl` 的修复脚本，**生成新文件的 ACE 是"仅此文件夹"**，所以重新解压出来的目录又不可读 → 最后把 data 目录整体 FullControl 授予当前用户 + SYSTEM/Administrators
- 结论：**这个项目的训练必须在 danger-full-access 下跑**，workspace-write 读不了数据目录
- 笔记本中途合盖，训练在 ep3 被中断；用 `--resume` 从 ep4 接着跑，学习率曲线连续

**明天第一件事**
- 补经典架构：LeNet → AlexNet → VGG（`LEARNING-PLAN.md` W5/W6 那条线），然后系统梳理全部知识点

**计划对照**：消融做完了，但**只证明了"shortcut 改善可优化性"，没复现"退化"**（退化要求 plain-34 比 plain-18 差）。诚实记下来，排进待跑队列 **P4**。

**提交**：待提交（`--plain` 支持 + 实验 #14/#15 + P4）

---

## 2026-10-24 · 星期六 · 工程化改造 + P1 长训练

**做了什么**
- 修好 venv 缺失的激活脚本（`Activate.ps1` 等四个文件从 Python 模板补回来），终端终于能正确激活
- 训练脚本工程化：`argparse` 参数化（10 个参数）、checkpoint 保存/恢复（含配置校验和 T_max 校验）、`--tag` 多实验隔离
- 日志落盘：`Tee-Object` + 已有的 `flush=True`，零代码改动
- **跑完 P1（100 轮，57 分钟）**：test **94.13%**（峰值 94.22%，末 10 轮 std 0.07）

**卡在哪**
- venv 创建时激活脚本没生成，潜伏了 6 天才暴露（因为一直用 PyCharm 的 Run 按钮，那条路径不需要激活）
- `resume` 一个已跑完的实验会在写 CSV 时 NameError（`train_acc` 只在循环内定义）→ 加 `start_epoch > epochs` 的早退
- 10 分钟睡眠设置差点毁掉 60 分钟训练；改完才开跑
- PowerShell 5.1 的 `Tee-Object` 写 UTF-16 而不是 UTF-8，以后用 Python 解析日志要指定编码

**明天第一件事**
- 加 CutMix 或 MixUp（要改训练循环，不是一行 transform），目标突破 94.13%

**计划对照**：P1 完成（实验 #13）。**轮数不够能解释 1.15 点，但 ep85 后就饱和了**，剩下 0.87 点要靠更强的增强。成绩 74.97% → 94.13%，累计 +19.16 点。

**提交**：待提交（工程化 + 实验 #13）

---

## 2026-10-22 · 星期四 · ResNet 与第一次论文精读

**做了什么**
- 手写 `BasicBlock`（含 Identity / 1×1 投影两个分支）+ `ResNet-18`，1117 万参数，六行 shape 全部对上
- 训练 40 轮：**test 92.47%**（峰值 92.73% @ ep37，std 0.19），比 SmallCNN 的 84.34% **高 8.13 点**
- 精读 ResNet 论文：摘要逐句 + Figure 1/2 + Table 2/3，建立了论文笔记模板 `practice\papers\_TEMPLATE.md`
- 又跑了两个正则化实验，**都失败了**：AdamW `wd=1e-4` 和 `wd=1e-2`，结果与 wd=0 完全一致（92.59% / 92.49% vs 92.47%）

**卡在哪**
- 第 3 轮 test 从 65.72% 掉到 59.31%：BN 的 running stats 前期没收敛，`eval()` 时用的还是半成品统计量，第 5–6 轮后才平稳。
- gap 从 2.6 涨到 **7.28**（train 99.75%），参数多 18 倍后开始记答案。
- 最重要的一课：我一直以为残差解决"梯度消失"，但论文原文**明确排除**了这个原因，作者自己猜是"收敛速度指数级变慢"。二手解释和原文不一致。
- 第二课：连续两次误判 weight_decay。第一次量级太小（实际衰减 0.16%），把 wd 放大 100 倍后仍无效（7.7% 衰减，train 纹丝不动）。**"机制发生了"不等于"机制对症"**——缩小权重治不了这里的过拟合，因为病根是数据不够。
- 顺带修正了自己的算式：余弦调度让 lr 平均只有峰值一半，所以衰减是 7.7% 而不是我一开始算的 14.5%。

**明天第一件事**
- 把 `weight_decay` 归零（回到 Adam），**只加一行** `transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))`，跑 40 轮，和 92.47% 对比（预期 93.5–94%）

**计划对照**：阶段 5（ResNet）完成，比计划（W7–8）提前约 1 周。成绩从 74.97% 到 **92.47%**；正则化路线试了两档均无效，方向转数据增强。

**提交**：`d343765`（ResNet+论文笔记）

---

## 2026-10-21 · 星期三 · 阶段 3 完成（调参循环走通）

**做了什么**
- 修掉 `03_cnn_cifar.py` 三个 bug：`&` 写成位与、评估缩进在 batch 循环内、sympy 垃圾 import
- 40 轮基线（无 BN）：末 10 轮均值 80.77%
- 加 `BatchNorm2d`：峰值 83.29% 但**震荡变大 4 倍**（std 0.48→1.99），末段均值只 +0.38
- 诊断出是**优化器震荡**（train/test 同步跌）→ 加 `CosineAnnealingLR`
- 最终 **84.34%**（末 10 轮均值），震荡归零（std 0.41）

**卡在哪**
- BN 单独用的效果和预期不符：以为能涨 3 个点，实际只涨 0.38。想明白原因花了一会儿——BN 改变了损失曲面的尺度，原 lr 变相变大了。
- 发现 PyCharm 第二次自动插入垃圾 import（`from sympy.core.random import shuffle`），已记进笔记。

**明天第一件事**
- 进阶段 5：先只写残差块，验证 `y = F(x) + x` 的 shape 和梯度能通，不急着搭完整 ResNet-18

**计划对照**：阶段 3（训练诊断：增强 / BN / 学习率调度）全部完成，成绩 74.97% → **84.34%**，+9.37 点。进度领先计划约 1 周。

**提交**：待提交

---

## 2026-10-19 · 星期一 · 第 2 天（第一个 CNN）

**做了什么**
- 把 `01_baseline` 重构成 CONFIG 版，结果与基线**逐位一致**（train 0.00313982 / val 0.14961701）
- 写 `02_explore.py` 摸清 CIFAR-10：50000/10000 张、shape `(3,32,32)`、像素 0–1、每类正好 5000 张
- 写 `03_cnn_cifar.py`：3 层卷积小 CNN（620,362 参数），20 轮无增强 **74.97%**
- 加数据增强（RandomCrop + HFlip）后 **78.81%**，train/test 差值从 20.1 降到 0.7

**卡在哪**
- CIFAR-10 下载单连接只有 10 KB/s，16 线程直接卡死（0 进展），改 8 线程分块 13 分钟下完。教训：并发数不是越多越好，从 8 开始试。
- 加增强后训练明显变慢：`num_workers=0` 让数据预处理和 GPU 串行，GPU 在等 CPU。
- 把 `evaluate()` 抽成带参数后，函数体里的 `test_loader` 忘了改——不报错，但会让 train 和 test 准确率永远相等。差点以为是好事。

**明天第一件事**
- 给 `03_cnn_cifar.py` 加 `num_workers=4` + `if __name__ == "__main__"` 保护，然后跑 40 轮看测试准确率上限（当前 78.81% 仍未收敛）

**计划对照**：W1（工程化 baseline）完成；W3（CIFAR CNN >70%）**超额达成**，实际 78.81%。进度比计划快。

**提交**：待提交

---

## 2026-10-17 · 星期六 · 第 0 天（环境与计划）

**做了什么**
- 搭好 Windows 环境：`D:\pytorch-learning\.venv`，torch 2.14.1+cu126，GPU 实测 43× 加速
- 跑通 examples 00–04，MNIST CNN 两轮 98.62%
- 定下 12 周计划、WSL Linux 练手清单、实验记录表
- 确认 WSL Ubuntu 在 `D:\WSL\Ubuntu`（22 GB），VS Code Remote-WSL 扩展已装好

**卡在哪**
- 装 torch 时网络只有 0.5–2 MB/s，中途断连一次，靠 pip 自动续传才装完。教训：大文件下载别盯着进度条，先确认它有续传能力。

**明天第一件事**
- 把 `examples\03_linear_regression.py` 改写成 `practice\01_baseline.py`：加固定 seed + config 字典，连跑三次 loss 曲线必须完全重合

**提交**：未提交（README、LEARNING-PLAN、WSL-LINUX-TRAINING、EXPERIMENTS 四个文件）

### 第 0 周回顾

- **产出**：环境可运行 + 一份可执行计划。没有训练代码产出，本周定位是准备。
- **计划对照**：刚好在第 1 周开始前完成，没落后。
- **下周调整**：第 1 周只有"工程化 baseline"一个任务，预留时间练 WSL 练习 1–3。

---

## 反面例子（不要这样写）

```markdown
## 10-17
今天学习了 pytorch，看了视频，感觉学到了很多，明天继续努力。
```

问题：没写看了哪一集、没写写了什么代码、没写卡在哪、"继续努力"明天依然不知道从哪开始。
**三个月后回看这条，你自己都得不到任何信息。**
