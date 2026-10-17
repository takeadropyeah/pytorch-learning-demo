# practice —— 你自己的练习代码放这里

`examples\` 是给你读和跑的参考实现，**不要往里写自己的代码**，理由见下面。你自己的东西全放在这个文件夹里。

## 为什么单独开一个文件夹

1. **`examples\` 是基准线。** 哪天你写的代码不收敛、报错看不懂，回到 `04_mnist_cnn.py` 跑一遍——它能正常跑，就说明是环境没问题、是你新写的代码有问题。如果你已经在 examples 里改得面目全非，这个对照就没用了。
2. **方便回头看进步。** 每个阶段一个文件，一个月后翻出来，能清楚看到自己从"只会复制"到"能改结构"的过程。
3. **`examples\` 可以随时整体删掉重装**，你的练习不会跟着丢。

## 命名规则

用「阶段号_主题」开头，一眼看出属于哪一课：

```
practice\
├── 01_shapes.py            # 对应第 1 课，练 shape 变换
├── 02_grad_manual.py       # 对应第 2 课，手动求导实验
├── 03_my_regression.py     # 对应第 3 课，自己改的超参
├── 04_cnn_variants.py      # 对应第 4 课，加层/删 Dropout 的对比
└── 05_mnist_experiments\   # 复杂了再升级成文件夹（见下）
    ├── model.py
    └── train.py
```

**一个实验一个文件，不要覆盖。** 新手最常见的坏习惯是永远在一个 `test.py` 里反复改，昨天那版跑出 98.6% 的代码今天就找不回来了。

## 什么时候该从「文件」升级成「文件夹」

满足下面任意一条就建文件夹：

- 代码超过 200 行，或者一个文件里塞了三件事（定义模型 + 加载数据 + 训练）
- 想在两个脚本之间复用同一个模型（`from model import MyCNN`）
- 同一个实验要出多个版本的结果、多份数据

否则就老实待在一个 `.py` 里。过早拆分对入门阶段是负担。

## 一个马上会踩的坑：相对路径

`04_mnist_cnn.py` 里写的是 `root="./data"`。PyCharm 运行脚本时，**默认工作目录是脚本所在文件夹**，不是项目根目录。所以你在 `practice\` 里写：

```python
datasets.MNIST(root="./data", ...)   # 数据会下到 practice\data\，而不是项目根的 data\
```

结果就是明明下过一次 MNIST，这里又下一遍（你的网速下这个要两分钟）。

改成用文件位置定位，在哪个目录跑都对：

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # 项目根目录
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"                 # 存模型权重

datasets.MNIST(root=str(DATA_DIR), ...)
```

## 只是试一行代码？

别建文件。用 Jupyter（`notebooks\`）或者 PyCharm 底部的 `Python Console`（`Alt+Shift+E` 附近，或 `View → Tool Windows → Python Console`），试完关掉，不留垃圾。
