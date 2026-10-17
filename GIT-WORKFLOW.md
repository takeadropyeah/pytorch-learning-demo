# 日常更新 GitHub 操作手册

> 你现在这套环境已经完全配好了，下面是纯日常操作。

## 当前状态（不用再配置任何东西）

| 项目 | 状态 |
| --- | --- |
| 本地分支 | `main` |
| 远程 | `origin` → `github.com/takeadropyeah/pytorch-learning-demo` |
| 跟踪关系 | `main` ↔ `origin/main`（已建立，`git push` 不用带参数） |
| gh 登录 | 已登录 `takeadropyeah`，token 存在系统钥匙串 |
| 自动忽略 | `data/`、`.venv/`、`checkpoints/`、`runs/`、`install.log`、`__pycache__/`、`.idea/` |

---

## 一、90% 的情况：三条命令

写完代码，在项目根目录敲：

```powershell
cd D:\pytorch-learning
git add -A
git commit -m "完成张量形状变换练习"
git push
```

就完了。`git add -A` 会把所有新增和修改的文件加进来，`git push` 因为有跟踪关系，不需要写 `origin main`。

## 二、敲之前先看一眼（5 秒的习惯）

```powershell
git status
```

- **红色**的 `??` = 新文件，还没被跟踪
- **红色**的 `modified` = 改过但没提交
- **绿色** = 已经 `add` 进去，等着提交

这一步的意义是：**确认列表里没有 `data\`、`.venv\`、`checkpoints\`**。有的话说明 `.gitignore` 漏了规则，先别提交，去补规则。

## 三、提交信息怎么写

这是给你未来自己看的，不是给机器看的。

好的：

```
git commit -m "完成 MNIST CNN 练习，准确率 98.6%"
git commit -m "给网络加一层 Conv，观察参数量变化"
git commit -m "修复梯度不清零导致 loss 不下降的 bug"
```

坏的：

```
git commit -m "update"        ← 三个月后你完全不知道改了什么
git commit -m "111"           ← 同上
git commit -m "改了改"         ← 同上
```

一个练习完成就提交一次。别攒一天二十个改动一次提交——出问题时你没法回退到"还正常的那一刻"。

---

## 四、不想敲命令：在 PyCharm 里点鼠标

| 操作 | 快捷键 | 菜单 |
| --- | --- | --- |
| 提交 | `Ctrl+K` | `Git` → `Commit` |
| 推送 | `Ctrl+Shift+K` | `Git` → `Push` |
| 看改了什么 | `Ctrl+D` | 对比视图 |
| Git 面板 | `Alt+9` | 左下角 Git 页签 |

最常用的一条流程：

1. 改完代码，按 `Ctrl+K`
2. 左边勾选要提交的文件（默认全勾，取消勾 `data` 之类）
3. 中间填提交信息
4. 点 **`Commit and Push...`**（一键完成提交+推送，比分开点省事）

如果只是想提交、稍后再推，点 `Commit`；之后 `Ctrl+Shift+K` 推送。

---

## 五、场景速查表

| 你要做的事 | 命令 |
| --- | --- |
| 看改了哪些文件 | `git status` |
| 看具体改了什么内容 | `git diff` |
| 看已 add 的内容 | `git diff --staged` |
| 只提交某一个文件 | `git add practice/01_shapes.py` |
| 提交信息写错了（还没推） | `git commit --amend -m "新的信息"` |
| 改坏了想放弃（还没提交） | `git restore practice/01_shapes.py` |
| 误把大文件 add 了 | `git rm --cached 文件名` |
| 看历史 | `git log --oneline` |
| 本地和远程差多少 | `git status -sb`（`ahead 2` 就是本地多 2 个提交） |
| 推送被拒绝 rejected | `git pull --rebase origin main` 然后 `git push` |
| 浏览器打开仓库 | `gh repo view --web` |

---

## 六、三条红线

1. **单文件超过 100 MB 会被 GitHub 直接拒绝。** 模型权重（`.pth`）别提交，`checkpoints/` 已经在忽略列表里了。
2. **永远不要用 `git add -f` 强制添加被忽略的目录。** 一旦 `data\` 或 `.venv\` 进了仓库历史，清理起来非常麻烦（要重写历史）。
3. **不要把 token、密钥写进代码。** 你的 gh token 存在系统钥匙串里，本来就不会进仓库；但自己写代码时别把 API key 硬编码进去。

---

## 七、同步到另一台电脑

```powershell
git clone https://github.com/takeadropyeah/pytorch-learning-demo.git
cd pytorch-learning-demo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

数据集不用管，`04_mnist_cnn.py` 里写了 `download=True`，第一次运行会自动下载。这就是当初不上传 `data\` 的原因。
